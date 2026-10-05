//! Pure 32-bit (protected mode) instruction decoder for the interpreter's
//! decode cache, the protected-mode counterpart of `jit64::decode_block_parts`.
//!
//! The 32-bit JIT (`cpu/jit*.rs`) fuses decoding with wasm generation, so there
//! was no decoder to share; this one is pure, like the long-mode one. It covers
//! the common integer instructions and stops the block at anything else, which
//! the interpreter then runs through `run_instruction`.
//!
//! Instructions come out as the mode-agnostic `decode_cache::Instruction`
//! (operand widths 8/16/32, registers 0-7, `Mem.addr_size` 32 or 16 with 0x67).
//! Executing them in protected mode is `cpu::interp::interp32_core::jexec32`,
//! which pairs with this the same way `interp64::jexec` pairs with the
//! long-mode decoder.
//!
//! The decoder is mode-blind about segments: `Mem.segment` carries only the FS
//! (0x64) / GS (0x65) override, because the caller guarantees flat segmentation
//! (DS/SS/CS bases zero) and `jmem_addr32` adds those two bases itself. Any other
//! segment override ends the block.

use crate::cpu::interp::decode_cache::{ArithOp, Instruction, Mem, ShiftKind};


struct Dec<'a> {
    bytes: &'a [u8],
    i: usize,
    base: u64,
    osize: u8, // 16 (0x66) or 32
    asize: u8, // 16 (0x67) or 32
    segment: Option<u8>,
}

#[derive(Copy, Clone)]
enum Operand32 {
    Reg(u8),
    Mem(Mem),
}

impl<'a> Dec<'a> {
    #[inline]
    fn u8(&mut self) -> Option<u8> {
        let v = *self.bytes.get(self.i)?;
        self.i += 1;
        Some(v)
    }
    #[inline]
    fn imm(&mut self, width: u8) -> Option<u64> {
        match width {
            8 => Some(self.u8()? as u64),
            16 => {
                let a = self.u8()? as u64;
                let b = self.u8()? as u64;
                Some(a | b << 8)
            },
            _ => {
                let a = self.u8()? as u64;
                let b = self.u8()? as u64;
                let c = self.u8()? as u64;
                let d = self.u8()? as u64;
                Some(a | b << 8 | c << 16 | d << 24)
            },
        }
    }
    #[inline]
    fn rel(&mut self, width: u8) -> Option<i64> {
        let start = self.i;
        let raw = self.imm(width)?;
        // Sign-extend.
        let signed = match width {
            8 => raw as u8 as i8 as i64,
            16 => raw as u16 as i16 as i64,
            _ => raw as u32 as i32 as i64,
        };
        let _ = start;
        Some(signed)
    }

    // 32/16-bit ModRM; returns the mod field, reg field and the operand.
    fn modrm(&mut self) -> Option<(u8, u8, Operand32)> {
        let modrm = self.u8()?;
        let mod_bits = modrm >> 6;
        let reg = modrm >> 3 & 7;
        let rm = modrm & 7;
        if mod_bits == 3 {
            return Some((mod_bits, reg, Operand32::Reg(rm)));
        }
        if self.asize == 32 {
            let mut base = None;
            let mut index = None;
            let mut scale = 0u8;
            let mut disp = 0i64;
            if rm == 4 {
                let sib = self.u8()?;
                let ss = sib >> 6 & 3;
                let idx = sib >> 3 & 7;
                let b = sib & 7;
                if idx != 4 {
                    index = Some(idx);
                    scale = ss;
                }
                if b == 5 && mod_bits == 0 {
                    disp = self.imm(32)? as u32 as i32 as i64;
                }
                else {
                    base = Some(b);
                }
            }
            else if rm == 5 && mod_bits == 0 {
                disp = self.imm(32)? as u32 as i32 as i64;
            }
            else {
                base = Some(rm);
            }
            if mod_bits == 1 {
                disp = self.imm(8)? as u8 as i8 as i64;
            }
            else if mod_bits == 2 {
                disp = self.imm(32)? as u32 as i32 as i64;
            }
            Some((
                mod_bits,
                reg,
                Operand32::Mem(Mem {
                    base,
                    index,
                    scale,
                    disp,
                    addr_size: self.asize,
                    segment: self.segment,
                }),
            ))
        }
        else {
            // 16-bit addressing.
            let (b, idx) = match rm {
                0 => (Some(3), Some(6)), // BX+SI
                1 => (Some(3), Some(7)), // BX+DI
                2 => (Some(5), Some(6)), // BP+SI
                3 => (Some(5), Some(7)), // BP+DI
                4 => (Some(6), None),    // SI
                5 => (Some(7), None),    // DI
                6 => {
                    if mod_bits == 0 {
                        (None, None)
                    }
                    else {
                        (Some(5), None) // BP
                    }
                },
                _ => (Some(3), None), // BX
            };
            let mut disp = 0i64;
            if mod_bits == 0 && rm == 6 {
                disp = self.imm(16)? as u16 as i16 as i64;
            }
            else if mod_bits == 1 {
                disp = self.imm(8)? as u8 as i8 as i64;
            }
            else if mod_bits == 2 {
                disp = self.imm(16)? as u16 as i16 as i64;
            }
            Some((
                mod_bits,
                reg,
                Operand32::Mem(Mem {
                    base: b,
                    index: idx,
                    scale: 0,
                    disp,
                    addr_size: self.asize,
                    segment: self.segment,
                }),
            ))
        }
    }
}

fn alu_op(o: u8) -> ArithOp {
    match o {
        0 => ArithOp::Add,
        1 => ArithOp::Or,
        2 => ArithOp::Adc,
        3 => ArithOp::Sbb,
        4 => ArithOp::And,
        5 => ArithOp::Sub,
        6 => ArithOp::Xor,
        _ => ArithOp::Cmp,
    }
}

// Decode one instruction; `Err(())` means unsupported / truncated.
fn decode_one(d: &mut Dec, out: &mut Vec<Instruction>, rips: &mut Vec<u64>) -> Result<bool, ()> {
    let start = d.i;
    let cur_rip = d.base.wrapping_add(start as u64);
    d.osize = 32;
    d.asize = 32;
    d.segment = None;
    let mut p66 = false;
    let mut p67 = false;
    loop {
        match *d.bytes.get(d.i).ok_or(())? {
            0x66 => {
                p66 = true;
                d.i += 1;
            },
            0x67 => {
                p67 = true;
                d.i += 1;
            },
            0xF0 => d.i += 1,
            // rep/repe change how often the instruction runs, so a block that
            // contains one cannot be executed as if it were unprefixed.
            0xF2 | 0xF3 => return Err(()),
            // Only FS/GS bases are added by jmem_addr32; ES/CS/SS/DS overrides
            // are not modelled, so refuse the block.
            0x26 | 0x2E | 0x36 | 0x3E => return Err(()),
            0x64 | 0x65 => {
                d.segment = Some(d.bytes[d.i]);
                d.i += 1;
            },
            _ => break,
        }
    }
    d.osize = if p66 { 16 } else { 32 };
    d.asize = if p67 { 16 } else { 32 };

    let opcode = d.u8().ok_or(())?;
    // 0x0F escapes to the two-byte map (SSE/AVX/MMX/...); not decoded here, so
    // end the block and let run_instruction handle it.
    if opcode == 0x0F {
        return Err(());
    }
    // The stack/control instructions below have no operand-size in the IR: the
    // executor always pushes/pops 32 bits and writes the full EIP. A `66` there
    // changes the width (push/pop/ret/call) or masks EIP to 16 bits (near
    // branches), so leave those encodings to the interpreter.
    if p66
        && matches!(
            opcode,
            0x50..=0x5F | 0x70..=0x7F | 0x8F | 0xC2 | 0xC3 | 0xC9 | 0xE8 | 0xE9 | 0xEB
        )
    {
        return Err(());
    }
    let mut push = |ins: Instruction| {
        rips.push(cur_rip);
        out.push(ins);
    };

    if opcode < 0x40 {
        let op = alu_op(opcode >> 3);
        let form = opcode & 7;
        match form {
            0 | 1 => {
                // `op r/m, reg`: reg is the ModRM reg field, not an opcode
                // nibble -- the nibble only chose the operation above.
                let (_, reg, rm) = d.modrm().ok_or(())?;
                let width = if form == 0 { 8 } else { d.osize };
                match rm {
                    Operand32::Reg(r) => {
                        push(Instruction::ArithRegReg { op, dst: r, src: reg, width })
                    },
                    Operand32::Mem(mem) => {
                        push(Instruction::ArithMemReg { op, mem, src: reg, width })
                    },
                }
            },
            2 | 3 => {
                // `op reg, r/m`: here the reg field is the destination.
                let (_, reg, rm) = d.modrm().ok_or(())?;
                let width = if form == 2 { 8 } else { d.osize };
                match rm {
                    Operand32::Reg(r) => {
                        push(Instruction::ArithRegReg { op, dst: reg, src: r, width })
                    },
                    Operand32::Mem(mem) => {
                        push(Instruction::ArithRegMem { op, dst: reg, mem, width })
                    },
                }
            },
            4 => {
                let value = d.imm(8).ok_or(())?;
                push(Instruction::ArithRegImm { op, r: 0, value, width: 8 });
            },
            5 => {
                let value = d.imm(d.osize).ok_or(())?;
                push(Instruction::ArithRegImm { op, r: 0, value, width: d.osize });
            },
            6 | 7 => {
                // Not arithmetic: 0x06/0x07/0x0E are PUSH/POP ES/CS and 0x0F is
                // the two-byte escape. `op AL/eAX, imm` is forms 4 and 5.
                return Err(());
            },
            _ => return Err(()),
        }
        return Ok(true);
    }

    match opcode {
        0x40..=0x4F => push(Instruction::IncDecReg {
            r: opcode & 7,
            width: d.osize,
            decrement: opcode >= 0x48,
        }),
        0x50..=0x57 => push(Instruction::PushReg { r: opcode & 7 }),
        0x58..=0x5F => push(Instruction::PopReg { r: opcode & 7 }),
        0x70..=0x7F => {
            let disp = d.rel(8).ok_or(())?;
            let fallthrough = d.base.wrapping_add(d.i as u64);
            push(Instruction::Jcc {
                code: opcode & 0xF,
                target: fallthrough.wrapping_add(disp as u64),
                fallthrough,
            });
            return Ok(false);
        },
        0x80 | 0x81 | 0x83 => {
            let (_, reg, rm) = d.modrm().ok_or(())?;
            let width = if opcode == 0x80 { 8 } else { d.osize };
            // 0x80 and 0x83 carry a sign-extended imm8; only 0x81 takes a full
            // operand-width immediate.
            let value = if opcode == 0x81 {
                d.imm(d.osize).ok_or(())?
            }
            else {
                d.imm(8).ok_or(())? as u8 as i8 as i64 as u64
            };
            let op = alu_op(reg);
            match rm {
                Operand32::Reg(r) => push(Instruction::ArithRegImm { op, r, value, width }),
                Operand32::Mem(mem) => push(Instruction::ArithMemImm { op, mem, value, width }),
            }
        },
        0x84 | 0x85 => {
            // `test r/m, reg`: the second operand is the ModRM reg field.
            let (_, reg, rm) = d.modrm().ok_or(())?;
            let width = if opcode == 0x84 { 8 } else { d.osize };
            match rm {
                Operand32::Reg(r) => push(Instruction::ArithRegReg {
                    op: ArithOp::Test,
                    dst: r,
                    src: reg,
                    width,
                }),
                Operand32::Mem(mem) => push(Instruction::ArithMemReg {
                    op: ArithOp::Test,
                    mem,
                    src: reg,
                    width,
                }),
            }
        },
        0x86 | 0x87 => {
            // `xchg r/m, reg`: likewise.
            let (_, reg, rm) = d.modrm().ok_or(())?;
            let width = if opcode == 0x86 { 8 } else { d.osize };
            match rm {
                Operand32::Reg(r) => push(Instruction::XchgRegReg { a: r, b: reg, width }),
                // The memory form needs read-modify-write ordering the executor
                // does not model; leave it to the interpreter.
                Operand32::Mem(_) => return Err(()),
            }
        },
        0x88 | 0x89 | 0x8A | 0x8B => {
            // The second operand is always the ModRM reg field: 88/89 are
            // `mov r/m, reg` and 8A/8B are `mov reg, r/m`.
            let (_, reg, rm) = d.modrm().ok_or(())?;
            let width = if opcode & 1 == 0 { 8 } else { d.osize };
            match opcode {
                0x88 | 0x89 => match rm {
                    Operand32::Reg(r) => push(Instruction::MovRegReg { dst: r, src: reg, width }),
                    Operand32::Mem(mem) => push(Instruction::MovMemReg { mem, src: reg, width }),
                },
                _ => match rm {
                    Operand32::Reg(r) => push(Instruction::MovRegReg { dst: reg, src: r, width }),
                    Operand32::Mem(mem) => push(Instruction::MovRegMem { dst: reg, mem, width }),
                },
            }
        },
        0x8D => {
            let (mod_bits, reg, rm) = d.modrm().ok_or(())?;
            if mod_bits == 3 {
                return Err(());
            }
            let Operand32::Mem(mem) = rm else { return Err(()) };
            push(Instruction::Lea { dst: reg, mem, width: d.osize });
        },
        0x8F => {
            let (_, reg, rm) = d.modrm().ok_or(())?;
            if reg != 0 {
                return Err(());
            }
            match rm {
                Operand32::Reg(r) => push(Instruction::PopReg { r }),
                Operand32::Mem(_) => return Err(()),
            }
        },
        0x90..=0x97 => {
            let r = opcode & 7;
            if r == 0 {
                push(Instruction::Nop);
            }
            else {
                push(Instruction::XchgRegReg { a: 0, b: r, width: d.osize });
            }
        },
        0xA8 => {
            let value = d.imm(8).ok_or(())?;
            push(Instruction::ArithRegImm { op: ArithOp::Test, r: 0, value, width: 8 });
        },
        0xA9 => {
            let value = d.imm(d.osize).ok_or(())?;
            push(Instruction::ArithRegImm { op: ArithOp::Test, r: 0, value, width: d.osize });
        },
        0xB0..=0xB7 => {
            let value = d.imm(8).ok_or(())?;
            push(Instruction::MovRegImm { r: opcode & 7, value, width: 8 });
        },
        0xB8..=0xBF => {
            let value = d.imm(d.osize).ok_or(())?;
            push(Instruction::MovRegImm { r: opcode & 7, value, width: d.osize });
        },
        0xC0 | 0xC1 => {
            let (_, reg, rm) = d.modrm().ok_or(())?;
            let width = if opcode == 0xC0 { 8 } else { d.osize };
            let count = d.imm(8).ok_or(())? as u8;
            let kind = shift_kind(reg)?;
            match rm {
                Operand32::Reg(r) => push(Instruction::ShiftReg { kind, r, width, count }),
                Operand32::Mem(mem) => push(Instruction::ShiftMem { kind, mem, width, count }),
            }
        },
        0xC2 => {
            let adjustment = d.imm(16).ok_or(())? as u16;
            push(Instruction::Ret { adjustment });
            return Ok(false);
        },
        0xC3 => {
            push(Instruction::Ret { adjustment: 0 });
            return Ok(false);
        },
        0xC6 | 0xC7 => {
            let width = if opcode == 0xC6 { 8 } else { d.osize };
            let (_, reg, rm) = d.modrm().ok_or(())?;
            if reg != 0 {
                return Err(());
            }
            let value = d.imm(width).ok_or(())?;
            match rm {
                Operand32::Reg(r) => push(Instruction::MovRegImm { r, value, width }),
                Operand32::Mem(mem) => push(Instruction::MovMemImm { mem, value, width }),
            }
        },
        0xC9 => push(Instruction::Leave),
        0xD0 | 0xD1 | 0xD2 | 0xD3 => {
            let (_, reg, rm) = d.modrm().ok_or(())?;
            let width = if opcode & 1 == 0 { 8 } else { d.osize };
            let kind = shift_kind(reg)?;
            let by_cl = opcode >= 0xD2;
            match rm {
                Operand32::Reg(r) => {
                    if by_cl {
                        push(Instruction::ShiftRegCl { kind, r, width });
                    }
                    else {
                        push(Instruction::ShiftReg { kind, r, width, count: 1 });
                    }
                },
                Operand32::Mem(mem) => {
                    if by_cl {
                        push(Instruction::ShiftMemCl { kind, mem, width });
                    }
                    else {
                        push(Instruction::ShiftMem { kind, mem, width, count: 1 });
                    }
                },
            }
        },
        0xE8 => {
            let disp = d.rel(d.osize).ok_or(())?;
            let return_address = d.base.wrapping_add(d.i as u64);
            push(Instruction::Call { target: return_address.wrapping_add(disp as u64), return_address });
            return Ok(false);
        },
        0xE9 => {
            let disp = d.rel(d.osize).ok_or(())?;
            let target = d.base.wrapping_add(d.i as u64).wrapping_add(disp as u64);
            push(Instruction::Jmp { target });
            return Ok(false);
        },
        0xEB => {
            let disp = d.rel(8).ok_or(())?;
            let target = d.base.wrapping_add(d.i as u64).wrapping_add(disp as u64);
            push(Instruction::Jmp { target });
            return Ok(false);
        },
        0xF4 => {
            push(Instruction::Hlt);
            return Ok(false);
        },
        0xF6 | 0xF7 => {
            let width = if opcode == 0xF6 { 8 } else { d.osize };
            let (_, reg, rm) = d.modrm().ok_or(())?;
            match reg {
                0 => {
                    let value = d.imm(width).ok_or(())?;
                    match rm {
                        Operand32::Reg(r) => {
                            push(Instruction::ArithRegImm { op: ArithOp::Test, r, value, width })
                        },
                        Operand32::Mem(mem) => {
                            push(Instruction::ArithMemImm { op: ArithOp::Test, mem, value, width })
                        },
                    }
                },
                2 => match rm {
                    Operand32::Reg(r) => push(Instruction::NotReg { r, width }),
                    Operand32::Mem(mem) => push(Instruction::NotMem { mem, width }),
                },
                3 => match rm {
                    Operand32::Reg(r) => push(Instruction::NegReg { r, width }),
                    Operand32::Mem(mem) => push(Instruction::NegMem { mem, width }),
                },
                _ => return Err(()),
            }
        },
        0xFE => {
            let (_, reg, rm) = d.modrm().ok_or(())?;
            if reg > 1 {
                return Err(());
            }
            let decrement = reg == 1;
            match rm {
                Operand32::Reg(r) => push(Instruction::IncDecReg { r, width: 8, decrement }),
                Operand32::Mem(mem) => push(Instruction::IncDecMem { mem, width: 8, decrement }),
            }
        },
        0xFF => {
            let (_, reg, rm) = d.modrm().ok_or(())?;
            // /2 call, /4 jmp and /6 push use the operand size for the pushed
            // value and the branch offset; the executor only models 32-bit.
            if p66 && matches!(reg, 2 | 4 | 6) {
                return Err(());
            }
            match reg {
                0 | 1 => {
                    let decrement = reg == 1;
                    match rm {
                        Operand32::Reg(r) => {
                            push(Instruction::IncDecReg { r, width: d.osize, decrement })
                        },
                        Operand32::Mem(mem) => {
                            push(Instruction::IncDecMem { mem, width: d.osize, decrement })
                        },
                    }
                },
                2 => {
                    let ra = d.base.wrapping_add(d.i as u64);
                    match rm {
                        Operand32::Reg(r) => {
                            push(Instruction::CallReg { r, return_address: ra });
                            return Ok(false);
                        },
                        Operand32::Mem(mem) => {
                            push(Instruction::CallMem { mem, return_address: ra });
                            return Ok(false);
                        },
                    }
                },
                4 => match rm {
                    Operand32::Reg(r) => {
                        push(Instruction::JmpReg { r });
                        return Ok(false);
                    },
                    Operand32::Mem(mem) => {
                        push(Instruction::JmpMem { mem });
                        return Ok(false);
                    },
                },
                6 => match rm {
                    Operand32::Reg(r) => push(Instruction::PushReg { r }),
                    Operand32::Mem(_) => return Err(()),
                },
                _ => return Err(()),
            }
        },
        _ => return Err(()),
    }
    Ok(true)
}

fn shift_kind(reg: u8) -> Result<ShiftKind, ()> {
    Ok(match reg {
        4 => ShiftKind::Shl,
        5 => ShiftKind::Shr,
        7 => ShiftKind::Sar,
        0 => ShiftKind::Rol,
        1 => ShiftKind::Ror,
        _ => return Err(()),
    })
}
// Pure decoder entry point, mirroring jit64::decode_block_parts: given the code
// bytes starting at `base`, produce the block it decodes to. Only touches the
// byte slice, so it is testable without paging.
pub fn decode_block_parts(
    base: u64,
    bytes: &[u8],
) -> Result<(Vec<Instruction>, Vec<u64>, u64), String> {
    let mut d = Dec {
        bytes,
        i: 0,
        base,
        osize: 32,
        asize: 32,
        segment: None,
    };
    let mut instrs: Vec<Instruction> = Vec::new();
    let mut rips: Vec<u64> = Vec::new();
    loop {
        if instrs.len() >= BLOCK_INSTR_LIMIT {
            break;
        }
        let before = d.i;
        match decode_one(&mut d, &mut instrs, &mut rips) {
            // A non-terminator: keep going.
            Ok(true) => {},
            // A control transfer ends the block.
            Ok(false) => break,
            // Unsupported or truncated: drop it and end the block before it, so
            // the interpreter executes it itself.
            Err(()) => {
                d.i = before;
                break;
            },
        }
        if d.i >= bytes.len() {
            break;
        }
    }
    if instrs.is_empty() {
        return Err("no supported instruction".into());
    }
    Ok((instrs, rips, base.wrapping_add(d.i as u64)))
}

/// Cap on instructions per cached block, so a straight-line run of thousands of
/// instructions does not build one huge cache entry.
pub const BLOCK_INSTR_LIMIT: usize = 64;
