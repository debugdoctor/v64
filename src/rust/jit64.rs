//! Long mode JIT targeting wasm32: guest registers are i64 locals, physical
//! addresses stay 32-bit (below 4 GiB).
//! http://www.sandpile.org/x86/opra.htm

#![allow(dead_code)]

use crate::wasmgen::wasm_builder::{WasmBuilder, WasmLocalI64};

// Register file layout in linear memory (see global_pointers.rs):
//   r0-r7  low 32 bits at 64 + 4*r, high 32 bits at 128 + 4*r
//   r8-r15 full 64 bits at 160 + 8*(r-8)
const REG_LOW: i32 = 64;
const REG_HIGH: i32 = 128;
const REG_EXT: i32 = 160;
const IN_HLT: i32 = 616;
const FLAGS_ADDR: i32 = 120;
const FLAGS_CHANGED_ADDR: i32 = 100;
const IP_ADDR: i32 = 232;
const PREVIOUS_RIP_ADDR: i32 = 240;
const INSTRUCTION_COUNTER_ADDR: i32 = 664;

// CF | PF | AF | ZF | SF | OF
const FLAG_MASK: i32 = 0x8D5;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mem {
    pub base: Option<u8>,
    pub index: Option<u8>,
    pub scale: u8, // index << scale

    pub disp: i64,
    // Effective-address width: 64 by default, 32 when the `67` prefix is present.
    pub addr_size: u8,
}

// Mask of the bits a value of this operand width occupies.
fn width_mask(width: u8) -> u64 {
    match width {
        8 => 0xFF,
        16 => 0xFFFF,
        32 => 0xFFFF_FFFF,
        _ => u64::MAX,
    }
}

// The value of the sign bit for this operand width.
fn width_sign_bit(width: u8) -> u64 {
    match width {
        8 => 0x80,
        16 => 0x8000,
        32 => 0x8000_0000,
        _ => 1 << 63,
    }
}

// The `66` prefix selects a 16-bit operand in long mode even together with REX.W.
fn operand_width(prefix_66: bool, rex_w: bool) -> u8 {
    if prefix_66 {
        16
    }
    else if rex_w {
        64
    }
    else {
        32
    }
}

// Runtime (host-side) equivalents of the width helpers above, used by the
// imported JIT helpers.
fn operand_mask(width: u32) -> u64 {
    match width {
        8 => 0xFF,
        16 => 0xFFFF,
        32 => 0xFFFF_FFFF,
        _ => u64::MAX,
    }
}

fn sign_extend_to_i128(value: u64, width: u32) -> i128 {
    match width {
        8 => value as u8 as i8 as i128,
        16 => value as u16 as i16 as i128,
        32 => value as u32 as i32 as i128,
        _ => value as i64 as i128,
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Instr {
    MovRegImm { r: u8, value: u64, width: u8 },
    MovRegReg { dst: u8, src: u8, width: u8 },
    MovRegMem { dst: u8, mem: Mem, width: u8 },
    MovMemReg { mem: Mem, src: u8, width: u8 },
    MovMemImm { mem: Mem, value: u64, width: u8 },
    MovExtendReg { dst: u8, src: u8, src_width: u8, dst_width: u8, signed: bool, high8: bool },
    MovExtendMem { dst: u8, mem: Mem, src_width: u8, dst_width: u8, signed: bool },
    XchgRegReg { a: u8, b: u8, width: u8 },
    XchgMemReg { mem: Mem, r: u8, width: u8 },
    NotReg { r: u8, width: u8 },
    NotMem { mem: Mem, width: u8 },
    NegReg { r: u8, width: u8 },
    NegMem { mem: Mem, width: u8 },
    IncDecReg { r: u8, width: u8, decrement: bool },
    IncDecMem { mem: Mem, width: u8, decrement: bool },
    CmovRegReg { code: u8, dst: u8, src: u8, width: u8 },
    CmovRegMem { code: u8, dst: u8, mem: Mem, width: u8 },
    SetccReg { code: u8, dst: u8, high8: bool },
    SetccMem { code: u8, mem: Mem },
    ShiftReg { kind: ShiftKind, r: u8, width: u8, count: u8 },
    ShiftRegCl { kind: ShiftKind, r: u8, width: u8 },
    ShiftMem { kind: ShiftKind, mem: Mem, width: u8, count: u8 },
    ShiftMemCl { kind: ShiftKind, mem: Mem, width: u8 },
    Cpuid,
    ImulRegReg { dst: u8, lhs: u8, rhs: u8, width: u8 },
    ImulRegImm { dst: u8, src: u8, value: u64, width: u8 },
    ImulRegMem { dst: u8, mem: Mem, value: Option<u64>, width: u8 },
    Bswap { r: u8, width: u8 },
    Lea { dst: u8, mem: Mem, width: u8 },
    AddRegReg { dst: u8, src: u8, width: u8 },
    AddRegImm { r: u8, value: u64, width: u8 },
    // cmp/test update flags without writing dst
    ArithRegReg { op: ArithOp, dst: u8, src: u8, width: u8 },
    ArithRegImm { op: ArithOp, r: u8, value: u64, width: u8 },
    ArithRegMem { op: ArithOp, dst: u8, mem: Mem, width: u8 },
    ArithMemReg { op: ArithOp, mem: Mem, src: u8, width: u8 },
    ArithMemImm { op: ArithOp, mem: Mem, value: u64, width: u8 },
    PushReg { r: u8 },
    PopReg { r: u8 },
    PushImm { value: u64 },
    Call { target: u64, return_address: u64 },
    CallReg { r: u8, return_address: u64 },
    CallMem { mem: Mem, return_address: u64 },
    JmpReg { r: u8 },
    JmpMem { mem: Mem },
    Ret { adjustment: u16 },
    Leave,
    Nop,
    Jmp { target: u64 },
    // fallthrough is the next instruction
    Jcc { code: u8, target: u64, fallthrough: u64 },
    Hlt,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ArithOp {
    Add,
    Adc,
    Sub,
    Sbb,
    Cmp,
    Test,
    And,
    Or,
    Xor,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub enum ShiftKind { Shl, Shr, Sar }

struct DecodedBlock {
    instrs: Vec<Instr>,
    rips: Vec<u64>,
    current_rip: u64,
    end_rip: u64,
}

impl DecodedBlock {
    fn push(&mut self, instr: Instr) {
        self.instrs.push(instr);
        self.rips.push(self.current_rip);
    }
}

impl ArithOp {
    fn is_logic(self) -> bool { matches!(self, Self::Test | Self::And | Self::Or | Self::Xor) }
    fn writes_result(self) -> bool { !matches!(self, Self::Cmp | Self::Test) }
}

fn read_u32(bytes: &[u8], i: &mut usize) -> Result<u32, String> {
    if *i + 4 > bytes.len() {
        return Err("truncated imm32".into());
    }
    let v = u32::from_le_bytes([bytes[*i], bytes[*i + 1], bytes[*i + 2], bytes[*i + 3]]);
    *i += 4;
    Ok(v)
}

fn read_i8(bytes: &[u8], i: &mut usize) -> Result<i8, String> {
    if *i >= bytes.len() {
        return Err("truncated imm8".into());
    }
    let v = bytes[*i] as i8;
    *i += 1;
    Ok(v)
}

// Read an immediate whose encoded size follows the operand width: 16-bit
// operands use imm16, 32- and 64-bit operands use imm32 (sign-extended to 64).
fn read_imm_operand(bytes: &[u8], i: &mut usize, width: u8) -> Result<u64, String> {
    if width == 16 {
        if *i + 2 > bytes.len() {
            return Err("truncated imm16".into());
        }
        let v = u16::from_le_bytes([bytes[*i], bytes[*i + 1]]) as u64;
        *i += 2;
        Ok(v)
    }
    else {
        let raw = read_u32(bytes, i)?;
        Ok(if width == 64 { raw as i32 as i64 as u64 } else { raw as u64 })
    }
}

// Encoded immediate size for the operand width (imm16 vs imm32).
fn imm_operand_bytes(width: u8) -> usize {
    if width == 16 {
        2
    }
    else {
        4
    }
}

// mod != 3
fn decode_mem(
    modrm: u8,
    rex: u8,
    bytes: &[u8],
    i: &mut usize,
    block_base: u64,
    addr_size: u8,
    trailing_bytes: usize,
) -> Result<Mem, String> {
    let rex_b = (rex & 0x01 != 0) as u8;
    let rex_x = (rex & 0x02 != 0) as u8;
    let mod_bits = modrm >> 6;
    let rm_low = modrm & 7;

    let mut base = None;
    let mut index = None;
    let mut scale = 0;
    let mut disp: i64 = 0;

    if rm_low == 4 {
        if *i >= bytes.len() {
            return Err("truncated sib".into());
        }
        let sib = bytes[*i];
        *i += 1;
        scale = sib >> 6;
        let index_low = sib >> 3 & 7;
        let base_low = sib & 7;
        if index_low != 4 {
            index = Some(index_low | rex_x << 3);
        }
        if base_low == 5 && mod_bits == 0 {
            disp = read_u32(bytes, i)? as i32 as i64;
        }
        else {
            base = Some(base_low | rex_b << 3);
        }
    }
    else if rm_low == 5 && mod_bits == 0 {
        let relative = read_u32(bytes, i)? as i32 as i64;
        let after = block_base.wrapping_add((*i + trailing_bytes) as u64);
        // With a 32-bit address size, RIP-relative addressing uses EIP and wraps
        // at 2^32.
        disp = if addr_size == 32 {
            (after as u32).wrapping_add(relative as u32) as i64
        }
        else {
            after.wrapping_add(relative as u64) as i64
        };
    }
    else {
        base = Some(rm_low | rex_b << 3);
    }

    match mod_bits {
        0 => {},
        1 => disp = disp.wrapping_add(read_i8(bytes, i)? as i64),
        2 => disp = disp.wrapping_add(read_u32(bytes, i)? as i32 as i64),
        _ => unreachable!(),
    }

    Ok(Mem {
        base,
        index,
        scale,
        disp,
        addr_size,
    })
}

// Stops at the end of the basic block (after hlt, jmp or jcc).
fn decode_block_with_rips(base: u64, bytes: &[u8]) -> Result<DecodedBlock, String> {
    let mut out = DecodedBlock {
        instrs: Vec::new(),
        rips: Vec::new(),
        current_rip: base,
        end_rip: base,
    };
    let mut i = 0;

    while i < bytes.len() {
        let start = i;
        out.current_rip = base.wrapping_add(start as u64);
        let mut rex = 0u8;
        let mut prefix_f3 = false;
        let mut prefix_66 = false;
        let mut prefix_67 = false;
        loop {
            match bytes[i] {
                0xF3 => {
                    prefix_f3 = true;
                    i += 1;
                },
                0x66 => {
                    prefix_66 = true;
                    i += 1;
                },
                0x67 => {
                    prefix_67 = true;
                    i += 1;
                },
                // LOCK and REPNE do not change the semantics of the integer
                // instructions handled here (string operations are not JITted).
                0xF0 | 0xF2 => {
                    i += 1;
                },
                0x40..=0x4F => {
                    rex = bytes[i];
                    i += 1;
                },
                _ => break,
            }
            if i >= bytes.len() {
                return Err("truncated prefixes".into());
            }
        }

        let opcode = bytes[i];
        i += 1;
        if prefix_f3 && opcode != 0x0F {
            return Err(format!("unsupported f3 opcode {:02x} at {}", opcode, start));
        }
        let rex_w = rex & 0x08 != 0;
        let rex_r = (rex & 0x04 != 0) as u8;
        let rex_b = (rex & 0x01 != 0) as u8;
        let addr_size = if prefix_67 { 32 } else { 64 };

        match opcode {
            0x90 => out.push(Instr::Nop),

            // mov r, imm
            0xB8..=0xBF => {
                let r = (opcode - 0xB8) | rex_b << 3;
                let width = operand_width(prefix_66, rex_w);
                let value = match width {
                    64 => {
                        if i + 8 > bytes.len() {
                            return Err("truncated imm64".into());
                        }
                        let mut v = 0u64;
                        for k in 0..8 {
                            v |= (bytes[i + k] as u64) << (8 * k);
                        }
                        i += 8;
                        v
                    },
                    16 => {
                        if i + 2 > bytes.len() {
                            return Err("truncated imm16".into());
                        }
                        let v = u16::from_le_bytes([bytes[i], bytes[i + 1]]) as u64;
                        i += 2;
                        v
                    },
                    _ => read_u32(bytes, &mut i)? as u64,
                };
                out.push(Instr::MovRegImm { r, value, width });
            },

            // mov r/m, imm16/imm32
            0xC7 => {
                let width = operand_width(prefix_66, rex_w);
                let modrm = *bytes.get(i).ok_or("truncated modrm")?;
                i += 1;
                if modrm >> 3 & 7 != 0 {
                    return Err("unsupported mov immediate form".into());
                }
                let mem = if modrm >> 6 == 3 {
                    None
                }
                else {
                    Some(decode_mem(
                        modrm,
                        rex,
                        bytes,
                        &mut i,
                        base,
                        addr_size,
                        if width == 16 { 2 } else { 4 },
                    )?)
                };
                let value = if width == 16 {
                    if i + 2 > bytes.len() {
                        return Err("truncated imm16".into());
                    }
                    let v = u16::from_le_bytes([bytes[i], bytes[i + 1]]) as u64;
                    i += 2;
                    v
                }
                else {
                    let raw = read_u32(bytes, &mut i)?;
                    if width == 64 { raw as i32 as i64 as u64 } else { raw as u64 }
                };
                if let Some(mem) = mem {
                    out.push(Instr::MovMemImm { mem, value, width });
                }
                else {
                    out.push(Instr::MovRegImm { r: (modrm & 7) | rex_b << 3, value, width });
                }
            },

            // movsxd r64, r/m32
            0x63 => {
                if prefix_66 || !rex_w { return Err("movsxd requires rex.w".into()); }
                let modrm = *bytes.get(i).ok_or("truncated movsxd modrm")?;
                i += 1;
                let dst = (modrm >> 3 & 7) | rex_r << 3;
                if modrm >> 6 == 3 {
                    out.push(Instr::MovExtendReg {
                        dst,
                        src: (modrm & 7) | rex_b << 3,
                        src_width: 32,
                        dst_width: 64,
                        signed: true,
                        high8: false,
                    });
                }
                else {
                    let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, 0)?;
                    out.push(Instr::MovExtendMem {
                        dst, mem, src_width: 32, dst_width: 64, signed: true,
                    });
                }
            },

            // imul r, r/m, imm32/imm8
            0x69 | 0x6B => {
                let width = operand_width(prefix_66, rex_w);
                let modrm = *bytes.get(i).ok_or("truncated imul modrm")?;
                i += 1;
                let dst = (modrm >> 3 & 7) | rex_r << 3;
                let mem = if modrm >> 6 == 3 {
                    None
                }
                else {
                    Some(decode_mem(
                        modrm,
                        rex,
                        bytes,
                        &mut i,
                        base,
                        addr_size,
                        if opcode == 0x6B { 1 } else if width == 16 { 2 } else { 4 },
                    )?)
                };
                let value = if opcode == 0x6B {
                    read_i8(bytes, &mut i)? as i64 as u64
                }
                else if width == 16 {
                    if i + 2 > bytes.len() {
                        return Err("truncated imm16".into());
                    }
                    let v = u16::from_le_bytes([bytes[i], bytes[i + 1]]) as u64;
                    i += 2;
                    v
                }
                else {
                    read_u32(bytes, &mut i)? as i32 as i64 as u64
                };
                if let Some(mem) = mem {
                    out.push(Instr::ImulRegMem { dst, mem, value: Some(value), width });
                }
                else {
                    out.push(Instr::ImulRegImm {
                        dst,
                        src: (modrm & 7) | rex_b << 3,
                        value,
                        width,
                    });
                }
            },

            // xchg r/m, r (register form)
            0x87 => {
                let width = operand_width(prefix_66, rex_w);
                let modrm = *bytes.get(i).ok_or("truncated modrm")?;
                i += 1;
                let r = (modrm >> 3 & 7) | rex_r << 3;
                if modrm >> 6 == 3 {
                    out.push(Instr::XchgRegReg {
                        a: (modrm & 7) | rex_b << 3,
                        b: r,
                        width,
                    });
                }
                else {
                    let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, 0)?;
                    out.push(Instr::XchgMemReg { mem, r, width });
                }
            },

            // mov/lea/add/sub/cmp/test r/m, r and r, r/m
            0x89 | 0x8B | 0x8D | 0x01 | 0x03 | 0x09 | 0x0B | 0x11 | 0x13 | 0x19 | 0x1B
            | 0x21 | 0x23 | 0x29 | 0x2B | 0x31 | 0x33 | 0x39 | 0x3B | 0x85 => {
                let width = operand_width(prefix_66, rex_w);
                let modrm = *bytes.get(i).ok_or("truncated modrm")?;
                i += 1;
                let reg = (modrm >> 3 & 7) | rex_r << 3;
                let rm = (modrm & 7) | rex_b << 3;
                if modrm >> 6 == 3 {
                    if opcode == 0x8D {
                        return Err("lea requires a memory operand".into());
                    }
                    out.push(match opcode {
                        0x89 => Instr::MovRegReg { dst: rm, src: reg, width },
                        0x8B => Instr::MovRegReg { dst: reg, src: rm, width },
                        0x01 => Instr::AddRegReg { dst: rm, src: reg, width },
                        0x03 => Instr::AddRegReg { dst: reg, src: rm, width },
                        0x09 => Instr::ArithRegReg { op: ArithOp::Or, dst: rm, src: reg, width },
                        0x0B => Instr::ArithRegReg { op: ArithOp::Or, dst: reg, src: rm, width },
                        0x11 => Instr::ArithRegReg { op: ArithOp::Adc, dst: rm, src: reg, width },
                        0x13 => Instr::ArithRegReg { op: ArithOp::Adc, dst: reg, src: rm, width },
                        0x19 => Instr::ArithRegReg { op: ArithOp::Sbb, dst: rm, src: reg, width },
                        0x1B => Instr::ArithRegReg { op: ArithOp::Sbb, dst: reg, src: rm, width },
                        0x21 => Instr::ArithRegReg { op: ArithOp::And, dst: rm, src: reg, width },
                        0x23 => Instr::ArithRegReg { op: ArithOp::And, dst: reg, src: rm, width },
                        0x29 => Instr::ArithRegReg { op: ArithOp::Sub, dst: rm, src: reg, width },
                        0x2B => Instr::ArithRegReg { op: ArithOp::Sub, dst: reg, src: rm, width },
                        0x31 => Instr::ArithRegReg { op: ArithOp::Xor, dst: rm, src: reg, width },
                        0x33 => Instr::ArithRegReg { op: ArithOp::Xor, dst: reg, src: rm, width },
                        0x39 => Instr::ArithRegReg { op: ArithOp::Cmp, dst: rm, src: reg, width },
                        0x3B => Instr::ArithRegReg { op: ArithOp::Cmp, dst: reg, src: rm, width },
                        0x85 => Instr::ArithRegReg { op: ArithOp::Test, dst: rm, src: reg, width },
                        _ => unreachable!(),
                    });
                }
                else {
                    let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, 0)?;
                    out.push(match opcode {
                        0x89 => Instr::MovMemReg { mem, src: reg, width },
                        0x8B => Instr::MovRegMem { dst: reg, mem, width },
                        0x8D => Instr::Lea { dst: reg, mem, width },
                        0x01 => Instr::ArithMemReg { op: ArithOp::Add, mem, src: reg, width },
                        0x03 => Instr::ArithRegMem { op: ArithOp::Add, dst: reg, mem, width },
                        0x09 => Instr::ArithMemReg { op: ArithOp::Or, mem, src: reg, width },
                        0x0B => Instr::ArithRegMem { op: ArithOp::Or, dst: reg, mem, width },
                        0x11 => Instr::ArithMemReg { op: ArithOp::Adc, mem, src: reg, width },
                        0x13 => Instr::ArithRegMem { op: ArithOp::Adc, dst: reg, mem, width },
                        0x19 => Instr::ArithMemReg { op: ArithOp::Sbb, mem, src: reg, width },
                        0x1B => Instr::ArithRegMem { op: ArithOp::Sbb, dst: reg, mem, width },
                        0x21 => Instr::ArithMemReg { op: ArithOp::And, mem, src: reg, width },
                        0x23 => Instr::ArithRegMem { op: ArithOp::And, dst: reg, mem, width },
                        0x29 => Instr::ArithMemReg { op: ArithOp::Sub, mem, src: reg, width },
                        0x2B => Instr::ArithRegMem { op: ArithOp::Sub, dst: reg, mem, width },
                        0x31 => Instr::ArithMemReg { op: ArithOp::Xor, mem, src: reg, width },
                        0x33 => Instr::ArithRegMem { op: ArithOp::Xor, dst: reg, mem, width },
                        0x39 => Instr::ArithMemReg { op: ArithOp::Cmp, mem, src: reg, width },
                        0x3B => Instr::ArithRegMem { op: ArithOp::Cmp, dst: reg, mem, width },
                        0x85 => Instr::ArithMemReg { op: ArithOp::Test, mem, src: reg, width },
                        _ => unreachable!(),
                    });
                }
            },

            // add/or/and/sub/xor/cmp r/m, imm (group 1)
            0x81 | 0x83 => {
                let width = operand_width(prefix_66, rex_w);
                let modrm = *bytes.get(i).ok_or("truncated modrm")?;
                i += 1;
                let group = modrm >> 3 & 7;
                if !matches!(group, 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7) {
                    return Err("unsupported group 1 form".into());
                }
                let mem = if modrm >> 6 == 3 {
                    None
                }
                else {
                    Some(decode_mem(
                        modrm,
                        rex,
                        bytes,
                        &mut i,
                        base,
                        addr_size,
                        if opcode == 0x83 { 1 } else if width == 16 { 2 } else { 4 },
                    )?)
                };
                let value = if opcode == 0x83 {
                    read_i8(bytes, &mut i)? as i64 as u64
                }
                else if width == 16 {
                    if i + 2 > bytes.len() {
                        return Err("truncated imm16".into());
                    }
                    let v = u16::from_le_bytes([bytes[i], bytes[i + 1]]) as u64;
                    i += 2;
                    v
                }
                else {
                    read_u32(bytes, &mut i)? as i32 as i64 as u64
                };
                let op = match group {
                    0 => ArithOp::Add,
                    1 => ArithOp::Or,
                    2 => ArithOp::Adc,
                    3 => ArithOp::Sbb,
                    4 => ArithOp::And,
                    5 => ArithOp::Sub,
                    6 => ArithOp::Xor,
                    _ => ArithOp::Cmp,
                };
                if let Some(mem) = mem {
                    out.push(Instr::ArithMemImm { op, mem, value, width });
                }
                else {
                    let r = (modrm & 7) | rex_b << 3;
                    out.push(if op == ArithOp::Add {
                        Instr::AddRegImm { r, value, width }
                    }
                    else {
                        Instr::ArithRegImm { op, r, value, width }
                    });
                }
            },

            // test r/m64, imm32 (sign-extended)
            0xF7 => {
                let width = operand_width(prefix_66, rex_w);
                let modrm = *bytes.get(i).ok_or("truncated modrm")?;
                i += 1;
                let group = modrm >> 3 & 7;
                if matches!(group, 2 | 3) {
                    if modrm >> 6 == 3 {
                        let r = (modrm & 7) | rex_b << 3;
                        out.push(if group == 2 {
                            Instr::NotReg { r, width }
                        }
                        else {
                            Instr::NegReg { r, width }
                        });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, 0)?;
                        out.push(if group == 2 {
                            Instr::NotMem { mem, width }
                        }
                        else {
                            Instr::NegMem { mem, width }
                        });
                    }
                    continue;
                }
                if group != 0 {
                    return Err("unsupported f7 group form".into());
                }
                let mem = if modrm >> 6 == 3 {
                    None
                }
                else {
                    Some(decode_mem(
                        modrm,
                        rex,
                        bytes,
                        &mut i,
                        base,
                        addr_size,
                        imm_operand_bytes(width),
                    )?)
                };
                let value = read_imm_operand(bytes, &mut i, width)?;
                if let Some(mem) = mem {
                    out.push(Instr::ArithMemImm { op: ArithOp::Test, mem, value, width });
                }
                else {
                    let r = (modrm & 7) | rex_b << 3;
                    out.push(Instr::ArithRegImm { op: ArithOp::Test, r, value, width });
                }
            },

            // add rax, imm32
            0x05 => {
                let width = operand_width(prefix_66, rex_w);
                let value = read_imm_operand(bytes, &mut i, width)?;
                out.push(Instr::AddRegImm { r: 0, value, width });
            },

            // logical/sub/cmp rax, imm32 and test rax, imm32
            0x0D | 0x15 | 0x1D | 0x25 | 0x2D | 0x35 | 0x3D | 0xA9 => {
                let width = operand_width(prefix_66, rex_w);
                let value = read_imm_operand(bytes, &mut i, width)?;
                let op = match opcode {
                    0x0D => ArithOp::Or,
                    0x15 => ArithOp::Adc,
                    0x1D => ArithOp::Sbb,
                    0x25 => ArithOp::And,
                    0x2D => ArithOp::Sub,
                    0x35 => ArithOp::Xor,
                    0x3D => ArithOp::Cmp,
                    _ => ArithOp::Test,
                };
                out.push(Instr::ArithRegImm { op, r: 0, value, width });
            },

            // jmp rel32 / rel8
            0xE9 => {
                let rel = read_u32(bytes, &mut i)? as i32 as i64;
                out.push(Instr::Jmp {
                    target: base.wrapping_add(i as u64).wrapping_add(rel as u64),
                });
                break;
            },

            // push/pop r64
            0x50..=0x57 => {
                if prefix_66 {
                    return Err("16-bit push is not supported".into());
                }
                out.push(Instr::PushReg { r: opcode - 0x50 | rex_b << 3 });
            },
            0x58..=0x5F => {
                if prefix_66 {
                    return Err("16-bit pop is not supported".into());
                }
                out.push(Instr::PopReg { r: opcode - 0x58 | rex_b << 3 });
            },

            // push imm32/imm8 (sign-extended)
            0x68 => {
                if prefix_66 {
                    return Err("16-bit push is not supported".into());
                }
                let value = read_u32(bytes, &mut i)? as i32 as i64 as u64;
                out.push(Instr::PushImm { value });
            },
            0x6A => {
                if prefix_66 {
                    return Err("16-bit push is not supported".into());
                }
                let value = read_i8(bytes, &mut i)? as i64 as u64;
                out.push(Instr::PushImm { value });
            },

            // call rel32
            0xE8 => {
                let rel = read_u32(bytes, &mut i)? as i32 as i64;
                let return_address = base.wrapping_add(i as u64);
                out.push(Instr::Call {
                    target: return_address.wrapping_add(rel as u64),
                    return_address,
                });
                break;
            },

            // ret / ret imm16
            0xC3 | 0xC2 => {
                let adjustment = if opcode == 0xC2 {
                    if i + 2 > bytes.len() { return Err("truncated imm16".into()); }
                    let value = u16::from_le_bytes([bytes[i], bytes[i + 1]]);
                    value
                }
                else {
                    0
                };
                out.push(Instr::Ret { adjustment });
                break;
            },

            // leave
            0xC9 => out.push(Instr::Leave),

            // shl/shr/sar r/m, 1/imm8/cl
            0xD1 | 0xC1 | 0xD3 => {
                let width = operand_width(prefix_66, rex_w);
                let modrm = *bytes.get(i).ok_or("truncated shift modrm")?;
                i += 1;
                let kind = match modrm >> 3 & 7 {
                    4 => ShiftKind::Shl,
                    5 => ShiftKind::Shr,
                    7 => ShiftKind::Sar,
                    _ => return Err("unsupported shift form".into()),
                };
                let count_mask = if width == 64 { 63 } else { 31 };
                if modrm >> 6 != 3 {
                    let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, 0)?;
                    if opcode == 0xD3 {
                        out.push(Instr::ShiftMemCl { kind, mem, width });
                    }
                    else {
                        let count = if opcode == 0xD1 { 1 } else { read_i8(bytes, &mut i)? as u8 };
                        out.push(Instr::ShiftMem {
                            kind,
                            mem,
                            width,
                            count: count & count_mask,
                        });
                    }
                    continue;
                }
                let r = (modrm & 7) | rex_b << 3;
                if opcode == 0xD3 {
                    out.push(Instr::ShiftRegCl { kind, r, width });
                }
                else {
                    let count = if opcode == 0xD1 { 1 } else { read_i8(bytes, &mut i)? as u8 };
                    out.push(Instr::ShiftReg {
                        kind,
                        r,
                        width,
                        count: count & count_mask,
                    });
                }
            },

            // inc/dec/call/jmp register (FF group)
            0xFF => {
                let modrm = *bytes.get(i).ok_or("truncated modrm")?;
                i += 1;
                let group = modrm >> 3 & 7;
                if !matches!(group, 0 | 1 | 2 | 4) {
                    return Err("unsupported ff group form".into());
                }
                if modrm >> 6 != 3 {
                    let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, 0)?;
                    match group {
                        0 | 1 => {
                            out.push(Instr::IncDecMem {
                                mem,
                                width: operand_width(prefix_66, rex_w),
                                decrement: group == 1,
                            });
                        },
                        2 => {
                            out.push(Instr::CallMem { mem, return_address: base + i as u64 });
                            break;
                        },
                        4 => {
                            out.push(Instr::JmpMem { mem });
                            break;
                        },
                        _ => unreachable!(),
                    }
                    continue;
                }
                let r = (modrm & 7) | rex_b << 3;
                if group <= 1 {
                    out.push(Instr::IncDecReg {
                        r,
                        width: operand_width(prefix_66, rex_w),
                        decrement: group == 1,
                    });
                }
                else if group == 2 {
                    out.push(Instr::CallReg { r, return_address: base + i as u64 });
                    break;
                }
                else {
                    out.push(Instr::JmpReg { r });
                    break;
                }
            },
            0xEB => {
                let rel = read_i8(bytes, &mut i)? as i64;
                out.push(Instr::Jmp {
                    target: base.wrapping_add(i as u64).wrapping_add(rel as u64),
                });
                break;
            },

            // jcc rel8
            0x70..=0x7F => {
                let rel = read_i8(bytes, &mut i)? as i64;
                let after = base.wrapping_add(i as u64);
                out.push(Instr::Jcc {
                    code: opcode - 0x70,
                    target: after.wrapping_add(rel as u64),
                    fallthrough: after,
                });
                break;
            },

            // jcc rel32
            0x0F => {
                let second = *bytes.get(i).ok_or("truncated two-byte opcode")?;
                i += 1;
                if prefix_f3 && !(second == 0x1E && bytes.get(i) == Some(&0xFA)) {
                    return Err(format!("unsupported f3 0f opcode {:02x} at {}", second, start));
                }
                if (0x80..=0x8F).contains(&second) {
                    let rel = read_u32(bytes, &mut i)? as i32 as i64;
                    let after = base.wrapping_add(i as u64);
                    out.push(Instr::Jcc {
                        code: second - 0x80,
                        target: after.wrapping_add(rel as u64),
                        fallthrough: after,
                    });
                    break;
                }
                else if second == 0x1F {
                    let modrm = *bytes.get(i).ok_or("truncated nop modrm")?;
                    i += 1;
                    if modrm >> 6 != 3 {
                        let _ = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, 0)?;
                    }
                    out.push(Instr::Nop);
                }
                else if second == 0x1E && bytes.get(i) == Some(&0xFA) {
                    i += 1;
                    out.push(Instr::Nop); // ENDBR64
                }
                else if matches!(second, 0xB6 | 0xB7 | 0xBE | 0xBF) {
                    let modrm = *bytes.get(i).ok_or("truncated movx modrm")?;
                    i += 1;
                    let dst = (modrm >> 3 & 7) | rex_r << 3;
                    let src_width = if matches!(second, 0xB6 | 0xBE) { 8 } else { 16 };
                    let dst_width = operand_width(prefix_66, rex_w);
                    let signed = matches!(second, 0xBE | 0xBF);
                    if modrm >> 6 == 3 {
                        let rm = modrm & 7;
                        let high8 = src_width == 8 && rex == 0 && rm >= 4;
                        let src = if high8 { rm - 4 } else { rm | rex_b << 3 };
                        out.push(Instr::MovExtendReg {
                            dst, src, src_width, dst_width, signed, high8,
                        });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, 0)?;
                        out.push(Instr::MovExtendMem {
                            dst, mem, src_width, dst_width, signed,
                        });
                    }
                }
                else if (0x40..=0x4F).contains(&second) {
                    let modrm = *bytes.get(i).ok_or("truncated cmov modrm")?;
                    i += 1;
                    let dst = (modrm >> 3 & 7) | rex_r << 3;
                    let width = operand_width(prefix_66, rex_w);
                    if modrm >> 6 == 3 {
                        out.push(Instr::CmovRegReg {
                            code: second - 0x40,
                            dst,
                            src: (modrm & 7) | rex_b << 3,
                            width,
                        });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, 0)?;
                        out.push(Instr::CmovRegMem {
                            code: second - 0x40,
                            dst,
                            mem,
                            width,
                        });
                    }
                }
                else if second == 0xAF {
                    let modrm = *bytes.get(i).ok_or("truncated imul modrm")?;
                    i += 1;
                    let dst = (modrm >> 3 & 7) | rex_r << 3;
                    let width = operand_width(prefix_66, rex_w);
                    if modrm >> 6 == 3 {
                        out.push(Instr::ImulRegReg {
                            dst,
                            lhs: dst,
                            rhs: (modrm & 7) | rex_b << 3,
                            width,
                        });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, 0)?;
                        out.push(Instr::ImulRegMem { dst, mem, value: None, width });
                    }
                }
                else if (0x90..=0x9F).contains(&second) {
                    let modrm = *bytes.get(i).ok_or("truncated setcc modrm")?;
                    i += 1;
                    if modrm >> 6 == 3 {
                        let rm = modrm & 7;
                        let high8 = rex == 0 && rm >= 4;
                        out.push(Instr::SetccReg {
                            code: second - 0x90,
                            dst: if high8 { rm - 4 } else { rm | rex_b << 3 },
                            high8,
                        });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, 0)?;
                        out.push(Instr::SetccMem {
                            code: second - 0x90,
                            mem,
                        });
                    }
                }
                else if (0xC8..=0xCF).contains(&second) {
                    if prefix_66 {
                        return Err("bswap with a 16-bit operand is undefined".into());
                    }
                    out.push(Instr::Bswap {
                        r: second - 0xC8 | rex_b << 3,
                        width: if rex_w { 64 } else { 32 },
                    });
                }
                // CPUID
                else if second == 0xA2 {
                    out.push(Instr::Cpuid);
                }
                else {
                    return Err(format!("unsupported opcode 0f {:02x} at {}", second, start));
                }
            },

            0xF4 => {
                out.push(Instr::Hlt);
                break;
            },

            _ => return Err(format!("unsupported opcode {:02x} at {}", opcode, start)),
        }
    }

    out.end_rip = base.wrapping_add(i as u64);
    Ok(out)
}

pub fn decode_block(base: u64, bytes: &[u8]) -> Result<Vec<Instr>, String> {
    Ok(decode_block_with_rips(base, bytes)?.instrs)
}

fn find_reg(locals: &[(u8, WasmLocalI64)], r: u8) -> Option<usize> {
    locals.iter().position(|(rr, _)| *rr == r)
}

fn load_reg(b: &mut WasmBuilder, locals: &mut Vec<(u8, WasmLocalI64)>, r: u8) -> usize {
    if let Some(i) = find_reg(locals, r) {
        return i;
    }
    if r >= 8 {
        b.const_i32(REG_EXT + 8 * (r as i32 - 8));
        b.load_aligned_i64(0);
    }
    else {
        b.const_i32(REG_LOW + 4 * r as i32);
        b.load_aligned_i32(0);
        b.extend_unsigned_i32_to_i64();
        b.const_i32(REG_HIGH + 4 * r as i32);
        b.load_aligned_i32(0);
        b.extend_unsigned_i32_to_i64();
        b.const_i64(32);
        b.shl_i64();
        b.or_i64();
    }
    let local = b.set_new_local_i64();
    locals.push((r, local));
    locals.len() - 1
}

fn store_reg(b: &mut WasmBuilder, local: &WasmLocalI64, r: u8) {
    if r >= 8 {
        b.const_i32(REG_EXT + 8 * (r as i32 - 8));
        b.get_local_i64(local);
        b.store_aligned_i64(0);
    }
    else {
        b.const_i32(REG_LOW + 4 * r as i32);
        b.get_local_i64(local);
        b.wrap_i64_to_i32();
        b.store_aligned_i32(0);

        b.const_i32(REG_HIGH + 4 * r as i32);
        b.get_local_i64(local);
        b.const_i64(32);
        b.shr_u_i64();
        b.wrap_i64_to_i32();
        b.store_aligned_i32(0);
    }
}

fn gen_effective_addr(b: &mut WasmBuilder, locals: &mut Vec<(u8, WasmLocalI64)>, mem: &Mem) {
    let mask32 = mem.addr_size == 32;
    let mut have_term = false;
    if let Some(base) = mem.base {
        let i = load_reg(b, locals, base);
        b.get_local_i64(&locals[i].1);
        if mask32 {
            b.const_i64(0xFFFF_FFFF);
            b.and_i64();
        }
        have_term = true;
    }
    if let Some(index) = mem.index {
        let i = load_reg(b, locals, index);
        b.get_local_i64(&locals[i].1);
        if mask32 {
            b.const_i64(0xFFFF_FFFF);
            b.and_i64();
        }
        if mem.scale > 0 {
            b.const_i64(1 << mem.scale);
            b.mul_i64();
        }
        if have_term {
            b.add_i64();
        }
        have_term = true;
    }
    if !have_term {
        b.const_i64(0);
    }
    if mem.disp != 0 {
        b.const_i64(mem.disp);
        b.add_i64();
    }
    // A 32-bit effective address wraps at 2^32 and is zero-extended.
    if mask32 {
        b.const_i64(0xFFFF_FFFF);
        b.and_i64();
    }
}

fn gen_check_memory_fault(b: &mut WasmBuilder, locals: &[(u8, WasmLocalI64)]) {
    b.call_fn0_ret("jit64_memory_faulted");
    b.if_void();
    emit_registers_back(b, locals);
    b.return_();
    b.block_end();
}

fn gen_memory_address_local(
    b: &mut WasmBuilder,
    locals: &mut Vec<(u8, WasmLocalI64)>,
    mem: &Mem,
) -> WasmLocalI64 {
    gen_effective_addr(b, locals, mem);
    b.set_new_local_i64()
}

fn gen_memory_read(
    b: &mut WasmBuilder,
    locals: &[(u8, WasmLocalI64)],
    address: &WasmLocalI64,
    width: u8,
) -> WasmLocalI64 {
    b.get_local_i64(address);
    b.const_i32(width as i32);
    b.call_fn2_i64_i32_ret_i64("jit64_mem_read");
    let value = b.set_new_local_i64();
    gen_check_memory_fault(b, locals);
    value
}

fn gen_memory_write(
    b: &mut WasmBuilder,
    locals: &[(u8, WasmLocalI64)],
    address: &WasmLocalI64,
    value: &WasmLocalI64,
    width: u8,
) {
    b.get_local_i64(address);
    b.get_local_i64(value);
    b.const_i32(width as i32);
    b.call_fn3_i64_i64_i32("jit64_mem_write");
    gen_check_memory_fault(b, locals);
}

fn gen_memory_probe_write(
    b: &mut WasmBuilder,
    locals: &[(u8, WasmLocalI64)],
    address: &WasmLocalI64,
    width: u8,
) {
    b.get_local_i64(address);
    b.const_i32(width as i32 | 0x100);
    b.call_fn2_i64_i32_ret("jit64_mem_probe");
    b.if_void();
    emit_registers_back(b, locals);
    b.return_();
    b.block_end();
}

fn emit_push_value(
    b: &mut WasmBuilder,
    locals: &mut Vec<(u8, WasmLocalI64)>,
    value: &WasmLocalI64,
) {
    let rsp = load_reg(b, locals, 4);
    b.get_local_i64(&locals[rsp].1);
    b.const_i64(8);
    b.sub_i64();
    let new_rsp = b.set_new_local_i64();
    gen_memory_probe_write(b, locals, &new_rsp, 64);
    gen_memory_write(b, locals, &new_rsp, value, 64);
    b.get_local_i64(&new_rsp);
    b.set_local_i64(&locals[rsp].1);
    b.free_local_i64(new_rsp);
}

fn emit_pop_value(b: &mut WasmBuilder, locals: &mut Vec<(u8, WasmLocalI64)>) -> WasmLocalI64 {
    let rsp = load_reg(b, locals, 4);
    b.get_local_i64(&locals[rsp].1);
    let address = b.set_new_local_i64();
    let value = gen_memory_read(b, locals, &address, 64);
    b.free_local_i64(address);

    b.get_local_i64(&locals[rsp].1);
    b.const_i64(8);
    b.add_i64();
    b.set_local_i64(&locals[rsp].1);
    value
}

// Mask the i64 on the stack to the operand width.
fn emit_mask(b: &mut WasmBuilder, width: u8) {
    if width < 64 {
        b.const_i64(width_mask(width) as i64);
        b.and_i64();
    }
}

// Write `value` into the register local `dst`. A 16-bit write preserves the
// upper bits; 32-bit writes zero-extend (the caller masks to 32 bits).
fn emit_write_reg(b: &mut WasmBuilder, dst: &WasmLocalI64, value: &WasmLocalI64, width: u8) {
    if width == 16 {
        b.get_local_i64(dst);
        b.const_i64(!0xFFFF);
        b.and_i64();
        b.get_local_i64(value);
        b.const_i64(0xFFFF);
        b.and_i64();
        b.or_i64();
        b.set_local_i64(dst);
    }
    else {
        b.get_local_i64(value);
        b.set_local_i64(dst);
    }
}

// Write `value` into guest register `r`, loading its previous value when a
// 16-bit write must preserve the upper bits and the register is not resident.
fn write_reg_value(
    b: &mut WasmBuilder,
    locals: &mut Vec<(u8, WasmLocalI64)>,
    r: u8,
    value: &WasmLocalI64,
    width: u8,
) {
    if let Some(di) = find_reg(locals, r) {
        emit_write_reg(b, &locals[di].1, value, width);
    }
    else if width == 16 {
        let di = load_reg(b, locals, r);
        emit_write_reg(b, &locals[di].1, value, width);
    }
    else {
        b.get_local_i64(value);
        let local = b.set_new_local_i64();
        locals.push((r, local));
    }
}

fn gen_arith_flags(
    b: &mut WasmBuilder,
    dst: &WasmLocalI64,
    src: &WasmLocalI64,
    result: &WasmLocalI64,
    op: ArithOp,
    width: u8,
) {
    // stack: [addr]
    b.const_i32(FLAGS_ADDR);
    b.const_i32(FLAGS_ADDR);
    b.load_aligned_i32(0);
    b.const_i32(!FLAG_MASK);
    b.and_i32();
    // ZF
    b.get_local_i64(result);
    b.eqz_i64();
    b.const_i32(6);
    b.shl_i32();
    b.or_i32();
    // SF
    b.get_local_i64(result);
    b.const_i64(width_sign_bit(width) as i64);
    b.and_i64();
    b.const_i64(0);
    b.ne_i64();
    b.const_i32(7);
    b.shl_i32();
    b.or_i32();
    // CF
    match op {
        ArithOp::Test | ArithOp::And | ArithOp::Or | ArithOp::Xor => b.const_i32(0),
        ArithOp::Sub | ArithOp::Cmp => {
            b.get_local_i64(dst);
            b.get_local_i64(src);
            b.lt_u_i64();
        },
        ArithOp::Add => {
            b.get_local_i64(result);
            b.get_local_i64(dst);
            b.lt_u_i64();
        },
        ArithOp::Adc | ArithOp::Sbb => unreachable!(),
    }
    b.or_i32();
    // OF
    if op.is_logic() {
        b.const_i32(0);
    }
    else {
        b.get_local_i64(dst);
        b.get_local_i64(result);
        b.xor_i64();
        if matches!(op, ArithOp::Sub | ArithOp::Cmp) {
            b.get_local_i64(dst);
        }
        else {
            b.get_local_i64(result);
        }
        b.get_local_i64(src);
        b.xor_i64();
        b.and_i64();
        b.const_i64(width_sign_bit(width) as i64);
        b.and_i64();
        b.const_i64(0);
        b.ne_i64();
    }
    b.const_i32(11);
    b.shl_i32();
    b.or_i32();
    // AF: carry/borrow out of bit 3 (undefined for test).
    if !op.is_logic() {
        b.get_local_i64(dst);
        b.get_local_i64(src);
        b.xor_i64();
        b.get_local_i64(result);
        b.xor_i64();
        b.const_i64(0x10);
        b.and_i64();
        b.wrap_i64_to_i32();
        b.or_i32();
    }
    // PF: even parity of the low result byte.
    b.get_local_i64(result);
    b.wrap_i64_to_i32();
    b.const_i32(0xFF);
    b.and_i32();
    let parity = b.set_new_local();
    for shift in [4, 2, 1] {
        b.get_local(&parity);
        b.get_local(&parity);
        b.const_i32(shift);
        b.shr_u_i32();
        b.xor_i32();
        b.set_local(&parity);
    }
    b.get_local(&parity);
    b.const_i32(1);
    b.and_i32();
    b.eqz_i32();
    b.const_i32(2);
    b.shl_i32();
    b.or_i32();
    b.free_local(parity);
    b.store_aligned_i32(0);
    b.const_i32(FLAGS_CHANGED_ADDR);
    b.const_i32(0);
    b.store_aligned_i32(0);
}

fn emit_arith(
    b: &mut WasmBuilder,
    dst: &WasmLocalI64,
    src: &WasmLocalI64,
    op: ArithOp,
    width: u8,
) {
    if matches!(op, ArithOp::Adc | ArithOp::Sbb) {
        b.get_local_i64(dst);
        b.get_local_i64(src);
        b.const_i32(width as i32 | if op == ArithOp::Sbb { 0x100 } else { 0 });
        b.call_fn3_i64_i64_i32_ret_i64("jit64_adc_sbb");
        let result = b.set_new_local_i64();
        if op.writes_result() {
            emit_write_reg(b, dst, &result, width);
        }
        b.free_local_i64(result);
        return;
    }
    b.get_local_i64(dst);
    emit_mask(b, width);
    let lhs = b.set_new_local_i64();
    b.get_local_i64(src);
    emit_mask(b, width);
    let rhs = b.set_new_local_i64();
    b.get_local_i64(&lhs);
    b.get_local_i64(&rhs);
    match op {
        ArithOp::Add => b.add_i64(),
        ArithOp::Adc | ArithOp::Sbb => unreachable!(),
        ArithOp::Sub | ArithOp::Cmp => b.sub_i64(),
        ArithOp::Test | ArithOp::And => b.and_i64(),
        ArithOp::Or => b.or_i64(),
        ArithOp::Xor => b.xor_i64(),
    }
    emit_mask(b, width);
    let result = b.set_new_local_i64();
    gen_arith_flags(b, &lhs, &rhs, &result, op, width);
    if op.writes_result() {
        emit_write_reg(b, dst, &result, width);
    }
    b.free_local_i64(lhs);
    b.free_local_i64(rhs);
    b.free_local_i64(result);
}

fn emit_arith_mem(
    b: &mut WasmBuilder,
    locals: &mut Vec<(u8, WasmLocalI64)>,
    mem: &Mem,
    src: &WasmLocalI64,
    op: ArithOp,
    width: u8,
) {
    let address = gen_memory_address_local(b, locals, mem);
    if op.writes_result() {
        gen_memory_probe_write(b, locals, &address, width);
    }
    let dst = gen_memory_read(b, locals, &address, width);
    emit_arith(b, &dst, src, op, width);
    if op.writes_result() {
        gen_memory_write(b, locals, &address, &dst, width);
    }
    b.free_local_i64(address);
    b.free_local_i64(dst);
}

fn emit_extend(b: &mut WasmBuilder, src_width: u8, dst_width: u8, signed: bool) {
    let mask = match src_width {
        8 => 0xFF,
        16 => 0xFFFF,
        32 => 0xFFFF_FFFF,
        _ => unreachable!(),
    };
    b.const_i64(mask);
    b.and_i64();
    if signed {
        let sign = match src_width {
            8 => 0x80,
            16 => 0x8000,
            32 => 0x8000_0000,
            _ => unreachable!(),
        };
        b.const_i64(sign);
        b.xor_i64();
        b.const_i64(sign);
        b.sub_i64();
    }
    if dst_width == 32 {
        b.const_i64(0xFFFF_FFFF);
        b.and_i64();
    }
    else if dst_width == 16 {
        b.const_i64(0xFFFF);
        b.and_i64();
    }
}

fn emit_imul(
    b: &mut WasmBuilder,
    dst: &WasmLocalI64,
    lhs: &WasmLocalI64,
    rhs: &WasmLocalI64,
    width: u8,
) {
    b.get_local_i64(lhs);
    b.get_local_i64(rhs);
    b.const_i32(width as i32);
    b.call_fn3_i64_i64_i32_ret_i64("jit64_imul");
    let result = b.set_new_local_i64();
    emit_write_reg(b, dst, &result, width);
    b.free_local_i64(result);
}

/// Push (flags >> bit) & 1.
fn push_flag_bit(b: &mut WasmBuilder, bit: i32) {
    b.const_i32(FLAGS_ADDR);
    b.load_aligned_i32(0);
    b.const_i32(bit);
    b.shr_u_i32();
    b.const_i32(1);
    b.and_i32();
}

/// Push an i32 that is 1 if the condition holds, 0 otherwise.
fn gen_condition(b: &mut WasmBuilder, code: u8) {
    match code {
        0x0 => push_flag_bit(b, 11),                    // O
        0x1 => {
            push_flag_bit(b, 11);
            b.eqz_i32();
        },                                              // NO
        0x2 => push_flag_bit(b, 0),                     // B/C
        0x3 => {
            push_flag_bit(b, 0);
            b.eqz_i32();
        },                                              // AE/NC
        0x4 => push_flag_bit(b, 6),                     // E/Z
        0x5 => {
            push_flag_bit(b, 6);
            b.eqz_i32();
        },                                              // NE/NZ
        0x6 => {
            push_flag_bit(b, 0);
            push_flag_bit(b, 6);
            b.or_i32();
        },                                              // BE
        0x7 => {
            push_flag_bit(b, 0);
            push_flag_bit(b, 6);
            b.or_i32();
            b.eqz_i32();
        },                                              // A
        0x8 => push_flag_bit(b, 7),                     // S
        0x9 => {
            push_flag_bit(b, 7);
            b.eqz_i32();
        },                                              // NS
        0xA => push_flag_bit(b, 2),                     // P
        0xB => {
            push_flag_bit(b, 2);
            b.eqz_i32();
        },                                              // NP
        0xC => {
            push_flag_bit(b, 7);
            push_flag_bit(b, 11);
            b.ne_i32();
        },                                              // L
        0xD => {
            push_flag_bit(b, 7);
            push_flag_bit(b, 11);
            b.eq_i32();
        },                                              // GE
        0xE => {
            push_flag_bit(b, 6);
            push_flag_bit(b, 7);
            push_flag_bit(b, 11);
            b.ne_i32();
            b.or_i32();
        },                                              // LE
        0xF => {
            push_flag_bit(b, 6);
            b.eqz_i32();
            push_flag_bit(b, 7);
            push_flag_bit(b, 11);
            b.eq_i32();
            b.and_i32();
        },                                              // G
        _ => unreachable!(),
    }
}

fn gen_set_ip(b: &mut WasmBuilder, value: u64) {
    b.const_i32(IP_ADDR);
    b.const_i64(value as i64);
    b.store_aligned_i64(0);
}

// Early returns happen after this, so an instruction that faults or is skipped
// is not counted.
fn bump_instruction_counter(b: &mut WasmBuilder) {
    b.const_i32(INSTRUCTION_COUNTER_ADDR);
    b.load_fixed_i32(INSTRUCTION_COUNTER_ADDR as u32);
    b.const_i32(1);
    b.add_i32();
    b.store_aligned_i32(0);
}

fn emit_registers_back(b: &mut WasmBuilder, locals: &[(u8, WasmLocalI64)]) {
    for (r, local) in locals {
        store_reg(b, local, *r);
    }
}

// `block_end` is where execution falls through when the block ends normally.
fn compile_block_with_rips(instrs: &[Instr], rips: &[u64], block_end: u64) -> Vec<u8> {
    let mut b = WasmBuilder::new();
    let mut locals: Vec<(u8, WasmLocalI64)> = Vec::new();
    let mut terminated = false;

    for (index, instr) in instrs.iter().enumerate() {
        if let Some(instruction_rip) = rips.get(index) {
            b.const_i32(PREVIOUS_RIP_ADDR);
            b.const_i64(*instruction_rip as i64);
            b.store_aligned_i64(0);
        }
        bump_instruction_counter(&mut b);
        match *instr {
            Instr::MovRegImm { r, value, width } => {
                b.const_i64(value as i64);
                let local = b.set_new_local_i64();
                write_reg_value(&mut b, &mut locals, r, &local, width);
                b.free_local_i64(local);
            },
            Instr::MovRegReg { dst, src, width } => {
                let si = load_reg(&mut b, &mut locals, src);
                b.get_local_i64(&locals[si].1);
                emit_mask(&mut b, width);
                let value = b.set_new_local_i64();
                write_reg_value(&mut b, &mut locals, dst, &value, width);
                b.free_local_i64(value);
            },
            Instr::MovRegMem { dst, mem, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let value = gen_memory_read(&mut b, &locals, &address, width);
                write_reg_value(&mut b, &mut locals, dst, &value, width);
                b.free_local_i64(address);
                b.free_local_i64(value);
            },
            Instr::MovMemReg { mem, src, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let si = load_reg(&mut b, &mut locals, src);
                let value = locals[si].1.unsafe_clone();
                gen_memory_write(&mut b, &locals, &address, &value, width);
                b.free_local_i64(address);
            },
            Instr::MovMemImm { mem, value, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                b.const_i64(value as i64);
                let value = b.set_new_local_i64();
                gen_memory_write(&mut b, &locals, &address, &value, width);
                b.free_local_i64(value);
                b.free_local_i64(address);
            },
            Instr::MovExtendReg { dst, src, src_width, dst_width, signed, high8 } => {
                let si = load_reg(&mut b, &mut locals, src);
                b.get_local_i64(&locals[si].1);
                if high8 {
                    b.const_i64(8);
                    b.shr_u_i64();
                }
                emit_extend(&mut b, src_width, dst_width, signed);
                let value = b.set_new_local_i64();
                write_reg_value(&mut b, &mut locals, dst, &value, dst_width);
                b.free_local_i64(value);
            },
            Instr::MovExtendMem { dst, mem, src_width, dst_width, signed } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let value = gen_memory_read(&mut b, &locals, &address, src_width);
                b.get_local_i64(&value);
                emit_extend(&mut b, src_width, dst_width, signed);
                let extended = b.set_new_local_i64();
                write_reg_value(&mut b, &mut locals, dst, &extended, dst_width);
                b.free_local_i64(extended);
                b.free_local_i64(address);
                b.free_local_i64(value);
            },
            Instr::XchgRegReg { a, b: other, width } => {
                let ai = load_reg(&mut b, &mut locals, a);
                let bi = load_reg(&mut b, &mut locals, other);
                b.get_local_i64(&locals[ai].1);
                emit_mask(&mut b, width);
                let old_a = b.set_new_local_i64();
                b.get_local_i64(&locals[bi].1);
                emit_mask(&mut b, width);
                let old_b = b.set_new_local_i64();
                emit_write_reg(&mut b, &locals[ai].1, &old_b, width);
                emit_write_reg(&mut b, &locals[bi].1, &old_a, width);
                b.free_local_i64(old_a);
                b.free_local_i64(old_b);
            },
            Instr::XchgMemReg { mem, r, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                gen_memory_probe_write(&mut b, &locals, &address, width);
                let old_mem = gen_memory_read(&mut b, &locals, &address, width);
                let ri = load_reg(&mut b, &mut locals, r);
                let old_reg = locals[ri].1.unsafe_clone();
                gen_memory_write(&mut b, &locals, &address, &old_reg, width);
                b.get_local_i64(&old_mem);
                emit_mask(&mut b, width);
                let value = b.set_new_local_i64();
                emit_write_reg(&mut b, &locals[ri].1, &value, width);
                b.free_local_i64(value);
                b.free_local_i64(address);
                b.free_local_i64(old_mem);
            },
            Instr::NotReg { r, width } => {
                let ri = load_reg(&mut b, &mut locals, r);
                b.get_local_i64(&locals[ri].1);
                b.const_i64(width_mask(width) as i64);
                b.xor_i64();
                emit_mask(&mut b, width);
                b.set_local_i64(&locals[ri].1);
            },
            Instr::NotMem { mem, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                gen_memory_probe_write(&mut b, &locals, &address, width);
                let value = gen_memory_read(&mut b, &locals, &address, width);
                b.get_local_i64(&value);
                b.const_i64(width_mask(width) as i64);
                b.xor_i64();
                emit_mask(&mut b, width);
                b.set_local_i64(&value);
                gen_memory_write(&mut b, &locals, &address, &value, width);
                b.free_local_i64(address);
                b.free_local_i64(value);
            },
            Instr::NegReg { r, width } => {
                let ri = load_reg(&mut b, &mut locals, r);
                b.const_i64(0);
                let zero = b.set_new_local_i64();
                let source = locals[ri].1.unsafe_clone();
                emit_arith(&mut b, &zero, &source, ArithOp::Sub, width);
                emit_write_reg(&mut b, &locals[ri].1, &zero, width);
                b.free_local_i64(zero);
            },
            Instr::NegMem { mem, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                gen_memory_probe_write(&mut b, &locals, &address, width);
                let source = gen_memory_read(&mut b, &locals, &address, width);
                b.const_i64(0);
                let result = b.set_new_local_i64();
                emit_arith(&mut b, &result, &source, ArithOp::Sub, width);
                gen_memory_write(&mut b, &locals, &address, &result, width);
                b.free_local_i64(address);
                b.free_local_i64(source);
                b.free_local_i64(result);
            },
            Instr::IncDecReg { r, width, decrement } => {
                b.const_i32(FLAGS_ADDR);
                b.load_aligned_i32(0);
                b.const_i32(1);
                b.and_i32();
                let carry = b.set_new_local();
                let ri = load_reg(&mut b, &mut locals, r);
                b.const_i64(1);
                let one = b.set_new_local_i64();
                emit_arith(
                    &mut b,
                    &locals[ri].1,
                    &one,
                    if decrement { ArithOp::Sub } else { ArithOp::Add },
                    width,
                );
                b.const_i32(FLAGS_ADDR);
                b.const_i32(FLAGS_ADDR);
                b.load_aligned_i32(0);
                b.const_i32(!1);
                b.and_i32();
                b.get_local(&carry);
                b.or_i32();
                b.store_aligned_i32(0);
                b.free_local(carry);
                b.free_local_i64(one);
            },
            Instr::IncDecMem { mem, width, decrement } => {
                b.const_i32(FLAGS_ADDR);
                b.load_aligned_i32(0);
                b.const_i32(1);
                b.and_i32();
                let carry = b.set_new_local();
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                gen_memory_probe_write(&mut b, &locals, &address, width);
                let value = gen_memory_read(&mut b, &locals, &address, width);
                b.const_i64(1);
                let one = b.set_new_local_i64();
                emit_arith(
                    &mut b,
                    &value,
                    &one,
                    if decrement { ArithOp::Sub } else { ArithOp::Add },
                    width,
                );
                gen_memory_write(&mut b, &locals, &address, &value, width);
                b.const_i32(FLAGS_ADDR);
                b.const_i32(FLAGS_ADDR);
                b.load_aligned_i32(0);
                b.const_i32(!1);
                b.and_i32();
                b.get_local(&carry);
                b.or_i32();
                b.store_aligned_i32(0);
                b.free_local(carry);
                b.free_local_i64(address);
                b.free_local_i64(value);
                b.free_local_i64(one);
            },
            Instr::CmovRegReg { code, dst, src, width } => {
                let si = load_reg(&mut b, &mut locals, src);
                let di = load_reg(&mut b, &mut locals, dst);
                gen_condition(&mut b, code);
                b.if_void();
                b.get_local_i64(&locals[si].1);
                emit_mask(&mut b, width);
                let value = b.set_new_local_i64();
                emit_write_reg(&mut b, &locals[di].1, &value, width);
                b.free_local_i64(value);
                b.block_end();
            },
            Instr::CmovRegMem { code, dst, mem, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let value = gen_memory_read(&mut b, &locals, &address, width);
                let di = load_reg(&mut b, &mut locals, dst);
                gen_condition(&mut b, code);
                b.if_void();
                b.get_local_i64(&value);
                emit_mask(&mut b, width);
                let masked = b.set_new_local_i64();
                emit_write_reg(&mut b, &locals[di].1, &masked, width);
                b.free_local_i64(masked);
                b.block_end();
                b.free_local_i64(address);
                b.free_local_i64(value);
            },
            Instr::SetccReg { code, dst, high8 } => {
                let di = load_reg(&mut b, &mut locals, dst);
                b.get_local_i64(&locals[di].1);
                b.const_i64(if high8 { !0xFF00 } else { !0xFF });
                b.and_i64();
                gen_condition(&mut b, code);
                b.extend_unsigned_i32_to_i64();
                if high8 {
                    b.const_i64(8);
                    b.shl_i64();
                }
                b.or_i64();
                b.set_local_i64(&locals[di].1);
            },
            Instr::SetccMem { code, mem } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                gen_condition(&mut b, code);
                b.extend_unsigned_i32_to_i64();
                let value = b.set_new_local_i64();
                gen_memory_write(&mut b, &locals, &address, &value, 8);
                b.free_local_i64(value);
                b.free_local_i64(address);
            },
            Instr::ShiftReg { kind, r, width, count } => {
                if count == 0 {
                    continue;
                }
                let ri = load_reg(&mut b, &mut locals, r);
                b.get_local_i64(&locals[ri].1);
                emit_mask(&mut b, width);
                let old = b.set_new_local_i64();

                b.get_local_i64(&old);
                if kind == ShiftKind::Sar && width < 64 {
                    b.const_i64(width_sign_bit(width) as i64);
                    b.xor_i64();
                    b.const_i64(width_sign_bit(width) as i64);
                    b.sub_i64();
                }
                b.const_i64(count as i64);
                match kind {
                    ShiftKind::Shl => b.shl_i64(),
                    ShiftKind::Shr => b.shr_u_i64(),
                    ShiftKind::Sar => b.shr_s_i64(),
                }
                emit_mask(&mut b, width);
                let result = b.set_new_local_i64();

                gen_arith_flags(&mut b, &old, &old, &result, ArithOp::Test, width);

                // Carry is the last bit shifted out. A left shift by at least the
                // operand width is undefined; report no carry instead of shifting
                // by a negative amount.
                if kind == ShiftKind::Shl && count >= width {
                    b.const_i32(0);
                }
                else {
                    b.get_local_i64(&old);
                    if kind == ShiftKind::Shl {
                        b.const_i64((width - count) as i64);
                    }
                    else {
                        b.const_i64((count - 1) as i64);
                    }
                    b.shr_u_i64();
                    b.const_i64(1);
                    b.and_i64();
                    b.wrap_i64_to_i32();
                }
                let carry = b.set_new_local();

                b.const_i32(FLAGS_ADDR);
                b.const_i32(FLAGS_ADDR);
                b.load_aligned_i32(0);
                b.get_local(&carry);
                b.or_i32();
                match (kind, count) {
                    (_, 2..) => b.const_i32(0),
                    (ShiftKind::Shl, 1) => {
                        b.get_local_i64(&result);
                        b.const_i64((width - 1) as i64);
                        b.shr_u_i64();
                        b.wrap_i64_to_i32();
                        b.const_i32(1);
                        b.and_i32();
                        b.get_local(&carry);
                        b.xor_i32();
                    },
                    (ShiftKind::Shr, 1) => {
                        b.get_local_i64(&old);
                        b.const_i64((width - 1) as i64);
                        b.shr_u_i64();
                        b.wrap_i64_to_i32();
                        b.const_i32(1);
                        b.and_i32();
                    },
                    (ShiftKind::Sar, 1) => b.const_i32(0),
                    _ => unreachable!(),
                }
                b.const_i32(11);
                b.shl_i32();
                b.or_i32();
                b.store_aligned_i32(0);

                emit_write_reg(&mut b, &locals[ri].1, &result, width);
                b.free_local(carry);
                b.free_local_i64(old);
                b.free_local_i64(result);
            },
            Instr::ShiftRegCl { kind, r, width } => {
                let ri = load_reg(&mut b, &mut locals, r);
                let ci = load_reg(&mut b, &mut locals, 1);
                b.get_local_i64(&locals[ci].1);
                b.const_i64(if width == 64 { 63 } else { 31 });
                b.and_i64();
                b.eqz_i64();
                b.if_void();
                b.else_();
                b.get_local_i64(&locals[ri].1);
                b.get_local_i64(&locals[ci].1);
                b.const_i32(width as i32 | (kind as i32) << 8);
                b.call_fn3_i64_i64_i32_ret_i64("jit64_shift");
                let result = b.set_new_local_i64();
                emit_write_reg(&mut b, &locals[ri].1, &result, width);
                b.free_local_i64(result);
                b.block_end();
            },
            Instr::ShiftMem { kind, mem, width, count } => {
                if count == 0 {
                    continue;
                }
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                gen_memory_probe_write(&mut b, &locals, &address, width);
                let value = gen_memory_read(&mut b, &locals, &address, width);
                b.get_local_i64(&value);
                b.const_i64(count as i64);
                b.const_i32(width as i32 | (kind as i32) << 8);
                b.call_fn3_i64_i64_i32_ret_i64("jit64_shift");
                let result = b.set_new_local_i64();
                gen_memory_write(&mut b, &locals, &address, &result, width);
                b.free_local_i64(result);
                b.free_local_i64(address);
                b.free_local_i64(value);
            },
            Instr::ShiftMemCl { kind, mem, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let value = gen_memory_read(&mut b, &locals, &address, width);
                let ci = load_reg(&mut b, &mut locals, 1);
                b.get_local_i64(&locals[ci].1);
                b.const_i64(if width == 64 { 63 } else { 31 });
                b.and_i64();
                b.eqz_i64();
                b.if_void();
                b.else_();
                gen_memory_probe_write(&mut b, &locals, &address, width);
                b.get_local_i64(&value);
                b.get_local_i64(&locals[ci].1);
                b.const_i32(width as i32 | (kind as i32) << 8);
                b.call_fn3_i64_i64_i32_ret_i64("jit64_shift");
                let result = b.set_new_local_i64();
                gen_memory_write(&mut b, &locals, &address, &result, width);
                b.free_local_i64(result);
                b.block_end();
                b.free_local_i64(address);
                b.free_local_i64(value);
            },
            Instr::Cpuid => {
                // CPUID runs in the interpreter against the register file, so
                // flush the resident registers and reload them afterwards.
                emit_registers_back(&mut b, &locals);
                for (_, local) in &locals {
                    b.free_local_i64(local.unsafe_clone());
                }
                locals.clear();
                b.call_fn0("jit64_cpuid");
            },
            Instr::ImulRegReg { dst, lhs, rhs, width } => {
                let li = load_reg(&mut b, &mut locals, lhs);
                let ri = load_reg(&mut b, &mut locals, rhs);
                let di = load_reg(&mut b, &mut locals, dst);
                let left = locals[li].1.unsafe_clone();
                let right = locals[ri].1.unsafe_clone();
                let destination = locals[di].1.unsafe_clone();
                emit_imul(&mut b, &destination, &left, &right, width);
            },
            Instr::ImulRegImm { dst, src, value, width } => {
                let si = load_reg(&mut b, &mut locals, src);
                let di = load_reg(&mut b, &mut locals, dst);
                b.const_i64(value as i64);
                let immediate = b.set_new_local_i64();
                let source = locals[si].1.unsafe_clone();
                let destination = locals[di].1.unsafe_clone();
                emit_imul(&mut b, &destination, &source, &immediate, width);
                b.free_local_i64(immediate);
            },
            Instr::ImulRegMem { dst, mem, value, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let source = gen_memory_read(&mut b, &locals, &address, width);
                let rhs = if let Some(value) = value {
                    b.const_i64(value as i64);
                    b.set_new_local_i64()
                }
                else {
                    source.unsafe_clone()
                };
                let di = load_reg(&mut b, &mut locals, dst);
                let lhs = if value.is_some() {
                    source.unsafe_clone()
                }
                else {
                    locals[di].1.unsafe_clone()
                };
                let destination = locals[di].1.unsafe_clone();
                emit_imul(&mut b, &destination, &lhs, &rhs, width);
                if value.is_some() {
                    b.free_local_i64(rhs);
                }
                b.free_local_i64(address);
                b.free_local_i64(source);
            },
            Instr::Bswap { r, width } => {
                let ri = load_reg(&mut b, &mut locals, r);
                b.get_local_i64(&locals[ri].1);
                b.const_i32(width as i32);
                b.call_fn2_i64_i32_ret_i64("jit64_bswap");
                b.set_local_i64(&locals[ri].1);
            },
            Instr::Lea { dst, mem, width } => {
                gen_effective_addr(&mut b, &mut locals, &mem);
                emit_mask(&mut b, width);
                let value = b.set_new_local_i64();
                write_reg_value(&mut b, &mut locals, dst, &value, width);
                b.free_local_i64(value);
            },
            Instr::AddRegReg { dst, src, width } => {
                let di = load_reg(&mut b, &mut locals, dst);
                let si = load_reg(&mut b, &mut locals, src);
                emit_arith(&mut b, &locals[di].1, &locals[si].1, ArithOp::Add, width);
            },
            Instr::AddRegImm { r, value, width } => {
                let di = load_reg(&mut b, &mut locals, r);
                b.const_i64(value as i64);
                let src = b.set_new_local_i64();
                emit_arith(&mut b, &locals[di].1, &src, ArithOp::Add, width);
                b.free_local_i64(src);
            },
            Instr::ArithRegReg { op, dst, src, width } => {
                let di = load_reg(&mut b, &mut locals, dst);
                let si = load_reg(&mut b, &mut locals, src);
                emit_arith(&mut b, &locals[di].1, &locals[si].1, op, width);
            },
            Instr::ArithRegImm { op, r, value, width } => {
                let di = load_reg(&mut b, &mut locals, r);
                b.const_i64(value as i64);
                let src = b.set_new_local_i64();
                emit_arith(&mut b, &locals[di].1, &src, op, width);
                b.free_local_i64(src);
            },
            Instr::ArithRegMem { op, dst, mem, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let src = gen_memory_read(&mut b, &locals, &address, width);
                let di = load_reg(&mut b, &mut locals, dst);
                emit_arith(&mut b, &locals[di].1, &src, op, width);
                b.free_local_i64(address);
                b.free_local_i64(src);
            },
            Instr::ArithMemReg { op, mem, src, width } => {
                let si = load_reg(&mut b, &mut locals, src);
                let source = locals[si].1.unsafe_clone();
                emit_arith_mem(&mut b, &mut locals, &mem, &source, op, width);
            },
            Instr::ArithMemImm { op, mem, value, width } => {
                b.const_i64(value as i64);
                let src = b.set_new_local_i64();
                emit_arith_mem(&mut b, &mut locals, &mem, &src, op, width);
                b.free_local_i64(src);
            },
            Instr::PushReg { r } => {
                let ri = load_reg(&mut b, &mut locals, r);
                b.get_local_i64(&locals[ri].1);
                let value = b.set_new_local_i64();
                emit_push_value(&mut b, &mut locals, &value);
                b.free_local_i64(value);
            },
            Instr::PopReg { r } => {
                let value = emit_pop_value(&mut b, &mut locals);
                if let Some(ri) = find_reg(&locals, r) {
                    b.get_local_i64(&value);
                    b.set_local_i64(&locals[ri].1);
                }
                else {
                    locals.push((r, value));
                    continue;
                }
                b.free_local_i64(value);
            },
            Instr::PushImm { value } => {
                b.const_i64(value as i64);
                let local = b.set_new_local_i64();
                emit_push_value(&mut b, &mut locals, &local);
                b.free_local_i64(local);
            },
            Instr::Call { target, return_address } => {
                b.const_i64(return_address as i64);
                let local = b.set_new_local_i64();
                emit_push_value(&mut b, &mut locals, &local);
                b.free_local_i64(local);
                gen_set_ip(&mut b, target);
                emit_registers_back(&mut b, &locals);
                b.return_();
                terminated = true;
            },
            Instr::CallReg { r, return_address } => {
                let ri = load_reg(&mut b, &mut locals, r);
                b.get_local_i64(&locals[ri].1);
                let target = b.set_new_local_i64();
                b.const_i64(return_address as i64);
                let local = b.set_new_local_i64();
                emit_push_value(&mut b, &mut locals, &local);
                b.free_local_i64(local);
                b.const_i32(IP_ADDR);
                b.get_local_i64(&target);
                b.store_aligned_i64(0);
                b.free_local_i64(target);
                emit_registers_back(&mut b, &locals);
                b.return_();
                terminated = true;
            },
            Instr::CallMem { mem, return_address } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let target = gen_memory_read(&mut b, &locals, &address, 64);
                b.free_local_i64(address);
                b.const_i64(return_address as i64);
                let return_value = b.set_new_local_i64();
                emit_push_value(&mut b, &mut locals, &return_value);
                b.free_local_i64(return_value);
                b.const_i32(IP_ADDR);
                b.get_local_i64(&target);
                b.store_aligned_i64(0);
                b.free_local_i64(target);
                emit_registers_back(&mut b, &locals);
                b.return_();
                terminated = true;
            },
            Instr::JmpReg { r } => {
                let ri = load_reg(&mut b, &mut locals, r);
                b.const_i32(IP_ADDR);
                b.get_local_i64(&locals[ri].1);
                b.store_aligned_i64(0);
                emit_registers_back(&mut b, &locals);
                b.return_();
                terminated = true;
            },
            Instr::JmpMem { mem } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let target = gen_memory_read(&mut b, &locals, &address, 64);
                b.free_local_i64(address);
                b.const_i32(IP_ADDR);
                b.get_local_i64(&target);
                b.store_aligned_i64(0);
                b.free_local_i64(target);
                emit_registers_back(&mut b, &locals);
                b.return_();
                terminated = true;
            },
            Instr::Ret { adjustment } => {
                let value = emit_pop_value(&mut b, &mut locals);
                if adjustment != 0 {
                    let rsp = load_reg(&mut b, &mut locals, 4);
                    b.get_local_i64(&locals[rsp].1);
                    b.const_i64(adjustment as i64);
                    b.add_i64();
                    b.set_local_i64(&locals[rsp].1);
                }
                b.const_i32(IP_ADDR);
                b.get_local_i64(&value);
                b.store_aligned_i64(0);
                b.free_local_i64(value);
                emit_registers_back(&mut b, &locals);
                b.return_();
                terminated = true;
            },
            Instr::Leave => {
                let rbp = load_reg(&mut b, &mut locals, 5);
                let rsp = load_reg(&mut b, &mut locals, 4);
                b.get_local_i64(&locals[rbp].1);
                b.set_local_i64(&locals[rsp].1);
                let value = emit_pop_value(&mut b, &mut locals);
                b.get_local_i64(&value);
                b.set_local_i64(&locals[rbp].1);
                b.free_local_i64(value);
            },
            Instr::Nop => {},
            Instr::Jmp { target } => {
                gen_set_ip(&mut b, target);
                emit_registers_back(&mut b, &locals);
                b.return_();
                terminated = true;
            },
            Instr::Jcc {
                code,
                target,
                fallthrough,
            } => {
                gen_condition(&mut b, code);
                b.if_void();
                gen_set_ip(&mut b, target);
                emit_registers_back(&mut b, &locals);
                b.return_();
                b.else_();
                gen_set_ip(&mut b, fallthrough);
                emit_registers_back(&mut b, &locals);
                b.return_();
                b.block_end();
                terminated = true;
            },
            Instr::Hlt => {
                gen_set_ip(&mut b, block_end);
                b.const_i32(IN_HLT);
                b.const_i32(1);
                b.store_u8(0);
                emit_registers_back(&mut b, &locals);
                b.return_();
                terminated = true;
            },
        }
    }

    if !terminated {
        gen_set_ip(&mut b, block_end);
        emit_registers_back(&mut b, &locals);
        b.return_();
    }

    for (_, local) in &locals {
        b.free_local_i64(local.unsafe_clone());
    }

    b.finish();
    let ptr = b.get_output_ptr();
    let len = b.get_output_len() as usize;
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}

pub fn compile_block(instrs: &[Instr], block_end: u64) -> Vec<u8> {
    compile_block_with_rips(instrs, &[], block_end)
}

pub fn compile_bytes(base: u64, bytes: &[u8]) -> Result<Vec<u8>, String> {
    let decoded = decode_block_with_rips(base, bytes)?;
    Ok(compile_block_with_rips(
        &decoded.instrs,
        &decoded.rips,
        decoded.end_rip,
    ))
}

// Test harness: compile a fixed program and return a pointer to the module bytes.
const PROTO_MAX: usize = 8192;
static mut PROTOTYPE_BYTES: [u8; PROTO_MAX] = [0; PROTO_MAX];
static mut PROTOTYPE_LEN: usize = 0;

#[no_mangle]
pub unsafe fn jit64_proto() -> u32 {
    // mov rax,0; cmp rax,1 (CF=1); inc rax (preserves CF); jc +2 -> taken
    let program = [
        0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rax, 0
        0x48, 0x83, 0xF8, 0x01, // cmp rax, 1
        0x48, 0xFF, 0xC0, // inc rax
        0x72, 0x02, // jc +2
        0xF4, // hlt
    ];
    let bytes = compile_bytes(0, &program).unwrap();
    let len = bytes.len().min(PROTO_MAX);
    std::ptr::copy_nonoverlapping(
        bytes.as_ptr(),
        std::ptr::addr_of_mut!(PROTOTYPE_BYTES) as *mut u8,
        len,
    );
    PROTOTYPE_LEN = len;
    std::ptr::addr_of_mut!(PROTOTYPE_BYTES) as *mut u8 as u32
}

#[no_mangle]
pub unsafe fn jit64_proto_len() -> u32 { PROTOTYPE_LEN as u32 }

// --- Long-mode JIT cache -----------------------------------------------------

mod js {
    #[link(wasm_import_module = "env")]
    extern "C" {
        pub fn jit64_compile(index: u32, ptr: u32, len: u32);
    }
}

mod wasm {
    extern "C" {
        pub fn call_indirect1(f: i32, x: u16);
    }
}

use std::collections::{HashMap, HashSet};

const JIT64_THRESHOLD: u32 = 500; // interpreted runs before a block is compiled

const COMPILE_BUF_SIZE: usize = 65536;
static mut COMPILE_BUF: [u8; COMPILE_BUF_SIZE] = [0; COMPILE_BUF_SIZE];
static mut JIT64_MEMORY_FAULT: u8 = 0;
// Lets tests and benchmarks run the same code with the JIT off.
static mut JIT64_ENABLED: bool = true;
static mut HOTNESS: *mut HashMap<u64, u32> = std::ptr::null_mut();
// guest RIP -> wasm table index of the compiled block
static mut BLOCKS: *mut HashMap<u64, u16> = std::ptr::null_mut();
// physical code page -> guest RIPs compiled from it
static mut CODE_PAGES: *mut HashMap<u32, HashSet<u64>> = std::ptr::null_mut();

unsafe fn user_access() -> bool { *crate::cpu::global_pointers::cpl == 3 }

unsafe fn hotness() -> &'static mut HashMap<u64, u32> {
    if HOTNESS.is_null() {
        HOTNESS = Box::into_raw(Box::new(HashMap::new()));
    }
    &mut *HOTNESS
}

unsafe fn blocks() -> &'static mut HashMap<u64, u16> {
    if BLOCKS.is_null() {
        BLOCKS = Box::into_raw(Box::new(HashMap::new()));
    }
    &mut *BLOCKS
}

unsafe fn code_pages() -> &'static mut HashMap<u32, HashSet<u64>> {
    if CODE_PAGES.is_null() {
        CODE_PAGES = Box::into_raw(Box::new(HashMap::new()));
    }
    &mut *CODE_PAGES
}

// Does nothing if the bytes cannot be decoded yet.
unsafe fn compile_and_register(rip: u64) {
    let page_end = (rip | 0xFFF) + 1;
    let mut bytes = Vec::new();
    let mut addr = rip;
    while addr < page_end {
        match crate::cpu::cpu::translate_address_64(addr, false, user_access()) {
            Ok(phys) => bytes.push(crate::cpu::memory::read8(phys) as u8),
            Err(()) => break,
        }
        addr += 1;
    }

    let module = match compile_bytes(rip, &bytes) {
        Ok(module) => module,
        Err(_) => return,
    };
    if module.len() > COMPILE_BUF_SIZE {
        return;
    }
    let index = match crate::jit::jit64_allocate_table_index() {
        Some(index) => index,
        None => return,
    };
    let len = module.len();
    std::ptr::copy_nonoverlapping(
        module.as_ptr(),
        std::ptr::addr_of_mut!(COMPILE_BUF) as *mut u8,
        len,
    );
    js::jit64_compile(
        index as u32,
        std::ptr::addr_of_mut!(COMPILE_BUF) as *mut u8 as u32,
        len as u32,
    );
    blocks().insert(rip, index);
    if let Ok(phys) = crate::cpu::cpu::translate_address_64(rip, false, user_access()) {
        code_pages().entry(phys >> 12).or_default().insert(rip);
    }
}

pub unsafe fn try_run(rip: u64) -> bool {
    if !JIT64_ENABLED {
        return false;
    }
    let index = match blocks().get(&rip) {
        Some(&index) => index,
        None => return false,
    };
    let indirect = index as i32 + crate::cpu::cpu::WASM_TABLE_OFFSET as i32;
    wasm::call_indirect1(indirect, 0);
    jit64_finish_fault();
    true
}

pub unsafe fn note_interpreted(rip: u64) {
    if !JIT64_ENABLED {
        return;
    }
    if blocks().contains_key(&rip) {
        return;
    }
    let count = hotness().entry(rip).or_insert(0);
    *count += 1;
    if *count >= JIT64_THRESHOLD {
        compile_and_register(rip);
    }
}

pub unsafe fn clear_cache() {
    if !HOTNESS.is_null() {
        hotness().clear();
    }
    if !BLOCKS.is_null() {
        let indices: Vec<u16> = blocks().drain().map(|(_, index)| index).collect();
        for index in indices {
            crate::jit::jit64_free_table_index(index);
        }
    }
    if !CODE_PAGES.is_null() {
        code_pages().clear();
    }
}

// Drop only the blocks compiled from this physical page.
pub unsafe fn invalidate_physical_page(page: u32) {
    if CODE_PAGES.is_null() {
        return;
    }
    let rips = match code_pages().remove(&page) {
        Some(rips) => rips,
        None => return,
    };
    for rip in rips {
        if let Some(index) = blocks().remove(&rip) {
            crate::jit::jit64_free_table_index(index);
        }
        if !HOTNESS.is_null() {
            hotness().remove(&rip);
        }
    }
}

#[no_mangle]
pub unsafe fn jit64_compiled_count() -> u32 { blocks().len() as u32 }

// Enable or disable block compilation and dispatch (used by tests/benchmarks).
#[no_mangle]
pub unsafe fn jit64_set_enabled(enabled: u32) { JIT64_ENABLED = enabled != 0; }

#[no_mangle]
pub unsafe fn jit64_clear_cache() { clear_cache(); }

// u64::MAX means translation faulted; the block returns immediately.
#[no_mangle]
pub unsafe fn jit64_translate(vaddr: u64, for_writing: u32) -> u64 {
    match crate::cpu::cpu::translate_address_64_jit(vaddr, for_writing != 0, user_access()) {
        Ok(phys) => crate::cpu::memory::mem8 as u64 + phys as u64,
        Err(()) => u64::MAX,
    }
}

#[no_mangle]
pub unsafe fn jit64_mem_read(vaddr: u64, width: u32) -> u64 {
    JIT64_MEMORY_FAULT = 0;
    let bytes = (width / 8) as usize;
    if bytes == 0 || bytes > 8 {
        JIT64_MEMORY_FAULT = 1;
        return 0;
    }
    if (vaddr as usize & 0xFFF) + bytes <= 0x1000 {
        let phys = match crate::cpu::cpu::translate_address_64_jit(vaddr, false, user_access()) {
            Ok(phys) => phys,
            Err(()) => { JIT64_MEMORY_FAULT = 1; return 0; },
        };
        return match width {
            8 => crate::cpu::memory::read8(phys) as u8 as u64,
            16 => crate::cpu::memory::read16(phys) as u16 as u64,
            32 => crate::cpu::memory::read32s(phys) as u32 as u64,
            64 => crate::cpu::memory::read64s(phys) as u64,
            _ => unreachable!(),
        };
    }
    let mut physical = [0u32; 8];
    for offset in 0..bytes {
        physical[offset] = match crate::cpu::cpu::translate_address_64_jit(vaddr + offset as u64, false, user_access()) {
            Ok(phys) => phys,
            Err(()) => { JIT64_MEMORY_FAULT = 1; return 0; },
        };
    }
    let mut value = 0;
    for offset in 0..bytes {
        value |= (crate::cpu::memory::read8(physical[offset]) as u8 as u64) << (offset * 8);
    }
    value
}

#[no_mangle]
pub unsafe fn jit64_mem_probe(vaddr: u64, access: u32) -> u32 {
    JIT64_MEMORY_FAULT = 0;
    let width = access & 0xFF;
    let for_writing = access & 0x100 != 0;
    let bytes = (width / 8) as usize;
    if bytes == 0 || bytes > 8 {
        JIT64_MEMORY_FAULT = 1;
        return 1;
    }
    for offset in 0..bytes {
        if crate::cpu::cpu::translate_address_64_jit(vaddr + offset as u64, for_writing, user_access()).is_err() {
            JIT64_MEMORY_FAULT = 1;
            return 1;
        }
    }
    0
}

#[no_mangle]
pub unsafe fn jit64_imul(lhs: u64, rhs: u64, width: u32) -> u64 {
    let mask = operand_mask(width);
    let product = sign_extend_to_i128(lhs, width) * sign_extend_to_i128(rhs, width);
    let result = product as u64 & mask;
    let extended = sign_extend_to_i128(result, width);
    *crate::cpu::global_pointers::flags &= !(1 | 1 << 11);
    if product != extended {
        *crate::cpu::global_pointers::flags |= 1 | 1 << 11;
    }
    *crate::cpu::global_pointers::flags_changed = 0;
    result
}

#[no_mangle]
pub fn jit64_bswap(value: u64, width: u32) -> u64 {
    if width == 32 { (value as u32).swap_bytes() as u64 } else { value.swap_bytes() }
}

// CPUID: delegate to the interpreter implementation, which reads and writes the
// register file in linear memory.
#[no_mangle]
pub unsafe fn jit64_cpuid() {
    crate::cpu::instructions_0f::instr_0FA2();
}

#[no_mangle]
pub unsafe fn jit64_shift(value: u64, raw_count: u64, encoded: u32) -> u64 {
    let width = encoded & 0xFF;
    let kind = encoded >> 8;
    let count = raw_count as u32 & if width == 64 { 63 } else { 31 };
    let mask = operand_mask(width);
    let value = value & mask;
    if count == 0 {
        return value;
    }
    let sign = 1u64 << (width - 1);
    let (result, carry, overflow) = match kind {
        0 => {
            let result = value.wrapping_shl(count) & mask;
            // A left shift by at least the operand width shifts out every bit;
            // carry is undefined there, so report no carry.
            let carry = count < width && value >> (width - count) & 1 != 0;
            (result, carry, count == 1 && (result & sign != 0) ^ carry)
        },
        1 => (value >> count, value >> (count - 1) & 1 != 0, count == 1 && value & sign != 0),
        2 => {
            let signed = (value ^ sign).wrapping_sub(sign);
            (((signed as i64) >> count) as u64 & mask, value >> (count - 1) & 1 != 0, false)
        },
        _ => unreachable!(),
    };
    let mut new_flags = *crate::cpu::global_pointers::flags & !FLAG_MASK;
    if carry { new_flags |= 1; }
    if result == 0 { new_flags |= 1 << 6; }
    if result & sign != 0 { new_flags |= 1 << 7; }
    if (result as u8).count_ones() & 1 == 0 { new_flags |= 1 << 2; }
    if overflow { new_flags |= 1 << 11; }
    *crate::cpu::global_pointers::flags = new_flags;
    *crate::cpu::global_pointers::flags_changed = 0;
    result
}

#[no_mangle]
pub unsafe fn jit64_adc_sbb(dst: u64, src: u64, encoded: u32) -> u64 {
    let width = encoded & 0xFF;
    let subtract = encoded & 0x100 != 0;
    let mask = operand_mask(width);
    let sign = 1u64 << (width - 1);
    let dst = dst & mask;
    let src = src & mask;
    let signed_dst = sign_extend_to_i128(dst, width);
    let signed_src = sign_extend_to_i128(src, width);
    let carry = (*crate::cpu::global_pointers::flags as u32 & 1) as u64;
    let (result, carry_out, overflow, adjust) = if subtract {
        let rhs = src as u128 + carry as u128;
        let result = dst.wrapping_sub(src).wrapping_sub(carry) & mask;
        let signed = signed_dst - signed_src - carry as i128;
        let min = -(1i128 << (width - 1));
        let max = (1i128 << (width - 1)) - 1;
        (result, (dst as u128) < rhs, signed < min || signed > max, (dst & 0xF) < (src & 0xF) + carry)
    }
    else {
        let sum = dst as u128 + src as u128 + carry as u128;
        let result = sum as u64 & mask;
        let signed = signed_dst + signed_src + carry as i128;
        let min = -(1i128 << (width - 1));
        let max = (1i128 << (width - 1)) - 1;
        (result, sum > mask as u128, signed < min || signed > max, (dst & 0xF) + (src & 0xF) + carry > 0xF)
    };
    let mut new_flags = *crate::cpu::global_pointers::flags & !FLAG_MASK;
    if carry_out { new_flags |= 1; }
    if adjust { new_flags |= 1 << 4; }
    if result == 0 { new_flags |= 1 << 6; }
    if result & sign != 0 { new_flags |= 1 << 7; }
    if (result as u8).count_ones() & 1 == 0 { new_flags |= 1 << 2; }
    if overflow { new_flags |= 1 << 11; }
    *crate::cpu::global_pointers::flags = new_flags;
    *crate::cpu::global_pointers::flags_changed = 0;
    result
}

#[no_mangle]
pub unsafe fn jit64_mem_write(vaddr: u64, value: u64, width: u32) {
    JIT64_MEMORY_FAULT = 0;
    let bytes = (width / 8) as usize;
    if bytes == 0 || bytes > 8 {
        JIT64_MEMORY_FAULT = 1;
        return;
    }
    if (vaddr as usize & 0xFFF) + bytes <= 0x1000 {
        let phys = match crate::cpu::cpu::translate_address_64_jit(vaddr, true, user_access()) {
            Ok(phys) => phys,
            Err(()) => { JIT64_MEMORY_FAULT = 1; return; },
        };
        match width {
            8 => crate::cpu::memory::write8(phys, value as i32),
            16 => crate::cpu::memory::write16(phys, value as i32),
            32 => crate::cpu::memory::write32(phys, value as i32),
            64 => {
                crate::cpu::memory::write32(phys, value as u32 as i32);
                crate::cpu::memory::write32(phys + 4, (value >> 32) as u32 as i32);
            },
            _ => unreachable!(),
        }
        return;
    }
    let mut physical = [0u32; 8];
    for offset in 0..bytes {
        physical[offset] = match crate::cpu::cpu::translate_address_64_jit(
            vaddr + offset as u64,
            true,
            user_access(),
        ) {
            Ok(phys) => phys,
            Err(()) => { JIT64_MEMORY_FAULT = 1; return; },
        };
    }
    for offset in 0..bytes {
        crate::cpu::memory::write8(
            physical[offset],
            (value >> (offset * 8)) as u8 as i32,
        );
    }
}

#[no_mangle]
pub unsafe fn jit64_memory_faulted() -> u32 { JIT64_MEMORY_FAULT as u32 }

#[no_mangle]
pub unsafe fn jit64_finish_fault() {
    if JIT64_MEMORY_FAULT == 0 {
        return;
    }
    JIT64_MEMORY_FAULT = 0;
    *crate::cpu::global_pointers::rip = *crate::cpu::global_pointers::previous_rip;
    *crate::cpu::global_pointers::instruction_pointer = *crate::cpu::global_pointers::rip as i32;
    crate::cpu::cpu::exit_jit64();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_sub_cmp_and_test_register_forms() {
        let decoded = decode_block(0x1000, &[
            0x48, 0x29, 0xD8, // sub rax, rbx
            0x4D, 0x3B, 0xC1, // cmp r8, r9
            0x48, 0x85, 0xC9, // test rcx, rcx
        ]).unwrap();

        assert_eq!(decoded, vec![
            Instr::ArithRegReg { op: ArithOp::Sub, dst: 0, src: 3, width: 64 },
            Instr::ArithRegReg { op: ArithOp::Cmp, dst: 8, src: 9, width: 64 },
            Instr::ArithRegReg { op: ArithOp::Test, dst: 1, src: 1, width: 64 },
        ]);
    }

    #[test]
    fn decodes_sub_cmp_and_test_immediate_forms() {
        let decoded = decode_block(0, &[
            0x49, 0x83, 0xE8, 0xFF, // sub r8, -1
            0x49, 0x81, 0xF8, 1, 0, 0, 0, // cmp r8, 1
            0x49, 0xF7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // test r8, -1
        ]).unwrap();

        assert_eq!(decoded, vec![
            Instr::ArithRegImm { op: ArithOp::Sub, r: 8, value: u64::MAX, width: 64 },
            Instr::ArithRegImm { op: ArithOp::Cmp, r: 8, value: 1, width: 64 },
            Instr::ArithRegImm { op: ArithOp::Test, r: 8, value: u64::MAX, width: 64 },
        ]);
    }

    #[test]
    fn decodes_logic_register_and_immediate_forms() {
        let decoded = decode_block(0, &[
            0x4D, 0x09, 0xC8, // or r8, r9
            0x4D, 0x23, 0xC1, // and r8, r9
            0x4D, 0x31, 0xC8, // xor r8, r9
            0x49, 0x83, 0xE0, 0x7F, // and r8, 0x7f
        ]).unwrap();

        assert_eq!(decoded, vec![
            Instr::ArithRegReg { op: ArithOp::Or, dst: 8, src: 9, width: 64 },
            Instr::ArithRegReg { op: ArithOp::And, dst: 8, src: 9, width: 64 },
            Instr::ArithRegReg { op: ArithOp::Xor, dst: 8, src: 9, width: 64 },
            Instr::ArithRegImm { op: ArithOp::And, r: 8, value: 0x7F, width: 64 },
        ]);
    }

    #[test]
    fn decodes_rip_relative_lea_and_near_jcc() {
        let lea = decode_block(0x1000, &[
            0x48, 0x8D, 0x15, 0xF9, 0x00, 0x00, 0x00, // lea rdx, [rip + 0xf9]
        ]).unwrap();
        assert_eq!(lea, vec![Instr::Lea {
            dst: 2,
            mem: Mem { base: None, index: None, scale: 0, addr_size: 64, disp: 0x1100 },
            width: 64,
        }]);

        let branch = decode_block(0x2000, &[0x0F, 0x85, 0xFA, 0xFF, 0xFF, 0xFF]).unwrap();
        assert_eq!(branch, vec![Instr::Jcc {
            code: 5,
            target: 0x2000,
            fallthrough: 0x2006,
        }]);
    }

    #[test]
    fn decodes_stack_call_and_return() {
        let decoded = decode_block(0x1000, &[
            0x41, 0x50, // push r8
            0x41, 0x59, // pop r9
            0x6A, 0xFF, // push -1
            0xE8, 0x05, 0x00, 0x00, 0x00, // call 0x1010
        ]).unwrap();
        assert_eq!(decoded, vec![
            Instr::PushReg { r: 8 },
            Instr::PopReg { r: 9 },
            Instr::PushImm { value: u64::MAX },
            Instr::Call { target: 0x1010, return_address: 0x100B },
        ]);

        assert_eq!(decode_block(0, &[0xC2, 0x10, 0x00]).unwrap(), vec![
            Instr::Ret { adjustment: 16 },
        ]);
    }

    #[test]
    fn decodes_memory_arithmetic_forms() {
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 8 };
        let decoded = decode_block(0, &[
            0x48, 0x83, 0x44, 0x24, 0x08, 0x01, // add qword [rsp+8], 1
            0x4C, 0x31, 0x44, 0x24, 0x08, // xor qword [rsp+8], r8
            0x4C, 0x03, 0x5C, 0x24, 0x08, // add r11, [rsp+8]
            0x48, 0xF7, 0x44, 0x24, 0x08, 0xFF, 0x00, 0x00, 0x00, // test [rsp+8], 0xff
        ]).unwrap();

        assert_eq!(decoded, vec![
            Instr::ArithMemImm { op: ArithOp::Add, mem, value: 1, width: 64 },
            Instr::ArithMemReg { op: ArithOp::Xor, mem, src: 8, width: 64 },
            Instr::ArithRegMem { op: ArithOp::Add, dst: 11, mem, width: 64 },
            Instr::ArithMemImm { op: ArithOp::Test, mem, value: 0xFF, width: 64 },
        ]);
    }

    #[test]
    fn decodes_memory_immediate_and_indirect_control_flow() {
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 8 };
        assert_eq!(decode_block(0, &[
            0x48, 0xC7, 0x44, 0x24, 0x08, 0xFF, 0xFF, 0xFF, 0xFF,
        ]).unwrap(), vec![Instr::MovMemImm { mem, value: u64::MAX, width: 64 }]);

        assert_eq!(decode_block(0x1000, &[0x41, 0xFF, 0xD0]).unwrap(), vec![
            Instr::CallReg { r: 8, return_address: 0x1003 },
        ]);
        assert_eq!(decode_block(0, &[0x41, 0xFF, 0xE1]).unwrap(), vec![
            Instr::JmpReg { r: 9 },
        ]);
        assert_eq!(decode_block(0x1000, &[0xFF, 0x15, 0xFA, 0x00, 0x00, 0x00]).unwrap(), vec![
            Instr::CallMem {
                mem: Mem { base: None, index: None, scale: 0, addr_size: 64, disp: 0x1100 },
                return_address: 0x1006,
            },
        ]);
        assert_eq!(decode_block(0, &[0xFF, 0x64, 0x24, 0x08]).unwrap(), vec![
            Instr::JmpMem {
                mem: Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 8 },
            },
        ]);
        assert_eq!(decode_block(0, &[0xC9, 0xC3]).unwrap(), vec![
            Instr::Leave,
            Instr::Ret { adjustment: 0 },
        ]);
    }

    #[test]
    fn decodes_32_bit_register_operations() {
        assert_eq!(decode_block(0, &[
            0x89, 0xD8, // mov eax, ebx
            0x41, 0x83, 0xC5, 0x01, // add r13d, 1
            0x85, 0xC0, // test eax, eax
        ]).unwrap(), vec![
            Instr::MovRegReg { dst: 0, src: 3, width: 32 },
            Instr::AddRegImm { r: 13, value: 1, width: 32 },
            Instr::ArithRegReg { op: ArithOp::Test, dst: 0, src: 0, width: 32 },
        ]);
    }

    #[test]
    fn decodes_32_bit_memory_operations() {
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 24 };
        assert_eq!(decode_block(0, &[
            0xC7, 0x44, 0x24, 0x18, 0xFF, 0xFF, 0xFF, 0xFF,
            0x83, 0x44, 0x24, 0x18, 0x01,
            0x44, 0x8B, 0x74, 0x24, 0x18,
        ]).unwrap(), vec![
            Instr::MovMemImm { mem, value: 0xFFFF_FFFF, width: 32 },
            Instr::ArithMemImm { op: ArithOp::Add, mem, value: 1, width: 32 },
            Instr::MovRegMem { dst: 14, mem, width: 32 },
        ]);
    }

    #[test]
    fn decodes_nop_and_endbr64() {
        assert_eq!(decode_block(0, &[
            0x90,
            0x0F, 0x1F, 0x44, 0x00, 0x00,
            0xF3, 0x0F, 0x1E, 0xFA,
        ]).unwrap(), vec![Instr::Nop, Instr::Nop, Instr::Nop]);
    }

    #[test]
    fn decodes_movzx_and_movsx() {
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 16 };
        assert_eq!(decode_block(0, &[
            0x0F, 0xB6, 0xC4, // movzx eax, ah
            0x48, 0x0F, 0xBE, 0xCB, // movsx rcx, bl
            0x44, 0x0F, 0xB7, 0x7C, 0x24, 0x10, // movzx r15d, word [rsp+16]
        ]).unwrap(), vec![
            Instr::MovExtendReg {
                dst: 0, src: 0, src_width: 8, dst_width: 32, signed: false, high8: true,
            },
            Instr::MovExtendReg {
                dst: 1, src: 3, src_width: 8, dst_width: 64, signed: true, high8: false,
            },
            Instr::MovExtendMem {
                dst: 15, mem, src_width: 16, dst_width: 32, signed: false,
            },
        ]);
    }

    #[test]
    fn decodes_movsxd() {
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 16 };
        assert_eq!(decode_block(0, &[
            0x4D, 0x63, 0xC1, // movsxd r8, r9d
            0x4C, 0x63, 0x54, 0x24, 0x10, // movsxd r10, dword [rsp+16]
        ]).unwrap(), vec![
            Instr::MovExtendReg {
                dst: 8, src: 9, src_width: 32, dst_width: 64, signed: true, high8: false,
            },
            Instr::MovExtendMem {
                dst: 10, mem, src_width: 32, dst_width: 64, signed: true,
            },
        ]);
    }

    #[test]
    fn decodes_imul_forms() {
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 16 };
        assert_eq!(decode_block(0, &[
            0x45, 0x0F, 0xAF, 0xC1, // imul r8d, r9d
            0x4D, 0x6B, 0xC1, 0xFF, // imul r8, r9, -1
            0x48, 0x69, 0x5C, 0x24, 0x10, 2, 0, 0, 0, // imul rbx, [rsp+16], 2
        ]).unwrap(), vec![
            Instr::ImulRegReg { dst: 8, lhs: 8, rhs: 9, width: 32 },
            Instr::ImulRegImm { dst: 8, src: 9, value: u64::MAX, width: 64 },
            Instr::ImulRegMem { dst: 3, mem, value: Some(2), width: 64 },
        ]);
    }

    #[test]
    fn decodes_bswap() {
        assert_eq!(decode_block(0, &[
            0x41, 0x0F, 0xC8, // bswap r8d
            0x49, 0x0F, 0xCF, // bswap r15
        ]).unwrap(), vec![
            Instr::Bswap { r: 8, width: 32 },
            Instr::Bswap { r: 15, width: 64 },
        ]);
    }

    #[test]
    fn decodes_adc_and_sbb_forms() {
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 16 };
        assert_eq!(decode_block(0, &[
            0x4D, 0x11, 0xC8, // adc r8, r9
            0x49, 0x83, 0xD8, 0xFF, // sbb r8, -1
            0x48, 0x13, 0x5C, 0x24, 0x10, // adc rbx, [rsp+16]
            0x48, 0x83, 0x5C, 0x24, 0x10, 0x00, // sbb qword [rsp+16], 0
        ]).unwrap(), vec![
            Instr::ArithRegReg { op: ArithOp::Adc, dst: 8, src: 9, width: 64 },
            Instr::ArithRegImm { op: ArithOp::Sbb, r: 8, value: u64::MAX, width: 64 },
            Instr::ArithRegMem { op: ArithOp::Adc, dst: 3, mem, width: 64 },
            Instr::ArithMemImm { op: ArithOp::Sbb, mem, value: 0, width: 64 },
        ]);
    }

    #[test]
    fn decodes_register_mov_immediate_xchg_and_not() {
        assert_eq!(decode_block(0, &[
            0x41, 0xC7, 0xC5, 1, 0, 0, 0, // mov r13d, 1
            0x45, 0x87, 0xE8, // xchg r8d, r13d
            0x49, 0xF7, 0xD0, // not r8
        ]).unwrap(), vec![
            Instr::MovRegImm { r: 13, value: 1, width: 32 },
            Instr::XchgRegReg { a: 8, b: 13, width: 32 },
            Instr::NotReg { r: 8, width: 64 },
        ]);
    }

    #[test]
    fn decodes_neg() {
        assert_eq!(decode_block(0, &[0xF7, 0xD8, 0x49, 0xF7, 0xD8]).unwrap(), vec![
            Instr::NegReg { r: 0, width: 32 },
            Instr::NegReg { r: 8, width: 64 },
        ]);
    }

    #[test]
    fn decodes_memory_unary_and_xchg() {
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 16 };
        assert_eq!(decode_block(0, &[
            0x48, 0xF7, 0x54, 0x24, 0x10,
            0x48, 0xF7, 0x5C, 0x24, 0x10,
            0x48, 0xFF, 0x44, 0x24, 0x10,
            0x48, 0xFF, 0x4C, 0x24, 0x10,
            0x48, 0x87, 0x7C, 0x24, 0x10,
        ]).unwrap(), vec![
            Instr::NotMem { mem, width: 64 },
            Instr::NegMem { mem, width: 64 },
            Instr::IncDecMem { mem, width: 64, decrement: false },
            Instr::IncDecMem { mem, width: 64, decrement: true },
            Instr::XchgMemReg { mem, r: 7, width: 64 },
        ]);
    }

    #[test]
    fn decodes_inc_and_dec() {
        assert_eq!(decode_block(0, &[
            0xFF, 0xC0, // inc eax
            0x49, 0xFF, 0xC8, // dec r8
        ]).unwrap(), vec![
            Instr::IncDecReg { r: 0, width: 32, decrement: false },
            Instr::IncDecReg { r: 8, width: 64, decrement: true },
        ]);
    }

    #[test]
    fn decodes_cmov_and_setcc() {
        assert_eq!(decode_block(0, &[
            0x45, 0x0F, 0x44, 0xC1, // cmove r8d, r9d
            0x0F, 0x95, 0xC4, // setne ah
            0x41, 0x0F, 0x94, 0xC2, // sete r10b
        ]).unwrap(), vec![
            Instr::CmovRegReg { code: 4, dst: 8, src: 9, width: 32 },
            Instr::SetccReg { code: 5, dst: 0, high8: true },
            Instr::SetccReg { code: 4, dst: 10, high8: false },
        ]);
    }

    #[test]
    fn decodes_cmov_and_setcc_memory_forms() {
        let mem = Mem { base: Some(0), index: None, scale: 0, addr_size: 64, disp: 0 };
        assert_eq!(decode_block(0, &[
            0x0F, 0x44, 0x00, // cmove eax, [rax]
            0x0F, 0x94, 0x00, // sete byte [rax]
        ]).unwrap(), vec![
            Instr::CmovRegMem { code: 4, dst: 0, mem, width: 32 },
            Instr::SetccMem { code: 4, mem },
        ]);
    }

    #[test]
    fn decodes_single_bit_shifts() {
        assert_eq!(decode_block(0, &[
            0xD1, 0xE0, // shl eax, 1
            0x49, 0xD1, 0xE8, // shr r8, 1
            0x41, 0xD1, 0xF9, // sar r9d, 1
        ]).unwrap(), vec![
            Instr::ShiftReg { kind: ShiftKind::Shl, r: 0, width: 32, count: 1 },
            Instr::ShiftReg { kind: ShiftKind::Shr, r: 8, width: 64, count: 1 },
            Instr::ShiftReg { kind: ShiftKind::Sar, r: 9, width: 32, count: 1 },
        ]);
    }

    #[test]
    fn decodes_masked_immediate_shifts() {
        assert_eq!(decode_block(0, &[
            0xC1, 0xE0, 0x21, // shl eax, 33 -> 1
            0x49, 0xC1, 0xE8, 0x41, // shr r8, 65 -> 1
        ]).unwrap(), vec![
            Instr::ShiftReg { kind: ShiftKind::Shl, r: 0, width: 32, count: 1 },
            Instr::ShiftReg { kind: ShiftKind::Shr, r: 8, width: 64, count: 1 },
        ]);
    }

    #[test]
    fn decodes_cl_shifts() {
        assert_eq!(decode_block(0, &[
            0xD3, 0xE0, // shl eax, cl
            0x49, 0xD3, 0xE8, // shr r8, cl
            0x41, 0xD3, 0xF9, // sar r9d, cl
        ]).unwrap(), vec![
            Instr::ShiftRegCl { kind: ShiftKind::Shl, r: 0, width: 32 },
            Instr::ShiftRegCl { kind: ShiftKind::Shr, r: 8, width: 64 },
            Instr::ShiftRegCl { kind: ShiftKind::Sar, r: 9, width: 32 },
        ]);
    }

    #[test]
    fn decodes_16_bit_operand_forms() {
        assert_eq!(decode_block(0, &[
            0x66, 0x89, 0xD8, // mov ax, bx
            0x66, 0x01, 0xC8, // add ax, cx
            0x66, 0xB8, 0x34, 0x12, // mov ax, 0x1234
            0x66, 0x81, 0xC3, 0x78, 0x56, // add bx, 0x5678
            0x66, 0x83, 0xC0, 0xFF, // add ax, -1
            0x66, 0x0F, 0xB7, 0xC3, // movzx ax, bx
            0x66, 0xD1, 0xE0, // shl ax, 1
            0x66, 0xF7, 0xD8, // neg ax
        ]).unwrap(), vec![
            Instr::MovRegReg { dst: 0, src: 3, width: 16 },
            Instr::AddRegReg { dst: 0, src: 1, width: 16 },
            Instr::MovRegImm { r: 0, value: 0x1234, width: 16 },
            Instr::AddRegImm { r: 3, value: 0x5678, width: 16 },
            Instr::AddRegImm { r: 0, value: u64::MAX, width: 16 },
            Instr::MovExtendReg {
                dst: 0,
                src: 3,
                src_width: 16,
                dst_width: 16,
                signed: false,
                high8: false,
            },
            Instr::ShiftReg { kind: ShiftKind::Shl, r: 0, width: 16, count: 1 },
            Instr::NegReg { r: 0, width: 16 },
        ]);
    }

    #[test]
    fn decodes_address_size_override() {
        assert_eq!(decode_block(0, &[
            0x67, 0x48, 0x8B, 0x03, // mov rax, [ebx]
            0x66, 0x67, 0x8B, 0x03, // mov ax, [ebx]
        ]).unwrap(), vec![
            Instr::MovRegMem {
                dst: 0,
                mem: Mem { base: Some(3), index: None, scale: 0, disp: 0, addr_size: 32 },
                width: 64,
            },
            Instr::MovRegMem {
                dst: 0,
                mem: Mem { base: Some(3), index: None, scale: 0, disp: 0, addr_size: 32 },
                width: 16,
            },
        ]);
    }

    #[test]
    fn accepts_lock_and_repne_prefixes() {
        assert_eq!(decode_block(0, &[
            0xF0, 0x48, 0x01, 0xD8, // lock add rax, rbx
            0xF2, 0x48, 0x01, 0xD8, // repne add rax, rbx
        ]).unwrap(), vec![
            Instr::AddRegReg { dst: 0, src: 3, width: 64 },
            Instr::AddRegReg { dst: 0, src: 3, width: 64 },
        ]);
    }

    #[test]
    fn decodes_memory_form_shifts() {
        assert_eq!(decode_block(0, &[
            0x48, 0xD1, 0x20, // shl qword [rax], 1
            0x48, 0xC1, 0x20, 0x05, // shl qword [rax], 5
            0x48, 0xD3, 0x20, // shl qword [rax], cl
            0x66, 0xC1, 0x20, 0x04, // shl word [rax], 4
        ]).unwrap(), vec![
            Instr::ShiftMem {
                kind: ShiftKind::Shl,
                mem: Mem { base: Some(0), index: None, scale: 0, disp: 0, addr_size: 64 },
                width: 64,
                count: 1,
            },
            Instr::ShiftMem {
                kind: ShiftKind::Shl,
                mem: Mem { base: Some(0), index: None, scale: 0, disp: 0, addr_size: 64 },
                width: 64,
                count: 5,
            },
            Instr::ShiftMemCl {
                kind: ShiftKind::Shl,
                mem: Mem { base: Some(0), index: None, scale: 0, disp: 0, addr_size: 64 },
                width: 64,
            },
            Instr::ShiftMem {
                kind: ShiftKind::Shl,
                mem: Mem { base: Some(0), index: None, scale: 0, disp: 0, addr_size: 64 },
                width: 16,
                count: 4,
            },
        ]);
    }

    #[test]
    fn decodes_cpuid() {
        assert_eq!(decode_block(0, &[0x0F, 0xA2]).unwrap(), vec![Instr::Cpuid]);
    }
}
