//! Long mode JIT targeting wasm32: guest registers are i64 locals, physical
//! addresses stay 32-bit (below 4 GiB).
//! http://www.sandpile.org/x86/opra.htm

#![allow(dead_code)]

use std::sync::atomic::{AtomicU32, Ordering};

use crate::cpu::jit::emit;
use crate::cpu::interp::decode_cache::{ArithOp, Instruction, Mem, ShiftKind};
use crate::wasmgen::wasm_builder::{WasmBuilder, WasmLocal, WasmLocalI64};

// Instrumentation: log the first few blocks that fail to decode.
static LOGGED_FAILS: AtomicU32 = AtomicU32::new(0);

const REG_LOW: i32 = 64;
const REG_HIGH: i32 = 128;
const REG_EXT: i32 = 160;
const FS_BASE_ADDR: i32 = 1104;
const GS_BASE_ADDR: i32 = 1112;
const IN_HLT: i32 = 616;
const FLAGS_ADDR: i32 = 120;
const FLAGS_CHANGED_ADDR: i32 = 100;
const IP_ADDR: i32 = 232;
const PREVIOUS_RIP_ADDR: i32 = 240;
const INSTRUCTION_COUNTER_ADDR: i32 = 664;
const SREG_ADDR: i32 = 668;

// CF | PF | AF | ZF | SF | OF
const FLAG_MASK: i32 = 0x8D5;


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

// In long mode REX.W beats `66`: `66 4c 03 07` is `add r8, [rdi]`.
fn operand_width(prefix_66: bool, rex_w: bool) -> u8 {
    if rex_w {
        64
    }
    else if prefix_66 {
        16
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


struct DecodedBlock {
    instrs: Vec<Instruction>,
    rips: Vec<u64>,
    current_rip: u64,
    end_rip: u64,
}

impl DecodedBlock {
    fn push(&mut self, instr: Instruction) {
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
    if width == 8 {
        let v = *bytes.get(*i).ok_or("truncated imm8")? as u64;
        *i += 1;
        return Ok(v);
    }
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

// Encoded immediate size for the operand width (imm8 vs imm16 vs imm32).
fn imm_operand_bytes(width: u8) -> usize {
    if width == 8 {
        1
    }
    else if width == 16 {
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
    segment: Option<u8>,
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
        if index_low != 4 || rex_x != 0 {
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
        segment,
    })
}

// Decode one basic block into `out`; stops at hlt/jmp/jcc or the first
// undecodable instruction (whose start is `current_rip`).
fn superblock_limit() -> usize {
    if unsafe { JIT64_SUPERBLOCKS } { unsafe { JIT64_BLOCK_LIMIT } } else { usize::MAX }
}

// Length of the VEX-encoded instruction starting at bytes[i] (C4/C5).
fn vex_decode_len(bytes: &[u8], i: usize) -> Result<usize, String> {
    let first = bytes[i];
    let mut j = i + 1;
    let map;
    if first == 0xC5 {
        let _ = *bytes.get(j).ok_or("truncated vex")?;
        j += 1;
        map = 1;
    }
    else {
        let b1 = *bytes.get(j).ok_or("truncated vex")?;
        let _ = *bytes.get(j + 1).ok_or("truncated vex")?;
        j += 2;
        map = b1 & 0x1F;
    }
    let opcode = *bytes.get(j).ok_or("truncated vex opcode")?;
    j += 1;
    // VZEROUPPER/VZEROALL (VEX.128/256.0F.WIG 77) have no ModRM byte.
    if map == 1 && opcode == 0x77 {
        if j > bytes.len() {
            return Err("truncated vex instruction".into());
        }
        return Ok(j - i);
    }
    let modrm = *bytes.get(j).ok_or("truncated vex modrm")?;
    j += 1;
    let mod_bits = modrm >> 6;
    let rm_low = modrm & 7;
    if mod_bits != 3 {
        if rm_low == 4 {
            let sib = *bytes.get(j).ok_or("truncated vex sib")?;
            j += 1;
            if sib & 7 == 5 && mod_bits == 0 {
                j += 4;
            }
        }
        else if rm_low == 5 && mod_bits == 0 {
            j += 4;
        }
        match mod_bits {
            0 => {},
            1 => j += 1,
            2 => j += 4,
            _ => {},
        }
    }
    let has_imm = match map {
        3 => true,
        1 => matches!(opcode, 0x70 | 0x71 | 0x72 | 0x73 | 0xC2 | 0xC6),
        _ => false,
    };
    if has_imm {
        j += 1;
    }
    if j > bytes.len() {
        return Err("truncated vex instruction".into());
    }
    Ok(j - i)
}

fn decode_into(out: &mut DecodedBlock, base: u64, bytes: &[u8]) -> Result<(), String> {
    let mut i = 0;

    while i < bytes.len() {
        // stop at the superblock instruction cap.
        if out.instrs.len() >= superblock_limit() {
            break;
        }
        let start = i;
        out.current_rip = base.wrapping_add(start as u64);
        let mut rex = 0u8;
        let mut prefix_f3 = false;
        let mut prefix_66 = false;
        let mut prefix_67 = false;
        let mut segment_override = None;
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
                // In long mode the ES/CS/SS/DS overrides are ignored (their
                // bases are zero); FS/GS contribute to the effective address.
                0x26 | 0x2E | 0x36 | 0x3E => {
                    i += 1;
                },
                0x64 | 0x65 => {
                    segment_override = Some(bytes[i]);
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

        // VEX/AVX: executed by the interpreter (see Instruction::Avx).
        if bytes[i] == 0xC4 || bytes[i] == 0xC5 {
            if i != start {
                return Err("legacy prefix before VEX".into());
            }
            let len = vex_decode_len(bytes, i)?;
            out.push(Instruction::Avx { rip: out.current_rip });
            i += len;
            continue;
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
            // xchg rAX, r (0x90 is NOP only when the other register is rAX;
            // with REX.B it exchanges with r8-r15)
            0x90..=0x97 => {
                let r = (opcode - 0x90) | rex_b << 3;
                if r == 0 {
                    out.push(Instruction::Nop);
                }
                else {
                    out.push(Instruction::XchgRegReg {
                        a: 0,
                        b: r,
                        width: operand_width(prefix_66, rex_w),
                    });
                }
            },

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
                out.push(Instruction::MovRegImm { r, value, width });
            },

            // mov r/m, imm16/imm32
            0xC6 | 0xC7 => {
                let width = if opcode == 0xC6 { 8 } else { operand_width(prefix_66, rex_w) };
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
                        addr_size, segment_override,
                        if opcode == 0xC6 { 1 } else if width == 16 { 2 } else { 4 },
                    )?)
                };
                let value = if opcode == 0xC6 {
                    let v = *bytes.get(i).ok_or("truncated imm8")? as u64;
                    i += 1;
                    v
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
                    let raw = read_u32(bytes, &mut i)?;
                    if width == 64 { raw as i32 as i64 as u64 } else { raw as u64 }
                };
                if let Some(mem) = mem {
                    out.push(Instruction::MovMemImm { mem, value, width });
                }
                else {
                    let mut r = (modrm & 7) | rex_b << 3;
                    if width == 8 && rex == 0 && (4..8).contains(&r) {
                        r = 16 + (r - 4);
                    }
                    out.push(Instruction::MovRegImm { r, value, width });
                }
            },

            // in/out: E4-E7 use an imm8 port, EC-EF the DX port
            0xE4 | 0xE5 | 0xE6 | 0xE7 => {
                let port = *bytes.get(i).ok_or("truncated port")? as u16;
                i += 1;
                let width = if opcode & 1 == 0 { 8 } else { operand_width(prefix_66, rex_w) };
                out.push(if opcode & 2 == 0 {
                    Instruction::In { imm: Some(port), width }
                }
                else {
                    Instruction::Out { imm: Some(port), width }
                });
            },
            0xEC | 0xED | 0xEE | 0xEF => {
                let width = if opcode & 1 == 0 { 8 } else { operand_width(prefix_66, rex_w) };
                out.push(if opcode & 2 == 0 {
                    Instruction::In { imm: None, width }
                }
                else {
                    Instruction::Out { imm: None, width }
                });
            },
            // cli (0xFA)
            0xFA => out.push(Instruction::Cli),
            // pushfq (0x9C)
            0x9C => {
                if prefix_66 && !rex_w {
                    return Err("16-bit pushf is not supported".into());
                }
                out.push(Instruction::PushFlags);
            },

            // movsxd r64, r/m32
            0x63 => {
                if !rex_w { return Err("movsxd requires rex.w".into()); }
                let modrm = *bytes.get(i).ok_or("truncated movsxd modrm")?;
                i += 1;
                let dst = (modrm >> 3 & 7) | rex_r << 3;
                if modrm >> 6 == 3 {
                    out.push(Instruction::MovExtendReg {
                        dst,
                        src: (modrm & 7) | rex_b << 3,
                        src_width: 32,
                        dst_width: 64,
                        signed: true,
                        high8: false,
                    });
                }
                else {
                    let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                    out.push(Instruction::MovExtendMem {
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
                        addr_size, segment_override,
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
                    out.push(Instruction::ImulRegMem { dst, mem, value: Some(value), width });
                }
                else {
                    out.push(Instruction::ImulRegImm {
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
                    out.push(Instruction::XchgRegReg {
                        a: (modrm & 7) | rex_b << 3,
                        b: r,
                        width,
                    });
                }
                else {
                    let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                    out.push(Instruction::XchgMemReg { mem, r, width });
                }
            },

            // mov/lea/add/sub/cmp/test r/m, r and r, r/m
            0x88 | 0x8A | 0x89 | 0x8B | 0x8D | 0x01 | 0x03 | 0x09 | 0x0B | 0x11 | 0x13 | 0x19 | 0x1B
            | 0x21 | 0x23 | 0x29 | 0x2B | 0x31 | 0x33 | 0x39 | 0x3B | 0x85 | 0x00 | 0x02 | 0x08
            | 0x0A | 0x10 | 0x12 | 0x18 | 0x1A | 0x20 | 0x22 | 0x28 | 0x2A | 0x30 | 0x32 | 0x38
            | 0x3A | 0x84 => {
                let width = if matches!(
                    opcode,
                    0x88 | 0x8A | 0x00 | 0x02 | 0x08 | 0x0A | 0x10 | 0x12 | 0x18 | 0x1A | 0x20
                        | 0x22 | 0x28 | 0x2A | 0x30 | 0x32 | 0x38 | 0x3A | 0x84
                ) {
                    8
                }
                else {
                    operand_width(prefix_66, rex_w)
                };
                let modrm = *bytes.get(i).ok_or("truncated modrm")?;
                i += 1;
                let reg = (modrm >> 3 & 7) | rex_r << 3;
                let rm = (modrm & 7) | rex_b << 3;
                // Without a REX prefix the 8-bit registers 4..7 are AH/CH/DH/BH;
                // map them into the 16..19 index space.
                let map_high = |x: u8| if (4..8).contains(&x) { 16 + (x - 4) } else { x };
                let (reg, rm) = if width == 8 && rex == 0 {
                    (map_high(reg), if modrm >> 6 == 3 { map_high(rm) } else { rm })
                }
                else {
                    (reg, rm)
                };
                if modrm >> 6 == 3 {
                    if opcode == 0x8D {
                        return Err("lea requires a memory operand".into());
                    }
                    out.push(match opcode {
                        0x88 => Instruction::MovRegReg { dst: rm, src: reg, width },
                        0x8A => Instruction::MovRegReg { dst: reg, src: rm, width },
                        0x89 => Instruction::MovRegReg { dst: rm, src: reg, width },
                        0x8B => Instruction::MovRegReg { dst: reg, src: rm, width },
                        0x01 => Instruction::AddRegReg { dst: rm, src: reg, width },
                        0x03 => Instruction::AddRegReg { dst: reg, src: rm, width },
                        0x09 => Instruction::ArithRegReg { op: ArithOp::Or, dst: rm, src: reg, width },
                        0x0B => Instruction::ArithRegReg { op: ArithOp::Or, dst: reg, src: rm, width },
                        0x11 => Instruction::ArithRegReg { op: ArithOp::Adc, dst: rm, src: reg, width },
                        0x13 => Instruction::ArithRegReg { op: ArithOp::Adc, dst: reg, src: rm, width },
                        0x19 => Instruction::ArithRegReg { op: ArithOp::Sbb, dst: rm, src: reg, width },
                        0x1B => Instruction::ArithRegReg { op: ArithOp::Sbb, dst: reg, src: rm, width },
                        0x21 => Instruction::ArithRegReg { op: ArithOp::And, dst: rm, src: reg, width },
                        0x23 => Instruction::ArithRegReg { op: ArithOp::And, dst: reg, src: rm, width },
                        0x29 => Instruction::ArithRegReg { op: ArithOp::Sub, dst: rm, src: reg, width },
                        0x2B => Instruction::ArithRegReg { op: ArithOp::Sub, dst: reg, src: rm, width },
                        0x31 => Instruction::ArithRegReg { op: ArithOp::Xor, dst: rm, src: reg, width },
                        0x33 => Instruction::ArithRegReg { op: ArithOp::Xor, dst: reg, src: rm, width },
                        0x39 => Instruction::ArithRegReg { op: ArithOp::Cmp, dst: rm, src: reg, width },
                        0x3B => Instruction::ArithRegReg { op: ArithOp::Cmp, dst: reg, src: rm, width },
                        0x85 => Instruction::ArithRegReg { op: ArithOp::Test, dst: rm, src: reg, width },
                        0x00 => Instruction::ArithRegReg { op: ArithOp::Add, dst: rm, src: reg, width },
                        0x02 => Instruction::ArithRegReg { op: ArithOp::Add, dst: reg, src: rm, width },
                        0x08 => Instruction::ArithRegReg { op: ArithOp::Or, dst: rm, src: reg, width },
                        0x0A => Instruction::ArithRegReg { op: ArithOp::Or, dst: reg, src: rm, width },
                        0x10 => Instruction::ArithRegReg { op: ArithOp::Adc, dst: rm, src: reg, width },
                        0x12 => Instruction::ArithRegReg { op: ArithOp::Adc, dst: reg, src: rm, width },
                        0x18 => Instruction::ArithRegReg { op: ArithOp::Sbb, dst: rm, src: reg, width },
                        0x1A => Instruction::ArithRegReg { op: ArithOp::Sbb, dst: reg, src: rm, width },
                        0x20 => Instruction::ArithRegReg { op: ArithOp::And, dst: rm, src: reg, width },
                        0x22 => Instruction::ArithRegReg { op: ArithOp::And, dst: reg, src: rm, width },
                        0x28 => Instruction::ArithRegReg { op: ArithOp::Sub, dst: rm, src: reg, width },
                        0x2A => Instruction::ArithRegReg { op: ArithOp::Sub, dst: reg, src: rm, width },
                        0x30 => Instruction::ArithRegReg { op: ArithOp::Xor, dst: rm, src: reg, width },
                        0x32 => Instruction::ArithRegReg { op: ArithOp::Xor, dst: reg, src: rm, width },
                        0x38 => Instruction::ArithRegReg { op: ArithOp::Cmp, dst: rm, src: reg, width },
                        0x3A => Instruction::ArithRegReg { op: ArithOp::Cmp, dst: reg, src: rm, width },
                        0x84 => Instruction::ArithRegReg { op: ArithOp::Test, dst: rm, src: reg, width },
                        _ => unreachable!(),
                    });
                }
                else {
                    let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                    out.push(match opcode {
                        0x88 => Instruction::MovMemReg { mem, src: reg, width },
                        0x8A => Instruction::MovRegMem { dst: reg, mem, width },
                        0x89 => Instruction::MovMemReg { mem, src: reg, width },
                        0x8B => Instruction::MovRegMem { dst: reg, mem, width },
                        0x8D => Instruction::Lea { dst: reg, mem, width },
                        0x01 => Instruction::ArithMemReg { op: ArithOp::Add, mem, src: reg, width },
                        0x03 => Instruction::ArithRegMem { op: ArithOp::Add, dst: reg, mem, width },
                        0x09 => Instruction::ArithMemReg { op: ArithOp::Or, mem, src: reg, width },
                        0x0B => Instruction::ArithRegMem { op: ArithOp::Or, dst: reg, mem, width },
                        0x11 => Instruction::ArithMemReg { op: ArithOp::Adc, mem, src: reg, width },
                        0x13 => Instruction::ArithRegMem { op: ArithOp::Adc, dst: reg, mem, width },
                        0x19 => Instruction::ArithMemReg { op: ArithOp::Sbb, mem, src: reg, width },
                        0x1B => Instruction::ArithRegMem { op: ArithOp::Sbb, dst: reg, mem, width },
                        0x21 => Instruction::ArithMemReg { op: ArithOp::And, mem, src: reg, width },
                        0x23 => Instruction::ArithRegMem { op: ArithOp::And, dst: reg, mem, width },
                        0x29 => Instruction::ArithMemReg { op: ArithOp::Sub, mem, src: reg, width },
                        0x2B => Instruction::ArithRegMem { op: ArithOp::Sub, dst: reg, mem, width },
                        0x31 => Instruction::ArithMemReg { op: ArithOp::Xor, mem, src: reg, width },
                        0x33 => Instruction::ArithRegMem { op: ArithOp::Xor, dst: reg, mem, width },
                        0x39 => Instruction::ArithMemReg { op: ArithOp::Cmp, mem, src: reg, width },
                        0x3B => Instruction::ArithRegMem { op: ArithOp::Cmp, dst: reg, mem, width },
                        0x85 => Instruction::ArithMemReg { op: ArithOp::Test, mem, src: reg, width },
                        0x00 => Instruction::ArithMemReg { op: ArithOp::Add, mem, src: reg, width },
                        0x02 => Instruction::ArithRegMem { op: ArithOp::Add, dst: reg, mem, width },
                        0x08 => Instruction::ArithMemReg { op: ArithOp::Or, mem, src: reg, width },
                        0x0A => Instruction::ArithRegMem { op: ArithOp::Or, dst: reg, mem, width },
                        0x10 => Instruction::ArithMemReg { op: ArithOp::Adc, mem, src: reg, width },
                        0x12 => Instruction::ArithRegMem { op: ArithOp::Adc, dst: reg, mem, width },
                        0x18 => Instruction::ArithMemReg { op: ArithOp::Sbb, mem, src: reg, width },
                        0x1A => Instruction::ArithRegMem { op: ArithOp::Sbb, dst: reg, mem, width },
                        0x20 => Instruction::ArithMemReg { op: ArithOp::And, mem, src: reg, width },
                        0x22 => Instruction::ArithRegMem { op: ArithOp::And, dst: reg, mem, width },
                        0x28 => Instruction::ArithMemReg { op: ArithOp::Sub, mem, src: reg, width },
                        0x2A => Instruction::ArithRegMem { op: ArithOp::Sub, dst: reg, mem, width },
                        0x30 => Instruction::ArithMemReg { op: ArithOp::Xor, mem, src: reg, width },
                        0x32 => Instruction::ArithRegMem { op: ArithOp::Xor, dst: reg, mem, width },
                        0x38 => Instruction::ArithMemReg { op: ArithOp::Cmp, mem, src: reg, width },
                        0x3A => Instruction::ArithRegMem { op: ArithOp::Cmp, dst: reg, mem, width },
                        0x84 => Instruction::ArithMemReg { op: ArithOp::Test, mem, src: reg, width },
                        _ => unreachable!(),
                    });
                }
            },

            // add/or/and/sub/xor/cmp r/m, imm (group 1)
            0x80 | 0x81 | 0x83 => {
                let width = if opcode == 0x80 { 8 } else { operand_width(prefix_66, rex_w) };
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
                        addr_size, segment_override,
                        if opcode == 0x80 || opcode == 0x83 { 1 } else if width == 16 { 2 } else { 4 },
                    )?)
                };
                let value = if opcode == 0x83 {
                    read_i8(bytes, &mut i)? as i64 as u64
                }
                else if opcode == 0x80 {
                    let v = *bytes.get(i).ok_or("truncated imm8")? as u64;
                    i += 1;
                    v
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
                    out.push(Instruction::ArithMemImm { op, mem, value, width });
                }
                else {
                    let mut r = (modrm & 7) | rex_b << 3;
                    if width == 8 && rex == 0 && (4..8).contains(&r) {
                        r = 16 + (r - 4);
                    }
                    out.push(if op == ArithOp::Add {
                        Instruction::AddRegImm { r, value, width }
                    }
                    else {
                        Instruction::ArithRegImm { op, r, value, width }
                    });
                }
            },

            // test/not/neg r/m8, imm8 (0xF6) and r/m, imm (0xF7)
            0xF6 | 0xF7 => {
                let width = if opcode == 0xF6 {
                    8
                }
                else {
                    operand_width(prefix_66, rex_w)
                };
                let modrm = *bytes.get(i).ok_or("truncated modrm")?;
                i += 1;
                let group = modrm >> 3 & 7;
                // Without a REX prefix the 8-bit registers 4..7 are AH/CH/DH/BH;
                // map them into the 16..19 index space.
                let reg_rm = {
                    let x = (modrm & 7) | rex_b << 3;
                    if width == 8 && rex == 0 && (4..8).contains(&x) {
                        16 + (x - 4)
                    }
                    else {
                        x
                    }
                };
                if matches!(group, 2 | 3) {
                    if modrm >> 6 == 3 {
                        let r = reg_rm;
                        out.push(if group == 2 {
                            Instruction::NotReg { r, width }
                        }
                        else {
                            Instruction::NegReg { r, width }
                        });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                        out.push(if group == 2 {
                            Instruction::NotMem { mem, width }
                        }
                        else {
                            Instruction::NegMem { mem, width }
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
                        addr_size, segment_override,
                        imm_operand_bytes(width),
                    )?)
                };
                let value = read_imm_operand(bytes, &mut i, width)?;
                if let Some(mem) = mem {
                    out.push(Instruction::ArithMemImm { op: ArithOp::Test, mem, value, width });
                }
                else {
                    let r = reg_rm;
                    out.push(Instruction::ArithRegImm { op: ArithOp::Test, r, value, width });
                }
            },

            // add rax, imm32
            0x05 => {
                let width = operand_width(prefix_66, rex_w);
                let value = read_imm_operand(bytes, &mut i, width)?;
                out.push(Instruction::AddRegImm { r: 0, value, width });
            },

            // logical/sub/cmp rax, imm32 and test rax, imm32
            0x0D | 0x15 | 0x1D | 0x25 | 0x2D | 0x35 | 0x3D | 0xA9 | 0x3C | 0xA8 => {
                let width = if opcode == 0x3C || opcode == 0xA8 {
                    8
                }
                else {
                    operand_width(prefix_66, rex_w)
                };
                let value = read_imm_operand(bytes, &mut i, width)?;
                let op = match opcode {
                    0x0D => ArithOp::Or,
                    0x15 => ArithOp::Adc,
                    0x1D => ArithOp::Sbb,
                    0x25 => ArithOp::And,
                    0x2D => ArithOp::Sub,
                    0x35 => ArithOp::Xor,
                    0x3D | 0x3C => ArithOp::Cmp,
                    0xA8 => ArithOp::Test,
                    _ => ArithOp::Test,
                };
                out.push(Instruction::ArithRegImm { op, r: 0, value, width });
            },

            // jmp rel32 / rel8
            0xE9 => {
                let rel = read_u32(bytes, &mut i)? as i32 as i64;
                out.push(Instruction::Jmp {
                    target: base.wrapping_add(i as u64).wrapping_add(rel as u64),
                });
                break;
            },

            // push/pop r64
            0x50..=0x57 => {
                if prefix_66 && !rex_w {
                    return Err("16-bit push is not supported".into());
                }
                out.push(Instruction::PushReg { r: opcode - 0x50 | rex_b << 3 });
            },
            0x58..=0x5F => {
                if prefix_66 && !rex_w {
                    return Err("16-bit pop is not supported".into());
                }
                out.push(Instruction::PopReg { r: opcode - 0x58 | rex_b << 3 });
            },

            // push imm32/imm8 (sign-extended)
            0x68 => {
                if prefix_66 && !rex_w {
                    return Err("16-bit push is not supported".into());
                }
                let value = read_u32(bytes, &mut i)? as i32 as i64 as u64;
                out.push(Instruction::PushImm { value });
            },
            0x6A => {
                if prefix_66 && !rex_w {
                    return Err("16-bit push is not supported".into());
                }
                let value = read_i8(bytes, &mut i)? as i64 as u64;
                out.push(Instruction::PushImm { value });
            },

            // call rel32
            0xE8 => {
                let rel = read_u32(bytes, &mut i)? as i32 as i64;
                let return_address = base.wrapping_add(i as u64);
                out.push(Instruction::Call {
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
                out.push(Instruction::Ret { adjustment });
                break;
            },

            // leave
            0xC9 => out.push(Instruction::Leave),

            // shl/shr/sar r/m, 1/imm8/cl (the C0/D0/D2 forms are always 8-bit)
            0xC0 | 0xC1 | 0xD0 | 0xD1 | 0xD2 | 0xD3 => {
                let byte_form = matches!(opcode, 0xC0 | 0xD0 | 0xD2);
                let width = if byte_form { 8 } else { operand_width(prefix_66, rex_w) };
                let modrm = *bytes.get(i).ok_or("truncated shift modrm")?;
                i += 1;
                let kind = match modrm >> 3 & 7 {
                    0 => ShiftKind::Rol,
                    1 => ShiftKind::Ror,
                    4 => ShiftKind::Shl,
                    5 => ShiftKind::Shr,
                    7 => ShiftKind::Sar,
                    _ => return Err("unsupported shift form".into()),
                };
                let count_mask = if width == 64 { 63 } else { 31 };
                if modrm >> 6 != 3 {
                    let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                    if matches!(opcode, 0xD2 | 0xD3) {
                        out.push(Instruction::ShiftMemCl { kind, mem, width });
                    }
                    else {
                        let count = if matches!(opcode, 0xD0 | 0xD1) { 1 } else { read_i8(bytes, &mut i)? as u8 };
                        out.push(Instruction::ShiftMem {
                            kind,
                            mem,
                            width,
                            count: count & count_mask,
                        });
                    }
                    continue;
                }
                let r = (modrm & 7) | rex_b << 3;
                if matches!(opcode, 0xD2 | 0xD3) {
                    out.push(Instruction::ShiftRegCl { kind, r, width });
                }
                else {
                    let count = if matches!(opcode, 0xD0 | 0xD1) { 1 } else { read_i8(bytes, &mut i)? as u8 };
                    out.push(Instruction::ShiftReg {
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
                    let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                    match group {
                        0 | 1 => {
                            out.push(Instruction::IncDecMem {
                                mem,
                                width: operand_width(prefix_66, rex_w),
                                decrement: group == 1,
                            });
                        },
                        2 => {
                            out.push(Instruction::CallMem { mem, return_address: base + i as u64 });
                            break;
                        },
                        4 => {
                            out.push(Instruction::JmpMem { mem });
                            break;
                        },
                        _ => unreachable!(),
                    }
                    continue;
                }
                let r = (modrm & 7) | rex_b << 3;
                if group <= 1 {
                    out.push(Instruction::IncDecReg {
                        r,
                        width: operand_width(prefix_66, rex_w),
                        decrement: group == 1,
                    });
                }
                else if group == 2 {
                    out.push(Instruction::CallReg { r, return_address: base + i as u64 });
                    break;
                }
                else {
                    out.push(Instruction::JmpReg { r });
                    break;
                }
            },
            0xEB => {
                let rel = read_i8(bytes, &mut i)? as i64;
                out.push(Instruction::Jmp {
                    target: base.wrapping_add(i as u64).wrapping_add(rel as u64),
                });
                break;
            },

            // jcc rel8
            0x70..=0x7F => {
                let rel = read_i8(bytes, &mut i)? as i64;
                let after = base.wrapping_add(i as u64);
                out.push(Instruction::Jcc {
                    code: opcode - 0x70,
                    target: after.wrapping_add(rel as u64),
                    fallthrough: after,
                });
                if !unsafe { JIT64_SUPERBLOCKS } {
                    break;
                }
            },

            // jcc rel32
            0x0F => {
                let second = *bytes.get(i).ok_or("truncated two-byte opcode")?;
                i += 1;
                // F3 0F xx is only MOVDQU (6F/7F) here; everything else, plus
                // ENDBR64 (F3 0F 1E FA), is left to the interpreter.
                let endbr64 = second == 0x1E && bytes.get(i) == Some(&0xFA);
                let movdqu = unsafe { JIT64_SSE } && matches!(second, 0x6F | 0x7F);
                if prefix_f3 && !endbr64 && !movdqu {
                    return Err(format!("unsupported f3 0f opcode {:02x} at {}", second, start));
                }
                if (0x80..=0x8F).contains(&second) {
                    let rel = read_u32(bytes, &mut i)? as i32 as i64;
                    let after = base.wrapping_add(i as u64);
                    out.push(Instruction::Jcc {
                        code: second - 0x80,
                        target: after.wrapping_add(rel as u64),
                        fallthrough: after,
                    });
                    if !unsafe { JIT64_SUPERBLOCKS } {
                        break;
                    }
                }
                else if second == 0x1F {
                    let modrm = *bytes.get(i).ok_or("truncated nop modrm")?;
                    i += 1;
                    if modrm >> 6 != 3 {
                        let _ = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                    }
                    out.push(Instruction::Nop);
                }
                else if second == 0x1E && bytes.get(i) == Some(&0xFA) {
                    i += 1;
                    out.push(Instruction::Nop); // ENDBR64
                }
                else if second == 0x18 || second == 0x0D {
                    // PREFETCHT0/T1/T2/NTA (0F 18 /0../3) and PREFETCHW (0F 0D):
                    // no architectural effect, only the operand has to be decoded.
                    let modrm = *bytes.get(i).ok_or("truncated prefetch modrm")?;
                    i += 1;
                    if modrm >> 6 != 3 {
                        let _ = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                    }
                    out.push(Instruction::Nop);
                }
                else if second == 0xB0 || second == 0xB1 {
                    // CMPXCHG r/m, r (0F B1) and CMPXCHG r/m8, r8 (0F B0)
                    let modrm = *bytes.get(i).ok_or("truncated cmpxchg modrm")?;
                    i += 1;
                    let width = if second == 0xB0 { 8 } else { operand_width(prefix_66, rex_w) };
                    // AH/CH/DH/BH need the high-8 handling used elsewhere; let
                    // the interpreter take those encodings.
                    if width == 8 && rex == 0 && ((modrm >> 3 & 7) >= 4 || (modrm >> 6 == 3 && (modrm & 7) >= 4)) {
                        return Err("cmpxchg with a high byte register is not supported".into());
                    }
                    let src = (modrm >> 3 & 7) | rex_r << 3;
                    if modrm >> 6 == 3 {
                        out.push(Instruction::CmpxchgReg { dst: (modrm & 7) | rex_b << 3, src, width });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                        out.push(Instruction::CmpxchgMem { mem, src, width });
                    }
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
                        out.push(Instruction::MovExtendReg {
                            dst, src, src_width, dst_width, signed, high8,
                        });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                        out.push(Instruction::MovExtendMem {
                            dst, mem, src_width, dst_width, signed,
                        });
                    }
                }
                // MOVDQA/MOVDQU xmm, xmm/m128 (66 0F 6F, F3 0F 6F)
                else if unsafe { JIT64_SSE } && second == 0x6F && (prefix_66 || prefix_f3) {
                    let modrm = *bytes.get(i).ok_or("truncated movdqa modrm")?;
                    i += 1;
                    let dst = (modrm >> 3 & 7) | rex_r << 3;
                    if modrm >> 6 == 3 {
                        out.push(Instruction::XmmCopy { dst, src: (modrm & 7) | rex_b << 3 });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                        out.push(Instruction::XmmLoad { dst, mem });
                    }
                }
                // MOVDQA/MOVDQU xmm/m128, xmm (66 0F 7F, F3 0F 7F)
                else if unsafe { JIT64_SSE } && second == 0x7F && (prefix_66 || prefix_f3) {
                    let modrm = *bytes.get(i).ok_or("truncated movdqa modrm")?;
                    i += 1;
                    let src = (modrm >> 3 & 7) | rex_r << 3;
                    if modrm >> 6 == 3 {
                        out.push(Instruction::XmmCopy { dst: (modrm & 7) | rex_b << 3, src });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                        out.push(Instruction::XmmStore { mem, src });
                    }
                }
                // PXOR xmm, xmm/m128 (66 0F EF)
                else if unsafe { JIT64_SSE } && second == 0xEF && prefix_66 {
                    let modrm = *bytes.get(i).ok_or("truncated pxor modrm")?;
                    i += 1;
                    let dst = (modrm >> 3 & 7) | rex_r << 3;
                    if modrm >> 6 == 3 {
                        out.push(Instruction::XmmXor { dst, src: (modrm & 7) | rex_b << 3 });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                        out.push(Instruction::XmmXorMem { dst, mem });
                    }
                }
                // Integer SSE2 (66 0F xx), via interp64's sse_int_apply.
                else if unsafe { JIT64_SSE } && prefix_66 && is_sse_int_op(second) {
                    let modrm = *bytes.get(i).ok_or("truncated sse modrm")?;
                    i += 1;
                    let dst = (modrm >> 3 & 7) | rex_r << 3;
                    if modrm >> 6 == 3 {
                        out.push(Instruction::XmmInt { op: second, dst, src: (modrm & 7) | rex_b << 3 });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                        out.push(Instruction::XmmIntMem { op: second, dst, mem });
                    }
                }
                // Packed shift by imm (66 0F 71/72/73); register form only.
                else if unsafe { JIT64_SSE } && prefix_66 && matches!(second, 0x71 | 0x72 | 0x73) {
                    let modrm = *bytes.get(i).ok_or("truncated sse shift modrm")?;
                    i += 1;
                    if modrm >> 6 != 3 {
                        return Err("sse shift by immediate with a memory operand".into());
                    }
                    let count = *bytes.get(i).ok_or("truncated sse shift imm8")?;
                    i += 1;
                    out.push(Instruction::XmmShiftImm {
                        op: second,
                        group: modrm >> 3 & 7,
                        count,
                        dst: (modrm & 7) | rex_b << 3,
                    });
                }
                // PSHUFD (66 0F 70); PSHUFHW/PSHUFLW stay in the interpreter.
                else if unsafe { JIT64_SSE } && prefix_66 && second == 0x70 {
                    let modrm = *bytes.get(i).ok_or("truncated pshufd modrm")?;
                    i += 1;
                    let dst = (modrm >> 3 & 7) | rex_r << 3;
                    let control = *bytes.get(i).ok_or("truncated pshufd imm8")?;
                    i += 1;
                    if modrm >> 6 == 3 {
                        out.push(Instruction::XmmShuf { control, dst, src: (modrm & 7) | rex_b << 3 });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                        out.push(Instruction::XmmShufMem { control, dst, mem });
                    }
                }
                else if (0x40..=0x4F).contains(&second) {
                    let modrm = *bytes.get(i).ok_or("truncated cmov modrm")?;
                    i += 1;
                    let dst = (modrm >> 3 & 7) | rex_r << 3;
                    let width = operand_width(prefix_66, rex_w);
                    if modrm >> 6 == 3 {
                        out.push(Instruction::CmovRegReg {
                            code: second - 0x40,
                            dst,
                            src: (modrm & 7) | rex_b << 3,
                            width,
                        });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                        out.push(Instruction::CmovRegMem {
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
                        out.push(Instruction::ImulRegReg {
                            dst,
                            lhs: dst,
                            rhs: (modrm & 7) | rex_b << 3,
                            width,
                        });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                        out.push(Instruction::ImulRegMem { dst, mem, value: None, width });
                    }
                }
                else if (0x90..=0x9F).contains(&second) {
                    let modrm = *bytes.get(i).ok_or("truncated setcc modrm")?;
                    i += 1;
                    if modrm >> 6 == 3 {
                        let rm = modrm & 7;
                        let high8 = rex == 0 && rm >= 4;
                        out.push(Instruction::SetccReg {
                            code: second - 0x90,
                            dst: if high8 { rm - 4 } else { rm | rex_b << 3 },
                            high8,
                        });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                        out.push(Instruction::SetccMem {
                            code: second - 0x90,
                            mem,
                        });
                    }
                }
                else if (0xC8..=0xCF).contains(&second) {
                    if prefix_66 {
                        return Err("bswap with a 16-bit operand is undefined".into());
                    }
                    out.push(Instruction::Bswap {
                        r: second - 0xC8 | rex_b << 3,
                        width: if rex_w { 64 } else { 32 },
                    });
                }
                // RDTSC (0F 31)
                else if second == 0x31 {
                    out.push(Instruction::Rdtsc);
                }
                // BT r/m, r (0F A3), register form only
                else if second == 0xA3 {
                    let modrm = *bytes.get(i).ok_or("truncated modrm")?;
                    i += 1;
                    if modrm >> 6 != 3 {
                        return Err("bt with a memory operand is not supported".into());
                    }
                    let ir = (modrm >> 3 & 7) | rex_r << 3;
                    out.push(Instruction::BitTestReg {
                        r: (modrm & 7) | rex_b << 3,
                        index_r: ir,
                        op: 0,
                        width: operand_width(prefix_66, rex_w),
                    });
                }
                // group 8: BT/BTS/BTR/BTC r/m, imm8 (0F BA /4../7)
                else if second == 0xBA {
                    let width = operand_width(prefix_66, rex_w);
                    let modrm = *bytes.get(i).ok_or("truncated modrm")?;
                    i += 1;
                    let group = modrm >> 3 & 7;
                    if !(4..=7).contains(&group) {
                        return Err("unsupported group 8 form".into());
                    }
                    if modrm >> 6 != 3 {
                        return Err("0f ba with a memory operand is not supported".into());
                    }
                    let index = *bytes.get(i).ok_or("truncated imm8")? as u64;
                    i += 1;
                    out.push(Instruction::BitTestImm {
                        r: (modrm & 7) | rex_b << 3,
                        index,
                        op: group - 4,
                        width,
                    });
                }
                // CPUID (0F A2)
                else if second == 0xA2 {
                    out.push(Instruction::Cpuid);
                }
                else {
                    return Err(format!("unsupported opcode 0f {:02x} at {}", second, start));
                }
            },

            0xF4 => {
                out.push(Instruction::Hlt);
                break;
            },

            _ => return Err(format!("unsupported opcode {:02x} at {}", opcode, start)),
        }
    }

    out.end_rip = base.wrapping_add(i as u64);
    Ok(())
}

fn new_decoded_block(base: u64) -> DecodedBlock {
    DecodedBlock {
        instrs: Vec::new(),
        rips: Vec::new(),
        current_rip: base,
        end_rip: base,
    }
}

// Strict decoding: any undecodable instruction fails the block (decoder tests).
fn decode_block_strict(base: u64, bytes: &[u8]) -> Result<DecodedBlock, String> {
    let mut out = new_decoded_block(base);
    decode_into(&mut out, base, bytes)?;
    Ok(out)
}

// Compile the decodable prefix, resume at the first undecodable instruction.
fn decode_block_with_rips(base: u64, bytes: &[u8]) -> Result<DecodedBlock, String> {
    let mut out = new_decoded_block(base);
    match decode_into(&mut out, base, bytes) {
        Ok(()) => Ok(out),
        Err(e) =>
        {
            if !unsafe { JIT64_PARTIAL_BLOCKS } || out.instrs.is_empty() {
                return Err(e);
            }
            // Resume in the interpreter at `current_rip`, where decode stopped.
            out.end_rip = out.current_rip;
            Ok(out)
        },
    }
}

pub fn decode_block(base: u64, bytes: &[u8]) -> Result<Vec<Instruction>, String> {
    Ok(decode_block_strict(base, bytes)?.instrs)
}

// Decode a block into (instructions, per-instruction RIPs, end RIP), for the
// interpreter's decode cache.
pub fn decode_block_parts(
    base: u64,
    bytes: &[u8],
) -> Result<(Vec<Instruction>, Vec<u64>, u64), String> {
    let d = decode_block_with_rips(base, bytes)?;
    Ok((d.instrs, d.rips, d.end_rip))
}

fn find_reg(locals: &[(u8, WasmLocalI64)], r: u8) -> Option<usize> {
    locals.iter().position(|(rr, _)| *rr == r)
}

fn load_reg(b: &mut WasmBuilder, locals: &mut Vec<(u8, WasmLocalI64)>, r: u8) -> usize {
    if let Some(i) = find_reg(locals, r) {
        return i;
    }
    if r >= 16 {
        // AH/CH/DH/BH: the high byte of register r-16. Never cached, so it is
        // always derived from the base register's current local.
        let bi = load_reg(b, locals, r - 16);
        b.get_local_i64(&locals[bi].1);
        b.const_i64(8);
        b.shr_u_i64();
        b.const_i64(0xFF);
        b.and_i64();
        let local = b.set_new_local_i64();
        // Key 0xFF never matches a register index and is skipped on write-back.
        locals.push((0xFF, local));
        return locals.len() - 1;
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
    // The FS/GS base is added after the wrap, matching the interpreter.
    match mem.segment {
        Some(0x64) => {
            b.const_i32(FS_BASE_ADDR);
            b.load_aligned_i64(0);
            b.add_i64();
        },
        Some(0x65) => {
            b.const_i32(GS_BASE_ADDR);
            b.load_aligned_i64(0);
            b.add_i64();
        },
        _ => {},
    }
}

fn gen_check_memory_fault(b: &mut WasmBuilder, locals: &[(u8, WasmLocalI64)]) {
    b.call_fn0_ret("jit64_memory_faulted");
    b.if_void();
    emit_registers_back(b, locals);
    b.return_();
    b.block_end();
    // An SMC write may have invalidated this block; stop before running on.
    if unsafe { JIT64_SMC_BAIL } {
        b.call_fn0_ret("jit64_code_write_bail");
        b.if_void();
        emit_registers_back(b, locals);
        b.return_();
        b.block_end();
    }
}

fn gen_memory_address_local(
    b: &mut WasmBuilder,
    locals: &mut Vec<(u8, WasmLocalI64)>,
    mem: &Mem,
) -> WasmLocalI64 {
    gen_effective_addr(b, locals, mem);
    b.set_new_local_i64()
}

// Integer SSE2 opcodes (66 0F xx) shared with interp64's sse_int_apply.
fn is_sse_int_op(op: u8) -> bool {
    matches!(
        op,
        0x60 | 0x61 | 0x62 | 0x63 | 0x64 | 0x65 | 0x66 | 0x67 | 0x68 | 0x69 | 0x6A | 0x6B
            | 0x6C | 0x6D | 0x74 | 0x75 | 0x76 | 0xD1 | 0xD2 | 0xD3 | 0xD4 | 0xD5 | 0xD8
            | 0xD9 | 0xDA | 0xDB | 0xDC | 0xDD | 0xDE | 0xDF | 0xE1 | 0xE2 | 0xE4 | 0xE5
            | 0xE8 | 0xE9 | 0xEA | 0xEB | 0xEC | 0xED | 0xEE | 0xEF | 0xF1 | 0xF2 | 0xF3
            | 0xF5 | 0xF6 | 0xF8 | 0xF9 | 0xFA | 0xFB | 0xFC | 0xFD | 0xFE
    )
}

// Long-mode inline memory fast path.
const CPL_ADDR: i32 = 612; // global_pointers::cpl
const CR0_ADDR: i32 = 580; // global_pointers::cr
const MEMORY_SIZE_ADDR: i32 = 812; // global_pointers::memory_size
const CR0_WP: i32 = 1 << 16;

fn tlb64_base() -> i32 { std::ptr::addr_of!(crate::cpu::core::tlb64) as u32 as i32 }
fn tlb64_generation_addr() -> i32 {
    std::ptr::addr_of!(crate::cpu::core::tlb64_generation) as u32 as i32
}
fn mem8_static_addr() -> i32 {
    std::ptr::addr_of!(crate::memory::mem8) as u32 as i32
}

// Leaves a local holding &tlb64[(vaddr >> 12) & mask].
fn gen_tlb64_entry(b: &mut WasmBuilder, address: &WasmLocalI64) -> WasmLocal {
    use crate::cpu::core::*;
    b.get_local_i64(address);
    b.const_i64(12);
    b.shr_u_i64();
    b.wrap_i64_to_i32();
    b.const_i32(TLB64_INDEX_MASK as i32);
    b.and_i32();
    b.const_i32(TLB64_ENTRY_SIZE as i32);
    b.mul_i32();
    b.const_i32(tlb64_base());
    b.add_i32();
    b.set_new_local()
}

// Leaves i32 (0/1): can this read be inline? `entry` = TLB for `address`.
fn gen_memory_read_fast_condition(
    b: &mut WasmBuilder,
    entry: &WasmLocal,
    address: &WasmLocalI64,
    width: u8,
) {
    use crate::cpu::core::*;
    let bytes = (width / 8) as i32;

    // generation matches
    b.get_local(entry);
    b.load_aligned_i32(TLB64_OFF_GENERATION);
    b.const_i32(tlb64_generation_addr());
    b.load_aligned_i32(0);
    b.eq_i32();
    // tag matches vaddr & !0xFFF
    b.get_local(entry);
    b.load_aligned_i64(TLB64_OFF_TAG);
    b.get_local_i64(address);
    b.const_i64(!0xFFFi64);
    b.and_i64();
    b.eq_i64();
    b.and_i32();
    // the access stays inside one page
    b.get_local_i64(address);
    b.wrap_i64_to_i32();
    b.const_i32(0xFFF);
    b.and_i32();
    b.const_i32(0x1000 - bytes);
    b.leu_i32();
    b.and_i32();
    // a user access needs the user bit
    b.const_i32(CPL_ADDR);
    b.load_u8(0);
    b.eqz_i32();
    b.get_local(entry);
    b.load_u8(TLB64_OFF_FLAGS);
    b.const_i32(TLB64_USER as i32);
    b.and_i32();
    b.const_i32(0);
    b.ne_i32();
    b.or_i32();
    b.and_i32();
    // phys is plain RAM: below memory_size and outside the 0xA0000..0xC0000 hole
    gen_tlb_is_ram(b, entry);
    b.and_i32();
}

// Leaves an i32 (0/1) on the stack: whether this write can be served inline.
fn gen_memory_write_fast_condition(
    b: &mut WasmBuilder,
    entry: &WasmLocal,
    address: &WasmLocalI64,
    width: u8,
) {
    use crate::cpu::core::*;
    let bytes = (width / 8) as i32;

    b.get_local(entry);
    b.load_aligned_i32(TLB64_OFF_GENERATION);
    b.const_i32(tlb64_generation_addr());
    b.load_aligned_i32(0);
    b.eq_i32();
    b.get_local(entry);
    b.load_aligned_i64(TLB64_OFF_TAG);
    b.get_local_i64(address);
    b.const_i64(!0xFFFi64);
    b.and_i64();
    b.eq_i64();
    b.and_i32();
    b.get_local_i64(address);
    b.wrap_i64_to_i32();
    b.const_i32(0xFFF);
    b.and_i32();
    b.const_i32(0x1000 - bytes);
    b.leu_i32();
    b.and_i32();
    // user bit (for a user access) and the write bit (or a supervisor write with
    // CR0.WP clear)
    b.const_i32(CPL_ADDR);
    b.load_u8(0);
    b.eqz_i32(); // !user
    b.get_local(entry);
    b.load_u8(TLB64_OFF_FLAGS);
    b.const_i32(TLB64_USER as i32);
    b.and_i32();
    b.const_i32(0);
    b.ne_i32();
    b.or_i32();
    b.and_i32();
    // writable = (flags & WRITABLE) || (!user && CR0.WP == 0)
    b.get_local(entry);
    b.load_u8(TLB64_OFF_FLAGS);
    b.const_i32(TLB64_WRITABLE as i32);
    b.and_i32();
    b.const_i32(0);
    b.ne_i32();
    b.const_i32(CPL_ADDR);
    b.load_u8(0);
    b.eqz_i32();
    b.const_i32(CR0_ADDR);
    b.load_aligned_i32(0);
    b.const_i32(CR0_WP);
    b.and_i32();
    b.const_i32(0);
    b.ne_i32();
    b.eqz_i32();
    b.and_i32();
    b.or_i32();
    b.and_i32();
    // no compiled code on this physical page (self-modifying code stays correct)
    b.get_local(entry);
    b.load_u8(TLB64_OFF_FLAGS);
    b.const_i32(TLB64_HAS_CODE as i32);
    b.and_i32();
    b.const_i32(0);
    b.eq_i32();
    b.and_i32();
    gen_tlb_is_ram(b, entry);
    b.and_i32();
}

// Leaves an i32 (0/1): the TLB entry maps plain RAM (not MMIO/device). 
fn gen_tlb_is_ram(b: &mut WasmBuilder, entry: &WasmLocal) {
    use crate::cpu::core::*;
    b.get_local(entry);
    b.load_u8(TLB64_OFF_FLAGS);
    b.const_i32(TLB64_RAM as i32);
    b.and_i32();
    b.const_i32(0);
    b.ne_i32();
}

// Leaves the linear address (i32); `entry` = TLB entry for `address`.
fn gen_inline_address(b: &mut WasmBuilder, entry: &WasmLocal, address: &WasmLocalI64) {
    use crate::cpu::core::*;
    b.get_local(entry);
    b.load_aligned_i32(TLB64_OFF_PHYS_PAGE);
    b.get_local_i64(address);
    b.wrap_i64_to_i32();
    b.const_i32(0xFFF);
    b.and_i32();
    b.or_i32();
    b.const_i32(mem8_static_addr());
    b.load_aligned_i32(0);
    b.add_i32();
}

fn gen_inline_load(b: &mut WasmBuilder, entry: &WasmLocal, address: &WasmLocalI64, width: u8) {
    gen_inline_address(b, entry, address);
    match width {
        8 =>
        {
            b.load_u8(0);
            b.extend_unsigned_i32_to_i64();
        },
        16 =>
        {
            b.load_aligned_u16(0);
            b.extend_unsigned_i32_to_i64();
        },
        32 =>
        {
            b.load_aligned_i32(0);
            b.extend_unsigned_i32_to_i64();
        },
        _ => { b.load_aligned_i64(0); },
    }
}

fn gen_inline_store(
    b: &mut WasmBuilder,
    entry: &WasmLocal,
    address: &WasmLocalI64,
    value: &WasmLocalI64,
    width: u8,
) {
    gen_inline_address(b, entry, address);
    b.get_local_i64(value);
    match width {
        8 =>
        {
            b.wrap_i64_to_i32();
            b.store_u8(0);
        },
        16 =>
        {
            b.wrap_i64_to_i32();
            b.store_unaligned_u16(0);
        },
        32 =>
        {
            b.wrap_i64_to_i32();
            b.store_unaligned_i32(0);
        },
        _ => { b.store_unaligned_i64(0); },
    }
}

fn gen_memory_read(
    b: &mut WasmBuilder,
    locals: &[(u8, WasmLocalI64)],
    address: &WasmLocalI64,
    width: u8,
) -> WasmLocalI64 {
    if unsafe { JIT64_INLINE_MEMORY } && matches!(width, 8 | 16 | 32 | 64)
    {
        // One TLB entry for both the fast-path check and the inline address.
        let entry = gen_tlb64_entry(b, address);
        gen_memory_read_fast_condition(b, &entry, address, width);
        b.if_i64();
        gen_inline_load(b, &entry, address, width);
        b.else_();
        b.get_local_i64(address);
        b.const_i32(width as i32);
        b.call_fn2_i64_i32_ret_i64("jit64_mem_read");
        gen_check_memory_fault(b, locals);
        b.block_end();
        let value = b.set_new_local_i64();
        b.free_local(entry);
        return value;
    }
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
    if unsafe { JIT64_SELFCHECK } {
        // Log the store for the self-check (the re-run may overwrite it).
        b.get_local_i64(address);
        b.get_local_i64(value);
        b.const_i32(width as i32);
        b.call_fn3_i64_i64_i32("jit64_selfcheck_log_write");
    }
    if unsafe { JIT64_INLINE_MEMORY && JIT64_INLINE_MEMORY_WRITE } && matches!(width, 8 | 16 | 32 | 64)
    {
        // One TLB entry for both the fast-path check and the inline address.
        let entry = gen_tlb64_entry(b, address);
        gen_memory_write_fast_condition(b, &entry, address, width);
        b.if_void();
        // A plain RAM write has no side effects, so no register flush is needed.
        gen_inline_store(b, &entry, address, value, width);
        b.else_();
        gen_memory_write_slow(b, locals, address, value, width);
        b.block_end();
        b.free_local(entry);
        return;
    }
    gen_memory_write_slow(b, locals, address, value, width);
}

fn gen_memory_write_slow(
    b: &mut WasmBuilder,
    locals: &[(u8, WasmLocalI64)],
    address: &WasmLocalI64,
    value: &WasmLocalI64,
    width: u8,
) {
    // A slow (MMIO/device) write must not be applied twice by the self-check.
    unsafe { BLOCK_NO_SELFCHECK = true; }
    // An MMIO write can interrupt; flush registers first so RSP is current.
    emit_registers_back(b, locals);
    b.get_local_i64(address);
    b.get_local_i64(value);
    b.const_i32(width as i32);
    b.call_fn3_i64_i64_i32("jit64_mem_write");
    gen_check_memory_fault(b, locals);
    // It changed *rip and RSP: stop the block, but keep this block's register
    // updates (RSP excepted, since the delivery already set it).
    b.call_fn0_ret("jit64_exception_delivered");
    b.if_void();
    emit_registers_back_except_rsp(b, locals);
    b.return_();
    b.block_end();
}

// SSE2: the XMM file is in emulated memory, so 128-bit operands are pairs of
// 64-bit halves.
fn gen_xmm_reg_load(b: &mut WasmBuilder, r: u8, offset: u32) {
    b.const_i32(crate::cpu::global_pointers::get_reg_xmm_addr(r) as i32);
    b.load_aligned_i64(offset);
}

fn gen_xmm_store_half(b: &mut WasmBuilder, r: u8, value: &WasmLocalI64, offset: u32) {
    b.const_i32(crate::cpu::global_pointers::get_reg_xmm_addr(r) as i32);
    b.get_local_i64(value);
    b.store_aligned_i64(offset);
}

fn gen_xmm_mem_half_read(
    b: &mut WasmBuilder,
    locals: &[(u8, WasmLocalI64)],
    address: &WasmLocalI64,
    offset: i64,
) -> WasmLocalI64 {
    if offset != 0 {
        b.get_local_i64(address);
        b.const_i64(offset);
        b.add_i64();
        let addr = b.set_new_local_i64();
        let value = gen_memory_read(b, locals, &addr, 64);
        b.free_local_i64(addr);
        value
    }
    else {
        gen_memory_read(b, locals, address, 64)
    }
}

fn gen_xmm_mem_half_write(
    b: &mut WasmBuilder,
    locals: &[(u8, WasmLocalI64)],
    address: &WasmLocalI64,
    offset: i64,
    value: &WasmLocalI64,
) {
    if offset != 0 {
        b.get_local_i64(address);
        b.const_i64(offset);
        b.add_i64();
        let addr = b.set_new_local_i64();
        gen_memory_write(b, locals, &addr, value, 64);
        b.free_local_i64(addr);
    }
    else {
        gen_memory_write(b, locals, address, value, 64);
    }
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
    if width == 8 {
        // An 8-bit write preserves the upper 56 bits.
        b.get_local_i64(dst);
        b.const_i64(!0xFF);
        b.and_i64();
        b.get_local_i64(value);
        b.const_i64(0xFF);
        b.and_i64();
        b.or_i64();
        b.set_local_i64(dst);
    }
    else if width == 16 {
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
    if r >= 16 {
        // AH/CH/DH/BH: merge into bits 8..15 of the base register.
        let bi = load_reg(b, locals, r - 16);
        b.get_local_i64(&locals[bi].1);
        b.const_i64(!0xFF00);
        b.and_i64();
        b.get_local_i64(value);
        b.const_i64(0xFF);
        b.and_i64();
        b.const_i64(8);
        b.shl_i64();
        b.or_i64();
        b.set_local_i64(&locals[bi].1);
        return;
    }
    if let Some(di) = find_reg(locals, r) {
        emit_write_reg(b, &locals[di].1, value, width);
    }
    else if width == 8 || width == 16 {
        let di = load_reg(b, locals, r);
        emit_write_reg(b, &locals[di].1, value, width);
    }
    else {
        b.get_local_i64(value);
        let local = b.set_new_local_i64();
        locals.push((r, local));
    }
}

// AH/CH/DH/BH use temporary locals; merge ALU results into their parent
// registers before another instruction reads them.
fn write_back_high_byte(
    b: &mut WasmBuilder,
    locals: &mut Vec<(u8, WasmLocalI64)>,
    r: u8,
    index: usize,
) {
    if r >= 16 {
        let value = locals[index].1.unsafe_clone();
        write_reg_value(b, locals, r, &value, 8);
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
    emit::gen_get_flags(b);
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
    // 64-bit operands need no masking, so use the register locals directly.
    let owned = width != 64;
    let lhs = if owned {
        b.get_local_i64(dst);
        emit_mask(b, width);
        b.set_new_local_i64()
    }
    else {
        dst.unsafe_clone()
    };
    let rhs = if owned {
        b.get_local_i64(src);
        emit_mask(b, width);
        b.set_new_local_i64()
    }
    else {
        src.unsafe_clone()
    };
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
    if owned {
        b.free_local_i64(lhs);
        b.free_local_i64(rhs);
    }
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
    emit::gen_get_flags(b);
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
        if *r >= 16 {
            // High-byte temporaries (and their base is written back separately).
            continue;
        }
        store_reg(b, local, *r);
    }
}

// Hand off to the successor block. Emitted only when chaining is enabled.
fn emit_chain_exit(b: &mut WasmBuilder) {
    if unsafe { JIT64_CHAIN_CODEGEN } {
        b.call_fn0("jit64_chain");
    }
}

// Block entry setup (flags + exception flag). Normally in run_block() (A/B).
fn emit_block_prologue(b: &mut WasmBuilder) {
    if unsafe { JIT64_BLOCK_PROLOGUE } {
        b.call_fn0("jit64_sync_flags");
        b.call_fn0("jit64_clear_exception_flag");
    }
}

// Count a block exit by reason (only when a diagnostic build asks for it).
fn emit_exit_stat(b: &mut WasmBuilder, reason: u32) {
    if unsafe { JIT64_EXIT_STATS } {
        b.const_i32(reason as i32);
        b.call_fn1("jit64_note_exit");
    }
}

// Like emit_registers_back, but leaves RSP alone: used on the early-return path
// after a helper delivered an exception, whose frame push already set RSP.
fn emit_registers_back_except_rsp(b: &mut WasmBuilder, locals: &[(u8, WasmLocalI64)]) {
    for (r, local) in locals {
        if *r >= 16 || *r == 4 {
            continue;
        }
        store_reg(b, local, *r);
    }
}

// A register loaded inside one arm is only valid there; the other path still
// reads its local's default 0. At the join, flush this arm's locals to memory.
fn end_branch_arm(b: &mut WasmBuilder, locals: &mut Vec<(u8, WasmLocalI64)>, saved: usize) {
    while locals.len() > saved
    {
        let (r, local) = locals.pop().unwrap();
        if r < 16
        {
            store_reg(b, &local, r);
        }
        b.free_local_i64(local);
    }
}

// `block_end` is where execution falls through when the block ends normally.
fn compile_block_with_rips(instrs: &[Instruction], rips: &[u64], block_end: u64) -> Vec<u8> {
    let mut b = WasmBuilder::new();
    let mut locals: Vec<(u8, WasmLocalI64)> = Vec::new();
    let mut terminated = false;
    unsafe { BLOCK_NO_SELFCHECK = false; }

    // Materialise lazy flags first; normally run_block() does this host-side.
    emit_block_prologue(&mut b);

    for (index, instr) in instrs.iter().enumerate() {
        if let Some(instruction_rip) = rips.get(index) {
            b.const_i32(PREVIOUS_RIP_ADDR);
            b.const_i64(*instruction_rip as i64);
            b.store_aligned_i64(0);
            // A helper's interrupt resumes after the effects, so point at the
            // next instruction.
            let resume = rips.get(index + 1).copied().unwrap_or(block_end);
            b.const_i32(IP_ADDR);
            b.const_i64(resume as i64);
            b.store_aligned_i64(0);
        }
        bump_instruction_counter(&mut b);
        match *instr {
            Instruction::MovRegImm { r, value, width } => {
                b.const_i64(value as i64);
                let local = b.set_new_local_i64();
                write_reg_value(&mut b, &mut locals, r, &local, width);
                b.free_local_i64(local);
            },
            Instruction::MovRegReg { dst, src, width } => {
                let si = load_reg(&mut b, &mut locals, src);
                b.get_local_i64(&locals[si].1);
                emit_mask(&mut b, width);
                let value = b.set_new_local_i64();
                write_reg_value(&mut b, &mut locals, dst, &value, width);
                b.free_local_i64(value);
            },
            Instruction::MovRegMem { dst, mem, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let value = gen_memory_read(&mut b, &locals, &address, width);
                write_reg_value(&mut b, &mut locals, dst, &value, width);
                b.free_local_i64(address);
                b.free_local_i64(value);
            },
            Instruction::MovMemReg { mem, src, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let si = load_reg(&mut b, &mut locals, src);
                let value = locals[si].1.unsafe_clone();
                gen_memory_write(&mut b, &locals, &address, &value, width);
                b.free_local_i64(address);
            },
            Instruction::MovMemImm { mem, value, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                b.const_i64(value as i64);
                let value = b.set_new_local_i64();
                gen_memory_write(&mut b, &locals, &address, &value, width);
                b.free_local_i64(value);
                b.free_local_i64(address);
            },
            Instruction::MovExtendReg { dst, src, src_width, dst_width, signed, high8 } => {
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
            Instruction::MovExtendMem { dst, mem, src_width, dst_width, signed } => {
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
            Instruction::XchgRegReg { a, b: other, width } => {
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
            Instruction::CmpxchgReg { dst, src, width } => {
                let di = load_reg(&mut b, &mut locals, dst);
                let ai = load_reg(&mut b, &mut locals, 0);
                b.get_local_i64(&locals[di].1);
                emit_mask(&mut b, width);
                b.get_local_i64(&locals[ai].1);
                emit_mask(&mut b, width);
                b.eq_i64();
                let zf = b.set_new_local();
                b.const_i32(FLAGS_ADDR);
                emit::gen_get_flags(&mut b);
                b.const_i32(!0x40);
                b.and_i32();
                b.get_local(&zf);
                b.const_i32(6);
                b.shl_i32();
                b.or_i32();
                b.store_aligned_i32(0);
                let saved = locals.len();
                b.get_local(&zf);
                b.if_void();
                {
                    let si = load_reg(&mut b, &mut locals, src);
                    let sv = locals[si].1.unsafe_clone();
                    b.get_local_i64(&sv);
                    emit_mask(&mut b, width);
                    let masked = b.set_new_local_i64();
                    emit_write_reg(&mut b, &locals[di].1, &masked, width);
                    b.free_local_i64(masked);
                }
                end_branch_arm(&mut b, &mut locals, saved);
                b.else_();
                {
                    b.get_local_i64(&locals[di].1);
                    emit_mask(&mut b, width);
                    let masked = b.set_new_local_i64();
                    emit_write_reg(&mut b, &locals[ai].1, &masked, width);
                    b.free_local_i64(masked);
                }
                end_branch_arm(&mut b, &mut locals, saved);
                b.block_end();
                b.free_local(zf);
            },
            Instruction::CmpxchgMem { mem, src, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let current = gen_memory_read(&mut b, &locals, &address, width);
                let ai = load_reg(&mut b, &mut locals, 0);
                b.get_local_i64(&current);
                emit_mask(&mut b, width);
                b.get_local_i64(&locals[ai].1);
                emit_mask(&mut b, width);
                b.eq_i64();
                let zf = b.set_new_local();
                b.const_i32(FLAGS_ADDR);
                emit::gen_get_flags(&mut b);
                b.const_i32(!0x40);
                b.and_i32();
                b.get_local(&zf);
                b.const_i32(6);
                b.shl_i32();
                b.or_i32();
                b.store_aligned_i32(0);
                let saved = locals.len();
                b.get_local(&zf);
                b.if_void();
                {
                    gen_memory_probe_write(&mut b, &locals, &address, width);
                    let si = load_reg(&mut b, &mut locals, src);
                    let sv = locals[si].1.unsafe_clone();
                    b.get_local_i64(&sv);
                    emit_mask(&mut b, width);
                    let masked = b.set_new_local_i64();
                    gen_memory_write(&mut b, &locals, &address, &masked, width);
                    b.free_local_i64(masked);
                }
                end_branch_arm(&mut b, &mut locals, saved);
                b.else_();
                {
                    b.get_local_i64(&current);
                    emit_mask(&mut b, width);
                    let masked = b.set_new_local_i64();
                    emit_write_reg(&mut b, &locals[ai].1, &masked, width);
                    b.free_local_i64(masked);
                }
                end_branch_arm(&mut b, &mut locals, saved);
                b.block_end();
                b.free_local(zf);
                b.free_local_i64(address);
                b.free_local_i64(current);
            },
            Instruction::XchgMemReg { mem, r, width } => {
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
            Instruction::NotReg { r, width } => {
                let ri = load_reg(&mut b, &mut locals, r);
                b.get_local_i64(&locals[ri].1);
                b.const_i64(width_mask(width) as i64);
                b.xor_i64();
                emit_mask(&mut b, width);
                b.set_local_i64(&locals[ri].1);
                write_back_high_byte(&mut b, &mut locals, r, ri);
            },
            Instruction::NotMem { mem, width } => {
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
            Instruction::NegReg { r, width } => {
                let ri = load_reg(&mut b, &mut locals, r);
                b.const_i64(0);
                let zero = b.set_new_local_i64();
                let source = locals[ri].1.unsafe_clone();
                emit_arith(&mut b, &zero, &source, ArithOp::Sub, width);
                emit_write_reg(&mut b, &locals[ri].1, &zero, width);
                write_back_high_byte(&mut b, &mut locals, r, ri);
                b.free_local_i64(zero);
            },
            Instruction::NegMem { mem, width } => {
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
            Instruction::IncDecReg { r, width, decrement } => {
                emit::gen_get_flags(&mut b);
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
                emit::gen_get_flags(&mut b);
                b.const_i32(!1);
                b.and_i32();
                b.get_local(&carry);
                b.or_i32();
                b.store_aligned_i32(0);
                b.free_local(carry);
                b.free_local_i64(one);
            },
            Instruction::IncDecMem { mem, width, decrement } => {
                emit::gen_get_flags(&mut b);
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
                emit::gen_get_flags(&mut b);
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
            Instruction::CmovRegReg { code, dst, src, width } => {
                let si = load_reg(&mut b, &mut locals, src);
                let di = load_reg(&mut b, &mut locals, dst);
                let saved = locals.len();
                gen_condition(&mut b, code);
                b.if_void();
                b.get_local_i64(&locals[si].1);
                emit_mask(&mut b, width);
                let value = b.set_new_local_i64();
                emit_write_reg(&mut b, &locals[di].1, &value, width);
                b.free_local_i64(value);
                end_branch_arm(&mut b, &mut locals, saved);
                b.block_end();
            },
            Instruction::CmovRegMem { code, dst, mem, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let value = gen_memory_read(&mut b, &locals, &address, width);
                let di = load_reg(&mut b, &mut locals, dst);
                let saved = locals.len();
                gen_condition(&mut b, code);
                b.if_void();
                b.get_local_i64(&value);
                emit_mask(&mut b, width);
                let masked = b.set_new_local_i64();
                emit_write_reg(&mut b, &locals[di].1, &masked, width);
                b.free_local_i64(masked);
                end_branch_arm(&mut b, &mut locals, saved);
                b.block_end();
                b.free_local_i64(address);
                b.free_local_i64(value);
            },
            Instruction::SetccReg { code, dst, high8 } => {
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
            Instruction::SetccMem { code, mem } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                gen_condition(&mut b, code);
                b.extend_unsigned_i32_to_i64();
                let value = b.set_new_local_i64();
                gen_memory_write(&mut b, &locals, &address, &value, 8);
                b.free_local_i64(value);
                b.free_local_i64(address);
            },
            Instruction::ShiftReg { kind, r, width, count } => {
                if count == 0 {
                    continue;
                }
                if kind == ShiftKind::Rol || kind == ShiftKind::Ror {
                    // Rotates reset CF/OF but leave the other flags; the runtime
                    // helper handles that, so delegate instead of inlining.
                    let ri = load_reg(&mut b, &mut locals, r);
                    b.get_local_i64(&locals[ri].1);
                    b.const_i64(count as i64);
                    b.const_i32(width as i32 | (kind as i32) << 8);
                    b.call_fn3_i64_i64_i32_ret_i64("jit64_shift");
                    let result = b.set_new_local_i64();
                    emit_write_reg(&mut b, &locals[ri].1, &result, width);
                    b.free_local_i64(result);
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
                    _ => unreachable!(),
                }
                emit_mask(&mut b, width);
                let result = b.set_new_local_i64();

                gen_arith_flags(&mut b, &old, &old, &result, ArithOp::Test, width);

                // Carry is the last bit out; shifts >= width report none.
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
                emit::gen_get_flags(&mut b);
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
            Instruction::ShiftRegCl { kind, r, width } => {
                let ri = load_reg(&mut b, &mut locals, r);
                let ci = load_reg(&mut b, &mut locals, 1);
                b.get_local_i64(&locals[ci].1);
                b.const_i64(if width == 64 { 63 } else { 31 });
                b.and_i64();
                let saved = locals.len();
                b.eqz_i64();
                b.if_void();
                end_branch_arm(&mut b, &mut locals, saved);
                b.else_();
                b.get_local_i64(&locals[ri].1);
                b.get_local_i64(&locals[ci].1);
                b.const_i32(width as i32 | (kind as i32) << 8);
                b.call_fn3_i64_i64_i32_ret_i64("jit64_shift");
                let result = b.set_new_local_i64();
                emit_write_reg(&mut b, &locals[ri].1, &result, width);
                b.free_local_i64(result);
                end_branch_arm(&mut b, &mut locals, saved);
                b.block_end();
            },
            Instruction::ShiftMem { kind, mem, width, count } => {
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
            Instruction::ShiftMemCl { kind, mem, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let value = gen_memory_read(&mut b, &locals, &address, width);
                let ci = load_reg(&mut b, &mut locals, 1);
                b.get_local_i64(&locals[ci].1);
                b.const_i64(if width == 64 { 63 } else { 31 });
                b.and_i64();
                let saved = locals.len();
                b.eqz_i64();
                b.if_void();
                end_branch_arm(&mut b, &mut locals, saved);
                b.else_();
                gen_memory_probe_write(&mut b, &locals, &address, width);
                b.get_local_i64(&value);
                b.get_local_i64(&locals[ci].1);
                b.const_i32(width as i32 | (kind as i32) << 8);
                b.call_fn3_i64_i64_i32_ret_i64("jit64_shift");
                let result = b.set_new_local_i64();
                gen_memory_write(&mut b, &locals, &address, &result, width);
                b.free_local_i64(result);
                end_branch_arm(&mut b, &mut locals, saved);
                b.block_end();
                b.free_local_i64(address);
                b.free_local_i64(value);
            },
            Instruction::Cpuid => {
                // CPUID runs in the interpreter against the register file, so
                // flush the resident registers and reload them afterwards.
                emit_registers_back(&mut b, &locals);
                for (_, local) in &locals {
                    b.free_local_i64(local.unsafe_clone());
                }
                locals.clear();
                b.call_fn0("jit64_cpuid");
            },
            Instruction::Avx { rip } => {
                // VEX/AVX runs in the interpreter at `rip` (the shared
                // implementation), so flush the resident registers first and
                // reload them afterwards.
                emit_registers_back(&mut b, &locals);
                for (_, local) in &locals {
                    b.free_local_i64(local.unsafe_clone());
                }
                locals.clear();
                b.const_i64(rip as i64);
                b.const_i32(0);
                b.call_fn2_i64_i32("jit64_avx");
                b.call_fn0_ret("jit64_exception_delivered");
                b.if_void();
                emit_registers_back_except_rsp(&mut b, &locals);
                b.return_();
                b.block_end();
            },
            Instruction::BitTestReg { r, index_r, op, width } => {
                let ri = load_reg(&mut b, &mut locals, r);
                b.get_local_i64(&locals[ri].1);
                let ii = load_reg(&mut b, &mut locals, index_r);
                b.get_local_i64(&locals[ii].1);
                b.const_i32(op as i32 | (width as i32) << 8);
                b.call_fn3_i64_i64_i32_ret_i64("jit64_bt");
                let value = b.set_new_local_i64();
                write_reg_value(&mut b, &mut locals, r, &value, width);
                b.free_local_i64(value);
            },
            Instruction::BitTestImm { r, index, op, width } => {
                let ri = load_reg(&mut b, &mut locals, r);
                b.get_local_i64(&locals[ri].1);
                b.const_i64(index as i64);
                b.const_i32(op as i32 | (width as i32) << 8);
                b.call_fn3_i64_i64_i32_ret_i64("jit64_bt");
                let value = b.set_new_local_i64();
                write_reg_value(&mut b, &mut locals, r, &value, width);
                b.free_local_i64(value);
            },
            Instruction::In { imm, width } => {
                // I/O-dependent, so not reproducible in the self-check.
                unsafe { BLOCK_NO_SELFCHECK = true; }
                // The helper can #GP (unprivileged port I/O); flush first.
                emit_registers_back(&mut b, &locals);
                match imm {
                    Some(port) => b.const_i64(port as i64),
                    None => {
                        let ri = load_reg(&mut b, &mut locals, 2);
                        b.get_local_i64(&locals[ri].1);
                        b.const_i64(0xFFFF);
                        b.and_i64();
                    },
                }
                b.const_i32(width as i32);
                b.call_fn2_i64_i32_ret_i64("jit64_in");
                let value = b.set_new_local_i64();
                write_reg_value(&mut b, &mut locals, 0, &value, width);
                b.free_local_i64(value);
                b.call_fn0_ret("jit64_exception_delivered");
                b.if_void();
                emit_registers_back_except_rsp(&mut b, &locals);
                b.return_();
                b.block_end();
            },
            Instruction::Out { imm, width } => {
                // I/O-dependent, so not reproducible in the self-check.
                unsafe { BLOCK_NO_SELFCHECK = true; }
                emit_registers_back(&mut b, &locals);
                match imm {
                    Some(port) => b.const_i64(port as i64),
                    None => {
                        let ri = load_reg(&mut b, &mut locals, 2);
                        b.get_local_i64(&locals[ri].1);
                        b.const_i64(0xFFFF);
                        b.and_i64();
                    },
                }
                let ai = load_reg(&mut b, &mut locals, 0);
                b.get_local_i64(&locals[ai].1);
                emit_mask(&mut b, width);
                b.const_i32(width as i32);
                b.call_fn3_i64_i64_i32("jit64_out");
                b.call_fn0_ret("jit64_exception_delivered");
                b.if_void();
                b.return_();
                b.block_end();
            },
            Instruction::Cli => {
                b.call_fn0("jit64_cli");
            },
            Instruction::PushFlags => {
                // push64 touches rsp, so flush the resident registers.
                emit_registers_back(&mut b, &locals);
                for (_, local) in &locals {
                    b.free_local_i64(local.unsafe_clone());
                }
                locals.clear();
                b.call_fn0("jit64_pushfq");
            },
            Instruction::XmmLoad { dst, mem } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let lo = gen_xmm_mem_half_read(&mut b, &locals, &address, 0);
                let hi = gen_xmm_mem_half_read(&mut b, &locals, &address, 8);
                gen_xmm_store_half(&mut b, dst, &lo, 0);
                gen_xmm_store_half(&mut b, dst, &hi, 8);
                b.free_local_i64(lo);
                b.free_local_i64(hi);
                b.free_local_i64(address);
            },
            Instruction::XmmStore { mem, src } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                gen_xmm_reg_load(&mut b, src, 0);
                let lo = b.set_new_local_i64();
                gen_xmm_reg_load(&mut b, src, 8);
                let hi = b.set_new_local_i64();
                gen_xmm_mem_half_write(&mut b, &locals, &address, 0, &lo);
                gen_xmm_mem_half_write(&mut b, &locals, &address, 8, &hi);
                b.free_local_i64(lo);
                b.free_local_i64(hi);
                b.free_local_i64(address);
            },
            Instruction::XmmCopy { dst, src } => {
                gen_xmm_reg_load(&mut b, src, 0);
                let lo = b.set_new_local_i64();
                gen_xmm_reg_load(&mut b, src, 8);
                let hi = b.set_new_local_i64();
                gen_xmm_store_half(&mut b, dst, &lo, 0);
                gen_xmm_store_half(&mut b, dst, &hi, 8);
                b.free_local_i64(lo);
                b.free_local_i64(hi);
            },
            Instruction::XmmXor { dst, src } => {
                for offset in [0u32, 8] {
                    b.const_i32(crate::cpu::global_pointers::get_reg_xmm_addr(dst) as i32);
                    gen_xmm_reg_load(&mut b, dst, offset);
                    gen_xmm_reg_load(&mut b, src, offset);
                    b.xor_i64();
                    b.store_aligned_i64(offset);
                }
            },
            Instruction::XmmXorMem { dst, mem } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let lo = gen_xmm_mem_half_read(&mut b, &locals, &address, 0);
                let hi = gen_xmm_mem_half_read(&mut b, &locals, &address, 8);
                for (offset, value) in [(0u32, &lo), (8u32, &hi)] {
                    b.const_i32(crate::cpu::global_pointers::get_reg_xmm_addr(dst) as i32);
                    b.get_local_i64(value);
                    gen_xmm_reg_load(&mut b, dst, offset);
                    b.xor_i64();
                    b.store_aligned_i64(offset);
                }
                b.free_local_i64(lo);
                b.free_local_i64(hi);
                b.free_local_i64(address);
            },
            Instruction::XmmInt { op, dst, src } => {
                gen_xmm_reg_load(&mut b, src, 0);
                let lo = b.set_new_local_i64();
                gen_xmm_reg_load(&mut b, src, 8);
                let hi = b.set_new_local_i64();
                b.const_i32(op as i32);
                b.get_local_i64(&lo);
                b.get_local_i64(&hi);
                b.const_i32(dst as i32);
                b.call_fn4_i32_i64_i64_i32_ret("jit64_sse_int");
                b.drop_();
                b.free_local_i64(lo);
                b.free_local_i64(hi);
            },
            Instruction::XmmIntMem { op, dst, mem } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let lo = gen_xmm_mem_half_read(&mut b, &locals, &address, 0);
                let hi = gen_xmm_mem_half_read(&mut b, &locals, &address, 8);
                b.const_i32(op as i32);
                b.get_local_i64(&lo);
                b.get_local_i64(&hi);
                b.const_i32(dst as i32);
                b.call_fn4_i32_i64_i64_i32_ret("jit64_sse_int");
                b.drop_();
                b.free_local_i64(lo);
                b.free_local_i64(hi);
                b.free_local_i64(address);
            },
            Instruction::XmmShiftImm { op, group, count, dst } => {
                gen_xmm_reg_load(&mut b, dst, 0);
                let lo = b.set_new_local_i64();
                gen_xmm_reg_load(&mut b, dst, 8);
                let hi = b.set_new_local_i64();
                b.const_i32(op as i32 | (group as i32) << 8 | (count as i32) << 16);
                b.get_local_i64(&lo);
                b.get_local_i64(&hi);
                b.const_i32(dst as i32);
                b.call_fn4_i32_i64_i64_i32_ret("jit64_sse_shift_imm");
                b.drop_();
                b.free_local_i64(lo);
                b.free_local_i64(hi);
            },
            Instruction::XmmShuf { control, dst, src } => {
                gen_xmm_reg_load(&mut b, src, 0);
                let lo = b.set_new_local_i64();
                gen_xmm_reg_load(&mut b, src, 8);
                let hi = b.set_new_local_i64();
                b.const_i32(control as i32);
                b.get_local_i64(&lo);
                b.get_local_i64(&hi);
                b.const_i32(dst as i32);
                b.call_fn4_i32_i64_i64_i32_ret("jit64_sse_pshuf");
                b.drop_();
                b.free_local_i64(lo);
                b.free_local_i64(hi);
            },
            Instruction::XmmShufMem { control, dst, mem } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let lo = gen_xmm_mem_half_read(&mut b, &locals, &address, 0);
                let hi = gen_xmm_mem_half_read(&mut b, &locals, &address, 8);
                b.const_i32(control as i32);
                b.get_local_i64(&lo);
                b.get_local_i64(&hi);
                b.const_i32(dst as i32);
                b.call_fn4_i32_i64_i64_i32_ret("jit64_sse_pshuf");
                b.drop_();
                b.free_local_i64(lo);
                b.free_local_i64(hi);
                b.free_local_i64(address);
            },
            Instruction::Rdtsc => {
                // Time-dependent, so not reproducible in the self-check.
                unsafe { BLOCK_NO_SELFCHECK = true; }
                b.call_fn0_ret_i64("jit64_rdtsc");
                let raw = b.set_new_local_i64();
                b.get_local_i64(&raw);
                b.const_i64(0xFFFF_FFFF);
                b.and_i64();
                let low = b.set_new_local_i64();
                write_reg_value(&mut b, &mut locals, 0, &low, 32);
                b.free_local_i64(low);
                b.get_local_i64(&raw);
                b.const_i64(32);
                b.shr_u_i64();
                let high = b.set_new_local_i64();
                write_reg_value(&mut b, &mut locals, 2, &high, 32);
                b.free_local_i64(high);
                b.free_local_i64(raw);
            },
            Instruction::ImulRegReg { dst, lhs, rhs, width } => {
                let li = load_reg(&mut b, &mut locals, lhs);
                let ri = load_reg(&mut b, &mut locals, rhs);
                let di = load_reg(&mut b, &mut locals, dst);
                let left = locals[li].1.unsafe_clone();
                let right = locals[ri].1.unsafe_clone();
                let destination = locals[di].1.unsafe_clone();
                emit_imul(&mut b, &destination, &left, &right, width);
            },
            Instruction::ImulRegImm { dst, src, value, width } => {
                let si = load_reg(&mut b, &mut locals, src);
                let di = load_reg(&mut b, &mut locals, dst);
                b.const_i64(value as i64);
                let immediate = b.set_new_local_i64();
                let source = locals[si].1.unsafe_clone();
                let destination = locals[di].1.unsafe_clone();
                emit_imul(&mut b, &destination, &source, &immediate, width);
                b.free_local_i64(immediate);
            },
            Instruction::ImulRegMem { dst, mem, value, width } => {
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
            Instruction::Bswap { r, width } => {
                let ri = load_reg(&mut b, &mut locals, r);
                b.get_local_i64(&locals[ri].1);
                b.const_i32(width as i32);
                b.call_fn2_i64_i32_ret_i64("jit64_bswap");
                b.set_local_i64(&locals[ri].1);
            },
            Instruction::Lea { dst, mem, width } => {
                gen_effective_addr(&mut b, &mut locals, &mem);
                emit_mask(&mut b, width);
                let value = b.set_new_local_i64();
                write_reg_value(&mut b, &mut locals, dst, &value, width);
                b.free_local_i64(value);
            },
            Instruction::AddRegReg { dst, src, width } => {
                let di = load_reg(&mut b, &mut locals, dst);
                let si = load_reg(&mut b, &mut locals, src);
                emit_arith(&mut b, &locals[di].1, &locals[si].1, ArithOp::Add, width);
                write_back_high_byte(&mut b, &mut locals, dst, di);
            },
            Instruction::AddRegImm { r, value, width } => {
                let di = load_reg(&mut b, &mut locals, r);
                b.const_i64(value as i64);
                let src = b.set_new_local_i64();
                emit_arith(&mut b, &locals[di].1, &src, ArithOp::Add, width);
                write_back_high_byte(&mut b, &mut locals, r, di);
                b.free_local_i64(src);
            },
            Instruction::ArithRegReg { op, dst, src, width } => {
                let di = load_reg(&mut b, &mut locals, dst);
                let si = load_reg(&mut b, &mut locals, src);
                emit_arith(&mut b, &locals[di].1, &locals[si].1, op, width);
                if op.writes_result() {
                    write_back_high_byte(&mut b, &mut locals, dst, di);
                }
            },
            Instruction::ArithRegImm { op, r, value, width } => {
                let di = load_reg(&mut b, &mut locals, r);
                b.const_i64(value as i64);
                let src = b.set_new_local_i64();
                emit_arith(&mut b, &locals[di].1, &src, op, width);
                if op.writes_result() {
                    write_back_high_byte(&mut b, &mut locals, r, di);
                }
                b.free_local_i64(src);
            },
            Instruction::ArithRegMem { op, dst, mem, width } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let src = gen_memory_read(&mut b, &locals, &address, width);
                let di = load_reg(&mut b, &mut locals, dst);
                emit_arith(&mut b, &locals[di].1, &src, op, width);
                if op.writes_result() {
                    write_back_high_byte(&mut b, &mut locals, dst, di);
                }
                b.free_local_i64(address);
                b.free_local_i64(src);
            },
            Instruction::ArithMemReg { op, mem, src, width } => {
                let si = load_reg(&mut b, &mut locals, src);
                let source = locals[si].1.unsafe_clone();
                emit_arith_mem(&mut b, &mut locals, &mem, &source, op, width);
            },
            Instruction::ArithMemImm { op, mem, value, width } => {
                b.const_i64(value as i64);
                let src = b.set_new_local_i64();
                emit_arith_mem(&mut b, &mut locals, &mem, &src, op, width);
                b.free_local_i64(src);
            },
            Instruction::PushReg { r } => {
                let ri = load_reg(&mut b, &mut locals, r);
                b.get_local_i64(&locals[ri].1);
                let value = b.set_new_local_i64();
                emit_push_value(&mut b, &mut locals, &value);
                b.free_local_i64(value);
            },
            Instruction::PopReg { r } => {
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
            Instruction::PushImm { value } => {
                b.const_i64(value as i64);
                let local = b.set_new_local_i64();
                emit_push_value(&mut b, &mut locals, &local);
                b.free_local_i64(local);
            },
            Instruction::Call { target, return_address } => {
                b.const_i64(return_address as i64);
                let local = b.set_new_local_i64();
                emit_push_value(&mut b, &mut locals, &local);
                b.free_local_i64(local);
                gen_set_ip(&mut b, target);
                emit_registers_back(&mut b, &locals);
                emit_exit_stat(&mut b, 4);
                emit_chain_exit(&mut b);
                b.return_();
                terminated = true;
            },
            Instruction::CallReg { r, return_address } => {
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
                emit_exit_stat(&mut b, 4);
                emit_chain_exit(&mut b);
                b.return_();
                terminated = true;
            },
            Instruction::CallMem { mem, return_address } => {
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
                emit_exit_stat(&mut b, 4);
                emit_chain_exit(&mut b);
                b.return_();
                terminated = true;
            },
            Instruction::JmpReg { r } => {
                let ri = load_reg(&mut b, &mut locals, r);
                b.const_i32(IP_ADDR);
                b.get_local_i64(&locals[ri].1);
                b.store_aligned_i64(0);
                emit_registers_back(&mut b, &locals);
                emit_exit_stat(&mut b, 1);
                emit_chain_exit(&mut b);
                b.return_();
                terminated = true;
            },
            Instruction::JmpMem { mem } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let target = gen_memory_read(&mut b, &locals, &address, 64);
                b.free_local_i64(address);
                b.const_i32(IP_ADDR);
                b.get_local_i64(&target);
                b.store_aligned_i64(0);
                b.free_local_i64(target);
                emit_registers_back(&mut b, &locals);
                emit_exit_stat(&mut b, 1);
                emit_chain_exit(&mut b);
                b.return_();
                terminated = true;
            },
            Instruction::Ret { adjustment } => {
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
                emit_exit_stat(&mut b, 5);
                emit_chain_exit(&mut b);
                b.return_();
                terminated = true;
            },
            Instruction::Leave => {
                let rbp = load_reg(&mut b, &mut locals, 5);
                let rsp = load_reg(&mut b, &mut locals, 4);
                b.get_local_i64(&locals[rbp].1);
                b.set_local_i64(&locals[rsp].1);
                let value = emit_pop_value(&mut b, &mut locals);
                b.get_local_i64(&value);
                b.set_local_i64(&locals[rbp].1);
                b.free_local_i64(value);
            },
            Instruction::Nop => {},
            Instruction::Jmp { target } => {
                gen_set_ip(&mut b, target);
                emit_registers_back(&mut b, &locals);
                emit_exit_stat(&mut b, 1);
                emit_chain_exit(&mut b);
                b.return_();
                terminated = true;
            },
            Instruction::Jcc {
                code,
                target,
                fallthrough,
            } => {
                // Non-final branches exit only when taken; else fall through.
                let final_branch = index + 1 == instrs.len();
                gen_condition(&mut b, code);
                b.if_void();
                gen_set_ip(&mut b, target);
                emit_registers_back(&mut b, &locals);
                emit_exit_stat(&mut b, 2);
                emit_chain_exit(&mut b);
                b.return_();
                b.block_end();
                if final_branch {
                    gen_set_ip(&mut b, fallthrough);
                    emit_registers_back(&mut b, &locals);
                    emit_exit_stat(&mut b, 3);
                    emit_chain_exit(&mut b);
                    b.return_();
                    terminated = true;
                }
            },
            Instruction::Hlt => {
                gen_set_ip(&mut b, block_end);
                // Publish registers before entering the halted state.
                emit_registers_back(&mut b, &locals);
                emit_exit_stat(&mut b, 6);
                b.call_fn0("jit64_hlt");
                b.return_();
                terminated = true;
            },
        }
    }

    if !terminated {
        gen_set_ip(&mut b, block_end);
        emit_registers_back(&mut b, &locals);
        emit_exit_stat(&mut b, 0);
        emit_chain_exit(&mut b);
        b.return_();
    }

    for (_, local) in &locals {
        b.free_local_i64(local.unsafe_clone());
    }

    if instrs.len() >= 15 {
        unsafe { LAST_BIG_BLOCK_LOCALS = b.local_count() as u32; }
    }
    b.finish();
    let ptr = b.get_output_ptr();
    let len = b.get_output_len() as usize;
    unsafe { std::slice::from_raw_parts(ptr, len).to_vec() }
}

pub fn compile_block(instrs: &[Instruction], block_end: u64) -> Vec<u8> {
    compile_block_with_rips(instrs, &[], block_end)
}
// Stable instruction-class bits for whole-block interpreter fallback.
fn bail_class(instr: &Instruction) -> u64 {
    match instr {
        Instruction::MovRegImm { .. } | Instruction::MovRegReg { .. } | Instruction::MovRegMem { .. }
        | Instruction::MovMemReg { .. } | Instruction::MovMemImm { .. } | Instruction::MovExtendReg { .. }
        | Instruction::MovExtendMem { .. } => 1 << 0,
        Instruction::XchgRegReg { .. } | Instruction::XchgMemReg { .. } => 1 << 1,
        Instruction::NotReg { .. } | Instruction::NotMem { .. } | Instruction::NegReg { .. } | Instruction::NegMem { .. } => 1 << 2,
        Instruction::IncDecReg { .. } | Instruction::IncDecMem { .. } => 1 << 3,
        Instruction::CmovRegReg { .. } | Instruction::CmovRegMem { .. } => 1 << 4,
        Instruction::SetccReg { .. } | Instruction::SetccMem { .. } => 1 << 5,
        Instruction::ShiftReg { .. } | Instruction::ShiftRegCl { .. } | Instruction::ShiftMem { .. }
        | Instruction::ShiftMemCl { .. } => 1 << 6,
        Instruction::Cpuid => 1 << 7,
        Instruction::Rdtsc => 1 << 8,
        Instruction::XmmCopy { .. } | Instruction::XmmLoad { .. } | Instruction::XmmStore { .. }
        | Instruction::XmmXor { .. } | Instruction::XmmXorMem { .. } | Instruction::XmmInt { .. }
        | Instruction::XmmIntMem { .. } | Instruction::XmmShiftImm { .. } | Instruction::XmmShuf { .. }
        | Instruction::XmmShufMem { .. } => 1 << 9,
        Instruction::In { .. } | Instruction::Out { .. } => 1 << 10,
        Instruction::Cli => 1 << 11,
        Instruction::PushFlags => 1 << 12,
        Instruction::CmpxchgReg { .. } | Instruction::CmpxchgMem { .. } => 1 << 13,
        Instruction::BitTestReg { .. } | Instruction::BitTestImm { .. } => 1 << 14,
        Instruction::ImulRegReg { .. } | Instruction::ImulRegImm { .. } | Instruction::ImulRegMem { .. } => 1 << 15,
        Instruction::Bswap { .. } => 1 << 16,
        Instruction::Lea { .. } => 1 << 17,
        Instruction::AddRegReg { .. } | Instruction::AddRegImm { .. } => 1 << 18,
        Instruction::ArithRegReg { .. } | Instruction::ArithRegImm { .. } | Instruction::ArithRegMem { .. }
        | Instruction::ArithMemReg { .. } | Instruction::ArithMemImm { .. } => 1 << 19,
        Instruction::PushReg { .. } | Instruction::PopReg { .. } | Instruction::PushImm { .. } => 1 << 20,
        Instruction::Ret { .. } => 1 << 23,
        Instruction::Jcc { .. } => 1 << 24,
        Instruction::Jmp { .. } | Instruction::JmpReg { .. } | Instruction::JmpMem { .. } => 1 << 25,
        Instruction::Call { .. } | Instruction::CallReg { .. } | Instruction::CallMem { .. } => 1 << 26,
        Instruction::Leave => 1 << 27,
        Instruction::Nop => 1 << 22,
        Instruction::Avx { .. } => 1 << 21,
        Instruction::Hlt => 1 << 28,
    }
}

#[no_mangle]
pub unsafe fn jit64_set_bail(mask: u64) { JIT64_BAIL = mask; }

#[no_mangle]
pub unsafe fn jit64_set_bail_rip(lo: u64, hi: u64) {
    JIT64_BAIL_RIP_LO = lo;
    JIT64_BAIL_RIP_HI = hi;
}

#[no_mangle]
pub unsafe fn jit64_set_bail_link(lo: u64, hi: u64) {
    JIT64_BAIL_LINK_LO = lo;
    JIT64_BAIL_LINK_HI = hi;
}

pub fn compile_bytes(base: u64, bytes: &[u8]) -> Result<Vec<u8>, String> {
    let decoded = decode_block_with_rips(base, bytes)?;
    if decoded.instrs.len() >= 15 {
        unsafe {
            if base < BIG_BLOCK_LO { BIG_BLOCK_LO = base; }
            if decoded.end_rip > BIG_BLOCK_HI { BIG_BLOCK_HI = decoded.end_rip; }
        }
    }
    if decoded.instrs.len() >= 15 && unsafe { BIG_BLOCK_N } < BIG_BLOCK_MAX as u32 {
        unsafe {
            let slot = BIG_BLOCK_N as usize;
            BIG_BLOCK_RIP[slot] = base;
            BIG_BLOCK_LEN[slot] = decoded.instrs.len() as u32;
            BIG_BLOCK_N += 1;
            if BIG_BLOCK_N == 1 {
                let n = bytes.len().min(BIG_BLOCK_BYTES_SIZE);
                BIG_BLOCK_BYTES[..n].copy_from_slice(&bytes[..n]);
                BIG_BLOCK_BYTES_LEN = n as u32;
            }
        }
    }
    let bail = unsafe { JIT64_BAIL };
    if bail != 0 && decoded.instrs.iter().any(|i| bail_class(i) & bail != 0) {
        return Err("bailed".into());
    }
    let bail_lo = unsafe { JIT64_BAIL_RIP_LO };
    if bail_lo != 0 && base >= bail_lo && base < unsafe { JIT64_BAIL_RIP_HI } {
        return Err("bailed rip".into());
    }
    let link_lo = unsafe { JIT64_BAIL_LINK_LO };
    if link_lo != 0 {
        let width = unsafe { JIT64_BAIL_LINK_HI }.wrapping_sub(link_lo);
        if base.wrapping_sub(link_lo) & 0x1F_FFFF < width {
            return Err("bailed link".into());
        }
    }
    if decoded.instrs.is_empty() {
        // Otherwise try_run returns true forever without advancing the
        // instruction counter, starving timers and interrupts.
        return Err("empty block".into());
    }
    unsafe { LAST_COMPILED_END_RIP = decoded.end_rip; }
    unsafe { LAST_COMPILED_LEN = decoded.instrs.len() as u16; }
    let module = compile_block_with_rips(
        &decoded.instrs,
        &decoded.rips,
        decoded.end_rip,
    );
    unsafe { LAST_COMPILED_NO_SELFCHECK = BLOCK_NO_SELFCHECK; }
    Ok(module)
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

use crate::hash::{FastMap, FastSet};

const JIT64_THRESHOLD: u32 = 500; // interpreted runs before a block is compiled
const JIT64_MAX_BLOCK_INSTRS: usize = 32; // hard cap per compiled block
// Fall-through superblocks: decode past a conditional branch up to this many
// instructions (0 in jit64_set_block_limit means JIT64_MAX_BLOCK_INSTRS).
static mut JIT64_SUPERBLOCKS: bool = true;
static mut JIT64_BLOCK_LIMIT: usize = 16;

// Long-mode block cache cap (configurable so tests can force eviction).
static mut JIT64_MAX_BLOCKS: usize = 28000;
// Diagnostic self-check: re-run each block in the interpreter and compare.
static mut JIT64_SELFCHECK: bool = false;
static mut LAST_COMPILED_END_RIP: u64 = 0;
static mut LAST_COMPILED_LEN: u16 = 0;
// Set for blocks that must not be self-checked (MMIO/RDTSC/port I/O).
static mut BLOCK_NO_SELFCHECK: bool = false;
static mut LAST_COMPILED_NO_SELFCHECK: bool = false;
// Only self-check blocks at least this long (the J7 failure needs >= 15).
static mut SELFCHECK_MIN_INSTRS: u16 = 15;
// Check the first N executions of a block (diagnostics).
static mut SELFCHECK_REPEAT: u16 = 1;

fn max_blocks() -> usize { unsafe { JIT64_MAX_BLOCKS } }

const COMPILE_BUF_SIZE: usize = 65536;
static mut COMPILE_BUF: [u8; COMPILE_BUF_SIZE] = [0; COMPILE_BUF_SIZE];
// Scratch for a block's code bytes (avoids a per-block allocation).
const CODE_BUF_SIZE: usize = 0x1000;
static mut CODE_BUF: [u8; CODE_BUF_SIZE] = [0; CODE_BUF_SIZE];
static mut JIT64_MEMORY_FAULT: u8 = 0;
// Inline RAM loads/stores in generated code (see gen_memory_read).
static mut JIT64_INLINE_MEMORY: bool = true;
// Separate switch for inline stores (self-modifying code makes them riskier).
static mut JIT64_INLINE_MEMORY_WRITE: bool = true;
// compile the decodable prefix, resume at the undecodable instruction.
static mut JIT64_PARTIAL_BLOCKS: bool = true;
// read a block's page with one translation (A/B).
static mut JIT64_DIRECT_CODE_READ: bool = true;
// Count interpreted hotness in a direct-mapped cache; off uses the map.
static mut JIT64_HOT_CACHE: bool = true;

// SSE2 codegen (movdqa/movdqu/pxor), validated by the differential suite.
static mut JIT64_SSE: bool = true;
// Dispatch counters (diagnostics; see jit64_stat).
static mut JIT64_RUN_HITS: u64 = 0;
static mut JIT64_RUN_MISSES: u64 = 0;
static mut JIT64_INTERP: u64 = 0;
static mut JIT64_COMPILES: u64 = 0;
// blocks entered directly from a predecessor block (not the main loop).
static mut JIT64_CHAIN_HOPS: u64 = 0;
// Memory slow-path (helper) calls and how many were cross-page (profiling).
static mut JIT64_MEM_SLOW_READ: u64 = 0;
static mut JIT64_MEM_SLOW_WRITE: u64 = 0;
static mut JIT64_MEM_SLOW_CROSS: u64 = 0;
// JIT block exit reasons: fallthrough, jmp, jcc taken/not, call, ret, hlt.
const JIT64_EXIT_REASON_COUNT: usize = 8;
static mut JIT64_EXIT_REASON: [u64; JIT64_EXIT_REASON_COUNT] = [0; JIT64_EXIT_REASON_COUNT];
// Compile-time gate: blocks compiled while this is set count their exits.
static mut JIT64_EXIT_STATS: bool = false;

// Bisect switch: classes of instructions whose blocks are left to the
// interpreter (see jit64_set_bail).
static mut JIT64_BAIL: u64 = 0;
// Leave blocks in the RIP range [lo, hi) to the interpreter.
static mut JIT64_BAIL_RIP_LO: u64 = 0;
static mut JIT64_BAIL_RIP_HI: u64 = 0;
// Match a link address modulo 2 MiB (KASLR slides the kernel
// by a 2 MiB-aligned amount, so this matches modulo 2 MiB).
static mut JIT64_BAIL_LINK_LO: u64 = 0;
static mut JIT64_BAIL_LINK_HI: u64 = 0;
// Synchronous delivery marks the active block so it exits at the next helper check.
static mut IN_JIT_BLOCK: bool = false;

// Set when a block delivered an exception (e.g. #GP): it must stop, since *rip
// changed and a frame was pushed.
static mut EXCEPTION_IN_BLOCK: bool = false;

// Set when a guest write invalidates a code page (SMC); blocks check it after
// each store and stop before running stale instructions.
static mut JIT64_CODE_WRITE_BAIL: bool = false;
// A/B switch for the SMC bail (diagnostics).
static mut JIT64_SMC_BAIL: bool = true;

pub unsafe fn note_exception_in_block() {
    EXCEPTION_IN_BLOCK = true;
}

#[no_mangle]
pub unsafe fn jit64_hlt() {
    *crate::cpu::global_pointers::in_hlt = true;
    // Like the interpreter's hlt: the main loop delivers the IRQ; otherwise tell
    // the host that execution cannot resume.
    if *crate::cpu::global_pointers::flags & crate::cpu::core::FLAG_INTERRUPT == 0 {
        crate::cpu::core::cpu_event_halt();
    }
}

#[no_mangle]
pub unsafe fn jit64_clear_exception_flag() { EXCEPTION_IN_BLOCK = false; }

#[no_mangle]
pub unsafe fn jit64_exception_delivered() -> u32 {
    let v = EXCEPTION_IN_BLOCK;
    EXCEPTION_IN_BLOCK = false;
    v as u32
}

// A code page was written: stop the block before it runs stale instructions.
#[no_mangle]
pub unsafe fn jit64_code_write_bail() -> u32 {
    let v = JIT64_CODE_WRITE_BAIL;
    JIT64_CODE_WRITE_BAIL = false;
    v as u32
}

#[no_mangle]
pub unsafe fn jit64_in_block() -> u32 { IN_JIT_BLOCK as u32 }

static mut JIT64_COMPILE_FAILS: u64 = 0;
// Diagnostic: first byte of blocks that failed to decode, by opcode.
static mut JIT64_FAIL_OP: [u32; 256] = [0; 256];
// Diagnostic: why a hot block did not register.
static mut JIT64_COMPILE_NO_TRANSLATE: u64 = 0;
static mut JIT64_COMPILE_TOO_BIG: u64 = 0;
static mut JIT64_COMPILE_NO_INDEX: u64 = 0;
static mut JIT64_REGISTERED: u64 = 0;
static mut JIT64_REPLACED: u64 = 0;
// Diagnostic: why blocks were dropped.
static mut JIT64_FORGET_EVICT: u64 = 0;
static mut JIT64_FORGET_STALE: u64 = 0;
static mut JIT64_FORGET_SMC: u64 = 0;
static mut JIT64_FORGET_INVLPG: u64 = 0;
static mut JIT64_FAULTS: u64 = 0;
static mut HOTNESS: *mut FastMap<u64, u32> = std::ptr::null_mut();
// Direct-mapped hotness cache (avoids a map op per interpretation); colliding
// RIPs fall back to the map.
const HOT_CACHE_SIZE: usize = 1 << 13;
static mut HOT_CACHE_RIP: [u64; HOT_CACHE_SIZE] = [u64::MAX; HOT_CACHE_SIZE];
static mut HOT_CACHE_COUNT: [u32; HOT_CACHE_SIZE] = [0; HOT_CACHE_SIZE];

#[inline]
fn hot_slot(rip: u64) -> usize { (rip >> 2) as usize & (HOT_CACHE_SIZE - 1) }

unsafe fn hot_cache_reset(rip: u64) {
    let slot = hot_slot(rip);
    if HOT_CACHE_RIP[slot] == rip {
        HOT_CACHE_RIP[slot] = u64::MAX;
        HOT_CACHE_COUNT[slot] = 0;
    }
}
// Block-cap trim request; run from try_run, outside hotness borrows.
static mut EVICT_PENDING: bool = false;
// Remaining chain dispatches for this loop entry; zero disables chaining.
static mut CHAIN_BUDGET: u32 = 0;
static mut JIT64_CHAIN_BUDGET: u32 = 0;
// Whether blocks emit the chain hand-off at exits (off costs nothing).
static mut JIT64_CHAIN_CODEGEN: bool = false;
// Whether the prologue (flags + exception flag) runs in wasm; default host.
static mut JIT64_BLOCK_PROLOGUE: bool = false;
// Compilation order for the clock hand; stale rips are skipped lazily.
static mut ORDER: *mut Vec<u64> = std::ptr::null_mut();
static mut HAND: usize = 0;
static mut LAST_VALID_PAGE: u64 = u64::MAX;
static mut LAST_VALID_PHYS: u32 = 0;
#[derive(Copy, Clone)]
struct BlockInfo {
    index: u16,
    // Physical page it was compiled from.
    phys_page: u32,
    // Decoded end RIP and instruction count (self-check only).
    end_rip: u64,
    instr_count: u16,
    // Self-check budget left (diagnostics only).
    check_left: u16,
    no_selfcheck: bool,
    // Second-chance bit for the clock eviction below.
    used: bool,
}

static mut BLOCKS: *mut FastMap<u64, Box<BlockInfo>> = std::ptr::null_mut();

// Direct-mapped block entry cache; forget()/clear_cache() invalidate slots, the
// `rip` tag catches reuse.
const ENTRY_CACHE_SIZE: usize = 1 << 12;
#[derive(Copy, Clone)]
struct EntrySlot {
    rip: u64,
    info: *mut BlockInfo,
}
static mut ENTRY_CACHE: [EntrySlot; ENTRY_CACHE_SIZE] = [EntrySlot {
    rip: 0,
    info: std::ptr::null_mut(),
}; ENTRY_CACHE_SIZE];
static mut JIT64_ENTRY_CACHE: bool = true;

#[inline]
fn entry_slot(rip: u64) -> usize { (rip >> 2) as usize & (ENTRY_CACHE_SIZE - 1) }

unsafe fn entry_cache_clear_rip(rip: u64) {
    let slot = entry_slot(rip);
    if !ENTRY_CACHE[slot].info.is_null() && ENTRY_CACHE[slot].rip == rip {
        ENTRY_CACHE[slot].info = std::ptr::null_mut();
    }
}
// physical code page -> guest RIPs compiled from it
static mut CODE_PAGES: *mut FastMap<u32, FastSet<u64>> = std::ptr::null_mut();
// virtual code page -> guest RIPs compiled from it (INVLPG invalidation)
static mut VIRT_PAGES: *mut FastMap<u64, FastSet<u64>> = std::ptr::null_mut();

unsafe fn user_access() -> bool { *crate::cpu::global_pointers::cpl == 3 }

unsafe fn hotness() -> &'static mut FastMap<u64, u32> {
    if HOTNESS.is_null() {
        HOTNESS = Box::into_raw(Box::new(FastMap::default()));
    }
    &mut *HOTNESS
}

unsafe fn blocks() -> &'static mut FastMap<u64, Box<BlockInfo>> {
    if BLOCKS.is_null() {
        BLOCKS = Box::into_raw(Box::new(FastMap::default()));
    }
    &mut *BLOCKS
}

unsafe fn code_pages() -> &'static mut FastMap<u32, FastSet<u64>> {
    if CODE_PAGES.is_null() {
        CODE_PAGES = Box::into_raw(Box::new(FastMap::default()));
    }
    &mut *CODE_PAGES
}

unsafe fn virt_pages() -> &'static mut FastMap<u64, FastSet<u64>> {
    if VIRT_PAGES.is_null() {
        VIRT_PAGES = Box::into_raw(Box::new(FastMap::default()));
    }
    &mut *VIRT_PAGES
}

unsafe fn order() -> &'static mut Vec<u64> {
    if ORDER.is_null() {
        ORDER = Box::into_raw(Box::new(Vec::new()));
    }
    &mut *ORDER
}

// Evict the least recently used blocks (clock/second chance)
unsafe fn evict_some(count: usize) {
    let mut freed = 0;
    // Bounded: one full sweep may only clear `used` bits.
    let mut scanned = 0;
    let limit = order().len() * 2 + 16;
    while freed < count && scanned < limit && !order().is_empty() {
        if HAND >= order().len() {
            HAND = 0;
        }
        let rip = order()[HAND];
        HAND += 1;
        scanned += 1;
        match blocks().get_mut(&rip) {
            None => continue,
            Some(info) if info.used =>
            {
                info.used = false;
                continue;
            },
            Some(_) =>
            {
                JIT64_FORGET_EVICT += 1;
                forget(rip);
            },
        }
        freed += 1;
    }
}

// Drop one block; it can be compiled again once hot.
unsafe fn forget(rip: u64) {
    if let Some(info) = blocks().remove(&rip) {
        crate::cpu::jit::jit64_free_table_index(info.index);
    }
    entry_cache_clear_rip(rip);
    hot_cache_reset(rip);
    hotness().remove(&rip);
    if !VIRT_PAGES.is_null() {
        if let Some(rips) = virt_pages().get_mut(&(rip >> 12)) {
            rips.remove(&rip);
        }
    }
}

// Does nothing if the bytes cannot be decoded yet.
unsafe fn compile_and_register(rip: u64) {
    // At the cap: request a trim at the next dispatch (see evict_some).
    if blocks().len() >= max_blocks() {
        EVICT_PENDING = true;
    }
    JIT64_COMPILES += 1;
    // Hotness follows interpretation (which may have raised or switched CR3);
    // compiling is only a probe.
    let phys = match crate::cpu::core::translate_address_64_no_side_effects(rip) {
        Ok(phys) => phys,
        Err(()) =>
        {
            JIT64_COMPILE_NO_TRANSLATE += 1;
            return;
        },
    };
    let phys_page = phys >> 12;
    let code = std::slice::from_raw_parts_mut(
        std::ptr::addr_of_mut!(CODE_BUF) as *mut u8,
        CODE_BUF_SIZE,
    );
    let code_len = if JIT64_DIRECT_CODE_READ {
        // A mapped page is contiguous, so translate once and read on.
        let page_off = (rip & 0xFFF) as u32;
        let n = (0x1000 - page_off) as usize;
        for i in 0..n {
            code[i] = crate::memory::read8(phys.wrapping_add(i as u32)) as u8;
        }
        n
    }
    else {
        let page_end = (rip | 0xFFF) + 1;
        let mut addr = rip;
        let mut n = 0;
        while addr < page_end {
            match crate::cpu::core::translate_address_64_no_side_effects(addr) {
                Ok(p) => {
                    code[n] = crate::memory::read8(p) as u8;
                    n += 1;
                },
                Err(()) => break,
            }
            addr += 1;
        }
        n
    };
    let bytes = &code[..code_len];

    let module = match compile_bytes(rip, bytes) {
        Ok(module) => module,
        Err(e) =>
        {
            JIT64_COMPILE_FAILS += 1;
            if let Some(&first) = bytes.first() {
                JIT64_FAIL_OP[first as usize] += 1;
            }
            if LOGGED_FAILS.fetch_add(1, Ordering::Relaxed) < 40
            {
                dbg_log!("jit64: cannot compile 0x{:x}: {}", rip, e);
            }
            return;
        },
    };
    if module.len() > COMPILE_BUF_SIZE {
        JIT64_COMPILE_TOO_BIG += 1;
        return;
    }
    let index = match crate::cpu::jit::jit64_allocate_table_index() {
        Some(index) => index,
        None => {
            // No slot free: re-arm hotness so the block is retried later.
            JIT64_COMPILE_NO_INDEX += 1;
            hotness().remove(&rip);
            return;
        },
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
    JIT64_REGISTERED += 1;
    if let Some(old) = blocks().insert(
        rip,
        Box::new(BlockInfo {
            index,
            phys_page,
            end_rip: LAST_COMPILED_END_RIP,
            instr_count: LAST_COMPILED_LEN,
            check_left: SELFCHECK_REPEAT,
            no_selfcheck: LAST_COMPILED_NO_SELFCHECK,
            used: false,
        }),
    ) {
        // Replaced block: free its table slot so it is not leaked.
        JIT64_REPLACED += 1;
        crate::cpu::jit::jit64_free_table_index(old.index);
    }
    order().push(rip);
    // Compact the append-only order once stale entries dominate.
    if order().len() > blocks().len() * 4 {
        order().clear();
        order().extend(blocks().keys().copied());
        HAND = 0;
    }
    virt_pages().entry(rip >> 12).or_default().insert(rip);
    code_pages().entry(phys_page).or_default().insert(rip);
    // Flag as code so a guest write invalidates the block (SMC), and so inline
    // writes on this physical page fall back to the helper.
    crate::cpu::core::tlb_set_has_code(crate::memory::page::Page::page_of(phys_page << 12), true);
    crate::cpu::core::tlb64_refresh_code_flag(phys_page);
    if (BLOCK_LOG_N as usize) < BLOCK_LOG_MAX {
        BLOCK_LOG_RIP[BLOCK_LOG_N as usize] = rip;
        BLOCK_LOG_N += 1;
    }
}

// Physical page backing this guest page (one-entry cache).
unsafe fn phys_page_of(rip: u64) -> Option<u32> {
    let virt_page = rip >> 12;
    if virt_page == LAST_VALID_PAGE {
        return Some(LAST_VALID_PHYS);
    }
    // Between instructions: a faulting walk here would corrupt guest state.
    match crate::cpu::core::translate_address_64_no_side_effects(rip) {
        Ok(phys) =>
        {
            LAST_VALID_PAGE = virt_page;
            LAST_VALID_PHYS = phys >> 12;
            Some(LAST_VALID_PHYS)
        },
        Err(()) => None,
    }
}

/// A TLB flush: stop trusting the cached translation.
pub unsafe fn note_mapping_changed() {
    LAST_VALID_PAGE = u64::MAX;
}

pub unsafe fn try_run(rip: u64) -> bool {
    if !jit64_enabled() {
        return false;
    }
    // Refill the block-chaining budget once per main-loop entry.
    refill_chain_budget();
    if EVICT_PENDING {
        EVICT_PENDING = false;
        evict_some(max_blocks() / 4);
    }
    match lookup_compiled(rip) {
        Some((index, end_rip, check)) =>
        {
            JIT64_RUN_HITS += 1;
            log_exec(rip);
            dispatch_block(index, rip, end_rip, check);
            true
        },
        None =>
        {
            JIT64_RUN_MISSES += 1;
            false
        },
    }
}

// Resolve `rip` to a live block, validating the entry cache and physical page.
unsafe fn lookup_compiled(rip: u64) -> Option<(u16, u64, bool)> {
    // Fast path: an entry-cache hit skips the map lookup.
    if JIT64_ENTRY_CACHE {
        let entry = ENTRY_CACHE[entry_slot(rip)];
        if !entry.info.is_null() && entry.rip == rip {
            let info = &mut *entry.info;
            if phys_page_of(rip) == Some(info.phys_page) {
                info.used = true;
                let check = JIT64_SELFCHECK
                    && info.instr_count >= SELFCHECK_MIN_INSTRS
                    && info.check_left > 0
                    && !info.no_selfcheck;
                if check {
                    info.check_left -= 1;
                }
                return Some((info.index, info.end_rip, check));
            }
        }
    }
    let info_ptr: *mut BlockInfo = match blocks().get_mut(&rip) {
        Some(info) =>
        {
            info.used = true;
            &mut **info as *mut BlockInfo
        },
        None => return None,
    };
    // Stale block (its RIP maps elsewhere now): drop it.
    if phys_page_of(rip) != Some((*info_ptr).phys_page) {
        JIT64_FORGET_STALE += 1;
        forget(rip);
        return None;
    }
    if JIT64_ENTRY_CACHE {
        ENTRY_CACHE[entry_slot(rip)] = EntrySlot { rip, info: info_ptr };
    }
    let check = JIT64_SELFCHECK
        && (*info_ptr).instr_count >= SELFCHECK_MIN_INSTRS
        && (*info_ptr).check_left > 0
        && !(*info_ptr).no_selfcheck;
    if check {
        (*info_ptr).check_left -= 1;
    }
    Some(((*info_ptr).index, (*info_ptr).end_rip, check))
}

// At a block's exit, run its successor directly (budget bounds wasm recursion).
#[no_mangle]
pub unsafe fn jit64_chain() {
    if CHAIN_BUDGET == 0 || !jit64_enabled() {
        return;
    }
    if EVICT_PENDING {
        EVICT_PENDING = false;
        evict_some(max_blocks() / 4);
    }
    let target = *crate::cpu::global_pointers::rip;
    if let Some((index, end_rip, check)) = lookup_compiled(target) {
        CHAIN_BUDGET -= 1;
        JIT64_RUN_HITS += 1;
        JIT64_CHAIN_HOPS += 1;
        log_exec(target);
        dispatch_block(index, target, end_rip, check);
    }
}

// Refill the chaining budget per main-loop entry (disabled path costs nothing).
#[inline]
pub unsafe fn refill_chain_budget() {
    if JIT64_CHAIN_CODEGEN {
        CHAIN_BUDGET = JIT64_CHAIN_BUDGET;
    }
}

// --- Diagnostic self-check: run each block through the interpreter too ------

static mut JIT64_SELFCHECK_MISMATCH: u64 = 0;
static mut SELFCHECK_RUNS: u64 = 0;
static mut SELFCHECK_INSTRS: u64 = 0;
static mut SELFCHECK_BYTES: [u8; 128] = [0; 128];
static mut SELFCHECK_BYTES_LEN: u32 = 0;
static mut SELFCHECK_START: u64 = 0;
static mut SELFCHECK_END: u64 = 0;
static mut SELFCHECK_JIT_RIP: u64 = 0;
static mut SELFCHECK_INTERP_RIP: u64 = 0;
static mut SELFCHECK_JIT_FLAGS: i32 = 0;
static mut SELFCHECK_INTERP_FLAGS: i32 = 0;
static mut SELFCHECK_REG: i32 = -1;
static mut SELFCHECK_JIT_REG: u64 = 0;
static mut SELFCHECK_INTERP_REG: u64 = 0;
// Memory the block wrote, so a wrong store is caught too (not just registers).
const SELFCHECK_LOG_MAX: usize = 64;
static mut SELFCHECK_LOG_ENABLED: bool = false;
static mut SELFCHECK_LOG_N: u32 = 0;
static mut SELFCHECK_LOG_OVERFLOW: u32 = 0;
static mut SELFCHECK_LOG_PHYS: [u64; SELFCHECK_LOG_MAX] = [0; SELFCHECK_LOG_MAX];
static mut SELFCHECK_LOG_PRE: [u64; SELFCHECK_LOG_MAX] = [0; SELFCHECK_LOG_MAX];
static mut SELFCHECK_LOG_WIDTH: [u8; SELFCHECK_LOG_MAX] = [0; SELFCHECK_LOG_MAX];
static mut SELFCHECK_MEM_MISMATCH: u64 = 0;
static mut SELFCHECK_SKIPPED_FAULT: u64 = 0;
static mut SELFCHECK_MEM_PHYS: u64 = 0;
static mut SELFCHECK_MEM_JIT: u64 = 0;
static mut SELFCHECK_MEM_INTERP: u64 = 0;
static mut SELFCHECK_MEM_WIDTH: u8 = 0;
static mut SELFCHECK_PRE_REG: u64 = 0;
static mut SELFCHECK_RETIRED: u32 = 0;

unsafe fn sc_read_mem(phys: u32, width: u8) -> u64 {
    match width {
        8 => crate::memory::read8(phys) as u8 as u64,
        16 => crate::memory::read16(phys) as u16 as u64,
        32 => crate::memory::read32s(phys) as u32 as u64,
        _ => crate::memory::read64s(phys) as u64,
    }
}

unsafe fn sc_write_mem(phys: u32, value: u64, width: u8) {
    match width {
        8 => crate::memory::write8(phys, value as i32),
        16 => crate::memory::write16(phys, value as i32),
        32 => crate::memory::write32(phys, value as i32),
        _ =>
        {
            crate::memory::write32(phys, value as u32 as i32);
            crate::memory::write32(phys + 4, (value >> 32) as u32 as i32);
        },
    }
}

// Called before every generated store while a block is being self-checked.
#[no_mangle]
pub unsafe fn jit64_selfcheck_log_write(vaddr: u64, _value: u64, width: u32) {
    if !SELFCHECK_LOG_ENABLED {
        return;
    }
    let bytes = width / 8;
    if bytes == 0 || bytes > 8 || (vaddr as usize & 0xFFF) + bytes as usize > 0x1000 {
        SELFCHECK_LOG_OVERFLOW += 1;
        return;
    }
    let n = SELFCHECK_LOG_N as usize;
    if n >= SELFCHECK_LOG_MAX {
        SELFCHECK_LOG_OVERFLOW += 1;
        return;
    }
    let phys = match crate::cpu::core::translate_address_64_no_side_effects(vaddr) {
        Ok(phys) => phys,
        Err(_) =>
        {
            SELFCHECK_LOG_OVERFLOW += 1;
            return;
        },
    };
    SELFCHECK_LOG_PHYS[n] = phys as u64;
    SELFCHECK_LOG_PRE[n] = sc_read_mem(phys, width as u8);
    SELFCHECK_LOG_WIDTH[n] = width as u8;
    SELFCHECK_LOG_N += 1;
}

// 0 count, 1..6 first mismatch info, 7..9 first differing register.
#[no_mangle]
pub unsafe fn jit64_selfcheck_info(i: u32) -> u64 {
    match i {
        0 => JIT64_SELFCHECK_MISMATCH,
        1 => SELFCHECK_START,
        2 => SELFCHECK_END,
        3 => SELFCHECK_JIT_RIP,
        4 => SELFCHECK_INTERP_RIP,
        5 => SELFCHECK_JIT_FLAGS as u32 as u64,
        6 => SELFCHECK_INTERP_FLAGS as u32 as u64,
        7 => SELFCHECK_REG as u64,
        8 => SELFCHECK_JIT_REG,
        9 => SELFCHECK_INTERP_REG,
        10 => SELFCHECK_RUNS,
        11 => SELFCHECK_INSTRS,
        12 => SELFCHECK_BYTES_LEN as u64,
        13..=28 => {
            let w = (i - 13) as usize;
            let mut b = [0u8; 8];
            b.copy_from_slice(&SELFCHECK_BYTES[w * 8..w * 8 + 8]);
            u64::from_le_bytes(b)
        },
        29 => SELFCHECK_MEM_MISMATCH,
        30 => SELFCHECK_MEM_PHYS,
        31 => SELFCHECK_MEM_JIT,
        32 => SELFCHECK_MEM_INTERP,
        33 => SELFCHECK_MEM_WIDTH as u64,
        34 => SELFCHECK_SKIPPED_FAULT,
        35 => SELFCHECK_PRE_REG,
        36 => SELFCHECK_RETIRED as u64,
        _ => 0,
    }
}

pub unsafe fn jit64_set_selfcheck(enabled: u32) { JIT64_SELFCHECK = enabled != 0; }
pub unsafe fn jit64_set_selfcheck_min(n: u32) { SELFCHECK_MIN_INSTRS = n.min(255) as u16; }
pub unsafe fn jit64_set_selfcheck_repeat(n: u32) { SELFCHECK_REPEAT = n.min(65535) as u16; }

#[derive(Copy, Clone)]
struct GuestSnap {
    regs: [u64; 16],
    flags: i32,
    rip: u64,
    xmm: [u64; 32],
}

// AF is undefined for logic/shift/inc/dec, so ignore it even in registers.
fn guest_eq(a: &GuestSnap, b: &GuestSnap) -> bool {
    const UNDEFINED_FLAGS: i32 = 1 << 4;
    if a.rip != b.rip || a.xmm != b.xmm {
        return false;
    }
    if (a.flags & !UNDEFINED_FLAGS) != (b.flags & !UNDEFINED_FLAGS) {
        return false;
    }
    let af = ((a.flags ^ b.flags) & UNDEFINED_FLAGS) as u64;
    for i in 0..16usize {
        if (a.regs[i] ^ b.regs[i]) & !af != 0 {
            return false;
        }
    }
    true
}

unsafe fn snap_guest() -> GuestSnap {
    use crate::cpu::global_pointers::*;
    // Materialise lazy flags so the raw word is current.
    jit64_sync_flags();
    let mut regs = [0u64; 16];
    for i in 0..8usize {
        regs[i] = *reg32.add(i) as u32 as u64 | ((*reg_high.add(i) as u64) << 32);
    }
    for i in 0..8usize {
        regs[8 + i] = *reg64_ext.add(i);
    }
    let mut xmm = [0u64; 32];
    for i in 0..16usize {
        let v = crate::cpu::core::read_xmm128s(i as i32);
        xmm[i * 2] = v.u64[0];
        xmm[i * 2 + 1] = v.u64[1];
    }
    GuestSnap { regs, flags: *flags, rip: *rip, xmm }
}

unsafe fn restore_guest(s: &GuestSnap) {
    use crate::cpu::global_pointers::*;
    for i in 0..8usize {
        *reg32.add(i) = s.regs[i] as u32 as i32;
        *reg_high.add(i) = (s.regs[i] >> 32) as u32;
    }
    for i in 0..8usize {
        *reg64_ext.add(i) = s.regs[8 + i];
    }
    for i in 0..16usize {
        crate::cpu::core::write_xmm128_2(i as i32, s.xmm[i * 2], s.xmm[i * 2 + 1]);
    }
    *flags = s.flags;
    *flags_changed = 0;
    *rip = s.rip;
}

// Run the block, re-run it in the interpreter from the pre-state, then compare.
unsafe fn run_block_selfcheck(index: u16, start: u64, end: u64) {
    use crate::cpu::global_pointers::*;
    SELFCHECK_RUNS += 1;
    let pre = snap_guest();
    let ic0 = *instruction_counter;
    SELFCHECK_LOG_N = 0;
    SELFCHECK_LOG_OVERFLOW = 0;
    SELFCHECK_LOG_ENABLED = true;
    let faults0 = JIT64_FAULTS;
    run_block(index);
    let jit_faulted = JIT64_FAULTS != faults0;
    SELFCHECK_LOG_ENABLED = false;
    let ic1 = *instruction_counter;
    if jit_faulted
    {
        // A faulting block is not comparable (delivery differs); leave it.
        SELFCHECK_SKIPPED_FAULT += 1;
        return;
    }
    let post_jit = snap_guest();
    // Snapshot the JIT's memory, then restore the pre-state for the re-run.
    let nlog = SELFCHECK_LOG_N as usize;
    let mut jit_mem = [0u64; SELFCHECK_LOG_MAX];
    for i in 0..nlog {
        let w = SELFCHECK_LOG_WIDTH[i];
        jit_mem[i] = sc_read_mem(SELFCHECK_LOG_PHYS[i] as u32, w);
    }
    for i in (0..nlog).rev() {
        let w = SELFCHECK_LOG_WIDTH[i];
        sc_write_mem(SELFCHECK_LOG_PHYS[i] as u32, SELFCHECK_LOG_PRE[i], w);
    }
    // Re-run exactly as many instructions as the block retired.
    restore_guest(&pre);
    *instruction_counter = ic0;
    let n = ic1.wrapping_sub(ic0).min(256);
    SELFCHECK_INSTRS += n as u64;
    for _ in 0..n {
        crate::cpu::interp::interp64::run_one();
    }
    let post_interp = snap_guest();
    *instruction_counter = ic1;
    let mut mem_bad = false;
    for i in 0..nlog {
        let w = SELFCHECK_LOG_WIDTH[i];
        let interp_mem = sc_read_mem(SELFCHECK_LOG_PHYS[i] as u32, w);
        if interp_mem != jit_mem[i] && !mem_bad {
            mem_bad = true;
            SELFCHECK_MEM_PHYS = SELFCHECK_LOG_PHYS[i];
            SELFCHECK_MEM_JIT = jit_mem[i];
            SELFCHECK_MEM_INTERP = interp_mem;
            SELFCHECK_MEM_WIDTH = w;
        }
    }
    if !guest_eq(&post_jit, &post_interp) || mem_bad {
        if mem_bad {
            SELFCHECK_MEM_MISMATCH += 1;
        }
        if JIT64_SELFCHECK_MISMATCH == 0 {
            SELFCHECK_START = start;
            SELFCHECK_END = end;
            SELFCHECK_JIT_RIP = post_jit.rip;
            SELFCHECK_INTERP_RIP = post_interp.rip;
            SELFCHECK_JIT_FLAGS = post_jit.flags;
            SELFCHECK_INTERP_FLAGS = post_interp.flags;
            SELFCHECK_REG = -1;
            let blen = ((end.wrapping_sub(start)) as usize).min(128);
            for i in 0..blen {
                SELFCHECK_BYTES[i] = match crate::cpu::core::translate_address_64_no_side_effects(start + i as u64) {
                    Ok(p) => crate::memory::read8(p) as u8,
                    Err(_) => 0,
                };
            }
            SELFCHECK_BYTES_LEN = blen as u32;
            SELFCHECK_RETIRED = n as u32;
            for r in 0..16usize {
                let af = ((post_jit.flags ^ post_interp.flags) & (1 << 4)) as u64;
                if (post_jit.regs[r] ^ post_interp.regs[r]) & !af != 0 {
                    SELFCHECK_REG = r as i32;
                    SELFCHECK_JIT_REG = post_jit.regs[r];
                    SELFCHECK_INTERP_REG = post_interp.regs[r];
                    break;
                }
            }
            if SELFCHECK_REG == -1 {
                // 100 + n identifies xmm_n.
                for r in 0..16usize {
                    if post_jit.xmm[r * 2] != post_interp.xmm[r * 2]
                        || post_jit.xmm[r * 2 + 1] != post_interp.xmm[r * 2 + 1]
                    {
                        SELFCHECK_REG = 100 + r as i32;
                        SELFCHECK_JIT_REG = post_jit.xmm[r * 2];
                        SELFCHECK_INTERP_REG = post_interp.xmm[r * 2];
                        break;
                    }
                }
            }
            let first_reg = SELFCHECK_REG;
            if (0..16).contains(&first_reg) {
                SELFCHECK_PRE_REG = pre.regs[first_reg as usize];
            }
        }
        JIT64_SELFCHECK_MISMATCH += 1;
        *crate::cpu::global_pointers::in_hlt = true;
    }
    restore_guest(&post_jit);
}

unsafe fn dispatch_block(index: u16, rip: u64, end_rip: u64, check: bool) {
    if check {
        run_block_selfcheck(index, rip, end_rip);
    }
    else {
        run_block(index);
    }
}

unsafe fn run_block(index: u16) {
    let indirect = index as i32 + crate::cpu::core::WASM_TABLE_OFFSET as i32;
    if !JIT64_BLOCK_PROLOGUE {
        // Entry setup (emit_block_prologue): sync flags, clear exception flag.
        jit64_sync_flags();
        jit64_clear_exception_flag();
    }
    // Clear any code-write bail left by an earlier block, so this one can run.
    JIT64_CODE_WRITE_BAIL = false;
    IN_JIT_BLOCK = true;
    wasm::call_indirect1(indirect, 0);
    IN_JIT_BLOCK = false;
    jit64_finish_fault();
}

pub unsafe fn note_interpreted(rip: u64) {
    if !jit64_enabled() {
        return;
    }
    // The STI-shadow path can reach here with a compiled RIP; skip recompiling.
    if blocks().contains_key(&rip) {
        return;
    }
    JIT64_INTERP += 1;
    let count = if JIT64_HOT_CACHE {
        let slot = hot_slot(rip);
        if HOT_CACHE_RIP[slot] == rip {
            let c = HOT_CACHE_COUNT[slot].wrapping_add(1);
            HOT_CACHE_COUNT[slot] = c;
            c
        }
        else if HOT_CACHE_RIP[slot] == u64::MAX {
            HOT_CACHE_RIP[slot] = rip;
            HOT_CACHE_COUNT[slot] = 1;
            1
        }
        else {
            // Slot held by another RIP: keep the exact count in the map.
            let e = hotness().entry(rip).or_insert(0);
            *e += 1;
            *e
        }
    }
    else {
        let e = hotness().entry(rip).or_insert(0);
        *e += 1;
        *e
    };
    // Compile once; retrying an undecodable block is expensive.
    if count == JIT64_THRESHOLD {
        compile_and_register(rip);
    }
}

pub unsafe fn clear_cache() {
    if !HOTNESS.is_null() {
        hotness().clear();
    }
    let hot_rip = std::ptr::addr_of_mut!(HOT_CACHE_RIP) as *mut u64;
    let hot_count = std::ptr::addr_of_mut!(HOT_CACHE_COUNT) as *mut u32;
    for i in 0..HOT_CACHE_SIZE {
        *hot_rip.add(i) = u64::MAX;
        *hot_count.add(i) = 0;
    }
    if !BLOCKS.is_null() {
        let indices: Vec<u16> = blocks().drain().map(|(_, info)| info.index).collect();
        for index in indices {
            crate::cpu::jit::jit64_free_table_index(index);
        }
    }
    let cache = std::ptr::addr_of_mut!(ENTRY_CACHE) as *mut EntrySlot;
    for i in 0..ENTRY_CACHE_SIZE {
        (*cache.add(i)).info = std::ptr::null_mut();
    }
    BLOCK_LOG_N = 0;
    EXEC_LOG_N = 0;
    if !CODE_PAGES.is_null() {
        code_pages().clear();
    }
    if !VIRT_PAGES.is_null() {
        virt_pages().clear();
    }
    crate::cpu::core::tlb64_clear_has_code();
}

// Drop only the blocks compiled from this physical page.
pub unsafe fn invalidate_physical_page(page: u32) {
    // Every RAM write funnels through here, so this is where the decode cache's
    // per-page write counter moves (even for pages with no compiled code).
    crate::cpu::core::bump_page_write_count(page);
    if CODE_PAGES.is_null() {
        return;
    }
    let rips = match code_pages().remove(&page) {
        Some(rips) => rips,
        None => return,
    };
    // This page's block may hold stale instructions: make it stop.
    if JIT64_SMC_BAIL {
        JIT64_CODE_WRITE_BAIL = true;
    }
    for rip in rips {
        JIT64_FORGET_SMC += 1;
        forget(rip);
    }
    crate::cpu::core::tlb64_refresh_code_flag(page);
}

// Whether this physical page holds compiled code (inline writes must fall back).
pub unsafe fn physical_page_has_code(page: u32) -> bool {
    !CODE_PAGES.is_null() && code_pages().contains_key(&page)
}

// INVLPG: drop this virtual page's blocks (each block is one page).
pub unsafe fn invalidate_virtual_page(vaddr: u64) {
    if VIRT_PAGES.is_null() {
        return;
    }
    let rips = match virt_pages().remove(&(vaddr >> 12)) {
        Some(rips) => rips,
        None => return,
    };
    for rip in rips {
        JIT64_FORGET_INVLPG += 1;
        forget(rip);
    }
}

// Diagnostic (tests): RIPs of compiled blocks in compile order.
const BLOCK_LOG_MAX: usize = 256;
static mut BLOCK_LOG_N: u32 = 0;
static mut BLOCK_LOG_RIP: [u64; BLOCK_LOG_MAX] = [0; BLOCK_LOG_MAX];

// Diagnostic (tests): RIPs of executed blocks in execution order.
const EXEC_LOG_MAX: usize = 8192;
static mut EXEC_LOG: [u64; EXEC_LOG_MAX] = [0; EXEC_LOG_MAX];
static mut EXEC_LOG_N: u32 = 0;

unsafe fn log_exec(rip: u64) {
    if (EXEC_LOG_N as usize) < EXEC_LOG_MAX {
        EXEC_LOG[EXEC_LOG_N as usize] = rip;
        EXEC_LOG_N += 1;
    }
}

#[no_mangle]
pub unsafe fn jit64_exec_log_len() -> u32 { EXEC_LOG_N }

#[no_mangle]
pub unsafe fn jit64_exec_log(i: u32) -> u64 {
    if i >= EXEC_LOG_N || i as usize >= EXEC_LOG_MAX {
        return u64::MAX;
    }
    EXEC_LOG[i as usize]
}

#[no_mangle]
pub unsafe fn jit64_block_log_len() -> u32 { BLOCK_LOG_N }

#[no_mangle]
pub unsafe fn jit64_block_log_rip(i: u32) -> u64 {
    if i >= BLOCK_LOG_N || i as usize >= BLOCK_LOG_MAX {
        return u64::MAX;
    }
    BLOCK_LOG_RIP[i as usize]
}

// Diagnostic (tests): decode the block at `rip`; (end_rip << 16) | count.
#[no_mangle]
pub unsafe fn jit64_probe_block(rip: u64) -> u64 {
    let phys = match crate::cpu::core::translate_address_64_no_side_effects(rip) {
        Ok(phys) => phys,
        Err(()) => return u64::MAX,
    };
    let page_off = (rip & 0xFFF) as u32;
    let mut bytes = Vec::with_capacity((0x1000 - page_off) as usize);
    for i in 0..0x1000 - page_off {
        bytes.push(crate::memory::read8(phys.wrapping_add(i)) as u8);
    }
    match decode_block_with_rips(rip, &bytes) {
        Ok(d) => (d.end_rip << 16) | d.instrs.len() as u64,
        Err(_) => u64::MAX,
    }
}

// Diagnostic (tests): compile `rip` with the given partial-prefix setting.
static mut PROBE_LEN: u32 = 0;
const PROBE_BUF_SIZE: usize = 65536;
static mut PROBE_BUF: [u8; PROBE_BUF_SIZE] = [0; PROBE_BUF_SIZE];

#[no_mangle]
pub unsafe fn jit64_compile_probe(rip: u64, partial: u32) -> u32 {
    let saved = JIT64_PARTIAL_BLOCKS;
    JIT64_PARTIAL_BLOCKS = partial != 0;
    let module = match crate::cpu::core::translate_address_64_no_side_effects(rip) {
        Ok(phys) =>
        {
            let page_off = (rip & 0xFFF) as u32;
            let mut bytes = Vec::with_capacity((0x1000 - page_off) as usize);
            for i in 0..0x1000 - page_off {
                bytes.push(crate::memory::read8(phys.wrapping_add(i)) as u8);
            }
            compile_bytes(rip, &bytes).ok()
        },
        Err(()) => None,
    };
    JIT64_PARTIAL_BLOCKS = saved;
    match module {
        Some(m) =>
        {
            let len = m.len().min(PROBE_BUF_SIZE);
            std::ptr::copy_nonoverlapping(
                m.as_ptr(),
                std::ptr::addr_of_mut!(PROBE_BUF) as *mut u8,
                len,
            );
            PROBE_LEN = len as u32;
            std::ptr::addr_of!(PROBE_BUF) as u32
        },
        None => 0,
    }
}

#[no_mangle]
pub unsafe fn jit64_compile_probe_len() -> u32 { PROBE_LEN }

#[no_mangle]
pub unsafe fn jit64_fail_op(op: u32) -> u32 {
    if op < 256 { JIT64_FAIL_OP[op as usize] } else { 0 }
}

// Recent blocks at or above the superblock threshold (diagnostics).
const BIG_BLOCK_MAX: usize = 64;
static mut BIG_BLOCK_RIP: [u64; BIG_BLOCK_MAX] = [0; BIG_BLOCK_MAX];
static mut BIG_BLOCK_LEN: [u32; BIG_BLOCK_MAX] = [0; BIG_BLOCK_MAX];
static mut BIG_BLOCK_N: u32 = 0;
const BIG_BLOCK_BYTES_SIZE: usize = 128;
static mut BIG_BLOCK_BYTES: [u8; BIG_BLOCK_BYTES_SIZE] = [0; BIG_BLOCK_BYTES_SIZE];
static mut BIG_BLOCK_BYTES_LEN: u32 = 0;
static mut BIG_BLOCK_LO: u64 = u64::MAX;
static mut BIG_BLOCK_HI: u64 = 0;
static mut LAST_BIG_BLOCK_LOCALS: u32 = 0;

#[no_mangle]
pub unsafe fn jit64_last_big_block_locals() -> u32 { LAST_BIG_BLOCK_LOCALS }

#[no_mangle]
pub unsafe fn jit64_big_block_lo() -> u64 { BIG_BLOCK_LO }
#[no_mangle]
pub unsafe fn jit64_big_block_hi() -> u64 { BIG_BLOCK_HI }

#[no_mangle]
pub unsafe fn jit64_big_block_bytes(i: u32) -> u32 {
    if (i as usize) < BIG_BLOCK_BYTES_SIZE { BIG_BLOCK_BYTES[i as usize] as u32 } else { 0 }
}

#[no_mangle]
pub unsafe fn jit64_big_block_bytes_len() -> u32 { BIG_BLOCK_BYTES_LEN }

#[no_mangle]
pub unsafe fn jit64_big_block_count() -> u32 { BIG_BLOCK_N }

#[no_mangle]
pub unsafe fn jit64_big_block_rip(i: u32) -> u64 {
    if (i as usize) < BIG_BLOCK_MAX { BIG_BLOCK_RIP[i as usize] } else { 0 }
}

#[no_mangle]
pub unsafe fn jit64_big_block_len(i: u32) -> u32 {
    if (i as usize) < BIG_BLOCK_MAX { BIG_BLOCK_LEN[i as usize] } else { 0 }
}

#[no_mangle]
pub unsafe fn jit64_compiled_count() -> u32 { blocks().len() as u32 }

// Block exit reason histogram (see emit_exit_stat).
#[no_mangle]
pub unsafe fn jit64_note_exit(reason: u32) {
    if (reason as usize) < JIT64_EXIT_REASON_COUNT {
        JIT64_EXIT_REASON[reason as usize] += 1;
    }
}

#[no_mangle]
pub unsafe fn jit64_exit_reason(i: u32) -> u64 {
    if (i as usize) < JIT64_EXIT_REASON_COUNT { JIT64_EXIT_REASON[i as usize] } else { 0 }
}

pub unsafe fn jit64_set_exit_stats(enabled: u32) {
    JIT64_EXIT_STATS = enabled != 0;
    clear_cache();
}

// Test helper: zero all XMM registers so both engines start alike.
#[no_mangle]
pub unsafe fn jit64_clear_xmm() {
    for r in 0..16 {
        crate::cpu::core::write_xmm128_2(r, 0, 0);
    }
}

// Diagnostics: 0 run hits, 1 run misses, 2 interpreted, 3 compiles, 4 failures,
// 5 compiled block count.
#[no_mangle]
pub unsafe fn jit64_stat(index: u32) -> u64 {
    match index {
        0 => JIT64_RUN_HITS,
        1 => JIT64_RUN_MISSES,
        2 => JIT64_INTERP,
        3 => JIT64_COMPILES,
        4 => JIT64_COMPILE_FAILS,
        5 => blocks().len() as u64,
        6 => JIT64_FAULTS,
        7 => JIT64_COMPILE_NO_TRANSLATE,
        8 => JIT64_COMPILE_TOO_BIG,
        9 => JIT64_COMPILE_NO_INDEX,
        10 => JIT64_REGISTERED,
        11 => JIT64_REPLACED,
        12 => JIT64_FORGET_EVICT,
        13 => JIT64_FORGET_STALE,
        14 => JIT64_FORGET_SMC,
        15 => JIT64_FORGET_INVLPG,
        16 => JIT64_CHAIN_HOPS,
        17 => JIT64_MEM_SLOW_READ,
        18 => JIT64_MEM_SLOW_WRITE,
        19 => JIT64_MEM_SLOW_CROSS,
        _ => 0,
    }
}

// Enable or disable block compilation and dispatch (used by tests/benchmarks).
#[inline]
pub unsafe fn jit64_enabled() -> bool {
    !crate::config::jit_disabled(crate::config::JIT_DISABLE_64)
}

pub unsafe fn jit64_is_enabled() -> bool { jit64_enabled() }

pub unsafe fn jit64_set_inline_memory(enabled: u32) { JIT64_INLINE_MEMORY = enabled != 0; }

pub unsafe fn jit64_set_inline_write(enabled: u32) { JIT64_INLINE_MEMORY_WRITE = enabled != 0; }

pub unsafe fn jit64_set_sse(enabled: u32) { JIT64_SSE = enabled != 0; }

pub unsafe fn jit64_set_partial_blocks(enabled: u32) { JIT64_PARTIAL_BLOCKS = enabled != 0; }

pub unsafe fn jit64_set_entry_cache(enabled: u32) { JIT64_ENTRY_CACHE = enabled != 0; }

pub unsafe fn jit64_set_max_blocks(blocks: u32) { JIT64_MAX_BLOCKS = blocks as usize; }

pub unsafe fn jit64_set_direct_code_read(enabled: u32) { JIT64_DIRECT_CODE_READ = enabled != 0; }

pub unsafe fn jit64_set_hot_cache(enabled: u32) { JIT64_HOT_CACHE = enabled != 0; }

pub unsafe fn jit64_set_chain_budget(n: u32) {
    JIT64_CHAIN_BUDGET = n;
    JIT64_CHAIN_CODEGEN = n > 0;
    // Blocks compiled without the hand-off call cannot chain, so drop them.
    clear_cache();
}

pub unsafe fn jit64_set_block_prologue(enabled: u32) {
    JIT64_BLOCK_PROLOGUE = enabled != 0;
    clear_cache();
}

pub unsafe fn jit64_set_superblocks(enabled: u32) {
    JIT64_SUPERBLOCKS = enabled != 0;
    clear_cache();
}

pub unsafe fn jit64_set_block_limit(n: u32) {
    JIT64_BLOCK_LIMIT = if n == 0 { JIT64_MAX_BLOCK_INSTRS } else { n as usize };
    clear_cache();
}

pub unsafe fn jit64_set_smc_bail(enabled: u32) {
    JIT64_SMC_BAIL = enabled != 0;
    clear_cache();
}

// Answers for its own key range; `crate::config` chains the subsystems.
pub unsafe fn set_config(index: u32, value: u32) -> bool {
    use crate::config as cfg;

    match index {
        cfg::JIT64_INLINE_MEMORY => jit64_set_inline_memory(value),
        cfg::JIT64_INLINE_WRITE => jit64_set_inline_write(value),
        cfg::JIT64_SSE => jit64_set_sse(value),
        cfg::JIT64_PARTIAL_BLOCKS => jit64_set_partial_blocks(value),
        cfg::JIT64_ENTRY_CACHE => jit64_set_entry_cache(value),
        cfg::JIT64_HOT_CACHE => jit64_set_hot_cache(value),
        cfg::JIT64_DIRECT_CODE_READ => jit64_set_direct_code_read(value),
        cfg::JIT64_MAX_BLOCKS => jit64_set_max_blocks(value),
        cfg::JIT64_CHAIN_BUDGET => jit64_set_chain_budget(value),
        cfg::JIT64_BLOCK_PROLOGUE => jit64_set_block_prologue(value),
        cfg::JIT64_SUPERBLOCKS => jit64_set_superblocks(value),
        cfg::JIT64_BLOCK_LIMIT => jit64_set_block_limit(value),
        cfg::JIT64_SMC_BAIL => jit64_set_smc_bail(value),
        cfg::JIT64_SELFCHECK => jit64_set_selfcheck(value),
        cfg::JIT64_SELFCHECK_MIN => jit64_set_selfcheck_min(value),
        cfg::JIT64_SELFCHECK_REPEAT => jit64_set_selfcheck_repeat(value),
        cfg::JIT64_EXIT_STATS => jit64_set_exit_stats(value),
        _ => return false,
    }

    true
}

pub unsafe fn get_config(index: u32) -> Option<u32> {
    use crate::config as cfg;

    Some(match index {
        cfg::JIT64_INLINE_MEMORY => JIT64_INLINE_MEMORY as u32,
        cfg::JIT64_INLINE_WRITE => JIT64_INLINE_MEMORY_WRITE as u32,
        cfg::JIT64_SSE => JIT64_SSE as u32,
        cfg::JIT64_PARTIAL_BLOCKS => JIT64_PARTIAL_BLOCKS as u32,
        cfg::JIT64_ENTRY_CACHE => JIT64_ENTRY_CACHE as u32,
        cfg::JIT64_HOT_CACHE => JIT64_HOT_CACHE as u32,
        cfg::JIT64_DIRECT_CODE_READ => JIT64_DIRECT_CODE_READ as u32,
        cfg::JIT64_MAX_BLOCKS => JIT64_MAX_BLOCKS as u32,
        cfg::JIT64_CHAIN_BUDGET => JIT64_CHAIN_BUDGET,
        cfg::JIT64_BLOCK_PROLOGUE => JIT64_BLOCK_PROLOGUE as u32,
        cfg::JIT64_SUPERBLOCKS => JIT64_SUPERBLOCKS as u32,
        cfg::JIT64_BLOCK_LIMIT => JIT64_BLOCK_LIMIT as u32,
        cfg::JIT64_SMC_BAIL => JIT64_SMC_BAIL as u32,
        cfg::JIT64_SELFCHECK => JIT64_SELFCHECK as u32,
        cfg::JIT64_SELFCHECK_MIN => SELFCHECK_MIN_INSTRS as u32,
        cfg::JIT64_SELFCHECK_REPEAT => SELFCHECK_REPEAT as u32,
        cfg::JIT64_EXIT_STATS => JIT64_EXIT_STATS as u32,
        _ => return None,
    })
}

#[no_mangle]
pub unsafe fn jit64_clear_cache() { clear_cache(); }

// u64::MAX means translation faulted; the block returns immediately.
// Debuggers must not use jit64_translate: its failure delivers a guest #PF.
#[no_mangle]
pub unsafe fn jit64_debug_translate(vaddr: u64) -> u64 {
    match crate::cpu::core::translate_address_64_no_side_effects(vaddr) {
        Ok(phys) => crate::memory::mem8 as u64 + phys as u64,
        Err(()) => u64::MAX,
    }
}

#[no_mangle]
pub unsafe fn jit64_translate(vaddr: u64, for_writing: u32) -> u64 {
    match crate::cpu::core::translate_address_64_jit(vaddr, for_writing != 0, user_access()) {
        Ok(phys) => crate::memory::mem8 as u64 + phys as u64,
        Err(()) => u64::MAX,
    }
}

#[no_mangle]
pub unsafe fn jit64_mem_read(vaddr: u64, width: u32) -> u64 {
    JIT64_MEMORY_FAULT = 0;
    JIT64_MEM_SLOW_READ += 1;
    let bytes = (width / 8) as usize;
    if bytes == 0 || bytes > 8 {
        JIT64_MEMORY_FAULT = 1;
        return 0;
    }
    if (vaddr as usize & 0xFFF) + bytes <= 0x1000 {
        let phys = match crate::cpu::core::translate_address_64_jit(vaddr, false, user_access()) {
            Ok(phys) => phys,
            Err(()) => { JIT64_MEMORY_FAULT = 1; return 0; },
        };
        return match width {
            8 => crate::memory::read8(phys) as u8 as u64,
            16 => crate::memory::read16(phys) as u16 as u64,
            32 => crate::memory::read32s(phys) as u32 as u64,
            64 => crate::memory::read64s(phys) as u64,
            _ => unreachable!(),
        };
    }
    let mut physical = [0u32; 8];
    for offset in 0..bytes {
        physical[offset] = match crate::cpu::core::translate_address_64_jit(vaddr + offset as u64, false, user_access()) {
            Ok(phys) => phys,
            Err(()) => { JIT64_MEMORY_FAULT = 1; return 0; },
        };
    }
    JIT64_MEM_SLOW_CROSS += 1;
    let mut value = 0;
    for offset in 0..bytes {
        value |= (crate::memory::read8(physical[offset]) as u8 as u64) << (offset * 8);
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
        if crate::cpu::core::translate_address_64_jit(vaddr + offset as u64, for_writing, user_access()).is_err() {
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
    crate::cpu::interp::instructions_0f::instr_0FA2();
}

#[inline(always)]
pub(crate) unsafe fn shift_apply(value: u64, raw_count: u64, encoded: u32) -> u64 {
    let width = encoded & 0xFF;
    let kind = encoded >> 8;
    let count = raw_count as u32 & if width == 64 { 63 } else { 31 };
    let mask = operand_mask(width);
    let value = value & mask;
    if count == 0 {
        return value;
    }
    let sign = 1u64 << (width - 1);
    let (result, carry, overflow, rotate) = match kind {
        0 => {
            let result = value.wrapping_shl(count) & mask;
            // A left shift by at least the operand width shifts out every bit;
            // carry is undefined there, so report no carry.
            let carry = count < width && value >> (width - count) & 1 != 0;
            (result, carry, count == 1 && (result & sign != 0) ^ carry, false)
        },
        1 => (value >> count, value >> (count - 1) & 1 != 0, count == 1 && value & sign != 0, false),
        2 => {
            let signed = (value ^ sign).wrapping_sub(sign);
            (((signed as i64) >> count) as u64 & mask, value >> (count - 1) & 1 != 0, false, false)
        },
        3 => {
            // ROL: only CF and OF change.
            let result = ((value << count) | (value >> (width - count))) & mask;
            let carry = value >> (width - count) & 1 != 0;
            (result, carry, count == 1 && (result & sign != 0) != carry, true)
        },
        4 => {
            // ROR: only CF and OF change.
            let result = ((value >> count) | (value << (width - count))) & mask;
            let carry = value >> (count - 1) & 1 != 0;
            let msb = result & sign != 0;
            let next = result >> (width - 2) & 1 != 0;
            (result, carry, count == 1 && msb != next, true)
        },
        _ => unreachable!(),
    };
    // A rotate only touches CF and OF; it leaves the other flags alone.
    let mut new_flags = *crate::cpu::global_pointers::flags & !if rotate { 1 | 1 << 11 } else { FLAG_MASK };
    if carry { new_flags |= 1; }
    if !rotate {
        if result == 0 { new_flags |= 1 << 6; }
        if result & sign != 0 { new_flags |= 1 << 7; }
        if (result as u8).count_ones() & 1 == 0 { new_flags |= 1 << 2; }
    }
    if overflow { new_flags |= 1 << 11; }
    *crate::cpu::global_pointers::flags = new_flags;
    *crate::cpu::global_pointers::flags_changed = 0;
    result
}

#[no_mangle]
pub unsafe fn jit64_shift(value: u64, raw_count: u64, encoded: u32) -> u64 {
    shift_apply(value, raw_count, encoded)
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
    JIT64_MEM_SLOW_WRITE += 1;
    let bytes = (width / 8) as usize;
    if bytes == 0 || bytes > 8 {
        JIT64_MEMORY_FAULT = 1;
        return;
    }
    if (vaddr as usize & 0xFFF) + bytes <= 0x1000 {
        let phys = match crate::cpu::core::translate_address_64_jit(vaddr, true, user_access()) {
            Ok(phys) => phys,
            Err(()) => { JIT64_MEMORY_FAULT = 1; return; },
        };
        match width {
            8 => crate::memory::write8(phys, value as i32),
            16 => crate::memory::write16(phys, value as i32),
            32 => crate::memory::write32(phys, value as i32),
            64 => {
                crate::memory::write32(phys, value as u32 as i32);
                crate::memory::write32(phys + 4, (value >> 32) as u32 as i32);
            },
            _ => unreachable!(),
        }
        return;
    }
    let mut physical = [0u32; 8];
    for offset in 0..bytes {
        physical[offset] = match crate::cpu::core::translate_address_64_jit(
            vaddr + offset as u64,
            true,
            user_access(),
        ) {
            Ok(phys) => phys,
            Err(()) => { JIT64_MEMORY_FAULT = 1; return; },
        };
    }
    JIT64_MEM_SLOW_CROSS += 1;
    for offset in 0..bytes {
        crate::memory::write8(
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
    JIT64_FAULTS += 1;
    JIT64_MEMORY_FAULT = 0;
    *crate::cpu::global_pointers::rip = *crate::cpu::global_pointers::previous_rip;
    *crate::cpu::global_pointers::instruction_pointer = *crate::cpu::global_pointers::rip as i32;
    crate::cpu::core::exit_jit64();
}

// in/out. A failed translation sets the fault flag so the block rewinds.
#[no_mangle]
pub unsafe fn jit64_in(port: u64, width: u32) -> u64 {
    let port = port as i32;
    // `width` is in bits; treating it as bytes made `in al, dx` read 32 bits.
    if !crate::cpu::core::test_privileges_for_io(port, (width / 8) as i32) {
        return 0;
    }
    match width {
        8 => crate::cpu::core::io_port_read8(port) as u32 as u64,
        16 => crate::cpu::core::io_port_read16(port) as u16 as u64,
        _ => crate::cpu::core::io_port_read32(port) as u32 as u64,
    }
}

#[no_mangle]
pub unsafe fn jit64_out(port: u64, value: u64, width: u32) {
    let port = port as i32;
    if !crate::cpu::core::test_privileges_for_io(port, (width / 8) as i32) {
        return;
    }
    match width {
        8 => crate::cpu::core::io_port_write8(port, value as i32),
        16 => crate::cpu::core::io_port_write16(port, value as i32),
        _ => crate::cpu::core::io_port_write32(port, value as i32),
    }
}

// Materialise the interpreter's pending lazy flags. No-op if none are pending.
#[no_mangle]
pub unsafe fn jit64_sync_flags() {
    if *crate::cpu::global_pointers::flags_changed != 0 {
        *crate::cpu::global_pointers::flags = crate::cpu::core::get_eflags();
        *crate::cpu::global_pointers::flags_changed = 0;
    }
}

#[no_mangle]
pub unsafe fn jit64_cli() {
    *crate::cpu::global_pointers::flags &= !crate::cpu::core::FLAG_INTERRUPT;
    *crate::cpu::global_pointers::flags_changed = 0;
}

#[no_mangle]
pub unsafe fn jit64_pushfq() {
    let rsp = crate::cpu::core::read_reg64(4).wrapping_sub(8);
    let phys = match crate::cpu::core::translate_address_64_jit(rsp, true, false) {
        Ok(phys) => phys,
        Err(()) => {
            JIT64_MEMORY_FAULT = 1;
            return;
        },
    };
    crate::memory::write32(phys, *crate::cpu::global_pointers::flags);
    crate::memory::write32(phys + 4, 0);
    crate::cpu::core::write_reg64(4, rsp);
}

// rdtsc: the value the interpreter's RDTSC would return.
#[no_mangle]
pub unsafe fn jit64_rdtsc() -> u64 { crate::cpu::core::read_tsc() }

// BT/BTS/BTR/BTC (op: 0..3), matching interp64::bit_test: only CF is touched.
#[no_mangle]
pub unsafe fn jit64_bt(value: u64, index: u64, encoded: u32) -> u64 {
    let op = encoded & 0xFF;
    let width = (encoded >> 8) & 0xFF;
    let bit = index as u32 & (width - 1);
    let old = (value >> bit) & 1 != 0;
    let flags = crate::cpu::global_pointers::flags;
    *flags &= !1; // CF
    if old {
        *flags |= 1;
    }
    *crate::cpu::global_pointers::flags_changed = 0;
    if op == 0 {
        return value;
    }
    let result = match op {
        1 => value | 1u64 << bit,
        2 => value & !(1u64 << bit),
        _ => value ^ 1u64 << bit,
    };
    if width == 64 {
        result
    }
    else {
        result & (1u64 << width) - 1
    }
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
            Instruction::ArithRegReg { op: ArithOp::Sub, dst: 0, src: 3, width: 64 },
            Instruction::ArithRegReg { op: ArithOp::Cmp, dst: 8, src: 9, width: 64 },
            Instruction::ArithRegReg { op: ArithOp::Test, dst: 1, src: 1, width: 64 },
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
            Instruction::ArithRegImm { op: ArithOp::Sub, r: 8, value: u64::MAX, width: 64 },
            Instruction::ArithRegImm { op: ArithOp::Cmp, r: 8, value: 1, width: 64 },
            Instruction::ArithRegImm { op: ArithOp::Test, r: 8, value: u64::MAX, width: 64 },
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
            Instruction::ArithRegReg { op: ArithOp::Or, dst: 8, src: 9, width: 64 },
            Instruction::ArithRegReg { op: ArithOp::And, dst: 8, src: 9, width: 64 },
            Instruction::ArithRegReg { op: ArithOp::Xor, dst: 8, src: 9, width: 64 },
            Instruction::ArithRegImm { op: ArithOp::And, r: 8, value: 0x7F, width: 64 },
        ]);
    }

    #[test]
    fn decodes_rip_relative_lea_and_near_jcc() {
        let lea = decode_block(0x1000, &[
            0x48, 0x8D, 0x15, 0xF9, 0x00, 0x00, 0x00, // lea rdx, [rip + 0xf9]
        ]).unwrap();
        assert_eq!(lea, vec![Instruction::Lea {
            dst: 2,
            mem: Mem { base: None, index: None, scale: 0, addr_size: 64, disp: 0x1100, segment: None },
            width: 64,
        }]);

        let branch = decode_block(0x2000, &[0x0F, 0x85, 0xFA, 0xFF, 0xFF, 0xFF]).unwrap();
        assert_eq!(branch, vec![Instruction::Jcc {
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
            Instruction::PushReg { r: 8 },
            Instruction::PopReg { r: 9 },
            Instruction::PushImm { value: u64::MAX },
            Instruction::Call { target: 0x1010, return_address: 0x100B },
        ]);

        assert_eq!(decode_block(0, &[0xC2, 0x10, 0x00]).unwrap(), vec![
            Instruction::Ret { adjustment: 16 },
        ]);
    }

    #[test]
    fn decodes_memory_arithmetic_forms() {
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 8, segment: None };
        let decoded = decode_block(0, &[
            0x48, 0x83, 0x44, 0x24, 0x08, 0x01, // add qword [rsp+8], 1
            0x4C, 0x31, 0x44, 0x24, 0x08, // xor qword [rsp+8], r8
            0x4C, 0x03, 0x5C, 0x24, 0x08, // add r11, [rsp+8]
            0x48, 0xF7, 0x44, 0x24, 0x08, 0xFF, 0x00, 0x00, 0x00, // test [rsp+8], 0xff
        ]).unwrap();

        assert_eq!(decoded, vec![
            Instruction::ArithMemImm { op: ArithOp::Add, mem, value: 1, width: 64 },
            Instruction::ArithMemReg { op: ArithOp::Xor, mem, src: 8, width: 64 },
            Instruction::ArithRegMem { op: ArithOp::Add, dst: 11, mem, width: 64 },
            Instruction::ArithMemImm { op: ArithOp::Test, mem, value: 0xFF, width: 64 },
        ]);
    }

    #[test]
    fn decodes_memory_immediate_and_indirect_control_flow() {
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 8, segment: None };
        assert_eq!(decode_block(0, &[
            0x48, 0xC7, 0x44, 0x24, 0x08, 0xFF, 0xFF, 0xFF, 0xFF,
        ]).unwrap(), vec![Instruction::MovMemImm { mem, value: u64::MAX, width: 64 }]);

        assert_eq!(decode_block(0x1000, &[0x41, 0xFF, 0xD0]).unwrap(), vec![
            Instruction::CallReg { r: 8, return_address: 0x1003 },
        ]);
        assert_eq!(decode_block(0, &[0x41, 0xFF, 0xE1]).unwrap(), vec![
            Instruction::JmpReg { r: 9 },
        ]);
        assert_eq!(decode_block(0x1000, &[0xFF, 0x15, 0xFA, 0x00, 0x00, 0x00]).unwrap(), vec![
            Instruction::CallMem {
                mem: Mem { base: None, index: None, scale: 0, addr_size: 64, disp: 0x1100, segment: None },
                return_address: 0x1006,
            },
        ]);
        assert_eq!(decode_block(0, &[0xFF, 0x64, 0x24, 0x08]).unwrap(), vec![
            Instruction::JmpMem {
                mem: Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 8, segment: None },
            },
        ]);
        assert_eq!(decode_block(0, &[0xC9, 0xC3]).unwrap(), vec![
            Instruction::Leave,
            Instruction::Ret { adjustment: 0 },
        ]);
    }

    #[test]
    fn decodes_32_bit_register_operations() {
        assert_eq!(decode_block(0, &[
            0x89, 0xD8, // mov eax, ebx
            0x41, 0x83, 0xC5, 0x01, // add r13d, 1
            0x85, 0xC0, // test eax, eax
        ]).unwrap(), vec![
            Instruction::MovRegReg { dst: 0, src: 3, width: 32 },
            Instruction::AddRegImm { r: 13, value: 1, width: 32 },
            Instruction::ArithRegReg { op: ArithOp::Test, dst: 0, src: 0, width: 32 },
        ]);
    }

    #[test]
    fn decodes_32_bit_memory_operations() {
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 24, segment: None };
        assert_eq!(decode_block(0, &[
            0xC7, 0x44, 0x24, 0x18, 0xFF, 0xFF, 0xFF, 0xFF,
            0x83, 0x44, 0x24, 0x18, 0x01,
            0x44, 0x8B, 0x74, 0x24, 0x18,
        ]).unwrap(), vec![
            Instruction::MovMemImm { mem, value: 0xFFFF_FFFF, width: 32 },
            Instruction::ArithMemImm { op: ArithOp::Add, mem, value: 1, width: 32 },
            Instruction::MovRegMem { dst: 14, mem, width: 32 },
        ]);
    }

    #[test]
    fn decodes_nop_and_endbr64() {
        assert_eq!(decode_block(0, &[
            0x90,
            0x0F, 0x1F, 0x44, 0x00, 0x00,
            0xF3, 0x0F, 0x1E, 0xFA,
        ]).unwrap(), vec![Instruction::Nop, Instruction::Nop, Instruction::Nop]);
    }

    #[test]
    fn decodes_movzx_and_movsx() {
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 16, segment: None };
        assert_eq!(decode_block(0, &[
            0x0F, 0xB6, 0xC4, // movzx eax, ah
            0x48, 0x0F, 0xBE, 0xCB, // movsx rcx, bl
            0x44, 0x0F, 0xB7, 0x7C, 0x24, 0x10, // movzx r15d, word [rsp+16]
        ]).unwrap(), vec![
            Instruction::MovExtendReg {
                dst: 0, src: 0, src_width: 8, dst_width: 32, signed: false, high8: true,
            },
            Instruction::MovExtendReg {
                dst: 1, src: 3, src_width: 8, dst_width: 64, signed: true, high8: false,
            },
            Instruction::MovExtendMem {
                dst: 15, mem, src_width: 16, dst_width: 32, signed: false,
            },
        ]);
    }

    #[test]
    fn decodes_movsxd() {
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 16, segment: None };
        assert_eq!(decode_block(0, &[
            0x4D, 0x63, 0xC1, // movsxd r8, r9d
            0x4C, 0x63, 0x54, 0x24, 0x10, // movsxd r10, dword [rsp+16]
        ]).unwrap(), vec![
            Instruction::MovExtendReg {
                dst: 8, src: 9, src_width: 32, dst_width: 64, signed: true, high8: false,
            },
            Instruction::MovExtendMem {
                dst: 10, mem, src_width: 32, dst_width: 64, signed: true,
            },
        ]);
    }

    #[test]
    fn decodes_imul_forms() {
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 16, segment: None };
        assert_eq!(decode_block(0, &[
            0x45, 0x0F, 0xAF, 0xC1, // imul r8d, r9d
            0x4D, 0x6B, 0xC1, 0xFF, // imul r8, r9, -1
            0x48, 0x69, 0x5C, 0x24, 0x10, 2, 0, 0, 0, // imul rbx, [rsp+16], 2
        ]).unwrap(), vec![
            Instruction::ImulRegReg { dst: 8, lhs: 8, rhs: 9, width: 32 },
            Instruction::ImulRegImm { dst: 8, src: 9, value: u64::MAX, width: 64 },
            Instruction::ImulRegMem { dst: 3, mem, value: Some(2), width: 64 },
        ]);
    }

    #[test]
    fn decodes_bswap() {
        assert_eq!(decode_block(0, &[
            0x41, 0x0F, 0xC8, // bswap r8d
            0x49, 0x0F, 0xCF, // bswap r15
        ]).unwrap(), vec![
            Instruction::Bswap { r: 8, width: 32 },
            Instruction::Bswap { r: 15, width: 64 },
        ]);
    }

    #[test]
    fn decodes_adc_and_sbb_forms() {
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 16, segment: None };
        assert_eq!(decode_block(0, &[
            0x4D, 0x11, 0xC8, // adc r8, r9
            0x49, 0x83, 0xD8, 0xFF, // sbb r8, -1
            0x48, 0x13, 0x5C, 0x24, 0x10, // adc rbx, [rsp+16]
            0x48, 0x83, 0x5C, 0x24, 0x10, 0x00, // sbb qword [rsp+16], 0
        ]).unwrap(), vec![
            Instruction::ArithRegReg { op: ArithOp::Adc, dst: 8, src: 9, width: 64 },
            Instruction::ArithRegImm { op: ArithOp::Sbb, r: 8, value: u64::MAX, width: 64 },
            Instruction::ArithRegMem { op: ArithOp::Adc, dst: 3, mem, width: 64 },
            Instruction::ArithMemImm { op: ArithOp::Sbb, mem, value: 0, width: 64 },
        ]);
    }

    #[test]
    fn decodes_register_mov_immediate_xchg_and_not() {
        assert_eq!(decode_block(0, &[
            0x41, 0xC7, 0xC5, 1, 0, 0, 0, // mov r13d, 1
            0x45, 0x87, 0xE8, // xchg r8d, r13d
            0x49, 0xF7, 0xD0, // not r8
        ]).unwrap(), vec![
            Instruction::MovRegImm { r: 13, value: 1, width: 32 },
            Instruction::XchgRegReg { a: 8, b: 13, width: 32 },
            Instruction::NotReg { r: 8, width: 64 },
        ]);
    }

    #[test]
    fn decodes_neg() {
        assert_eq!(decode_block(0, &[0xF7, 0xD8, 0x49, 0xF7, 0xD8]).unwrap(), vec![
            Instruction::NegReg { r: 0, width: 32 },
            Instruction::NegReg { r: 8, width: 64 },
        ]);
    }

    #[test]
    fn decodes_memory_unary_and_xchg() {
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 16, segment: None };
        assert_eq!(decode_block(0, &[
            0x48, 0xF7, 0x54, 0x24, 0x10,
            0x48, 0xF7, 0x5C, 0x24, 0x10,
            0x48, 0xFF, 0x44, 0x24, 0x10,
            0x48, 0xFF, 0x4C, 0x24, 0x10,
            0x48, 0x87, 0x7C, 0x24, 0x10,
        ]).unwrap(), vec![
            Instruction::NotMem { mem, width: 64 },
            Instruction::NegMem { mem, width: 64 },
            Instruction::IncDecMem { mem, width: 64, decrement: false },
            Instruction::IncDecMem { mem, width: 64, decrement: true },
            Instruction::XchgMemReg { mem, r: 7, width: 64 },
        ]);
    }

    #[test]
    fn decodes_inc_and_dec() {
        assert_eq!(decode_block(0, &[
            0xFF, 0xC0, // inc eax
            0x49, 0xFF, 0xC8, // dec r8
        ]).unwrap(), vec![
            Instruction::IncDecReg { r: 0, width: 32, decrement: false },
            Instruction::IncDecReg { r: 8, width: 64, decrement: true },
        ]);
    }

    #[test]
    fn decodes_cmov_and_setcc() {
        assert_eq!(decode_block(0, &[
            0x45, 0x0F, 0x44, 0xC1, // cmove r8d, r9d
            0x0F, 0x95, 0xC4, // setne ah
            0x41, 0x0F, 0x94, 0xC2, // sete r10b
        ]).unwrap(), vec![
            Instruction::CmovRegReg { code: 4, dst: 8, src: 9, width: 32 },
            Instruction::SetccReg { code: 5, dst: 0, high8: true },
            Instruction::SetccReg { code: 4, dst: 10, high8: false },
        ]);
    }

    #[test]
    fn decodes_cmov_and_setcc_memory_forms() {
        let mem = Mem { base: Some(0), index: None, scale: 0, addr_size: 64, disp: 0, segment: None };
        assert_eq!(decode_block(0, &[
            0x0F, 0x44, 0x00, // cmove eax, [rax]
            0x0F, 0x94, 0x00, // sete byte [rax]
        ]).unwrap(), vec![
            Instruction::CmovRegMem { code: 4, dst: 0, mem, width: 32 },
            Instruction::SetccMem { code: 4, mem },
        ]);
    }

    #[test]
    fn decodes_single_bit_shifts() {
        assert_eq!(decode_block(0, &[
            0xD1, 0xE0, // shl eax, 1
            0x49, 0xD1, 0xE8, // shr r8, 1
            0x41, 0xD1, 0xF9, // sar r9d, 1
        ]).unwrap(), vec![
            Instruction::ShiftReg { kind: ShiftKind::Shl, r: 0, width: 32, count: 1 },
            Instruction::ShiftReg { kind: ShiftKind::Shr, r: 8, width: 64, count: 1 },
            Instruction::ShiftReg { kind: ShiftKind::Sar, r: 9, width: 32, count: 1 },
        ]);
    }

    #[test]
    fn decodes_masked_immediate_shifts() {
        assert_eq!(decode_block(0, &[
            0xC1, 0xE0, 0x21, // shl eax, 33 -> 1
            0x49, 0xC1, 0xE8, 0x41, // shr r8, 65 -> 1
        ]).unwrap(), vec![
            Instruction::ShiftReg { kind: ShiftKind::Shl, r: 0, width: 32, count: 1 },
            Instruction::ShiftReg { kind: ShiftKind::Shr, r: 8, width: 64, count: 1 },
        ]);
    }

    #[test]
    fn decodes_cl_shifts() {
        assert_eq!(decode_block(0, &[
            0xD3, 0xE0, // shl eax, cl
            0x49, 0xD3, 0xE8, // shr r8, cl
            0x41, 0xD3, 0xF9, // sar r9d, cl
        ]).unwrap(), vec![
            Instruction::ShiftRegCl { kind: ShiftKind::Shl, r: 0, width: 32 },
            Instruction::ShiftRegCl { kind: ShiftKind::Shr, r: 8, width: 64 },
            Instruction::ShiftRegCl { kind: ShiftKind::Sar, r: 9, width: 32 },
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
            Instruction::MovRegReg { dst: 0, src: 3, width: 16 },
            Instruction::AddRegReg { dst: 0, src: 1, width: 16 },
            Instruction::MovRegImm { r: 0, value: 0x1234, width: 16 },
            Instruction::AddRegImm { r: 3, value: 0x5678, width: 16 },
            Instruction::AddRegImm { r: 0, value: u64::MAX, width: 16 },
            Instruction::MovExtendReg {
                dst: 0,
                src: 3,
                src_width: 16,
                dst_width: 16,
                signed: false,
                high8: false,
            },
            Instruction::ShiftReg { kind: ShiftKind::Shl, r: 0, width: 16, count: 1 },
            Instruction::NegReg { r: 0, width: 16 },
        ]);
    }

    #[test]
    fn decodes_address_size_override() {
        assert_eq!(decode_block(0, &[
            0x67, 0x48, 0x8B, 0x03, // mov rax, [ebx]
            0x66, 0x67, 0x8B, 0x03, // mov ax, [ebx]
        ]).unwrap(), vec![
            Instruction::MovRegMem {
                dst: 0,
                mem: Mem { base: Some(3), index: None, scale: 0, disp: 0, addr_size: 32, segment: None },
                width: 64,
            },
            Instruction::MovRegMem {
                dst: 0,
                mem: Mem { base: Some(3), index: None, scale: 0, disp: 0, addr_size: 32, segment: None },
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
            Instruction::AddRegReg { dst: 0, src: 3, width: 64 },
            Instruction::AddRegReg { dst: 0, src: 3, width: 64 },
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
            Instruction::ShiftMem {
                kind: ShiftKind::Shl,
                mem: Mem { base: Some(0), index: None, scale: 0, disp: 0, addr_size: 64, segment: None },
                width: 64,
                count: 1,
            },
            Instruction::ShiftMem {
                kind: ShiftKind::Shl,
                mem: Mem { base: Some(0), index: None, scale: 0, disp: 0, addr_size: 64, segment: None },
                width: 64,
                count: 5,
            },
            Instruction::ShiftMemCl {
                kind: ShiftKind::Shl,
                mem: Mem { base: Some(0), index: None, scale: 0, disp: 0, addr_size: 64, segment: None },
                width: 64,
            },
            Instruction::ShiftMem {
                kind: ShiftKind::Shl,
                mem: Mem { base: Some(0), index: None, scale: 0, disp: 0, addr_size: 64, segment: None },
                width: 16,
                count: 4,
            },
        ]);
    }

    #[test]
    fn decodes_cpuid() {
        assert_eq!(decode_block(0, &[0x0F, 0xA2]).unwrap(), vec![Instruction::Cpuid]);
    }
}
