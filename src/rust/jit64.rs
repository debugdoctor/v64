//! Long mode JIT targeting wasm32: guest registers are i64 locals, physical
//! addresses stay 32-bit (below 4 GiB).
//! http://www.sandpile.org/x86/opra.htm

#![allow(dead_code)]

use std::sync::atomic::{AtomicU32, Ordering};

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

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Mem {
    pub base: Option<u8>,
    pub index: Option<u8>,
    pub scale: u8, // index << scale

    pub disp: i64,
    // Effective-address width: 64 by default, 32 when the `67` prefix is present.
    pub addr_size: u8,
    // FS (0x64) / GS (0x65) override, whose base is added to the address.
    pub segment: Option<u8>,
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
    Rdtsc,

    // SSE2. The XMM file lives in emulated memory, so these are pairs of
    // 64-bit accesses rather than 128-bit values.
    XmmCopy { dst: u8, src: u8 },
    XmmLoad { dst: u8, mem: Mem },
    XmmStore { mem: Mem, src: u8 },
    XmmXor { dst: u8, src: u8 },
    XmmXorMem { dst: u8, mem: Mem },

    In { imm: Option<u16>, width: u8 },
    Out { imm: Option<u16>, width: u8 },
    Cli,
    PushFlags,
    CmpxchgReg { dst: u8, src: u8, width: u8 },
    CmpxchgMem { mem: Mem, src: u8, width: u8 },
    BitTestReg { r: u8, index_r: u8, op: u8, width: u8 },
    BitTestImm { r: u8, index: u64, op: u8, width: u8 },
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
pub enum ShiftKind { Shl, Shr, Sar, Rol, Ror }

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
                    out.push(Instr::MovMemImm { mem, value, width });
                }
                else {
                    let mut r = (modrm & 7) | rex_b << 3;
                    if width == 8 && rex == 0 && (4..8).contains(&r) {
                        r = 16 + (r - 4);
                    }
                    out.push(Instr::MovRegImm { r, value, width });
                }
            },

            // in/out: E4-E7 use an imm8 port, EC-EF the DX port
            0xE4 | 0xE5 | 0xE6 | 0xE7 => {
                let port = *bytes.get(i).ok_or("truncated port")? as u16;
                i += 1;
                let width = if opcode & 1 == 0 { 8 } else { operand_width(prefix_66, rex_w) };
                out.push(if opcode & 2 == 0 {
                    Instr::In { imm: Some(port), width }
                }
                else {
                    Instr::Out { imm: Some(port), width }
                });
            },
            0xEC | 0xED | 0xEE | 0xEF => {
                let width = if opcode & 1 == 0 { 8 } else { operand_width(prefix_66, rex_w) };
                out.push(if opcode & 2 == 0 {
                    Instr::In { imm: None, width }
                }
                else {
                    Instr::Out { imm: None, width }
                });
            },
            // cli (0xFA)
            0xFA => out.push(Instr::Cli),
            // pushfq (0x9C)
            0x9C => {
                if prefix_66 {
                    return Err("16-bit pushf is not supported".into());
                }
                out.push(Instr::PushFlags);
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
                    let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
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
                    let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                    out.push(Instr::XchgMemReg { mem, r, width });
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
                        0x88 => Instr::MovRegReg { dst: rm, src: reg, width },
                        0x8A => Instr::MovRegReg { dst: reg, src: rm, width },
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
                        0x00 => Instr::ArithRegReg { op: ArithOp::Add, dst: rm, src: reg, width },
                        0x02 => Instr::ArithRegReg { op: ArithOp::Add, dst: reg, src: rm, width },
                        0x08 => Instr::ArithRegReg { op: ArithOp::Or, dst: rm, src: reg, width },
                        0x0A => Instr::ArithRegReg { op: ArithOp::Or, dst: reg, src: rm, width },
                        0x10 => Instr::ArithRegReg { op: ArithOp::Adc, dst: rm, src: reg, width },
                        0x12 => Instr::ArithRegReg { op: ArithOp::Adc, dst: reg, src: rm, width },
                        0x18 => Instr::ArithRegReg { op: ArithOp::Sbb, dst: rm, src: reg, width },
                        0x1A => Instr::ArithRegReg { op: ArithOp::Sbb, dst: reg, src: rm, width },
                        0x20 => Instr::ArithRegReg { op: ArithOp::And, dst: rm, src: reg, width },
                        0x22 => Instr::ArithRegReg { op: ArithOp::And, dst: reg, src: rm, width },
                        0x28 => Instr::ArithRegReg { op: ArithOp::Sub, dst: rm, src: reg, width },
                        0x2A => Instr::ArithRegReg { op: ArithOp::Sub, dst: reg, src: rm, width },
                        0x30 => Instr::ArithRegReg { op: ArithOp::Xor, dst: rm, src: reg, width },
                        0x32 => Instr::ArithRegReg { op: ArithOp::Xor, dst: reg, src: rm, width },
                        0x38 => Instr::ArithRegReg { op: ArithOp::Cmp, dst: rm, src: reg, width },
                        0x3A => Instr::ArithRegReg { op: ArithOp::Cmp, dst: reg, src: rm, width },
                        0x84 => Instr::ArithRegReg { op: ArithOp::Test, dst: rm, src: reg, width },
                        _ => unreachable!(),
                    });
                }
                else {
                    let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                    out.push(match opcode {
                        0x88 => Instr::MovMemReg { mem, src: reg, width },
                        0x8A => Instr::MovRegMem { dst: reg, mem, width },
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
                        0x00 => Instr::ArithMemReg { op: ArithOp::Add, mem, src: reg, width },
                        0x02 => Instr::ArithRegMem { op: ArithOp::Add, dst: reg, mem, width },
                        0x08 => Instr::ArithMemReg { op: ArithOp::Or, mem, src: reg, width },
                        0x0A => Instr::ArithRegMem { op: ArithOp::Or, dst: reg, mem, width },
                        0x10 => Instr::ArithMemReg { op: ArithOp::Adc, mem, src: reg, width },
                        0x12 => Instr::ArithRegMem { op: ArithOp::Adc, dst: reg, mem, width },
                        0x18 => Instr::ArithMemReg { op: ArithOp::Sbb, mem, src: reg, width },
                        0x1A => Instr::ArithRegMem { op: ArithOp::Sbb, dst: reg, mem, width },
                        0x20 => Instr::ArithMemReg { op: ArithOp::And, mem, src: reg, width },
                        0x22 => Instr::ArithRegMem { op: ArithOp::And, dst: reg, mem, width },
                        0x28 => Instr::ArithMemReg { op: ArithOp::Sub, mem, src: reg, width },
                        0x2A => Instr::ArithRegMem { op: ArithOp::Sub, dst: reg, mem, width },
                        0x30 => Instr::ArithMemReg { op: ArithOp::Xor, mem, src: reg, width },
                        0x32 => Instr::ArithRegMem { op: ArithOp::Xor, dst: reg, mem, width },
                        0x38 => Instr::ArithMemReg { op: ArithOp::Cmp, mem, src: reg, width },
                        0x3A => Instr::ArithRegMem { op: ArithOp::Cmp, dst: reg, mem, width },
                        0x84 => Instr::ArithMemReg { op: ArithOp::Test, mem, src: reg, width },
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
                    out.push(Instr::ArithMemImm { op, mem, value, width });
                }
                else {
                    let mut r = (modrm & 7) | rex_b << 3;
                    if width == 8 && rex == 0 && (4..8).contains(&r) {
                        r = 16 + (r - 4);
                    }
                    out.push(if op == ArithOp::Add {
                        Instr::AddRegImm { r, value, width }
                    }
                    else {
                        Instr::ArithRegImm { op, r, value, width }
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
                            Instr::NotReg { r, width }
                        }
                        else {
                            Instr::NegReg { r, width }
                        });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
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
                        addr_size, segment_override,
                        imm_operand_bytes(width),
                    )?)
                };
                let value = read_imm_operand(bytes, &mut i, width)?;
                if let Some(mem) = mem {
                    out.push(Instr::ArithMemImm { op: ArithOp::Test, mem, value, width });
                }
                else {
                    let r = reg_rm;
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
                        out.push(Instr::ShiftMemCl { kind, mem, width });
                    }
                    else {
                        let count = if matches!(opcode, 0xD0 | 0xD1) { 1 } else { read_i8(bytes, &mut i)? as u8 };
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
                if matches!(opcode, 0xD2 | 0xD3) {
                    out.push(Instr::ShiftRegCl { kind, r, width });
                }
                else {
                    let count = if matches!(opcode, 0xD0 | 0xD1) { 1 } else { read_i8(bytes, &mut i)? as u8 };
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
                    let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
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
                        let _ = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                    }
                    out.push(Instr::Nop);
                }
                else if second == 0x1E && bytes.get(i) == Some(&0xFA) {
                    i += 1;
                    out.push(Instr::Nop); // ENDBR64
                }
                else if second == 0x18 || second == 0x0D {
                    // PREFETCHT0/T1/T2/NTA (0F 18 /0../3) and PREFETCHW (0F 0D):
                    // no architectural effect, only the operand has to be decoded.
                    let modrm = *bytes.get(i).ok_or("truncated prefetch modrm")?;
                    i += 1;
                    if modrm >> 6 != 3 {
                        let _ = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                    }
                    out.push(Instr::Nop);
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
                        out.push(Instr::CmpxchgReg { dst: (modrm & 7) | rex_b << 3, src, width });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                        out.push(Instr::CmpxchgMem { mem, src, width });
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
                        out.push(Instr::MovExtendReg {
                            dst, src, src_width, dst_width, signed, high8,
                        });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                        out.push(Instr::MovExtendMem {
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
                        out.push(Instr::XmmCopy { dst, src: (modrm & 7) | rex_b << 3 });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                        out.push(Instr::XmmLoad { dst, mem });
                    }
                }
                // MOVDQA/MOVDQU xmm/m128, xmm (66 0F 7F, F3 0F 7F)
                else if unsafe { JIT64_SSE } && second == 0x7F && (prefix_66 || prefix_f3) {
                    let modrm = *bytes.get(i).ok_or("truncated movdqa modrm")?;
                    i += 1;
                    let src = (modrm >> 3 & 7) | rex_r << 3;
                    if modrm >> 6 == 3 {
                        out.push(Instr::XmmCopy { dst: (modrm & 7) | rex_b << 3, src });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                        out.push(Instr::XmmStore { mem, src });
                    }
                }
                // PXOR xmm, xmm/m128 (66 0F EF)
                else if unsafe { JIT64_SSE } && second == 0xEF && prefix_66 {
                    let modrm = *bytes.get(i).ok_or("truncated pxor modrm")?;
                    i += 1;
                    let dst = (modrm >> 3 & 7) | rex_r << 3;
                    if modrm >> 6 == 3 {
                        out.push(Instr::XmmXor { dst, src: (modrm & 7) | rex_b << 3 });
                    }
                    else {
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
                        out.push(Instr::XmmXorMem { dst, mem });
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
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
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
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
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
                        let mem = decode_mem(modrm, rex, bytes, &mut i, base, addr_size, segment_override, 0)?;
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
                // RDTSC (0F 31)
                else if second == 0x31 {
                    out.push(Instr::Rdtsc);
                }
                // BT r/m, r (0F A3), register form only
                else if second == 0xA3 {
                    let modrm = *bytes.get(i).ok_or("truncated modrm")?;
                    i += 1;
                    if modrm >> 6 != 3 {
                        return Err("bt with a memory operand is not supported".into());
                    }
                    let ir = (modrm >> 3 & 7) | rex_r << 3;
                    out.push(Instr::BitTestReg {
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
                    out.push(Instr::BitTestImm {
                        r: (modrm & 7) | rex_b << 3,
                        index,
                        op: group - 4,
                        width,
                    });
                }
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
}

fn gen_memory_address_local(
    b: &mut WasmBuilder,
    locals: &mut Vec<(u8, WasmLocalI64)>,
    mem: &Mem,
) -> WasmLocalI64 {
    gen_effective_addr(b, locals, mem);
    b.set_new_local_i64()
}

// Long-mode inline memory fast path.
const CPL_ADDR: i32 = 612; // global_pointers::cpl
const CR0_ADDR: i32 = 580; // global_pointers::cr
const MEMORY_SIZE_ADDR: i32 = 812; // global_pointers::memory_size
const CR0_WP: i32 = 1 << 16;

fn tlb64_base() -> i32 { std::ptr::addr_of!(crate::cpu::cpu::tlb64) as u32 as i32 }
fn tlb64_generation_addr() -> i32 {
    std::ptr::addr_of!(crate::cpu::cpu::tlb64_generation) as u32 as i32
}
fn mem8_static_addr() -> i32 {
    std::ptr::addr_of!(crate::cpu::memory::mem8) as u32 as i32
}

// Leaves a local holding &tlb64[(vaddr >> 12) & mask].
fn gen_tlb64_entry(b: &mut WasmBuilder, address: &WasmLocalI64) -> WasmLocal {
    use crate::cpu::cpu::*;
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

// Leaves an i32 (0/1) on the stack: whether this read can be served inline.
fn gen_memory_read_fast_condition(b: &mut WasmBuilder, address: &WasmLocalI64, width: u8) {
    use crate::cpu::cpu::*;
    let bytes = (width / 8) as i32;
    let entry = gen_tlb64_entry(b, address);

    // generation matches
    b.get_local(&entry);
    b.load_aligned_i32(TLB64_OFF_GENERATION);
    b.const_i32(tlb64_generation_addr());
    b.load_aligned_i32(0);
    b.eq_i32();
    // tag matches vaddr & !0xFFF
    b.get_local(&entry);
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
    b.get_local(&entry);
    b.load_u8(TLB64_OFF_FLAGS);
    b.const_i32(TLB64_USER as i32);
    b.and_i32();
    b.const_i32(0);
    b.ne_i32();
    b.or_i32();
    b.and_i32();
    // phys is plain RAM: below memory_size and outside the 0xA0000..0xC0000 hole
    gen_phys_is_ram(b, &entry, address);
    b.and_i32();

    b.free_local(entry);
}

// Leaves an i32 (0/1) on the stack: whether this write can be served inline.
fn gen_memory_write_fast_condition(b: &mut WasmBuilder, address: &WasmLocalI64, width: u8) {
    use crate::cpu::cpu::*;
    let bytes = (width / 8) as i32;
    let entry = gen_tlb64_entry(b, address);

    b.get_local(&entry);
    b.load_aligned_i32(TLB64_OFF_GENERATION);
    b.const_i32(tlb64_generation_addr());
    b.load_aligned_i32(0);
    b.eq_i32();
    b.get_local(&entry);
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
    b.get_local(&entry);
    b.load_u8(TLB64_OFF_FLAGS);
    b.const_i32(TLB64_USER as i32);
    b.and_i32();
    b.const_i32(0);
    b.ne_i32();
    b.or_i32();
    b.and_i32();
    // writable = (flags & WRITABLE) || (!user && CR0.WP == 0)
    b.get_local(&entry);
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
    b.get_local(&entry);
    b.load_u8(TLB64_OFF_FLAGS);
    b.const_i32(TLB64_HAS_CODE as i32);
    b.and_i32();
    b.const_i32(0);
    b.eq_i32();
    b.and_i32();
    gen_phys_is_ram(b, &entry, address);
    b.and_i32();

    b.free_local(entry);
}

// Leaves an i32 (0/1): the physical address is plain RAM (not MMIO).
fn gen_phys_is_ram(b: &mut WasmBuilder, entry: &WasmLocal, address: &WasmLocalI64) {
    use crate::cpu::cpu::*;
    b.get_local(entry);
    b.load_aligned_i32(TLB64_OFF_PHYS_PAGE);
    b.get_local_i64(address);
    b.wrap_i64_to_i32();
    b.const_i32(0xFFF);
    b.and_i32();
    b.or_i32();
    let phys = b.set_new_local();
    b.get_local(&phys);
    b.const_i32(0xA0000);
    b.ltu_i32();
    b.get_local(&phys);
    b.const_i32(0xC0000);
    b.geu_i32();
    b.get_local(&phys);
    b.const_i32(MEMORY_SIZE_ADDR);
    b.load_aligned_i32(0);
    b.ltu_i32();
    b.and_i32();
    b.or_i32();
    b.free_local(phys);
}

// Leaves the linear address (i32) of the access on the stack.
fn gen_inline_address(b: &mut WasmBuilder, address: &WasmLocalI64) {
    use crate::cpu::cpu::*;
    let entry = gen_tlb64_entry(b, address);
    b.get_local(&entry);
    b.load_aligned_i32(TLB64_OFF_PHYS_PAGE);
    b.get_local_i64(address);
    b.wrap_i64_to_i32();
    b.const_i32(0xFFF);
    b.and_i32();
    b.or_i32();
    b.const_i32(mem8_static_addr());
    b.load_aligned_i32(0);
    b.add_i32();
    b.free_local(entry);
}

fn gen_inline_load(b: &mut WasmBuilder, address: &WasmLocalI64, width: u8) {
    gen_inline_address(b, address);
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

fn gen_inline_store(b: &mut WasmBuilder, address: &WasmLocalI64, value: &WasmLocalI64, width: u8) {
    gen_inline_address(b, address);
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
        gen_memory_read_fast_condition(b, address, width);
        b.if_i64();
        gen_inline_load(b, address, width);
        b.else_();
        b.get_local_i64(address);
        b.const_i32(width as i32);
        b.call_fn2_i64_i32_ret_i64("jit64_mem_read");
        gen_check_memory_fault(b, locals);
        b.block_end();
        return b.set_new_local_i64();
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
    if unsafe { JIT64_INLINE_MEMORY && JIT64_INLINE_MEMORY_WRITE } && matches!(width, 8 | 16 | 32 | 64)
    {
        gen_memory_write_fast_condition(b, address, width);
        b.if_void();
        // A plain RAM write has no side effects, so no register flush is needed.
        gen_inline_store(b, address, value, width);
        b.else_();
        gen_memory_write_slow(b, locals, address, value, width);
        b.block_end();
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
    // An MMIO write (APIC/IOAPIC) can deliver an interrupt synchronously, and
    // the delivery pushes a frame using the register file — including RSP — so
    // flush the resident registers first, exactly like the CPU would have them.
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
    }
    else {
        b.get_local_i64(address);
    }
    b.const_i32(64);
    b.call_fn2_i64_i32_ret_i64("jit64_mem_read");
    let value = b.set_new_local_i64();
    gen_check_memory_fault(b, locals);
    value
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
    }
    else {
        b.get_local_i64(address);
    }
    b.get_local_i64(value);
    b.const_i32(64);
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
        if *r >= 16 {
            // High-byte temporaries (and their base is written back separately).
            continue;
        }
        store_reg(b, local, *r);
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
fn compile_block_with_rips(instrs: &[Instr], rips: &[u64], block_end: u64) -> Vec<u8> {
    let mut b = WasmBuilder::new();
    let mut locals: Vec<(u8, WasmLocalI64)> = Vec::new();
    let mut terminated = false;

    // The interpreter keeps the arithmetic flags lazily computed
    // (`flags_changed`, `last_op1`, ...); this block reads the raw flags word.
    // Materialise them first, otherwise a block entered right after an
    // interpreted instruction tests stale flags and mis-branches.
    b.call_fn0("jit64_sync_flags");
    b.call_fn0("jit64_clear_exception_flag");

    for (index, instr) in instrs.iter().enumerate() {
        if let Some(instruction_rip) = rips.get(index) {
            b.const_i32(PREVIOUS_RIP_ADDR);
            b.const_i64(*instruction_rip as i64);
            b.store_aligned_i64(0);
            // *rip is the address an interrupt taken during this instruction
            // resumes at. Such an interrupt is delivered from inside a helper,
            // i.e. after the instruction's effects are applied, so it must point
            // at the *next* instruction (a stale block-start value re-executed
            // the whole block; pointing at the current instruction double-ran
            // its store/out/inc).
            let resume = rips.get(index + 1).copied().unwrap_or(block_end);
            b.const_i32(IP_ADDR);
            b.const_i64(resume as i64);
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
            Instr::CmpxchgReg { dst, src, width } => {
                let di = load_reg(&mut b, &mut locals, dst);
                let ai = load_reg(&mut b, &mut locals, 0);
                b.get_local_i64(&locals[di].1);
                emit_mask(&mut b, width);
                b.get_local_i64(&locals[ai].1);
                emit_mask(&mut b, width);
                b.eq_i64();
                let zf = b.set_new_local();
                b.const_i32(FLAGS_ADDR);
                b.const_i32(FLAGS_ADDR);
                b.load_aligned_i32(0);
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
            Instr::CmpxchgMem { mem, src, width } => {
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
                b.const_i32(FLAGS_ADDR);
                b.load_aligned_i32(0);
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
                write_back_high_byte(&mut b, &mut locals, r, ri);
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
                write_back_high_byte(&mut b, &mut locals, r, ri);
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
            Instr::CmovRegMem { code, dst, mem, width } => {
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
            Instr::BitTestReg { r, index_r, op, width } => {
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
            Instr::BitTestImm { r, index, op, width } => {
                let ri = load_reg(&mut b, &mut locals, r);
                b.get_local_i64(&locals[ri].1);
                b.const_i64(index as i64);
                b.const_i32(op as i32 | (width as i32) << 8);
                b.call_fn3_i64_i64_i32_ret_i64("jit64_bt");
                let value = b.set_new_local_i64();
                write_reg_value(&mut b, &mut locals, r, &value, width);
                b.free_local_i64(value);
            },
            Instr::In { imm, width } => {
                // The helper can deliver a #GP itself (unprivileged port I/O),
                // which must end the block; flush first so the delivery sees the
                // up-to-date registers.
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
            Instr::Out { imm, width } => {
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
            Instr::Cli => {
                b.call_fn0("jit64_cli");
            },
            Instr::PushFlags => {
                // push64 touches rsp, so flush the resident registers.
                emit_registers_back(&mut b, &locals);
                for (_, local) in &locals {
                    b.free_local_i64(local.unsafe_clone());
                }
                locals.clear();
                b.call_fn0("jit64_pushfq");
            },
            Instr::XmmLoad { dst, mem } => {
                let address = gen_memory_address_local(&mut b, &mut locals, &mem);
                let lo = gen_xmm_mem_half_read(&mut b, &locals, &address, 0);
                let hi = gen_xmm_mem_half_read(&mut b, &locals, &address, 8);
                gen_xmm_store_half(&mut b, dst, &lo, 0);
                gen_xmm_store_half(&mut b, dst, &hi, 8);
                b.free_local_i64(lo);
                b.free_local_i64(hi);
                b.free_local_i64(address);
            },
            Instr::XmmStore { mem, src } => {
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
            Instr::XmmCopy { dst, src } => {
                gen_xmm_reg_load(&mut b, src, 0);
                let lo = b.set_new_local_i64();
                gen_xmm_reg_load(&mut b, src, 8);
                let hi = b.set_new_local_i64();
                gen_xmm_store_half(&mut b, dst, &lo, 0);
                gen_xmm_store_half(&mut b, dst, &hi, 8);
                b.free_local_i64(lo);
                b.free_local_i64(hi);
            },
            Instr::XmmXor { dst, src } => {
                for offset in [0u32, 8] {
                    b.const_i32(crate::cpu::global_pointers::get_reg_xmm_addr(dst) as i32);
                    gen_xmm_reg_load(&mut b, dst, offset);
                    gen_xmm_reg_load(&mut b, src, offset);
                    b.xor_i64();
                    b.store_aligned_i64(offset);
                }
            },
            Instr::XmmXorMem { dst, mem } => {
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
            Instr::Rdtsc => {
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
                write_back_high_byte(&mut b, &mut locals, dst, di);
            },
            Instr::AddRegImm { r, value, width } => {
                let di = load_reg(&mut b, &mut locals, r);
                b.const_i64(value as i64);
                let src = b.set_new_local_i64();
                emit_arith(&mut b, &locals[di].1, &src, ArithOp::Add, width);
                write_back_high_byte(&mut b, &mut locals, r, di);
                b.free_local_i64(src);
            },
            Instr::ArithRegReg { op, dst, src, width } => {
                let di = load_reg(&mut b, &mut locals, dst);
                let si = load_reg(&mut b, &mut locals, src);
                emit_arith(&mut b, &locals[di].1, &locals[si].1, op, width);
                if op.writes_result() {
                    write_back_high_byte(&mut b, &mut locals, dst, di);
                }
            },
            Instr::ArithRegImm { op, r, value, width } => {
                let di = load_reg(&mut b, &mut locals, r);
                b.const_i64(value as i64);
                let src = b.set_new_local_i64();
                emit_arith(&mut b, &locals[di].1, &src, op, width);
                if op.writes_result() {
                    write_back_high_byte(&mut b, &mut locals, r, di);
                }
                b.free_local_i64(src);
            },
            Instr::ArithRegMem { op, dst, mem, width } => {
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
                // Publish registers before entering the halted state.
                emit_registers_back(&mut b, &locals);
                b.call_fn0("jit64_hlt");
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
// Stable instruction-class bits for whole-block interpreter fallback.
fn bail_class(instr: &Instr) -> u64 {
    match instr {
        Instr::MovRegImm { .. } | Instr::MovRegReg { .. } | Instr::MovRegMem { .. }
        | Instr::MovMemReg { .. } | Instr::MovMemImm { .. } | Instr::MovExtendReg { .. }
        | Instr::MovExtendMem { .. } => 1 << 0,
        Instr::XchgRegReg { .. } | Instr::XchgMemReg { .. } => 1 << 1,
        Instr::NotReg { .. } | Instr::NotMem { .. } | Instr::NegReg { .. } | Instr::NegMem { .. } => 1 << 2,
        Instr::IncDecReg { .. } | Instr::IncDecMem { .. } => 1 << 3,
        Instr::CmovRegReg { .. } | Instr::CmovRegMem { .. } => 1 << 4,
        Instr::SetccReg { .. } | Instr::SetccMem { .. } => 1 << 5,
        Instr::ShiftReg { .. } | Instr::ShiftRegCl { .. } | Instr::ShiftMem { .. }
        | Instr::ShiftMemCl { .. } => 1 << 6,
        Instr::Cpuid => 1 << 7,
        Instr::Rdtsc => 1 << 8,
        Instr::XmmCopy { .. } | Instr::XmmLoad { .. } | Instr::XmmStore { .. }
        | Instr::XmmXor { .. } | Instr::XmmXorMem { .. } => 1 << 9,
        Instr::In { .. } | Instr::Out { .. } => 1 << 10,
        Instr::Cli => 1 << 11,
        Instr::PushFlags => 1 << 12,
        Instr::CmpxchgReg { .. } | Instr::CmpxchgMem { .. } => 1 << 13,
        Instr::BitTestReg { .. } | Instr::BitTestImm { .. } => 1 << 14,
        Instr::ImulRegReg { .. } | Instr::ImulRegImm { .. } | Instr::ImulRegMem { .. } => 1 << 15,
        Instr::Bswap { .. } => 1 << 16,
        Instr::Lea { .. } => 1 << 17,
        Instr::AddRegReg { .. } | Instr::AddRegImm { .. } => 1 << 18,
        Instr::ArithRegReg { .. } | Instr::ArithRegImm { .. } | Instr::ArithRegMem { .. }
        | Instr::ArithMemReg { .. } | Instr::ArithMemImm { .. } => 1 << 19,
        Instr::PushReg { .. } | Instr::PopReg { .. } | Instr::PushImm { .. } => 1 << 20,
        Instr::Ret { .. } => 1 << 23,
        Instr::Jcc { .. } => 1 << 24,
        Instr::Jmp { .. } | Instr::JmpReg { .. } | Instr::JmpMem { .. } => 1 << 25,
        Instr::Call { .. } | Instr::CallReg { .. } | Instr::CallMem { .. } => 1 << 26,
        Instr::Leave => 1 << 27,
        Instr::Nop => 1 << 22,
        Instr::Hlt => 1 << 28,
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

// Reserved for the 32-bit JIT: it asserts when the shared table runs out.
const JIT64_MAX_BLOCKS: usize = 3000;

// Evict a quarter of the cap per trim.
const JIT64_EVICT_BATCH: usize = JIT64_MAX_BLOCKS / 4;

const COMPILE_BUF_SIZE: usize = 65536;
static mut COMPILE_BUF: [u8; COMPILE_BUF_SIZE] = [0; COMPILE_BUF_SIZE];
static mut JIT64_MEMORY_FAULT: u8 = 0;
// Lets tests and benchmarks run the same code with the JIT off.
static mut JIT64_ENABLED: bool = true;
// Inline RAM loads/stores in generated code (see gen_memory_read).
static mut JIT64_INLINE_MEMORY: bool = true;
// Separate switch for inline stores (self-modifying code makes them riskier).
static mut JIT64_INLINE_MEMORY_WRITE: bool = true;
// SSE2 codegen remains opt-in while boot validation is incomplete.
static mut JIT64_SSE: bool = false;
// Dispatch counters (diagnostics; see jit64_stat).
static mut JIT64_RUN_HITS: u64 = 0;
static mut JIT64_RUN_MISSES: u64 = 0;
static mut JIT64_INTERP: u64 = 0;
static mut JIT64_COMPILES: u64 = 0;

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

// Set when an exception was delivered synchronously from inside a compiled
// block (e.g. a #GP from an unprivileged port access). The block must stop
// immediately: the delivery changed *rip and pushed a frame on the stack, so
// letting the rest of the block run (and write its registers back) would
// discard the exception and leak the frame.
static mut EXCEPTION_IN_BLOCK: bool = false;

pub unsafe fn note_exception_in_block() {
    EXCEPTION_IN_BLOCK = true;
}

#[no_mangle]
pub unsafe fn jit64_hlt() {
    *crate::cpu::global_pointers::in_hlt = true;
    // Parity with the interpreter's hlt: when interrupts are enabled the main
    // loop runs the timers and delivers a due IRQ (so don't do it here, it is
    // far too hot); when they are not, execution can never resume and the host
    // has to be told, exactly like instr_F4 does.
    if *crate::cpu::global_pointers::flags & crate::cpu::cpu::FLAG_INTERRUPT == 0 {
        crate::cpu::cpu::cpu_event_halt();
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

#[no_mangle]
pub unsafe fn jit64_in_block() -> u32 { IN_JIT_BLOCK as u32 }

static mut JIT64_COMPILE_FAILS: u64 = 0;
static mut JIT64_FAULTS: u64 = 0;
static mut HOTNESS: *mut HashMap<u64, u32> = std::ptr::null_mut();
// Block-cap trim request; run from try_run, outside hotness borrows.
static mut EVICT_PENDING: bool = false;
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
    // Second-chance bit for the clock eviction below.
    used: bool,
}

static mut BLOCKS: *mut HashMap<u64, BlockInfo> = std::ptr::null_mut();
// physical code page -> guest RIPs compiled from it
static mut CODE_PAGES: *mut HashMap<u32, HashSet<u64>> = std::ptr::null_mut();
// virtual code page -> guest RIPs compiled from it (INVLPG invalidation)
static mut VIRT_PAGES: *mut HashMap<u64, HashSet<u64>> = std::ptr::null_mut();

unsafe fn user_access() -> bool { *crate::cpu::global_pointers::cpl == 3 }

unsafe fn hotness() -> &'static mut HashMap<u64, u32> {
    if HOTNESS.is_null() {
        HOTNESS = Box::into_raw(Box::new(HashMap::new()));
    }
    &mut *HOTNESS
}

unsafe fn blocks() -> &'static mut HashMap<u64, BlockInfo> {
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

unsafe fn virt_pages() -> &'static mut HashMap<u64, HashSet<u64>> {
    if VIRT_PAGES.is_null() {
        VIRT_PAGES = Box::into_raw(Box::new(HashMap::new()));
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
            Some(_) => forget(rip),
        }
        freed += 1;
    }
}

// Drop one block; it can be compiled again once hot.
unsafe fn forget(rip: u64) {
    if let Some(info) = blocks().remove(&rip) {
        crate::jit::jit64_free_table_index(info.index);
    }
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
    if blocks().len() >= JIT64_MAX_BLOCKS {
        EVICT_PENDING = true;
    }
    JIT64_COMPILES += 1;
    // Hotness is recorded after interpretation, which may have switched CR3
    // or delivered an exception. Compilation is only a probe of the old RIP:
    // missing mappings must not inject a second guest exception.
    let phys_page = match crate::cpu::cpu::translate_address_64_no_side_effects(rip) {
        Ok(phys) => phys >> 12,
        Err(()) => return,
    };
    let page_end = (rip | 0xFFF) + 1;
    let mut bytes = Vec::new();
    let mut addr = rip;
    while addr < page_end {
        match crate::cpu::cpu::translate_address_64_no_side_effects(addr) {
            Ok(phys) => bytes.push(crate::cpu::memory::read8(phys) as u8),
            Err(()) => break,
        }
        addr += 1;
    }

    let module = match compile_bytes(rip, &bytes) {
        Ok(module) => module,
        Err(e) =>
        {
            JIT64_COMPILE_FAILS += 1;
            if LOGGED_FAILS.fetch_add(1, Ordering::Relaxed) < 40
            {
                dbg_log!("jit64: cannot compile 0x{:x}: {}", rip, e);
            }
            return;
        },
    };
    if module.len() > COMPILE_BUF_SIZE {
        return;
    }
    let index = match crate::jit::jit64_allocate_table_index() {
        Some(index) => index,
        None => {
            // No slot free: re-arm hotness so the block is retried later.
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
    blocks().insert(rip, BlockInfo { index, phys_page, used: false });
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
    crate::cpu::cpu::tlb_set_has_code(crate::page::Page::page_of(phys_page << 12), true);
    crate::cpu::cpu::tlb64_mark_code(phys_page, true);
}

// Physical page backing this guest page (one-entry cache).
unsafe fn phys_page_of(rip: u64) -> Option<u32> {
    let virt_page = rip >> 12;
    if virt_page == LAST_VALID_PAGE {
        return Some(LAST_VALID_PHYS);
    }
    // Between instructions: a faulting walk here would corrupt guest state.
    match crate::cpu::cpu::translate_address_64_no_side_effects(rip) {
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
    if !JIT64_ENABLED {
        return false;
    }
    if EVICT_PENDING {
        EVICT_PENDING = false;
        evict_some(JIT64_EVICT_BATCH);
    }
    let info = match blocks().get_mut(&rip) {
        Some(info) =>
        {
            info.used = true;
            *info
        },
        None =>
        {
            JIT64_RUN_MISSES += 1;
            return false;
        },
    };
    // Stale block (its RIP maps elsewhere now): drop it.
    if phys_page_of(rip) != Some(info.phys_page) {
        forget(rip);
        JIT64_RUN_MISSES += 1;
        return false;
    }
    JIT64_RUN_HITS += 1;
    let indirect = info.index as i32 + crate::cpu::cpu::WASM_TABLE_OFFSET as i32;
    IN_JIT_BLOCK = true;
    wasm::call_indirect1(indirect, 0);
    IN_JIT_BLOCK = false;
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
    JIT64_INTERP += 1;
    // End the hotness borrow before compile_and_register(): it touches the map.
    let should_compile = {
        let count = hotness().entry(rip).or_insert(0);
        *count += 1;
        // Compile once; retrying an undecodable block is expensive.
        *count == JIT64_THRESHOLD
    };
    if should_compile {
        compile_and_register(rip);
    }
}

pub unsafe fn clear_cache() {
    if !HOTNESS.is_null() {
        hotness().clear();
    }
    if !BLOCKS.is_null() {
        let indices: Vec<u16> = blocks().drain().map(|(_, info)| info.index).collect();
        for index in indices {
            crate::jit::jit64_free_table_index(index);
        }
    }
    if !CODE_PAGES.is_null() {
        code_pages().clear();
    }
    if !VIRT_PAGES.is_null() {
        virt_pages().clear();
    }
    crate::cpu::cpu::tlb64_clear_has_code();
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
        forget(rip);
    }
    crate::cpu::cpu::tlb64_mark_code(page, false);
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
        forget(rip);
    }
}

#[no_mangle]
pub unsafe fn jit64_compiled_count() -> u32 { blocks().len() as u32 }

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
        _ => 0,
    }
}

// Enable or disable block compilation and dispatch (used by tests/benchmarks).
#[no_mangle]
pub unsafe fn jit64_set_enabled(enabled: u32) { JIT64_ENABLED = enabled != 0; }

#[no_mangle]
pub unsafe fn jit64_set_inline_memory(enabled: u32) { JIT64_INLINE_MEMORY = enabled != 0; }

#[no_mangle]
pub unsafe fn jit64_set_inline_write(enabled: u32) { JIT64_INLINE_MEMORY_WRITE = enabled != 0; }

#[no_mangle]
pub unsafe fn jit64_set_sse(enabled: u32) { JIT64_SSE = enabled != 0; }

#[no_mangle]
pub unsafe fn jit64_clear_cache() { clear_cache(); }

// u64::MAX means translation faulted; the block returns immediately.
// Debuggers must not use jit64_translate: its failure delivers a guest #PF.
#[no_mangle]
pub unsafe fn jit64_debug_translate(vaddr: u64) -> u64 {
    match crate::cpu::cpu::translate_address_64_no_side_effects(vaddr) {
        Ok(phys) => crate::cpu::memory::mem8 as u64 + phys as u64,
        Err(()) => u64::MAX,
    }
}

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
    JIT64_FAULTS += 1;
    JIT64_MEMORY_FAULT = 0;
    *crate::cpu::global_pointers::rip = *crate::cpu::global_pointers::previous_rip;
    *crate::cpu::global_pointers::instruction_pointer = *crate::cpu::global_pointers::rip as i32;
    crate::cpu::cpu::exit_jit64();
}

// in/out. A failed translation sets the fault flag so the block rewinds.
#[no_mangle]
pub unsafe fn jit64_in(port: u64, width: u32) -> u64 {
    let port = port as i32;
    // `width` is in bits, as everywhere else here (the decode uses
    // operand_width). Matching on bytes silently turned `in al, dx` into a
    // 32-bit read.
    if !crate::cpu::cpu::test_privileges_for_io(port, (width / 8) as i32) {
        return 0;
    }
    match width {
        8 => crate::cpu::cpu::io_port_read8(port) as u32 as u64,
        16 => crate::cpu::cpu::io_port_read16(port) as u16 as u64,
        _ => crate::cpu::cpu::io_port_read32(port) as u32 as u64,
    }
}

#[no_mangle]
pub unsafe fn jit64_out(port: u64, value: u64, width: u32) {
    let port = port as i32;
    if !crate::cpu::cpu::test_privileges_for_io(port, (width / 8) as i32) {
        return;
    }
    match width {
        8 => crate::cpu::cpu::io_port_write8(port, value as i32),
        16 => crate::cpu::cpu::io_port_write16(port, value as i32),
        _ => crate::cpu::cpu::io_port_write32(port, value as i32),
    }
}

// Materialise the interpreter's pending lazy flags. No-op if none are pending.
#[no_mangle]
pub unsafe fn jit64_sync_flags() {
    if *crate::cpu::global_pointers::flags_changed != 0 {
        *crate::cpu::global_pointers::flags = crate::cpu::cpu::get_eflags();
        *crate::cpu::global_pointers::flags_changed = 0;
    }
}

#[no_mangle]
pub unsafe fn jit64_cli() {
    *crate::cpu::global_pointers::flags &= !crate::cpu::cpu::FLAG_INTERRUPT;
    *crate::cpu::global_pointers::flags_changed = 0;
}

#[no_mangle]
pub unsafe fn jit64_pushfq() {
    let rsp = crate::cpu::cpu::read_reg64(4).wrapping_sub(8);
    let phys = match crate::cpu::cpu::translate_address_64_jit(rsp, true, false) {
        Ok(phys) => phys,
        Err(()) => {
            JIT64_MEMORY_FAULT = 1;
            return;
        },
    };
    crate::cpu::memory::write32(phys, *crate::cpu::global_pointers::flags);
    crate::cpu::memory::write32(phys + 4, 0);
    crate::cpu::cpu::write_reg64(4, rsp);
}

// rdtsc: the value the interpreter's RDTSC would return.
#[no_mangle]
pub unsafe fn jit64_rdtsc() -> u64 { crate::cpu::cpu::read_tsc() }

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
            mem: Mem { base: None, index: None, scale: 0, addr_size: 64, disp: 0x1100, segment: None },
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
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 8, segment: None };
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
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 8, segment: None };
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
                mem: Mem { base: None, index: None, scale: 0, addr_size: 64, disp: 0x1100, segment: None },
                return_address: 0x1006,
            },
        ]);
        assert_eq!(decode_block(0, &[0xFF, 0x64, 0x24, 0x08]).unwrap(), vec![
            Instr::JmpMem {
                mem: Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 8, segment: None },
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
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 24, segment: None };
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
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 16, segment: None };
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
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 16, segment: None };
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
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 16, segment: None };
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
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 16, segment: None };
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
        let mem = Mem { base: Some(4), index: None, scale: 0, addr_size: 64, disp: 16, segment: None };
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
        let mem = Mem { base: Some(0), index: None, scale: 0, addr_size: 64, disp: 0, segment: None };
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
                mem: Mem { base: Some(3), index: None, scale: 0, disp: 0, addr_size: 32, segment: None },
                width: 64,
            },
            Instr::MovRegMem {
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
                mem: Mem { base: Some(0), index: None, scale: 0, disp: 0, addr_size: 64, segment: None },
                width: 64,
                count: 1,
            },
            Instr::ShiftMem {
                kind: ShiftKind::Shl,
                mem: Mem { base: Some(0), index: None, scale: 0, disp: 0, addr_size: 64, segment: None },
                width: 64,
                count: 5,
            },
            Instr::ShiftMemCl {
                kind: ShiftKind::Shl,
                mem: Mem { base: Some(0), index: None, scale: 0, disp: 0, addr_size: 64, segment: None },
                width: 64,
            },
            Instr::ShiftMem {
                kind: ShiftKind::Shl,
                mem: Mem { base: Some(0), index: None, scale: 0, disp: 0, addr_size: 64, segment: None },
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
