//! Long-mode 0F, SSE and AVX dispatch for `interp64` (the `interp64_0f` child
//! module). Shares the long-mode core (decode, memory, registers, flags) with
//! the base opcodes via `use super::*`.

#![allow(dead_code, unused_imports)]

use crate::cpu::core::{
    io_port_read8, io_port_write8, read_reg64, read_tsc, reg128, test_privileges_for_io, translate_address_64,
    write_reg64, APIC_MEM_ADDRESS, CS, FLAG_CARRY, FLAG_INTERRUPT, SS,
};
use crate::cpu::global_pointers::*;
use crate::cpu::interp::misc_instr::sign_extend;
use crate::cpu::interp::instructions_0f::XSAVE_YMM_OFFSET;
use crate::cpu::decode::op_size::OpSize;
use crate::cpu::interp::fpu::{
    f32_to_f80, f64_to_f80, f80_to_f32, f80_to_f64, fpu_convert_to_i16, fpu_convert_to_i32,
    fpu_convert_to_i64, fpu_fadd, fpu_fcom, fpu_fcomp, fpu_fdiv, fpu_fdivr, fpu_fmul, fpu_pop,
    fpu_push, fpu_fsub, fpu_fsubr, fpu_get_st0, fpu_truncate_to_i16, fpu_truncate_to_i32,
    fpu_finit, fpu_load_status_word, fpu_load_tag_word, fpu_set_status_word, fpu_set_tag_word,
    fpu_truncate_to_i64, i32_to_f80, i64_to_f80, set_control_word,
};
use crate::cpu::interp::softfloat::F80;
use crate::memory;
use crate::cpu::jit;
use crate::cpu::interp::decode_cache::{ArithOp as JArithOp, Instruction as JInstr, Mem as JMem};
use crate::memory::page::Page;
use crate::paging::OrPageFault;
use crate::cpu::decode::prefix;
use crate::profiler::stat;

// General-purpose register indices
const RAX: u8 = 0;
const RCX: u8 = 1;
const RDX: u8 = 2;
const RBX: u8 = 3;
const RSP: u8 = 4;
const RBP: u8 = 5;
const RSI: u8 = 6;
const RDI: u8 = 7;
const R11: u8 = 11;

// long mode MSRs
const MSR_EFER: i32 = 0xC0000080u32 as i32;
const MSR_STAR: i32 = 0xC0000081u32 as i32;
const MSR_LSTAR: i32 = 0xC0000082u32 as i32;
const MSR_SFMASK: i32 = 0xC0000084u32 as i32;
const MSR_FS_BASE: i32 = 0xC0000100u32 as i32;
const MSR_TSC_DEADLINE: i32 = 0x6E0;
const MSR_GS_BASE: i32 = 0xC0000101u32 as i32;
const MSR_KERNEL_GS_BASE: i32 = 0xC0000102u32 as i32;

const EFER_SCE: u64 = 1;
const EFER_LME: u64 = 1 << 8;
const EFER_LMA: u64 = 1 << 10;


use super::*;

pub(crate) unsafe fn xmm_get(r: u8) -> reg128 { *xmm_ptr(r as i32) }
pub(crate) unsafe fn xmm_set(r: u8, value: reg128) { *xmm_ptr(r as i32) = value; }

pub(crate) unsafe fn mem_read128(address: u64) -> OrPageFault<reg128> {
    let low = mem_read(address, OpSize::S64)?;
    let high = mem_read(address + 8, OpSize::S64)?;
    Ok(reg128 { u64: [low, high] })
}

pub(crate) unsafe fn mem_write128(address: u64, value: reg128) -> OrPageFault<()> {
    mem_write(address, OpSize::S64, value.u64[0])?;
    mem_write(address + 8, OpSize::S64, value.u64[1])?;
    Ok(())
}

pub(crate) unsafe fn sse_read(operand: Operand) -> OrPageFault<reg128> {
    match operand {
        Operand::Reg(r) => Ok(xmm_get(r)),
        Operand::Mem(address) => mem_read128(address),
    }
}


// JIT helper: apply an integer SSE op to xmm[dst]; source as two u64 halves.
#[no_mangle]
pub unsafe fn jit64_sse_int(op: i32, src_lo: u64, src_hi: u64, dst: i32) -> i32 {
    let src = reg128 { u64: [src_lo, src_hi] };
    let current = xmm_get(dst as u8);
    match crate::cpu::interp::simd_instr::int_apply(op as u8, current, src) {
        Some(result) => xmm_set(dst as u8, result),
        None => crate::cpu::core::trigger_ud(),
    }
    0
}

// JIT helper: packed shift-by-imm on xmm[dst]; `encoded` = opcode|group|imm8.
#[no_mangle]
pub unsafe fn jit64_sse_shift_imm(encoded: i32, lo: u64, hi: u64, dst: i32) -> i32 {
    let op = (encoded & 0xFF) as u8;
    let group = (encoded >> 8 & 0xFF) as u8;
    let count = (encoded >> 16 & 0xFF) as u64;
    let value = reg128 { u64: [lo, hi] };
    match crate::cpu::interp::simd_instr::shift_imm_apply(op, group, count, value) {
        Some(result) => xmm_set(dst as u8, result),
        None => crate::cpu::core::trigger_ud(),
    }
    0
}

pub(crate) unsafe fn sse_int(opcode: u8, pfx: &Prefixes) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let src = sse_read(operand)?;
    let dst = xmm_get(modrm.reg);
    match crate::cpu::interp::simd_instr::int_apply(opcode, dst, src) {
        Some(result) => xmm_set(modrm.reg, result),
        None => crate::cpu::core::trigger_ud(),
    }
    Ok(())
}

// kind: 0 logical right, 1 arithmetic right, 2 left; the count is per lane.


pub(crate) unsafe fn sse_shift_imm(opcode: u8, pfx: &Prefixes) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let count = fetch8()? as u64;
    let group = modrm.reg & 7;
    let dst = sse_read(operand)?;
    match crate::cpu::interp::simd_instr::shift_imm_apply(opcode, group, count, dst) {
        Some(result) => sse_shift_imm_store(operand, result),
        None =>
        {
            crate::cpu::core::trigger_ud();
            Ok(())
        },
    }
}

pub(crate) unsafe fn sse_shift_imm_store(operand: Operand, value: reg128) -> OrPageFault<()> {
    match operand {
        Operand::Reg(r) => xmm_set(r, value),
        Operand::Mem(address) => mem_write128(address, value)?,
    }
    Ok(())
}

// CVTPS2PD/CVTPD2PS/CVTSS2SD/CVTSD2SS (0F 5A) and the CVT*PS/DQ (0F 5B) forms.
pub(crate) unsafe fn sse_convert(width: u8, pfx: &Prefixes) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let src = sse_read(operand)?;
    let mut dst = xmm_get(modrm.reg);
    if width == 0x5A {
        if pfx.f3 {
            dst.f64[0] = src.f32[0] as f64;
        }
        else if pfx.f2 {
            dst.f32[0] = src.f64[0] as f32;
        }
        else if pfx.p66 {
            for i in 0..2 {
                dst.f32[i] = src.f64[i] as f32;
            }
            dst.u64[1] = 0;
        }
        else {
            for i in 0..2 {
                dst.f64[i] = src.f32[i] as f64;
            }
        }
    }
    else if pfx.f3 {
        for i in 0..4 {
            dst.i32[i] = if src.f32[i].is_nan() { i32::MIN } else { src.f32[i].trunc() as i32 };
        }
    }
    else if pfx.p66 {
        for i in 0..4 {
            dst.i32[i] = src.f32[i] as i32;
        }
    }
    else {
        for i in 0..4 {
            dst.f32[i] = src.i32[i] as f32;
        }
    }
    xmm_set(modrm.reg, dst);
    Ok(())
}

// CVTSI2SS/SD (0F 2A), CVTTSS2SI/SD (0F 2C) and CVTSS2SI/SD (0F 2D).
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/cvtsi2ss
pub(crate) unsafe fn sse_cvt_int(opcode: u8, pfx: &Prefixes) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let mut dst = xmm_get(modrm.reg);
    if opcode == 0x2A {
        let source = read_operand(operand, size_from_rex(pfx), pfx.has_rex())?;
        if pfx.f2 {
            dst.f64[0] = if pfx.has_rex_w() { source as i64 as f64 } else { source as u32 as i32 as f64 };
        }
        else {
            dst.f32[0] = if pfx.has_rex_w() { source as i64 as f32 } else { source as u32 as i32 as f32 };
        }
        xmm_set(modrm.reg, dst);
    }
    else {
        let src = sse_read(operand)?;
        let truncate = opcode == 0x2C;
        // Widening f32 to f64 is exact, so one helper covers both element sizes.
        let x = if pfx.f2 { src.f64[0] } else { src.f32[0] as f64 };
        let bits = if pfx.has_rex_w() {
            crate::cpu::interp::sse_instr::cvt_float_to_i64(x, truncate)
        }
        else {
            crate::cpu::interp::sse_instr::cvt_float_to_i32(x, truncate) as u64
        };
        write_reg64(modrm.reg as i32, bits);
    }
    Ok(())
}

pub(crate) fn size_from_rex(pfx: &Prefixes) -> OpSize {
    if pfx.has_rex_w() { OpSize::S64 } else { OpSize::S32 }
}

// RSQRTPS/RSQRTSS (0F 52) and RCPPS/RCPSS (0F 53).
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/rsqrtps
pub(crate) unsafe fn sse_rsqrt_rcp(opcode: u8, pfx: &Prefixes) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let src = sse_read(operand)?;
    let mut dst = xmm_get(modrm.reg);
    let compute = |x: f32| if opcode == 0x52 { 1.0 / x.sqrt() } else { 1.0 / x };
    if pfx.f3 {
        dst.f32[0] = compute(src.f32[0]);
    }
    else {
        for i in 0..4 {
            dst.f32[i] = compute(src.f32[i]);
        }
    }
    xmm_set(modrm.reg, dst);
    Ok(())
}

// --- three-byte escapes (0F 38 / 0F 3A) ----------------------------------

pub(crate) unsafe fn set_strcmp_flags(r: crate::cpu::interp::simd_instr::StrCmp) {
    *flags &= !((FLAG_CF | FLAG_ZF | FLAG_SF | FLAG_OF | FLAG_AF | FLAG_PF) as i32);
    if r.intres != 0 {
        *flags |= FLAG_CF as i32;
    }
    if r.zf {
        *flags |= FLAG_ZF as i32;
    }
    if r.sf {
        *flags |= FLAG_SF as i32;
    }
    if r.of {
        *flags |= FLAG_OF as i32;
    }
    *flags_changed = 0;
}

// MOVBE (F0/F1), ADCX/ADOX (F6), CRC32 (F2 F0/F1) and the 66 0F 38 group:
// SSSE3, SSE4.1/4.2 and AES-NI.
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/pshufb
pub(crate) unsafe fn run_0f38(pfx: &Prefixes) -> OrPageFault<()> {
    let opcode = fetch8()?;

    // MOVBE r, m / MOVBE m, r (no 66/F2/F3)
    if !pfx.p66 && !pfx.f2 && !pfx.f3 && (opcode == 0xF0 || opcode == 0xF1) {
        let (modrm, operand) = decode_operand(pfx)?;
        let width = if pfx.has_rex_w() { OpSize::S64 } else { OpSize::S32 };
        let swap = |value: u64| match width {
            OpSize::S64 => value.swap_bytes(),
            _ => (value as u32).swap_bytes() as u64,
        };
        if opcode == 0xF0 {
            let Operand::Mem(address) = operand else {
                crate::cpu::core::trigger_ud();
                return Ok(());
            };
            let value = swap(mem_read(address, width)?);
            write_reg(modrm.reg, width, pfx.has_rex(), value);
        }
        else {
            let value = read_reg(modrm.reg, width, pfx.has_rex());
            probe_operand_write(operand, width)?;
            write_operand(operand, width, pfx.has_rex(), swap(value))?;
        }
        return Ok(());
    }

    // ADCX (66) / ADOX (F3): add with carry, updating only one flag
    if (pfx.p66 || pfx.f3) && opcode == 0xF6 {
        let (modrm, operand) = decode_operand(pfx)?;
        let width = if pfx.has_rex_w() { OpSize::S64 } else { OpSize::S32 };
        let source = read_operand(operand, width, pfx.has_rex())? & width.mask();
        let destination = read_reg(modrm.reg, width, pfx.has_rex()) & width.mask();
        let carry = if pfx.p66 {
            *flags as u32 & FLAG_CF != 0
        }
        else {
            *flags as u32 & FLAG_OF != 0
        };
        let (sum, first) = destination.overflowing_add(source);
        let (sum, second) = sum.overflowing_add(carry as u64);
        write_reg(modrm.reg, width, pfx.has_rex(), sum & width.mask());
        let carry_out = (first || second) as i32;
        if pfx.p66 {
            *flags = (*flags & !(FLAG_CF as i32)) | carry_out * FLAG_CF as i32;
        }
        else {
            *flags = (*flags & !(FLAG_OF as i32)) | carry_out * FLAG_OF as i32;
        }
        *flags_changed = 0;
        return Ok(());
    }

    // CRC32 r32, r/m8/16/32/64 (F2 0F 38 F0/F1)
    if pfx.f2 && (opcode == 0xF0 || opcode == 0xF1) {
        let (modrm, operand) = decode_operand(pfx)?;
        let (size, bytes) = if opcode == 0xF0 {
            (OpSize::S8, 1usize)
        }
        else if pfx.p66 {
            (OpSize::S16, 2)
        }
        else if pfx.has_rex_w() {
            (OpSize::S64, 8)
        }
        else {
            (OpSize::S32, 4)
        };
        let source = read_operand(operand, size, pfx.has_rex())?;
        let crc = read_reg(modrm.reg, OpSize::S32, pfx.has_rex()) as u32;
        let value = crate::cpu::interp::simd_instr::crc32(crc, source, bytes);
        write_reg(modrm.reg, OpSize::S32, pfx.has_rex(), value as u64);
        return Ok(());
    }

    if pfx.p66 {
        let (modrm, operand) = decode_operand(pfx)?;
        match opcode {
            // PBLENDVB / BLENDVPS / BLENDVPD use XMM0 as the mask.
            0x10 | 0x14 | 0x15 => {
                let src = sse_read(operand)?;
                let mask = xmm_get(0);
                let dst = xmm_get(modrm.reg);
                let element_bytes = match opcode { 0x10 => 1, 0x14 => 4, _ => 8 };
                xmm_set(modrm.reg, crate::cpu::interp::simd_instr::blendv_apply(dst, src, mask, element_bytes));
            },
            // PTEST: ZF when (dst & src) == 0, CF when (~dst & src) == 0.
            0x17 => {
                let src = sse_read(operand)?;
                let (and, andn) = crate::cpu::interp::simd_instr::ptest(xmm_get(modrm.reg), src);
                *flags &= !((FLAG_CF | FLAG_ZF) as i32);
                if and == 0 {
                    *flags |= FLAG_ZF as i32;
                }
                if andn == 0 {
                    *flags |= FLAG_CF as i32;
                }
                *flags_changed = 0;
            },
            // MOVNTDQA xmm, m128
            0x2A => {
                let Operand::Mem(address) = operand else {
                    crate::cpu::core::trigger_ud();
                    return Ok(());
                };
                xmm_set(modrm.reg, mem_read128(address)?);
            },
            // AES-NI
            0xDB => {
                let src = sse_read(operand)?;
                xmm_set(modrm.reg, crate::cpu::interp::simd_instr::aesimc(src));
            },
            0xDC => {
                let src = sse_read(operand)?;
                let dst = xmm_get(modrm.reg);
                xmm_set(modrm.reg, crate::cpu::interp::simd_instr::aesenc(dst, src, false));
            },
            0xDD => {
                let src = sse_read(operand)?;
                let dst = xmm_get(modrm.reg);
                xmm_set(modrm.reg, crate::cpu::interp::simd_instr::aesenc(dst, src, true));
            },
            0xDE => {
                let src = sse_read(operand)?;
                let dst = xmm_get(modrm.reg);
                xmm_set(modrm.reg, crate::cpu::interp::simd_instr::aesdec(dst, src, false));
            },
            0xDF => {
                let src = sse_read(operand)?;
                let dst = xmm_get(modrm.reg);
                xmm_set(modrm.reg, crate::cpu::interp::simd_instr::aesdec(dst, src, true));
            },
            _ => {
                let src = sse_read(operand)?;
                let dst = xmm_get(modrm.reg);
                if let Some(result) = crate::cpu::interp::simd_instr::sse4_38_apply(opcode, dst, src) {
                    xmm_set(modrm.reg, result);
                }
                else if let Some(result) = crate::cpu::interp::simd_instr::ssse3_apply(opcode, dst, src) {
                    xmm_set(modrm.reg, result);
                }
                else {
                    crate::cpu::core::trigger_ud();
                }
            },
        }
        return Ok(());
    }

    crate::cpu::core::trigger_ud();
    Ok(())
}

// PALIGNR, the SSE4.1 0F3A ops, the SSE4.2 string compares and
// AESKEYGENASSIST (66 0F 3A xx).
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/palignr
pub(crate) unsafe fn run_0f3a(pfx: &Prefixes) -> OrPageFault<()> {
    let opcode = fetch8()?;
    if !pfx.p66 {
        crate::cpu::core::trigger_ud();
        return Ok(());
    }
    let (modrm, operand) = decode_operand_with_trailing(pfx, 1)?;
    let dst = xmm_get(modrm.reg);
    match opcode {
        // ROUNDPS/PD/SS/SD
        0x08 | 0x09 | 0x0A | 0x0B => {
            let src = sse_read(operand)?;
            let imm = fetch8()? as u8;
            let mut result = dst;
            if opcode == 0x08 || opcode == 0x0A {
                let lanes = if opcode == 0x08 { 4 } else { 1 };
                for i in 0..lanes {
                    result.f32[i] = crate::cpu::interp::simd_instr::round_apply(src.f32[i] as f64, imm) as f32;
                }
            }
            else {
                let lanes = if opcode == 0x09 { 2 } else { 1 };
                for i in 0..lanes {
                    result.f64[i] = crate::cpu::interp::simd_instr::round_apply(src.f64[i], imm);
                }
            }
            xmm_set(modrm.reg, result);
        },
        // BLENDPS/BLENDPD/PBLENDW
        0x0C | 0x0D | 0x0E => {
            let src = sse_read(operand)?;
            let imm = fetch8()? as u8;
            let (lanes, bits) = match opcode {
                0x0C => (4, 32),
                0x0D => (2, 64),
                _ => (8, 16),
            };
            xmm_set(modrm.reg, crate::cpu::interp::simd_instr::blend_imm_apply(dst, src, imm, lanes, bits));
        },
        // PALIGNR
        0x0F => {
            let src = sse_read(operand)?;
            let shift = fetch8()? as usize & 0x1F;
            let mut temp = [0u8; 32];
            temp[..16].copy_from_slice(&src.u8);
            temp[16..].copy_from_slice(&dst.u8);
            let mut result = reg128 { u64: [0, 0] };
            for i in 0..16 {
                result.u8[i] = if shift + i < 32 { temp[shift + i] } else { 0 };
            }
            xmm_set(modrm.reg, result);
        },
        // PEXTRB/PEXTRW/PEXTRD/PEXTRQ
        0x14 | 0x15 | 0x16 => {
            let imm = fetch8()? as u8;
            let (size, value) = match opcode {
                0x14 => (OpSize::S8, dst.u8[imm as usize] as u64),
                0x15 => (OpSize::S16, dst.u16[imm as usize & 7] as u64),
                _ =>
                    if pfx.has_rex_w() {
                        (OpSize::S64, dst.u64[imm as usize & 1])
                    }
                    else {
                        (OpSize::S32, dst.u32[imm as usize & 3] as u64)
                    },
            };
            probe_operand_write(operand, size)?;
            write_operand(operand, size, pfx.has_rex(), value)?;
        },
        // EXTRACTPS
        0x17 => {
            let imm = fetch8()? as u8;
            probe_operand_write(operand, OpSize::S32)?;
            write_operand(
                operand,
                OpSize::S32,
                pfx.has_rex(),
                crate::cpu::interp::simd_instr::extractps(dst, imm) as u64,
            )?;
        },
        // PINSRB/PINSRD/PINSRQ and INSERTPS
        0x20 | 0x21 | 0x22 => {
            let imm = fetch8()? as u8;
            if opcode == 0x21 {
                let src = sse_read(operand)?;
                xmm_set(modrm.reg, crate::cpu::interp::simd_instr::insertps(dst, src, imm));
            }
            else {
                let size = match opcode {
                    0x20 => OpSize::S8,
                    _ =>
                        if pfx.has_rex_w() {
                            OpSize::S64
                        }
                        else {
                            OpSize::S32
                        },
                };
                let value = read_operand(operand, size, pfx.has_rex())?;
                let mut result = dst;
                match size {
                    OpSize::S8 => result.u8[imm as usize] = value as u8,
                    OpSize::S32 => result.u32[imm as usize & 3] = value as u32,
                    _ => result.u64[imm as usize & 1] = value,
                }
                xmm_set(modrm.reg, result);
            }
        },
        // DPPS/DPPD
        0x40 | 0x41 => {
            let src = sse_read(operand)?;
            let imm = fetch8()? as u8;
            let result = if opcode == 0x40 {
                crate::cpu::interp::simd_instr::dpps(dst, src, imm)
            }
            else {
                crate::cpu::interp::simd_instr::dppd(dst, src, imm)
            };
            xmm_set(modrm.reg, result);
        },
        // MPSADBW
        0x42 => {
            let src = sse_read(operand)?;
            let imm = fetch8()? as u8;
            xmm_set(modrm.reg, crate::cpu::interp::simd_instr::mpsadbw(dst, src, imm));
        },
        // PCLMULQDQ
        0x44 => {
            let src = sse_read(operand)?;
            let imm = fetch8()? as u8;
            xmm_set(modrm.reg, crate::cpu::interp::simd_instr::pclmulqdq(dst, src, imm));
        },
        // PCMPESTRM/PCMPESTRI (explicit lengths in EAX/EDX)
        0x60 | 0x61 | 0x62 | 0x63 => {
            let src = sse_read(operand)?;
            let imm = fetch8()? as u8;
            let (la, lb) = if opcode == 0x60 || opcode == 0x61 {
                (
                    read_reg(RAX, OpSize::S32, false) as u32 as i32,
                    read_reg(RDX, OpSize::S32, false) as u32 as i32,
                )
            }
            else {
                (i32::MIN, i32::MIN)
            };
            let a = dst.u8;
            let b = src.u8;
            let r = crate::cpu::interp::simd_instr::strcmp_apply(&a, &b, la, lb, imm);
            if opcode == 0x60 || opcode == 0x62 {
                xmm_set(0, r.mask);
            }
            write_reg(RCX, OpSize::S32, false, r.index as u64);
            set_strcmp_flags(r);
        },
        // AESKEYGENASSIST
        0xDF => {
            let src = sse_read(operand)?;
            let imm = fetch8()? as u8;
            xmm_set(modrm.reg, crate::cpu::interp::simd_instr::aeskeygenassist(src, imm));
        },
        _ => crate::cpu::core::trigger_ud(),
    }
    Ok(())
}

// The base of a segment selector's descriptor, from the GDT (the only table
// we model).
pub(crate) unsafe fn sse_arith(opcode: u8, pfx: &Prefixes) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let src = sse_read(operand)?;
    let mut dst = xmm_get(modrm.reg);
    if pfx.f3 {
        dst.f32[0] = crate::cpu::interp::simd_instr::arith_f32(opcode, dst.f32[0], src.f32[0]);
    }
    else if pfx.f2 {
        dst.f64[0] = crate::cpu::interp::simd_instr::arith_f64(opcode, dst.f64[0], src.f64[0]);
    }
    else if pfx.p66 {
        for i in 0..2 {
            dst.f64[i] = crate::cpu::interp::simd_instr::arith_f64(opcode, dst.f64[i], src.f64[i]);
        }
    }
    else {
        for i in 0..4 {
            dst.f32[i] = crate::cpu::interp::simd_instr::arith_f32(opcode, dst.f32[i], src.f32[i]);
        }
    }
    xmm_set(modrm.reg, dst);
    Ok(())
}

// ANDPS/ANDNPS/ORPS/XORPS (0F 54-57), bitwise over the whole register.
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/andps
pub(crate) unsafe fn sse_logic(opcode: u8, pfx: &Prefixes) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let src = sse_read(operand)?;
    let mut dst = xmm_get(modrm.reg);
    for i in 0..2 {
        dst.u64[i] = match opcode {
            0x54 => dst.u64[i] & src.u64[i],
            0x55 => !dst.u64[i] & src.u64[i],
            0x56 => dst.u64[i] | src.u64[i],
            _ => dst.u64[i] ^ src.u64[i],
        };
    }
    xmm_set(modrm.reg, dst);
    Ok(())
}

// CMPPS/CMPPD/CMPSS/CMPSD (0F C2).
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/cmpps
pub(crate) unsafe fn sse_compare(pfx: &Prefixes) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let src = sse_read(operand)?;
    let mut dst = xmm_get(modrm.reg);
    let predicate = fetch8()? as u8 & 7;
    if pfx.f3 {
        dst.f32[0] = if crate::cpu::interp::simd_instr::compare_f32(predicate, dst.f32[0], src.f32[0]) { f32::from_bits(!0) } else { 0.0 };
    }
    else if pfx.f2 {
        dst.f64[0] = if crate::cpu::interp::simd_instr::compare_f64(predicate, dst.f64[0], src.f64[0]) { f64::from_bits(!0) } else { 0.0 };
    }
    else if pfx.p66 {
        for i in 0..2 {
            dst.f64[i] = if crate::cpu::interp::simd_instr::compare_f64(predicate, dst.f64[i], src.f64[i]) { f64::from_bits(!0) } else { 0.0 };
        }
    }
    else {
        for i in 0..4 {
            dst.f32[i] = if crate::cpu::interp::simd_instr::compare_f32(predicate, dst.f32[i], src.f32[i]) { f32::from_bits(!0) } else { 0.0 };
        }
    }
    xmm_set(modrm.reg, dst);
    Ok(())
}


// UCOMISS/UCOMISD and COMISS/COMISD (0F 2E/2F).
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/ucomiss
pub(crate) unsafe fn sse_ucomi(signalling: bool, pfx: &Prefixes) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let src = sse_read(operand)?;
    let dst = xmm_get(modrm.reg);
    let unordered;
    let less;
    let equal;
    if pfx.p66 || pfx.f2 {
        let a = dst.f64[0];
        let b = src.f64[0];
        unordered = a.is_nan() || b.is_nan();
        less = a < b;
        equal = a == b;
    }
    else {
        let a = dst.f32[0];
        let b = src.f32[0];
        unordered = a.is_nan() || b.is_nan();
        less = a < b;
        equal = a == b;
    }
    let _ = signalling;
    *flags &= !((FLAG_CF | FLAG_PF | FLAG_ZF | FLAG_OF | FLAG_SF | FLAG_AF) as i32);
    if unordered {
        *flags |= (FLAG_CF | FLAG_PF | FLAG_ZF) as i32;
    }
    else if less {
        *flags |= FLAG_CF as i32;
    }
    else if equal {
        *flags |= FLAG_ZF as i32;
    }
    *flags_changed = 0;
    Ok(())
}

// MOVMSKPS/MOVMSKPD (0F 50).
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/movmskps
pub(crate) unsafe fn sse_movmsk(pfx: &Prefixes) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let Operand::Reg(src) = operand else {
        crate::cpu::core::trigger_ud();
        return Ok(());
    };
    let value = xmm_get(src);
    let mask = if pfx.p66 {
        (value.f64[0].to_bits() >> 63) as u64 | (value.f64[1].to_bits() >> 63) << 1
    }
    else {
        let mut m = 0u64;
        for i in 0..4 {
            m |= ((value.f32[i].to_bits() >> 31) as u64) << i;
        }
        m
    };
    write_reg(modrm.reg, OpSize::S32, pfx.has_rex(), mask);
    Ok(())
}

// SHUFPS/SHUFPD (0F C6).
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/shufps
pub(crate) unsafe fn sse_shuf(pfx: &Prefixes) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let src = sse_read(operand)?;
    let dst = xmm_get(modrm.reg);
    let control = fetch8()? as u8;
    let mut result = dst;
    if pfx.p66 {
        result.f64[0] = dst.f64[(control & 1) as usize];
        result.f64[1] = src.f64[((control >> 1) & 1) as usize];
    }
    else {
        for i in 0..2 {
            result.f32[i] = dst.f32[((control >> (i * 2)) & 3) as usize];
            result.f32[i + 2] = src.f32[((control >> (i * 2 + 4)) & 3) as usize];
        }
    }
    xmm_set(modrm.reg, result);
    Ok(())
}


pub(crate) unsafe fn sse_pshuf(pfx: &Prefixes) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let src = sse_read(operand)?;
    let control = fetch8()? as u8;
    let variant = if pfx.f3 { 1 } else if pfx.f2 { 2 } else { 0 };
    xmm_set(modrm.reg, crate::cpu::interp::simd_instr::pshuf_apply(variant, control, src));
    Ok(())
}

// JIT helper: PSHUFD xmm[dst], xmm/m128, imm8 (66 0F 70).
#[no_mangle]
pub unsafe fn jit64_sse_pshuf(control: i32, lo: u64, hi: u64, dst: i32) -> i32 {
    let src = reg128 { u64: [lo, hi] };
    xmm_set(dst as u8, crate::cpu::interp::simd_instr::pshuf_apply(0, control as u8, src));
    0
}

pub(crate) unsafe fn run_sse(opcode: u8, pfx: &Prefixes) -> OrPageFault<bool> {
    let size = pfx.operand_size();
    match opcode {
        // MOVSS/MOVSD xmm, xmm/m32|m64 (F3/F2 0F 10): the memory form clears
        // the upper lanes, the register form keeps them.
        0x10 if pfx.f3 || pfx.f2 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let mut dst = xmm_get(modrm.reg);
            match (pfx.f3, operand) {
                (true, Operand::Reg(src)) => dst.u32[0] = xmm_get(src).u32[0],
                (true, Operand::Mem(address)) => {
                    dst = reg128 { u32: [mem_read(address, OpSize::S32)? as u32, 0, 0, 0] };
                },
                (false, Operand::Reg(src)) => dst.u64[0] = xmm_get(src).u64[0],
                (false, Operand::Mem(address)) => {
                    dst = reg128 { u64: [mem_read(address, OpSize::S64)?, 0] };
                },
            }
            xmm_set(modrm.reg, dst);
        },

        // MOVSS xmm/m32, xmm (F3 0F 11) and MOVSD xmm/m64, xmm (F2 0F 11)
        0x11 if pfx.f3 || pfx.f2 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = xmm_get(modrm.reg);
            match (pfx.f3, operand) {
                (true, Operand::Reg(r)) => {
                    let mut dst = xmm_get(r);
                    dst.u32[0] = value.u32[0];
                    xmm_set(r, dst);
                },
                (true, Operand::Mem(address)) => mem_write(address, OpSize::S32, value.u32[0] as u64)?,
                (false, Operand::Reg(r)) => {
                    let mut dst = xmm_get(r);
                    dst.u64[0] = value.u64[0];
                    xmm_set(r, dst);
                },
                (false, Operand::Mem(address)) => mem_write(address, OpSize::S64, value.u64[0])?,
            }
        },

        // MOVAPS/MOVUPS/MOVAPD/MOVUPD xmm, xmm/m128 (0F 10/28)
        0x10 | 0x28 if !pfx.f2 && !pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = sse_read(operand)?;
            xmm_set(modrm.reg, value);
        },

        // MOVAPS/MOVUPS/MOVAPD/MOVUPD xmm/m128, xmm (0F 11/29)
        0x11 | 0x29 if !pfx.f2 && !pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = xmm_get(modrm.reg);
            match operand {
                Operand::Reg(r) => xmm_set(r, value),
                Operand::Mem(address) => mem_write128(address, value)?,
            }
        },

        // MOVLPS/MOVLPD xmm, m64 (0F 12), MOVHLPS xmm, xmm (0F 12 mod=3)
        0x12 if !pfx.f2 && !pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let mut dst = xmm_get(modrm.reg);
            match operand {
                Operand::Reg(src) => {
                    if pfx.p66 {
                        crate::cpu::core::trigger_ud();
                        return Ok(true);
                    }
                    dst.u64[0] = xmm_get(src).u64[1];
                },
                Operand::Mem(address) => {
                    dst.u64[0] = mem_read(address, OpSize::S64)?;
                },
            }
            xmm_set(modrm.reg, dst);
        },

        // MOVLPS/MOVLPD m64, xmm (0F 13)
        0x13 if !pfx.f2 && !pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let Operand::Mem(address) = operand else {
                crate::cpu::core::trigger_ud();
                return Ok(true);
            };
            let value = xmm_get(modrm.reg);
            mem_write(address, OpSize::S64, value.u64[0])?;
        },

        // UNPCKLPS/UNPCKLPD (0F 14) / UNPCKHPS/UNPCKHPD (0F 15)
        0x14 | 0x15 if !pfx.f2 && !pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = sse_read(operand)?;
            let dst = xmm_get(modrm.reg);
            let mut result = dst;
            if opcode == 0x14 {
                result.u64[1] = src.u64[0];
            }
            else {
                result.u64[0] = dst.u64[1];
                result.u64[1] = src.u64[1];
            }
            xmm_set(modrm.reg, result);
        },

        // MOVHPS/MOVHPD xmm, m64 (0F 16), MOVLHPS xmm, xmm (0F 16 mod=3)
        0x16 if !pfx.f2 && !pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let mut dst = xmm_get(modrm.reg);
            match operand {
                Operand::Reg(src) => {
                    if pfx.p66 {
                        crate::cpu::core::trigger_ud();
                        return Ok(true);
                    }
                    dst.u64[1] = xmm_get(src).u64[0];
                },
                Operand::Mem(address) => {
                    dst.u64[1] = mem_read(address, OpSize::S64)?;
                },
            }
            xmm_set(modrm.reg, dst);
        },

        // MOVHPS/MOVHPD m64, xmm (0F 17)
        0x17 if !pfx.f2 && !pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let Operand::Mem(address) = operand else {
                crate::cpu::core::trigger_ud();
                return Ok(true);
            };
            let value = xmm_get(modrm.reg);
            mem_write(address, OpSize::S64, value.u64[1])?;
        },

        // ADDSUBPS (F2 0F D0)
        0xD0 if pfx.f2 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = sse_read(operand)?;
            let dst = xmm_get(modrm.reg);
            let mut result = dst;
            for i in 0..4 {
                result.f32[i] = if i % 2 == 0 { dst.f32[i] + src.f32[i] } else { dst.f32[i] - src.f32[i] };
            }
            xmm_set(modrm.reg, result);
        },

        // POPCNT r, r/m (F3 0F B8)
        0xB8 if pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = read_operand(operand, size, pfx.has_rex())? & size.mask();
            write_reg(modrm.reg, size, pfx.has_rex(), value.count_ones() as u64);
        },

        // MMX MOVD mm, r/m32 (0F 6E, no 66)
        0x6E if !pfx.p66 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = read_operand(operand, OpSize::S32, pfx.has_rex())?;
            crate::cpu::core::transition_fpu_to_mmx();
            crate::cpu::core::write_mmx_reg64(modrm.reg as i32, value);
        },

        // MMX MOVD r/m32, mm (0F 7E, no 66/F3)
        0x7E if !pfx.p66 && !pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = crate::cpu::core::read_mmx64s(modrm.reg as i32);
            write_operand(operand, OpSize::S32, pfx.has_rex(), value)?;
        },

        // MOVQ xmm, xmm/m64 (F3 0F 7E): SSE2 quadword load, upper 64 bits cleared.
        // cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/movq
        0x7E if pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = match operand {
                Operand::Reg(r) => xmm_get(r).u64[0],
                Operand::Mem(address) => mem_read(address, OpSize::S64)?,
            };
            xmm_set(modrm.reg, reg128 { u64: [value, 0] });
        },

        // MOVD/MOVQ xmm, r/m (66 0F 6E)
        0x6E if pfx.p66 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = read_operand(operand, if pfx.has_rex() { OpSize::S64 } else { OpSize::S32 }, pfx.has_rex())?;
            xmm_set(modrm.reg, reg128 { u64: [value, 0] });
        },

        // MOVD/MOVQ r/m, xmm (66 0F 7E)
        0x7E if pfx.p66 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = xmm_get(modrm.reg);
            write_operand(operand, if pfx.has_rex() { OpSize::S64 } else { OpSize::S32 }, pfx.has_rex(), value.u64[0])?;
        },

        // MOVDQA/MOVDQU xmm, xmm/m128 (66/F3 0F 6F)
        0x6F if pfx.p66 || pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = sse_read(operand)?;
            xmm_set(modrm.reg, value);
        },

        // MOVDQA/MOVDQU xmm/m128, xmm (66/F3 0F 7F)
        0x7F if pfx.p66 || pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = xmm_get(modrm.reg);
            match operand {
                Operand::Reg(r) => xmm_set(r, value),
                Operand::Mem(address) => mem_write128(address, value)?,
            }
        },

        // ADDSS (F3 0F 58) / ADDSD (F2 0F 58)
        0x58 if pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = sse_read(operand)?;
            let mut dst = xmm_get(modrm.reg);
            dst.f32[0] += src.f32[0];
            xmm_set(modrm.reg, dst);
        },
        0x58 if pfx.f2 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = sse_read(operand)?;
            let mut dst = xmm_get(modrm.reg);
            dst.f64[0] += src.f64[0];
            xmm_set(modrm.reg, dst);
        },

        // MULSS (F3 0F 59)
        0x59 if pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = sse_read(operand)?;
            let mut dst = xmm_get(modrm.reg);
            dst.f32[0] *= src.f32[0];
            xmm_set(modrm.reg, dst);
        },

        // CVTSI2SS (F3 0F 2A)
        0x2A if pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = read_operand(operand, if pfx.has_rex() { OpSize::S64 } else { OpSize::S32 }, pfx.has_rex())?;
            let mut dst = xmm_get(modrm.reg);
            dst.f32[0] = if pfx.has_rex() { value as i64 as f32 } else { value as u32 as i32 as f32 };
            xmm_set(modrm.reg, dst);
        },

        // CVTTSS2SI (F3 0F 2C)
        //
        // This used to be `src.f32[0] as i64 as u64`, and Rust's float-to-int
        // cast saturates rather than following the architecture: out-of-range
        // values came back as i64::MAX and NaN as 0, instead of the integer
        // indefinite value the SDM specifies. It also went through the i64 path
        // for the 32-bit form, so 3e9f wrapped to 0xB2D05E00.
        0x2C if pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = sse_read(operand)?;
            let x = src.f32[0] as f64;
            let value = if pfx.has_rex_w() {
                crate::cpu::interp::sse_instr::cvt_float_to_i64(x, true)
            }
            else {
                crate::cpu::interp::sse_instr::cvt_float_to_i32(x, true) as u64
            };
            write_reg(modrm.reg, if pfx.has_rex() { OpSize::S64 } else { OpSize::S32 }, pfx.has_rex(), value);
        },

        // PCMPEQD (66 0F 76)
        0x76 if pfx.p66 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = sse_read(operand)?;
            let dst = xmm_get(modrm.reg);
            let mut result = reg128 { u64: [0, 0] };
            for i in 0..4 {
                result.u32[i] = if dst.u32[i] == src.u32[i] { 0xFFFF_FFFF } else { 0 };
            }
            xmm_set(modrm.reg, result);
        },

        // PAND (66 0F DB) / POR (66 0F EB) / PXOR (66 0F EF)
        0xDB | 0xEB | 0xEF if pfx.p66 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = sse_read(operand)?;
            let dst = xmm_get(modrm.reg);
            let mut result = reg128 { u64: [0, 0] };
            for i in 0..2 {
                result.u64[i] = match opcode {
                    0xDB => dst.u64[i] & src.u64[i],
                    0xEB => dst.u64[i] | src.u64[i],
                    _ => dst.u64[i] ^ src.u64[i],
                };
            }
            xmm_set(modrm.reg, result);
        },

        // PSUBD (66 0F FA) / PADDD (66 0F FE)
        0xFA | 0xFE if pfx.p66 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = sse_read(operand)?;
            let dst = xmm_get(modrm.reg);
            let mut result = reg128 { u64: [0, 0] };
            for i in 0..4 {
                result.u32[i] = if opcode == 0xFE {
                    dst.u32[i].wrapping_add(src.u32[i])
                }
                else {
                    dst.u32[i].wrapping_sub(src.u32[i])
                };
            }
            xmm_set(modrm.reg, result);
        },

        // SQRTPS/PD/SS/SD, ADDPS/PD/SS/SD, MULPS..., SUBPS..., MINPS...,
        // DIVPS..., MAXPS... (0F 51/58/59/5C/5D/5E/5F)
        0x51 | 0x58 | 0x59 | 0x5C | 0x5D | 0x5E | 0x5F => sse_arith(opcode, pfx)?,

        // ANDPS/PD, ANDNPS/PD, ORPS/PD, XORPS/PD (0F 54-57)
        0x54 | 0x55 | 0x56 | 0x57 => sse_logic(opcode, pfx)?,

        // CMPPS/PD/SS/SD (0F C2)
        0xC2 => sse_compare(pfx)?,

        // UCOMISS/SD (0F 2E) and COMISS/SD (0F 2F)
        0x2E => sse_ucomi(false, pfx)?,
        0x2F => sse_ucomi(true, pfx)?,

        // MOVMSKPS/MOVMSKPD (0F 50)
        0x50 => sse_movmsk(pfx)?,

        // SHUFPS/SHUFPD (0F C6)
        0xC6 => sse_shuf(pfx)?,

        // PSHUFD/PSHUFHW/PSHUFLW (0F 70)
        0x70 => sse_pshuf(pfx)?,

        // SSE2 packed integer (0F D0-0xFF and 0x60-0x7F)
        0x60 | 0x61 | 0x62 | 0x63 | 0x64 | 0x65 | 0x66 | 0x67 | 0x68 | 0x69 | 0x6A
        | 0x6B | 0x6C | 0x6D | 0x74 | 0x75 | 0x76
        | 0xD1 | 0xD2 | 0xD3 | 0xD4 | 0xD5 | 0xD8 | 0xD9 | 0xDA | 0xDB | 0xDC | 0xDD
        | 0xDE | 0xDF | 0xE1 | 0xE2 | 0xE4 | 0xE5 | 0xE8 | 0xE9 | 0xEA | 0xEB | 0xEC
        | 0xED | 0xEE | 0xEF | 0xF1 | 0xF2 | 0xF3 | 0xF5 | 0xF6 | 0xF8 | 0xF9 | 0xFA
        | 0xFB | 0xFC | 0xFD | 0xFE if pfx.p66 => sse_int(opcode, pfx)?,

        // PSLLW/D/Q, PSRLW/D/Q, PSRAW/D with an immediate (0F 71/72/73)
        0x71 | 0x72 | 0x73 if pfx.p66 => sse_shift_imm(opcode, pfx)?,

        // CVTPS2PD/CVTPD2PS/CVTSS2SD/CVTSD2SS (0F 5A),
        // CVTDQ2PS/CVTPS2DQ/CVTTPS2DQ (0F 5B)
        0x5A => sse_convert(0x5A, pfx)?,
        0x5B => sse_convert(0x5B, pfx)?,

        // CVTSI2SS/SD (0F 2A), CVTTSS2SI/SD (0F 2C), CVTSS2SI/SD (0F 2D)
        0x2A if !pfx.p66 => sse_cvt_int(0x2A, pfx)?,
        0x2C | 0x2D => sse_cvt_int(opcode, pfx)?,

        // RSQRTPS/RSQRTSS (0F 52), RCPPS/RCPSS (0F 53)
        0x52 | 0x53 => sse_rsqrt_rcp(opcode, pfx)?,

        // MOVNTPS/MOVNTPD (0F 2B), a plain store here; cf. Intel SDM Vol. 2:
// https://www.felixcloutier.com/x86/movntps
        0x2B if !pfx.f2 && !pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let Operand::Mem(address) = operand else {
                crate::cpu::core::trigger_ud();
                return Ok(true);
            };
            mem_write128(address, xmm_get(modrm.reg))?;
        },

        // MOVDDUP (F2 0F 12) / MOVSLDUP (F3 0F 12), SSE3.
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/movddup
        0x12 if pfx.f2 || pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = sse_read(operand)?;
            let mut dst = src;
            if pfx.f2 {
                dst.u64[0] = src.u64[0];
                dst.u64[1] = src.u64[0];
            }
            else {
                dst.u64[0] = src.u32[0] as u64 | (src.u32[0] as u64) << 32;
                dst.u64[1] = src.u32[1] as u64 | (src.u32[1] as u64) << 32;
            }
            xmm_set(modrm.reg, dst);
        },

        // MOVSHDUP (F3 0F 16)
        0x16 if pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = sse_read(operand)?;
            let mut dst = reg128 { u64: [0, 0] };
            dst.u64[0] = src.u32[1] as u64 | (src.u32[1] as u64) << 32;
            dst.u64[1] = src.u32[3] as u64 | (src.u32[3] as u64) << 32;
            xmm_set(modrm.reg, dst);
        },

        // HADDPS/HADDPD (F2/66 0F 7C), HSUBPS/HSUBPD (F2/66 0F 7D), SSE3.
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/haddps
        0x7C | 0x7D if pfx.f2 || pfx.p66 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = sse_read(operand)?;
            let dst = xmm_get(modrm.reg);
            let mut result = dst;
            if pfx.p66 {
                let a = src.f64[0];
                let b = src.f64[1];
                result.f64[0] = if opcode == 0x7C { dst.f64[0] + dst.f64[1] } else { dst.f64[0] - dst.f64[1] };
                result.f64[1] = if opcode == 0x7C { a + b } else { a - b };
            }
            else {
                result.f32[0] = if opcode == 0x7C { dst.f32[0] + dst.f32[1] } else { dst.f32[0] - dst.f32[1] };
                result.f32[1] = if opcode == 0x7C { dst.f32[2] + dst.f32[3] } else { dst.f32[2] - dst.f32[3] };
                result.f32[2] = if opcode == 0x7C { src.f32[0] + src.f32[1] } else { src.f32[0] - src.f32[1] };
                result.f32[3] = if opcode == 0x7C { src.f32[2] + src.f32[3] } else { src.f32[2] - src.f32[3] };
            }
            xmm_set(modrm.reg, result);
        },

        // PINSRW (66 0F C4); cf. Intel SDM Vol. 2:
// https://www.felixcloutier.com/x86/pinsrw
        0xC4 if pfx.p66 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = read_operand(operand, size, pfx.has_rex())? as u16;
            let index = fetch8()? as usize & 7;
            let mut dst = xmm_get(modrm.reg);
            dst.u16[index] = value;
            xmm_set(modrm.reg, dst);
        },

        // PEXTRW (66 0F C5); cf. Intel SDM Vol. 2:
// https://www.felixcloutier.com/x86/pextrw
        0xC5 if pfx.p66 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let Operand::Reg(src) = operand else {
                crate::cpu::core::trigger_ud();
                return Ok(true);
            };
            let index = fetch8()? as usize & 7;
            let value = xmm_get(src).u16[index] as u64;
            write_reg(modrm.reg, OpSize::S32, pfx.has_rex(), value);
        },

        // MOVQ xmm/m64, xmm (66 0F D6) and PMOVMSKB (66 0F D7).
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/pmovmskb
        0xD6 if pfx.p66 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = xmm_get(modrm.reg);
            match operand {
                Operand::Reg(r) => {
                    let mut result = xmm_get(r);
                    result.u64[0] = value.u64[0];
                    xmm_set(r, result);
                },
                Operand::Mem(address) => mem_write(address, OpSize::S64, value.u64[0])?,
            }
        },
        0xD7 if pfx.p66 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let Operand::Reg(src) = operand else {
                crate::cpu::core::trigger_ud();
                return Ok(true);
            };
            let value = xmm_get(src);
            let mut mask = 0u64;
            for i in 0..16 {
                mask |= ((value.u8[i] >> 7) as u64) << i;
            }
            write_reg(modrm.reg, OpSize::S32, pfx.has_rex(), mask);
        },

        // PAVGB (66 0F E0) / PAVGW (66 0F E3).
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/pavgb
        0xE0 | 0xE3 if pfx.p66 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = sse_read(operand)?;
            let mut dst = xmm_get(modrm.reg);
            if opcode == 0xE0 {
                for i in 0..16 {
                    dst.u8[i] = ((dst.u8[i] as u16 + src.u8[i] as u16 + 1) / 2) as u8;
                }
            }
            else {
                for i in 0..8 {
                    dst.u16[i] = ((dst.u16[i] as u32 + src.u16[i] as u32 + 1) / 2) as u16;
                }
            }
            xmm_set(modrm.reg, dst);
        },

        // CVTTPD2DQ (66 0F E6) / CVTDQ2PD (F3 0F E6).
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/cvttpd2dq
        0xE6 if pfx.p66 || pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = sse_read(operand)?;
            let mut dst = xmm_get(modrm.reg);
            if pfx.p66 {
                for i in 0..2 {
                    dst.i32[i] = if src.f64[i].is_nan() { i32::MIN } else { src.f64[i].trunc() as i32 };
                }
                dst.u64[1] = 0;
            }
            else {
                for i in 0..2 {
                    dst.f64[i] = src.i32[i] as f64;
                }
            }
            xmm_set(modrm.reg, dst);
        },

        // MOVNTDQ (66 0F E7)
        0xE7 if pfx.p66 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let Operand::Mem(address) = operand else {
                crate::cpu::core::trigger_ud();
                return Ok(true);
            };
            mem_write128(address, xmm_get(modrm.reg))?;
        },

        // LDDQU (F2 0F F0), SSE3; cf. Intel SDM Vol. 2:
// https://www.felixcloutier.com/x86/lddqu
        0xF0 if pfx.f2 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = sse_read(operand)?;
            xmm_set(modrm.reg, value);
        },

        // MASKMOVDQU (66 0F F7); cf. Intel SDM Vol. 2:
// https://www.felixcloutier.com/x86/maskmovdqu
        0xF7 if pfx.p66 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let Operand::Reg(src) = operand else {
                crate::cpu::core::trigger_ud();
                return Ok(true);
            };
            let value = xmm_get(modrm.reg);
            let mask = xmm_get(src);
            let address = read_reg64(RDI as i32);
            for i in 0..16 {
                if mask.u8[i] & 0x80 != 0 {
                    mem_write(address + i as u64, OpSize::S8, value.u8[i] as u64)?;
                }
            }
        },

        _ => return Ok(false),
    }
    Ok(true)
}

pub(crate) struct Vex {
    pub(crate) map: u8,
    pub(crate) pp: u8,
    pub(crate) l: bool,
    pub(crate) w: bool,
    pub(crate) vvvv: u8,
    pub(crate) rex_r: u8,
    pub(crate) rex_x: u8,
    pub(crate) rex_b: u8,
    // FS/GS override written before the VEX prefix (the VEX prefix does not
    // carry it).
    pub(crate) segment: u8,
}

pub(crate) unsafe fn decode_vex(first: u8) -> OrPageFault<Vex> {
    let mut v = Vex { map: 1, pp: 0, l: false, w: false, vvvv: 0, rex_r: 0, rex_x: 0, rex_b: 0, segment: 0 };
    if first == 0xC5 {
        let b = fetch8()?;
        v.rex_r = if b & 0x80 == 0 { prefix::REX_R } else { 0 };
        v.vvvv = !(b >> 3) & 0xF;
        v.l = b & 4 != 0;
        v.pp = b & 3;
    }
    else {
        let b1 = fetch8()?;
        let b2 = fetch8()?;
        v.rex_r = if b1 & 0x80 == 0 { prefix::REX_R } else { 0 };
        v.rex_x = if b1 & 0x40 == 0 { prefix::REX_X } else { 0 };
        v.rex_b = if b1 & 0x20 == 0 { prefix::REX_B } else { 0 };
        v.map = b1 & 0x1F;
        v.w = b2 & 0x80 != 0;
        v.vvvv = !(b2 >> 3) & 0xF;
        v.l = b2 & 4 != 0;
        v.pp = b2 & 3;
    }
    Ok(v)
}

impl Vex {
    pub(crate) fn pfx(&self) -> Prefixes {
        let mut p = Prefixes::default();
        p.rex = self.rex_r | self.rex_x | self.rex_b | if self.w { prefix::REX_W } else { 0 };
        p.opsize_16 = self.pp == 1;
        p.p66 = self.pp == 1;
        p.f3 = self.pp == 2;
        p.f2 = self.pp == 3;
        p.segment = self.segment;
        p
    }
}

pub(crate) unsafe fn ymm_hi(r: u8) -> reg128 { *crate::cpu::global_pointers::ymm_high_ptr(r as i32) }
pub(crate) unsafe fn ymm_set_hi(r: u8, value: reg128) {
    *crate::cpu::global_pointers::ymm_high_ptr(r as i32) = value;
}

pub(crate) unsafe fn vex_rm(operand: Operand, l: bool) -> OrPageFault<(reg128, reg128)> {
    let lo = sse_read(operand)?;
    let hi = if l {
        match operand {
            Operand::Reg(r) => ymm_hi(r),
            Operand::Mem(a) => mem_read128(a + 16)?,
        }
    }
    else {
        reg128 { u64: [0, 0] }
    };
    Ok((lo, hi))
}

pub(crate) unsafe fn vex_set(dst: u8, lo: reg128, hi: reg128, l: bool) {
    xmm_set(dst, lo);
    ymm_set_hi(dst, if l { hi } else { reg128 { u64: [0, 0] } });
}

pub(crate) unsafe fn vex_store(operand: Operand, lo: reg128, hi: reg128, l: bool) -> OrPageFault<()> {
    match operand {
        Operand::Reg(r) =>
        {
            vex_set(r, lo, hi, l);
            Ok(())
        },
        Operand::Mem(a) =>
        {
            mem_write128(a, lo)?;
            if l {
                mem_write128(a + 16, hi)?;
            }
            Ok(())
        },
    }
}

pub(crate) unsafe fn vex_v(v: &Vex) -> (reg128, reg128) {
    (xmm_get(v.vvvv), if v.l { ymm_hi(v.vvvv) } else { reg128 { u64: [0, 0] } })
}

pub(crate) unsafe fn vex_arith(v: &Vex, opcode: u8) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let (s2lo, s2hi) = vex_rm(operand, v.l)?;
    let (s1lo, s1hi) = vex_v(v);
    if v.pp == 2 {
        let mut r = s1lo;
        r.f32[0] = crate::cpu::interp::simd_instr::arith_f32(opcode, s1lo.f32[0], s2lo.f32[0]);
        vex_set(modrm.reg, r, reg128 { u64: [0, 0] }, false);
    }
    else if v.pp == 3 {
        let mut r = s1lo;
        r.f64[0] = crate::cpu::interp::simd_instr::arith_f64(opcode, s1lo.f64[0], s2lo.f64[0]);
        vex_set(modrm.reg, r, reg128 { u64: [0, 0] }, false);
    }
    else if v.pp == 1 {
        let mut rlo = s1lo;
        let mut rhi = s1hi;
        for i in 0..2 {
            rlo.f64[i] = crate::cpu::interp::simd_instr::arith_f64(opcode, s1lo.f64[i], s2lo.f64[i]);
            if v.l {
                rhi.f64[i] = crate::cpu::interp::simd_instr::arith_f64(opcode, s1hi.f64[i], s2hi.f64[i]);
            }
        }
        vex_set(modrm.reg, rlo, rhi, v.l);
    }
    else {
        let mut rlo = s1lo;
        let mut rhi = s1hi;
        for i in 0..4 {
            rlo.f32[i] = crate::cpu::interp::simd_instr::arith_f32(opcode, s1lo.f32[i], s2lo.f32[i]);
            if v.l {
                rhi.f32[i] = crate::cpu::interp::simd_instr::arith_f32(opcode, s1hi.f32[i], s2hi.f32[i]);
            }
        }
        vex_set(modrm.reg, rlo, rhi, v.l);
    }
    Ok(())
}

pub(crate) unsafe fn vex_logic(v: &Vex, opcode: u8) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let (s2lo, s2hi) = vex_rm(operand, v.l)?;
    let (s1lo, s1hi) = vex_v(v);
    let apply = |a: reg128, b: reg128| -> reg128 {
        let mut r = a;
        for i in 0..2 {
            r.u64[i] = match opcode {
                0x54 => a.u64[i] & b.u64[i],
                0x55 => !a.u64[i] & b.u64[i],
                0x56 => a.u64[i] | b.u64[i],
                _ => a.u64[i] ^ b.u64[i],
            };
        }
        r
    };
    let rlo = apply(s1lo, s2lo);
    let rhi = if v.l { apply(s1hi, s2hi) } else { reg128 { u64: [0, 0] } };
    vex_set(modrm.reg, rlo, rhi, v.l);
    Ok(())
}

pub(crate) unsafe fn vex_int(v: &Vex, opcode: u8) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let (s2lo, s2hi) = vex_rm(operand, v.l)?;
    let (s1lo, s1hi) = vex_v(v);
    let rlo = match crate::cpu::interp::simd_instr::int_apply(opcode, s1lo, s2lo) {
        Some(r) => r,
        None =>
        {
            crate::cpu::core::trigger_ud();
            return Ok(());
        },
    };
    let rhi = if v.l {
        match crate::cpu::interp::simd_instr::int_apply(opcode, s1hi, s2hi) {
            Some(r) => r,
            None =>
            {
                crate::cpu::core::trigger_ud();
                return Ok(());
            },
        }
    }
    else {
        reg128 { u64: [0, 0] }
    };
    vex_set(modrm.reg, rlo, rhi, v.l);
    Ok(())
}

// VMOVD/VMOVQ (VEX.128): move 32/64 bits between an xmm and a GPR or memory;
// the upper xmm bits are zeroed on load. 6E/7E are the GPR forms (W picks the
// width), 7E with F3 loads from xmm/m64 and D6 stores to xmm/m64.
pub(crate) unsafe fn vex_movd(v: &Vex, opcode: u8) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    match (opcode, v.pp) {
        (0x6E, _) => {
            let size = if v.w { OpSize::S64 } else { OpSize::S32 };
            let value = read_operand(operand, size, pfx.has_rex())?;
            // VEX.128 zeroes bits 255:128 of the destination.
            vex_set(modrm.reg, reg128 { u64: [value, 0] }, reg128 { u64: [0, 0] }, false);
        },
        (0x7E, 1) => {
            let size = if v.w { OpSize::S64 } else { OpSize::S32 };
            let value = xmm_get(modrm.reg).u64[0];
            write_operand(operand, size, pfx.has_rex(), value)?;
        },
        (0x7E, 2) => {
            let value = match operand {
                Operand::Reg(r) => xmm_get(r).u64[0],
                Operand::Mem(a) => mem_read(a, OpSize::S64)?,
            };
            vex_set(modrm.reg, reg128 { u64: [value, 0] }, reg128 { u64: [0, 0] }, false);
        },
        (0xD6, _) => {
            let value = xmm_get(modrm.reg).u64[0];
            write_operand(operand, OpSize::S64, pfx.has_rex(), value)?;
        },
        _ => crate::cpu::core::trigger_ud(),
    }
    Ok(())
}

pub(crate) unsafe fn vex_mov(v: &Vex, opcode: u8) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    // vmovss/vmovsd are 0x10/0x11 with F3/F2; vmovdqu is 0x6F/0x7F with F3,
    // so restrict the scalar path to the scalar opcodes.
    if (v.pp == 2 || v.pp == 3) && (opcode == 0x10 || opcode == 0x11) {
        let bits = if v.pp == 2 { 32 } else { 64 };
        if opcode == 0x10 {
            let (s2lo, _) = vex_rm(operand, false)?;
            let (s1lo, _) = vex_v(v);
            let mut r = s1lo;
            if bits == 32 {
                r.u32[0] = s2lo.u32[0];
            }
            else {
                r.u64[0] = s2lo.u64[0];
            }
            vex_set(modrm.reg, r, reg128 { u64: [0, 0] }, false);
        }
        else {
            let src = xmm_get(modrm.reg);
            let (s1lo, _) = vex_v(v);
            match operand {
                Operand::Reg(r) =>
                {
                    let mut out = s1lo;
                    if bits == 32 {
                        out.u32[0] = src.u32[0];
                    }
                    else {
                        out.u64[0] = src.u64[0];
                    }
                    vex_set(r, out, reg128 { u64: [0, 0] }, false);
                },
                Operand::Mem(a) =>
                {
                    mem_write(a, if bits == 32 { OpSize::S32 } else { OpSize::S64 }, src.u64[0])?;
                },
            }
        }
        return Ok(());
    }
    if opcode == 0x10 || opcode == 0x28 || opcode == 0x6F {
        let (lo, hi) = vex_rm(operand, v.l)?;
        vex_set(modrm.reg, lo, hi, v.l);
    }
    else {
        let lo = xmm_get(modrm.reg);
        let hi = if v.l { ymm_hi(modrm.reg) } else { reg128 { u64: [0, 0] } };
        vex_store(operand, lo, hi, v.l)?;
    }
    Ok(())
}

pub(crate) unsafe fn vex_unpck(v: &Vex, opcode: u8) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let (s2lo, s2hi) = vex_rm(operand, v.l)?;
    let (s1lo, s1hi) = vex_v(v);
    let unpck = |a: reg128, b: reg128, high: bool, f64: bool| -> reg128 {
        let mut r = a;
        if f64 {
            if high {
                r.u64[0] = a.u64[1];
                r.u64[1] = b.u64[1];
            }
            else {
                r.u64[0] = a.u64[0];
                r.u64[1] = b.u64[0];
            }
        }
        else if high {
            r.u32[0] = a.u32[2];
            r.u32[1] = b.u32[2];
            r.u32[2] = a.u32[3];
            r.u32[3] = b.u32[3];
        }
        else {
            r.u32[0] = a.u32[0];
            r.u32[1] = b.u32[0];
            r.u32[2] = a.u32[1];
            r.u32[3] = b.u32[1];
        }
        r
    };
    let high = opcode == 0x15;
    let f64 = v.pp == 1;
    let rlo = unpck(s1lo, s2lo, high, f64);
    let rhi = if v.l { unpck(s1hi, s2hi, high, f64) } else { reg128 { u64: [0, 0] } };
    vex_set(modrm.reg, rlo, rhi, v.l);
    Ok(())
}

pub(crate) unsafe fn vex_cmp(v: &Vex) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let (s2lo, s2hi) = vex_rm(operand, v.l)?;
    let (s1lo, s1hi) = vex_v(v);
    let predicate = fetch8()? as u8 & 0x1F;
    let cmp32 = |a: f32, b: f32| if crate::cpu::interp::simd_instr::compare_f32(predicate, a, b) { f32::from_bits(!0) } else { 0.0 };
    let cmp64 = |a: f64, b: f64| if crate::cpu::interp::simd_instr::compare_f64(predicate, a, b) { f64::from_bits(!0) } else { 0.0 };
    if v.pp == 2 {
        let mut r = s1lo;
        r.f32[0] = cmp32(s1lo.f32[0], s2lo.f32[0]);
        vex_set(modrm.reg, r, reg128 { u64: [0, 0] }, false);
    }
    else if v.pp == 3 {
        let mut r = s1lo;
        r.f64[0] = cmp64(s1lo.f64[0], s2lo.f64[0]);
        vex_set(modrm.reg, r, reg128 { u64: [0, 0] }, false);
    }
    else if v.pp == 1 {
        let mut rlo = s1lo;
        let mut rhi = s1hi;
        for i in 0..2 {
            rlo.f64[i] = cmp64(s1lo.f64[i], s2lo.f64[i]);
            if v.l {
                rhi.f64[i] = cmp64(s1hi.f64[i], s2hi.f64[i]);
            }
        }
        vex_set(modrm.reg, rlo, rhi, v.l);
    }
    else {
        let mut rlo = s1lo;
        let mut rhi = s1hi;
        for i in 0..4 {
            rlo.f32[i] = cmp32(s1lo.f32[i], s2lo.f32[i]);
            if v.l {
                rhi.f32[i] = cmp32(s1hi.f32[i], s2hi.f32[i]);
            }
        }
        vex_set(modrm.reg, rlo, rhi, v.l);
    }
    Ok(())
}

pub(crate) unsafe fn vex_shuf(v: &Vex) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let (s2lo, s2hi) = vex_rm(operand, v.l)?;
    let (s1lo, s1hi) = vex_v(v);
    let control = fetch8()? as u8;
    let shuf = |a: reg128, b: reg128, f64: bool| -> reg128 {
        let mut r = a;
        if f64 {
            r.f64[0] = a.f64[(control & 1) as usize];
            r.f64[1] = b.f64[((control >> 1) & 1) as usize];
        }
        else {
            for i in 0..2 {
                r.f32[i] = a.f32[((control >> (i * 2)) & 3) as usize];
                r.f32[i + 2] = b.f32[((control >> (i * 2 + 4)) & 3) as usize];
            }
        }
        r
    };
    let f64 = v.pp == 1;
    let rlo = shuf(s1lo, s2lo, f64);
    let rhi = if v.l { shuf(s1hi, s2hi, f64) } else { reg128 { u64: [0, 0] } };
    vex_set(modrm.reg, rlo, rhi, v.l);
    Ok(())
}

pub(crate) unsafe fn vex_pshuf(v: &Vex) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let (slo, shi) = vex_rm(operand, v.l)?;
    let control = fetch8()? as u8;
    let variant = if v.pp == 2 { 1 } else if v.pp == 3 { 2 } else { 0 };
    let rlo = crate::cpu::interp::simd_instr::pshuf_apply(variant, control, slo);
    let rhi = if v.l { crate::cpu::interp::simd_instr::pshuf_apply(variant, control, shi) } else { reg128 { u64: [0, 0] } };
    vex_set(modrm.reg, rlo, rhi, v.l);
    Ok(())
}

pub(crate) unsafe fn vex_shift_imm(v: &Vex, opcode: u8) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let imm = fetch8()? as u64;
    let group = modrm.reg & 7;

    // VEX shift-by-immediate (VPSLLQ and friends): result in VEX.vvvv, source
    // ModRM.rm, ModRM.reg the group selector. Writing back to rm shifted the
    // source and left the destination untouched.
    let (slo, shi) = vex_rm(operand, v.l)?;
    let rlo = match crate::cpu::interp::simd_instr::shift_imm_apply(opcode, group, imm, slo) {
        Some(r) => r,
        None =>
        {
            crate::cpu::core::trigger_ud();
            return Ok(());
        },
    };
    let rhi = if v.l {
        match crate::cpu::interp::simd_instr::shift_imm_apply(opcode, group, imm, shi) {
            Some(r) => r,
            None =>
            {
                crate::cpu::core::trigger_ud();
                return Ok(());
            }
        }
    }
    else {
        reg128 { u64: [0, 0] }
    };
    vex_set(v.vvvv, rlo, rhi, v.l);
    Ok(())
}

pub(crate) unsafe fn vex_convert(v: &Vex, opcode: u8) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let (src, srchi) = vex_rm(operand, v.l)?;
    let (s1lo, s1hi) = vex_v(v);
    let dst = modrm.reg;
    let mut rlo = s1lo;
    let mut rhi = s1hi;
    match opcode {
        0x5A =>
        {
            if v.pp == 0 {
                // CVTPS2PD
                for i in 0..2 {
                    rlo.f64[i] = src.f32[i] as f64;
                    if v.l {
                        rhi.f64[i] = srchi.f32[i] as f64;
                    }
                }
            }
            else if v.pp == 1 {
                // CVTPD2PS
                for i in 0..2 {
                    rlo.f32[i] = src.f64[i] as f32;
                }
                rlo.u64[1] = 0;
                rhi = reg128 { u64: [0, 0] };
                if v.l {
                    for i in 0..2 {
                        rhi.f32[i] = srchi.f64[i] as f32;
                    }
                }
            }
            else if v.pp == 2 {
                // CVTSS2SD
                rlo.f64[0] = src.f32[0] as f64;
                vex_set(dst, rlo, reg128 { u64: [0, 0] }, false);
                return Ok(());
            }
            else {
                // CVTSD2SS
                rlo.f32[0] = src.f64[0] as f32;
                vex_set(dst, rlo, reg128 { u64: [0, 0] }, false);
                return Ok(());
            }
        },
        0x5B =>
        {
            if v.pp == 0 {
                // CVTDQ2PS
                for i in 0..4 {
                    rlo.f32[i] = src.i32[i] as f32;
                    if v.l {
                        rhi.f32[i] = srchi.i32[i] as f32;
                    }
                }
            }
            else if v.pp == 1 {
                // CVTPS2DQ
                for i in 0..4 {
                    rlo.i32[i] = src.f32[i].round() as i32;
                    if v.l {
                        rhi.i32[i] = srchi.f32[i].round() as i32;
                    }
                }
            }
            else {
                // CVTTPS2DQ
                for i in 0..4 {
                    rlo.i32[i] = if src.f32[i].is_nan() { i32::MIN } else { src.f32[i].trunc() as i32 };
                    if v.l {
                        rhi.i32[i] = if srchi.f32[i].is_nan() { i32::MIN } else { srchi.f32[i].trunc() as i32 };
                    }
                }
            }
        },
        0xE6 =>
        {
            if v.pp == 2 {
                // CVTDQ2PD
                for i in 0..2 {
                    rlo.f64[i] = src.i32[i] as f64;
                    if v.l {
                        rhi.f64[i] = srchi.i32[i] as f64;
                    }
                }
            }
            else {
                // CVTTPD2DQ (pp1) / CVTPD2DQ (pp3)
                let truncate = v.pp == 1;
                for i in 0..2 {
                    rlo.i32[i] = if truncate { src.f64[i].trunc() } else { src.f64[i].round() } as i32;
                }
                rlo.u64[1] = 0;
                rhi = reg128 { u64: [0, 0] };
                if v.l {
                    for i in 0..2 {
                        rhi.i32[i] = if truncate { srchi.f64[i].trunc() } else { srchi.f64[i].round() } as i32;
                    }
                }
            }
        },
        _ => {},
    }
    vex_set(dst, rlo, rhi, v.l);
    Ok(())
}

pub(crate) unsafe fn vex_cvt_int(v: &Vex, opcode: u8) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let dst = modrm.reg;
    let f64 = v.pp == 3;
    if opcode == 0x2A {
        // CVTSI2SS/SD: GPR -> scalar
        let size = if v.w { OpSize::S64 } else { OpSize::S32 };
        let source = read_operand(operand, size, true)?;
        let (s1lo, _) = vex_v(v);
        let mut r = s1lo;
        if f64 {
            r.f64[0] = if v.w { source as i64 as f64 } else { source as u32 as i32 as f64 };
        }
        else {
            r.f32[0] = if v.w { source as i64 as f32 } else { source as u32 as i32 as f32 };
        }
        vex_set(dst, r, reg128 { u64: [0, 0] }, false);
    }
    else {
        // CVTTSS2SI (2C) / CVTSS2SI (2D) and the SD forms: scalar -> GPR
        let (src, _) = vex_rm(operand, false)?;
        let truncate = opcode == 0x2C;
        let x = if f64 { src.f64[0] } else { src.f32[0] as f64 };
        let bits = if v.w {
            crate::cpu::interp::sse_instr::cvt_float_to_i64(x, truncate)
        }
        else {
            crate::cpu::interp::sse_instr::cvt_float_to_i32(x, truncate) as u64
        };
        write_reg64(dst as i32, bits);
    }
    Ok(())
}

pub(crate) unsafe fn vex_compare_scalar(v: &Vex, signalling: bool) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let (src, _) = vex_rm(operand, false)?;
    let a = xmm_get(modrm.reg);
    let (unordered, less, equal) = if v.pp == 1 {
        (a.f64[0].is_nan() || src.f64[0].is_nan(), a.f64[0] < src.f64[0], a.f64[0] == src.f64[0])
    }
    else {
        (a.f32[0].is_nan() || src.f32[0].is_nan(), a.f32[0] < src.f32[0], a.f32[0] == src.f32[0])
    };
    let _ = signalling;
    *flags &= !((FLAG_CF | FLAG_PF | FLAG_ZF | FLAG_OF | FLAG_SF | FLAG_AF) as i32);
    if unordered {
        *flags |= (FLAG_CF | FLAG_PF | FLAG_ZF) as i32;
    }
    else if less {
        *flags |= FLAG_CF as i32;
    }
    else if equal {
        *flags |= FLAG_ZF as i32;
    }
    *flags_changed = 0;
    Ok(())
}

pub(crate) unsafe fn vex_movmsk(v: &Vex) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let Operand::Reg(src) = operand else {
        crate::cpu::core::trigger_ud();
        return Ok(());
    };
    let value = xmm_get(src);
    let mask = if v.pp == 1 {
        (value.f64[0].to_bits() >> 63) as u64 | (value.f64[1].to_bits() >> 63) << 1
    }
    else {
        let mut m = 0u64;
        for i in 0..4 {
            m |= ((value.f32[i].to_bits() >> 31) as u64) << i;
        }
        m
    };
    write_reg(modrm.reg, OpSize::S32, true, mask);
    Ok(())
}

pub(crate) unsafe fn vex_pmovmskb(v: &Vex) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let (src, srchi) = vex_rm(operand, v.l)?;
    let mut mask = 0u64;
    for i in 0..16 {
        mask |= ((src.u8[i] >> 7) as u64) << i;
    }
    if v.l {
        for i in 0..16 {
            mask |= ((srchi.u8[i] >> 7) as u64) << (i + 16);
        }
    }
    write_reg(modrm.reg, OpSize::S32, true, mask);
    Ok(())
}

pub(crate) unsafe fn vex_movddup(v: &Vex, opcode: u8) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let (src, srchi) = vex_rm(operand, v.l)?;
    let mut rlo = src;
    let mut rhi = srchi;
    let dup = |s: reg128, odd: bool| -> reg128 {
        let mut r = s;
        if odd {
            // MOVSHDUP / MOVHLPS
            for i in 0..4 {
                r.u32[i] = s.u32[1 + (i & !1)];
            }
        }
        else {
            for i in 0..4 {
                r.u32[i] = s.u32[i & !1];
            }
        }
        r
    };
    if opcode == 0x12 && v.pp == 3 {
        // MOVDDUP (F2): duplicate even qwords
        rlo.u64[1] = src.u64[0];
        if v.l {
            rhi.u64[1] = srchi.u64[0];
        }
    }
    else if opcode == 0x12 {
        // MOVSLDUP (F3) / MOVHLPS (pp0)
        rlo = dup(src, false);
        if v.l {
            rhi = dup(srchi, false);
        }
    }
    else {
        // MOVSHDUP (F3 16)
        rlo = dup(src, true);
        if v.l {
            rhi = dup(srchi, true);
        }
    }
    vex_set(modrm.reg, rlo, rhi, v.l);
    Ok(())
}

pub(crate) unsafe fn run_vex_0f(v: &Vex) -> OrPageFault<()> {
    let opcode = fetch8()?;

    // VZEROUPPER (L=0) / VZEROALL (L=1)
    if opcode == 0x77 && v.pp == 0 {
        for r in 0..16u8 {
            ymm_set_hi(r, reg128 { u64: [0, 0] });
            if v.l {
                xmm_set(r, reg128 { u64: [0, 0] });
            }
        }
        return Ok(());
    }

    match opcode {
        0x10 | 0x11 | 0x28 | 0x29 => return vex_mov(v, opcode),
        0x51 | 0x58 | 0x59 | 0x5C | 0x5D | 0x5E | 0x5F => return vex_arith(v, opcode),
        0x5A | 0x5B | 0xE6 => return vex_convert(v, opcode),
        0x2A | 0x2C | 0x2D => return vex_cvt_int(v, opcode),
        0x2E | 0x2F => return vex_compare_scalar(v, opcode == 0x2F),
        0x50 => return vex_movmsk(v),
        // VMOVLPS/HPS and VMOVLPD/HPD (128-bit half moves)
        0x12 | 0x16 if v.pp <= 1 =>
        {
            let pfx = v.pfx();
            let (modrm, operand) = decode_operand(&pfx)?;
            let Operand::Mem(address) = operand else {
                crate::cpu::core::trigger_ud();
                return Ok(());
            };
            let mem = mem_read(address, OpSize::S64)?;
            let (s1lo, _) = vex_v(v);
            let mut r = s1lo;
            if opcode == 0x12 {
                r.u64[0] = mem;
            }
            else {
                r.u64[1] = mem;
            }
            vex_set(modrm.reg, r, reg128 { u64: [0, 0] }, false);
        },
        0x13 | 0x17 if v.pp <= 1 =>
        {
            let pfx = v.pfx();
            let (modrm, operand) = decode_operand(&pfx)?;
            let Operand::Mem(address) = operand else {
                crate::cpu::core::trigger_ud();
                return Ok(());
            };
            let value = xmm_get(modrm.reg);
            mem_write(address, OpSize::S64, if opcode == 0x13 { value.u64[0] } else { value.u64[1] })?;
        },
        0x12 | 0x16 if v.pp >= 2 => return vex_movddup(v, opcode),
        0xD7 if v.pp == 1 => return vex_pmovmskb(v),
        0x54 | 0x55 | 0x56 | 0x57 => return vex_logic(v, opcode),
        0x14 | 0x15 => return vex_unpck(v, opcode),
        0xC2 => return vex_cmp(v),
        0xC6 => return vex_shuf(v),
        0x70 => return vex_pshuf(v),
        0x71 | 0x72 | 0x73 if v.pp == 1 => return vex_shift_imm(v, opcode),
        0x6E if v.pp == 1 => return vex_movd(v, opcode),
        0x7E if v.pp == 1 || v.pp == 2 => return vex_movd(v, opcode),
        0xD6 if v.pp == 1 => return vex_movd(v, opcode),
        0x6F | 0x7F if v.pp == 1 || v.pp == 2 => return vex_mov(v, opcode),
        // VMOVNTPS/PD and VMOVNTDQ (non-temporal stores; a plain store here)
        0x2B if v.pp <= 1 =>
        {
            let pfx = v.pfx();
            let (modrm, operand) = decode_operand(&pfx)?;
            let lo = xmm_get(modrm.reg);
            let hi = if v.l { ymm_hi(modrm.reg) } else { reg128 { u64: [0, 0] } };
            vex_store(operand, lo, hi, v.l)?;
        },
        0xE7 if v.pp == 1 =>
        {
            let pfx = v.pfx();
            let (modrm, operand) = decode_operand(&pfx)?;
            let lo = xmm_get(modrm.reg);
            let hi = if v.l { ymm_hi(modrm.reg) } else { reg128 { u64: [0, 0] } };
            vex_store(operand, lo, hi, v.l)?;
        },
        // VLDDQU
        0xF0 if v.pp == 3 =>
        {
            let pfx = v.pfx();
            let (modrm, operand) = decode_operand(&pfx)?;
            let (lo, hi) = vex_rm(operand, v.l)?;
            vex_set(modrm.reg, lo, hi, v.l);
        },
        0x74 | 0x75 | 0x76 if v.pp == 1 => return vex_int(v, opcode),
        _ => {},
    }

    if v.pp == 1 {
        return vex_int(v, opcode);
    }

    crate::cpu::core::trigger_ud();
    Ok(())
}

pub(crate) unsafe fn fma_apply(v: &Vex, opcode: u8) -> OrPageFault<()> {
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let (s2lo, s2hi) = vex_rm(operand, v.l)?;
    let (s1lo, s1hi) = vex_v(v);
    let dst = modrm.reg;
    let dlo = xmm_get(dst);
    let dhi = if v.l { ymm_hi(dst) } else { reg128 { u64: [0, 0] } };
    let group = if opcode < 0xA0 { 0 } else if opcode < 0xB0 { 1 } else { 2 };
    let sub = (opcode & 0xF) - 6;
    let scalar = sub % 2 == 1;
    let f64 = v.w;
    // FMADDSUB/FMSUBADD alternate add/sub per element; the others use a fixed op.
    let combine = |p: f64, c: f64, lane: usize| -> f64 {
        match sub {
            0 => if lane % 2 == 0 { p + c } else { p - c },
            1 => if lane % 2 == 0 { p - c } else { p + c },
            2 | 3 => p + c,
            4 | 5 => p - c,
            6 | 7 => -p + c,
            _ => -p - c,
        }
    };
    let calc = |d: reg128, s1: reg128, s2: reg128, lane: usize| -> f64 {
        let (a, b, c) = match group {
            0 => (d, s2, s1),
            1 => (s1, d, s2),
            _ => (s1, s2, d),
        };
        let (p, cc) = if f64 {
            (a.f64[lane] * b.f64[lane], c.f64[lane])
        }
        else {
            (a.f32[lane] as f64 * b.f32[lane] as f64, c.f32[lane] as f64)
        };
        combine(p, cc, lane)
    };
    let mut rlo = dlo;
    let mut rhi = dhi;
    let lanes = if f64 { 2 } else { 4 };
    if scalar {
        if f64 {
            rlo.f64[0] = calc(dlo, s1lo, s2lo, 0);
        }
        else {
            rlo.f32[0] = calc(dlo, s1lo, s2lo, 0) as f32;
        }
        vex_set(dst, rlo, reg128 { u64: [0, 0] }, false);
        return Ok(());
    }
    for i in 0..lanes {
        if f64 {
            rlo.f64[i] = calc(dlo, s1lo, s2lo, i);
            if v.l {
                rhi.f64[i] = calc(dhi, s1hi, s2hi, i);
            }
        }
        else {
            rlo.f32[i] = calc(dlo, s1lo, s2lo, i) as f32;
            if v.l {
                rhi.f32[i] = calc(dhi, s1hi, s2hi, i) as f32;
            }
        }
    }
    vex_set(dst, rlo, rhi, v.l);
    Ok(())
}

pub(crate) unsafe fn ymm_lane(r: u8, i: usize, bytes: u64) -> u64 {
    let lo = xmm_get(r);
    let hi = ymm_hi(r);
    let bits = bytes * 8;
    let bit = i as u32 * bits as u32;
    let word = if bit < 128 { lo.u64[(bit / 64) as usize] } else { hi.u64[((bit - 128) / 64) as usize] };
    let mask = if bytes == 8 { u64::MAX } else { (1u64 << bits) - 1 };
    word >> (bit % 64) & mask
}

pub(crate) unsafe fn ymm_set_lane(r: u8, i: usize, bytes: u64, value: u64) {
    let mut lo = xmm_get(r);
    let mut hi = ymm_hi(r);
    let bits = bytes * 8;
    let bit = i as u32 * bits as u32;
    let mask = if bytes == 8 { u64::MAX } else { (1u64 << bits) - 1 };
    let shift = bit % 64;
    if bit < 128 {
        let w = &mut lo.u64[(bit / 64) as usize];
        *w = (*w & !(mask << shift)) | ((value & mask) << shift);
    }
    else {
        let w = &mut hi.u64[((bit - 128) / 64) as usize];
        *w = (*w & !(mask << shift)) | ((value & mask) << shift);
    }
    xmm_set(r, lo);
    ymm_set_hi(r, hi);
}

// AVX2 gather (VGATHER*/VPGATHER*) with VSIB addressing. The index register
// provides a per-lane offset; the mask (vvvv) selects lanes and is cleared for
// each gathered element.
pub(crate) unsafe fn vex_gather(v: &Vex, opcode: u8) -> OrPageFault<()> {
    let pfx = v.pfx();
    let modrm = decode_modrm(&pfx)?;
    if modrm.mod_bits == 3 || modrm.rm & 7 != 4 {
        crate::cpu::core::trigger_ud();
        return Ok(());
    }
    let sib = fetch8()?;
    let scale = 1u64 << (sib >> 6);
    let index_reg = ((sib >> 3 & 7) | ((pfx.rex & prefix::REX_X) << 2)) as u8;
    let base_low = sib & 7;
    let mut base = if base_low == 5 && modrm.mod_bits == 0 {
        fetch32()? as i32 as i64 as u64
    }
    else {
        read_reg64((base_low | (pfx.rex & prefix::REX_B) << 3) as i32)
    };
    match modrm.mod_bits {
        0 => {},
        1 => base = base.wrapping_add(fetch8()? as i8 as i64 as u64),
        _ => base = base.wrapping_add(fetch32()? as i32 as i64 as u64),
    }
    if pfx.segment == 4 {
        base = base.wrapping_add(*fs_base);
    }
    else if pfx.segment == 5 {
        base = base.wrapping_add(*gs_base);
    }

    let index64 = opcode == 0x91 || opcode == 0x93;
    let data64 = v.w;
    let idx_size = if index64 { 8u64 } else { 4u64 };
    let data_size = if data64 { 8u64 } else { 4u64 };
    let vec_bytes = if v.l { 32usize } else { 16usize };
    let count = core::cmp::min(vec_bytes / data_size as usize, vec_bytes / idx_size as usize);
    let mask_reg = v.vvvv;
    let dst = modrm.reg;
    let sign = 1u64 << (data_size * 8 - 1);
    for i in 0..count {
        let mask_elem = ymm_lane(mask_reg, i, data_size);
        if mask_elem & sign == 0 {
            continue;
        }
        let index_val = ymm_lane(index_reg, i, idx_size);
        let offset = if idx_size == 8 { index_val as i64 as u64 } else { index_val as u32 as i32 as i64 as u64 };
        let addr = base.wrapping_add(offset.wrapping_mul(scale));
        let value = mem_read(addr, if data64 { OpSize::S64 } else { OpSize::S32 })?;
        ymm_set_lane(dst, i, data_size, value);
        ymm_set_lane(mask_reg, i, data_size, 0);
    }
    if !v.l {
        ymm_set_hi(dst, reg128 { u64: [0, 0] });
    }
    Ok(())
}

pub(crate) fn bmi_mask(width: u32) -> u64 { if width == 64 { u64::MAX } else { (1u64 << width) - 1 } }

// BMI1/BMI2 in the VEX 0F38 map. Returns false if the opcode is not one.
pub(crate) unsafe fn vex_bmi_0f38(v: &Vex, opcode: u8) -> OrPageFault<bool> {
    let pfx = v.pfx();
    let size = if v.w { OpSize::S64 } else { OpSize::S32 };
    let width = if v.w { 64u32 } else { 32u32 };
    // BLSI/BLSMSK/BLSR use the ModRM.reg field as an opcode extension.
    if v.pp == 0 && opcode == 0xF3 {
        let (modrm, operand) = decode_operand(&pfx)?;
        let value = read_operand(operand, size, pfx.has_rex())? & bmi_mask(width);
        // BLSI isolates the lowest set bit; BLSMSK fills through it; BLSR clears it.
        let blsmsk = |x: u64| -> u64 {
            if x == 0 { 0 } else { ((1u64 << (64 - x.leading_zeros())) - 1) & bmi_mask(width) }
        };
        let r = match modrm.reg & 7 {
            1 => value & value.wrapping_sub(1), // BLSR
            2 => blsmsk(value),                 // BLSMSK
            3 => value & value.wrapping_neg(),  // BLSI
            _ =>
            {
                crate::cpu::core::trigger_ud();
                return Ok(true);
            },
        };
        write_reg(v.vvvv, size, true, r & bmi_mask(width));
        return Ok(true);
    }
    match (opcode, v.pp) {
        // ANDN: dst = ~vvvv & rm
        (0xF2, 0) =>
        {
            let (modrm, operand) = decode_operand(&pfx)?;
            let src = read_operand(operand, size, pfx.has_rex())?;
            let a = read_reg(v.vvvv, size, true);
            write_reg(modrm.reg, size, true, !a & src & bmi_mask(width));
        },
        // BEXTR: dst = bits [vvvv[7:0], +vvvv[15:8]) of rm
        (0xF7, 0) =>
        {
            let (modrm, operand) = decode_operand(&pfx)?;
            let src = read_operand(operand, size, pfx.has_rex())? & bmi_mask(width);
            let ctrl = read_reg(v.vvvv, size, true);
            let start = ctrl & 0xFF;
            let len = (ctrl >> 8) & 0xFF;
            let r = if start >= width as u64 {
                0
            }
            else if len == 0 {
                0
            }
            else if len >= width as u64 {
                src >> start
            }
            else {
                (src >> start) & bmi_mask(len as u32)
            };
            write_reg(modrm.reg, size, true, r);
        },
        // SHLX (66) / SHRX (F2) / SARX (F3): dst = rm shifted by the count in vvvv
        (0xF7, 1) | (0xF7, 2) | (0xF7, 3) =>
        {
            let (modrm, operand) = decode_operand(&pfx)?;
            let src = read_operand(operand, size, pfx.has_rex())? & bmi_mask(width);
            // countMASK is 3FH under VEX.W1 in 64-bit mode and 1FH otherwise;
            // there is no rule keyed on bit 5 of SRC2, which an earlier version
            // had.
            let index = read_reg(v.vvvv, size, true) as u32;
            // VEX.pp: 1 = 0x66 (SHLX), 3 = 0xF2 (SHRX), 2 = 0xF3 (SARX).
            let op_width = width;
            let count = index & (op_width - 1);
            let src = src & bmi_mask(op_width);
            let r = match v.pp {
                1 => src << count,  // SHLX (66)
                3 => src >> count,  // SHRX (F2)
                _ =>
                {
                    // SARX (F3): sign-extend from the top of the operation width,
                    // shift, then keep only the operation width.
                    let align = 64 - op_width;
                    ((src << align) as i64 >> align >> count) as u64
                },
            };
            write_reg(modrm.reg, size, true, r & bmi_mask(op_width));
        },
        // BZHI keeps SRC2[7:0] low bits.
        (0xF5, 0) =>
        {
            let (modrm, operand) = decode_operand(&pfx)?;
            let src = read_operand(operand, size, pfx.has_rex())? & bmi_mask(width);
            let n = read_reg(v.vvvv, size, true) as u32 & 0xff;
            let r = if n < width { src & ((1u64 << n) - 1) } else { src };
            write_reg(modrm.reg, size, true, r & bmi_mask(width));
            if n > width - 1 {
                *flags |= FLAG_CF as i32;
            }
            else {
                *flags &= !(FLAG_CF as i32);
            }
            *flags_changed = 0;
        },
        // PDEP (F2): deposit the low bits of vvvv into the rm mask
        (0xF5, 3) =>
        {
            let (modrm, operand) = decode_operand(&pfx)?;
            let mask = read_operand(operand, size, pfx.has_rex())? & bmi_mask(width);
            let src = read_reg(v.vvvv, size, true) & bmi_mask(width);
            let mut r = 0u64;
            let mut k = 0u32;
            for i in 0..width {
                if mask >> i & 1 != 0 {
                    if src >> k & 1 != 0 {
                        r |= 1 << i;
                    }
                    k += 1;
                }
            }
            write_reg(modrm.reg, size, true, r);
        },
        // PEXT (F3): gather the bits of vvvv selected by the rm mask
        (0xF5, 2) =>
        {
            let (modrm, operand) = decode_operand(&pfx)?;
            let mask = read_operand(operand, size, pfx.has_rex())? & bmi_mask(width);
            let src = read_reg(v.vvvv, size, true) & bmi_mask(width);
            let mut r = 0u64;
            let mut k = 0u32;
            for i in 0..width {
                if mask >> i & 1 != 0 {
                    if src >> i & 1 != 0 {
                        r |= 1 << k;
                    }
                    k += 1;
                }
            }
            write_reg(modrm.reg, size, true, r);
        },
        // MULX: implicit RDX times rm; reg receives the high half, vvvv the low.
        (0xF6, 3) =>
        {
            let (modrm, operand) = decode_operand(&pfx)?;
            let src2 = read_operand(operand, size, pfx.has_rex())? & bmi_mask(width);
            let src1 = read_reg(RDX, size, true) & bmi_mask(width);
            let product = (src1 as u128) * (src2 as u128);
            // Low half first, so the high half lands last when both destinations
            // are the same register -- which clang's division idiom relies on.
            write_reg(v.vvvv, size, true,
                      (product as u64) & bmi_mask(width));              // low half
            write_reg(modrm.reg, size, true,
                      ((product >> width) as u64) & bmi_mask(width));   // high half
        },
        _ => return Ok(false),
    }
    Ok(true)
}

pub(crate) unsafe fn run_vex_0f38(v: &Vex) -> OrPageFault<()> {
    let opcode = fetch8()?;
    if vex_bmi_0f38(v, opcode)? {
        return Ok(());
    }
    // FMA3 decodes its own operand below.
    if v.pp == 1 && ((0x96..=0x9F).contains(&opcode) || (0xA6..=0xAF).contains(&opcode) || (0xB6..=0xBF).contains(&opcode)) {
        return fma_apply(v, opcode);
    }
    // Gather uses VSIB addressing and decodes its own operand.
    if v.pp == 1 && (0x90..=0x93).contains(&opcode) {
        return vex_gather(v, opcode);
    }
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let (s2lo, s2hi) = vex_rm(operand, v.l)?;
    let (s1lo, s1hi) = vex_v(v);
    let dst = modrm.reg;

    match opcode {
        // VPSHUFB (66) / VPTEST
        0x00 if v.pp == 1 =>
        {
            let rlo = match crate::cpu::interp::simd_instr::ssse3_apply(0x00, s1lo, s2lo) { Some(r) => r, None => reg128 { u64: [0, 0] } };
            let rhi = if v.l {
                match crate::cpu::interp::simd_instr::ssse3_apply(0x00, s1hi, s2hi) { Some(r) => r, None => reg128 { u64: [0, 0] } }
            }
            else {
                reg128 { u64: [0, 0] }
            };
            vex_set(dst, rlo, rhi, v.l);
        },
        0x17 if v.pp == 1 =>
        {
            let (and, andn) = crate::cpu::interp::simd_instr::ptest(s1lo, s2lo);
            let (and2, andn2) = if v.l {
                crate::cpu::interp::simd_instr::ptest(s1hi, s2hi)
            }
            else {
                (0, 0)
            };
            *flags &= !((FLAG_CF | FLAG_ZF) as i32);
            if and == 0 && and2 == 0 {
                *flags |= FLAG_ZF as i32;
            }
            if andn == 0 && andn2 == 0 {
                *flags |= FLAG_CF as i32;
            }
            *flags_changed = 0;
        },
        // VPERMILPS/PD, variable form: dst = vvvv permuted by rm
        0x0C | 0x0D if v.pp == 1 =>
        {
            let perm = |data: reg128, ctrl: reg128, f64: bool| -> reg128 {
                let mut r = data;
                if f64 {
                    r.f64[0] = data.f64[(ctrl.u64[0] & 1) as usize];
                    r.f64[1] = data.f64[(ctrl.u64[1] & 1) as usize];
                }
                else {
                    for i in 0..4 {
                        r.f32[i] = data.f32[(ctrl.u32[i] & 3) as usize];
                    }
                }
                r
            };
            let f64 = opcode == 0x0D;
            let rlo = perm(s1lo, s2lo, f64);
            let rhi = if v.l { perm(s1hi, s2hi, f64) } else { reg128 { u64: [0, 0] } };
            vex_set(dst, rlo, rhi, v.l);
        },
        // VMASKMOVPS/PD load (2C/2D) and store (2E/2F)
        0x2C | 0x2D | 0x2E | 0x2F if v.pp == 1 =>
        {
            let Operand::Mem(address) = operand else {
                crate::cpu::core::trigger_ud();
                return Ok(());
            };
            let store = opcode == 0x2E || opcode == 0x2F;
            let f64 = opcode == 0x2D || opcode == 0x2F;
            let lanes = if v.l { 8 } else { 4 };
            let elem = if f64 { 8u64 } else { 4u64 };
            let mut mask = [0u32; 8];
            for i in 0..4 {
                mask[i] = s1lo.u32[i];
                mask[i + 4] = s1hi.u32[i];
            }
            let source = if store { xmm_get(dst) } else { reg128 { u64: [0, 0] } };
            let mut out = [0u32; 8];
            for i in 0..4 {
                out[i] = source.u32[i];
            }
            if store {
                let mut i = 0;
                while i < lanes {
                    let active = if f64 { mask[i | 1] & 0x8000_0000 != 0 } else { mask[i] & 0x8000_0000 != 0 };
                    if active {
                        if f64 {
                            let lo = source.u64[i / 2];
                            mem_write(address + (i as u64) * elem, OpSize::S64, lo)?;
                        }
                        else {
                            mem_write(address + (i as u64) * elem, OpSize::S32, source.u32[i] as u64)?;
                        }
                    }
                    i += if f64 { 2 } else { 1 };
                }
            }
            else {
                let mut rlo = reg128 { u64: [0, 0] };
                let mut rhi = reg128 { u64: [0, 0] };
                let mut i = 0;
                while i < lanes {
                    let active = if f64 { mask[i | 1] & 0x8000_0000 != 0 } else { mask[i] & 0x8000_0000 != 0 };
                    if active {
                        if f64 {
                            let v = mem_read(address + (i as u64) * elem, OpSize::S64)?;
                            if i < 4 {
                                rlo.u64[i / 2] = v;
                            }
                            else {
                                rhi.u64[(i - 4) / 2] = v;
                            }
                        }
                        else {
                            let v = mem_read(address + (i as u64) * elem, OpSize::S32)? as u32;
                            if i < 4 {
                                rlo.u32[i] = v;
                            }
                            else {
                                rhi.u32[i - 4] = v;
                            }
                        }
                    }
                    i += if f64 { 2 } else { 1 };
                }
                vex_set(dst, rlo, rhi, v.l);
            }
        },
        // VTESTPS/PD
        0x0E | 0x0F if v.pp == 1 =>
        {
            let (and, andn) = crate::cpu::interp::simd_instr::ptest(s1lo, s2lo);
            *flags &= !((FLAG_CF | FLAG_ZF) as i32);
            if and == 0 {
                *flags |= FLAG_ZF as i32;
            }
            if andn == 0 {
                *flags |= FLAG_CF as i32;
            }
            *flags_changed = 0;
        },
        // VBROADCASTSS / VBROADCASTSD / VBROADCASTF128
        0x18 if v.pp == 1 =>
        {
            let value = s2lo.f32[0];
            let mut rlo = reg128 { u64: [0, 0] };
            let mut rhi = reg128 { u64: [0, 0] };
            for i in 0..4 {
                rlo.f32[i] = value;
                rhi.f32[i] = value;
            }
            vex_set(dst, rlo, rhi, v.l);
        },
        0x19 if v.pp == 1 =>
        {
            let value = s2lo.f64[0];
            let mut rlo = reg128 { u64: [0, 0] };
            let mut rhi = reg128 { u64: [0, 0] };
            for i in 0..2 {
                rlo.f64[i] = value;
                rhi.f64[i] = value;
            }
            vex_set(dst, rlo, rhi, v.l);
        },
        0x1A if v.pp == 1 =>
        {
            vex_set(dst, s2lo, s2lo, v.l);
        },
        // VPERMPS/VPERMD (256-bit dword permute across the whole register)
        0x16 | 0x36 if v.pp == 1 =>
        {
            if !v.l {
                crate::cpu::core::trigger_ud();
                return Ok(());
            }
            let mut src = [0u32; 8];
            let mut idx = [0u32; 8];
            for i in 0..4 {
                src[i] = s2lo.u32[i];
                src[i + 4] = s2hi.u32[i];
                idx[i] = s1lo.u32[i];
                idx[i + 4] = s1hi.u32[i];
            }
            let mut rlo = reg128 { u64: [0, 0] };
            let mut rhi = reg128 { u64: [0, 0] };
            for i in 0..4 {
                rlo.u32[i] = src[(idx[i] & 7) as usize];
                rhi.u32[i] = src[(idx[i + 4] & 7) as usize];
            }
            vex_set(dst, rlo, rhi, true);
        },
        // Variable shifts: VEX.W selects 32-bit or 64-bit lanes.
        0x45 | 0x46 | 0x47 if v.pp == 1 =>
        {
            let kind = match opcode {
                0x45 => 0,
                0x46 => 1,
                _ => 2,
            };
            let shift_lanes = |value: reg128, counts: reg128| {
                if v.w {
                    crate::cpu::interp::simd_instr::variable_shift64(value, counts, kind)
                }
                else {
                    crate::cpu::interp::simd_instr::variable_shift32(value, counts, kind)
                }
            };
            let rlo = shift_lanes(s1lo, s2lo);
            let rhi = if v.l { shift_lanes(s1hi, s2hi) } else { reg128 { u64: [0, 0] } };
            vex_set(dst, rlo, rhi, v.l);
        },
        // VPBROADCASTD/Q/B/W and VBROADCASTI128
        0x58 | 0x59 | 0x78 | 0x79 | 0x5A if v.pp == 1 =>
        {
            let mut rlo = reg128 { u64: [0, 0] };
            let mut rhi = reg128 { u64: [0, 0] };
            match opcode {
                0x58 =>
                {
                    let value = s2lo.u32[0];
                    for i in 0..4 {
                        rlo.u32[i] = value;
                        rhi.u32[i] = value;
                    }
                },
                0x59 =>
                {
                    let value = s2lo.u64[0];
                    rlo.u64[0] = value;
                    rlo.u64[1] = value;
                    rhi.u64[0] = value;
                    rhi.u64[1] = value;
                },
                0x78 =>
                {
                    let value = s2lo.u8[0];
                    for i in 0..16 {
                        rlo.u8[i] = value;
                        rhi.u8[i] = value;
                    }
                },
                0x79 =>
                {
                    let value = s2lo.u16[0];
                    for i in 0..8 {
                        rlo.u16[i] = value;
                        rhi.u16[i] = value;
                    }
                },
                _ =>
                {
                    rlo = s2lo;
                    rhi = s2lo;
                },
            }
            vex_set(dst, rlo, rhi, v.l);
        },
        // VPMASKMOVD/Q load (0x8C/0x8D) and store (0x8E/0x8F)
        0x8C | 0x8D | 0x8E | 0x8F if v.pp == 1 =>
        {
            let Operand::Mem(address) = operand else {
                crate::cpu::core::trigger_ud();
                return Ok(());
            };
            let store = opcode == 0x8E || opcode == 0x8F;
            let qword = opcode == 0x8D || opcode == 0x8F;
            let lanes = if v.l { 8 } else { 4 };
            // Per-lane mask sign bits: qword uses the high dword of each qword.
            let mut mask = [0u32; 8];
            for i in 0..4 {
                mask[i] = s1lo.u32[i];
                mask[i + 4] = s1hi.u32[i];
            }
            let mut value = [0u32; 8];
            for i in 0..4 {
                value[i] = s2lo.u32[i];
                value[i + 4] = s2hi.u32[i];
            }
            if store {
                let src = xmm_get(dst);
                for i in 0..4 {
                    value[i] = src.u32[i];
                }
            }
            let active = |i: usize| -> bool {
                let sign = if qword { mask[i | 1] } else { mask[i] };
                sign & 0x8000_0000 != 0
            };
            if store {
                for i in 0..lanes {
                    if active(i) {
                        mem_write(address + (i as u64) * 4, OpSize::S32, value[i] as u64)?;
                    }
                }
            }
            else {
                let mut out = [0u32; 8];
                for i in 0..lanes {
                    if active(i) {
                        out[i] = mem_read(address + (i as u64) * 4, OpSize::S32)? as u32;
                    }
                }
                let mut rlo = reg128 { u64: [0, 0] };
                let mut rhi = reg128 { u64: [0, 0] };
                for i in 0..4 {
                    rlo.u32[i] = out[i];
                    rhi.u32[i] = out[i + 4];
                }
                vex_set(dst, rlo, rhi, v.l);
            }
        },
        // VAESIMC/VAESENC/VAESENCLAST/VAESDEC/VAESDECLAST
        0xDB | 0xDC | 0xDD | 0xDE | 0xDF if v.pp == 1 =>
        {
            let one = |s1: reg128, s2: reg128| -> reg128 {
                match opcode {
                    0xDB => crate::cpu::interp::simd_instr::aesimc(s2),
                    0xDC => crate::cpu::interp::simd_instr::aesenc(s1, s2, false),
                    0xDD => crate::cpu::interp::simd_instr::aesenc(s1, s2, true),
                    0xDE => crate::cpu::interp::simd_instr::aesdec(s1, s2, false),
                    _ => crate::cpu::interp::simd_instr::aesdec(s1, s2, true),
                }
            };
            let rlo = one(s1lo, s2lo);
            let rhi = if v.l { one(s1hi, s2hi) } else { reg128 { u64: [0, 0] } };
            vex_set(dst, rlo, rhi, v.l);
        },
        // SSE4.1/4.2 0F38 (three-operand): the shared apply takes (dst, src),
        // here src1 is vvvv and src2 is rm.
        _ if v.pp == 1 =>
        {
            if let Some(rlo) = crate::cpu::interp::simd_instr::sse4_38_apply(opcode, s1lo, s2lo) {
                let rhi = if v.l {
                    crate::cpu::interp::simd_instr::sse4_38_apply(opcode, s1hi, s2hi).unwrap_or(reg128 { u64: [0, 0] })
                }
                else {
                    reg128 { u64: [0, 0] }
                };
                vex_set(dst, rlo, rhi, v.l);
            }
            else if let Some(rlo) = crate::cpu::interp::simd_instr::ssse3_apply(opcode, s1lo, s2lo) {
                let rhi = if v.l {
                    crate::cpu::interp::simd_instr::ssse3_apply(opcode, s1hi, s2hi).unwrap_or(reg128 { u64: [0, 0] })
                }
                else {
                    reg128 { u64: [0, 0] }
                };
                vex_set(dst, rlo, rhi, v.l);
            }
            else {
                crate::cpu::core::trigger_ud();
            }
        },
        _ => crate::cpu::core::trigger_ud(),
    }
    Ok(())
}

pub(crate) unsafe fn run_vex_0f3a(v: &Vex) -> OrPageFault<()> {
    let opcode = fetch8()?;
    // RORX (VEX.LZ.F2.0F3A F0 /r ib): rotate rm right by imm8 into a GPR.
    if v.pp == 3 && opcode == 0xF0 {
        let pfx = v.pfx();
        let (modrm, operand) = decode_operand(&pfx)?;
        let size = if v.w { OpSize::S64 } else { OpSize::S32 };
        let width = if v.w { 64u32 } else { 32u32 };
        let src = read_operand(operand, size, pfx.has_rex())? & bmi_mask(width);
        let imm = fetch8()? as u32 & (width - 1);
        let r = if imm == 0 { src } else { (src >> imm) | (src << (width - imm)) };
        write_reg(modrm.reg, size, true, r & bmi_mask(width));
        return Ok(());
    }
    let pfx = v.pfx();
    let (modrm, operand) = decode_operand(&pfx)?;
    let (s2lo, s2hi) = vex_rm(operand, v.l)?;
    let (s1lo, s1hi) = vex_v(v);
    let dst = modrm.reg;
    let imm = fetch8()? as u8;

    match opcode {
        // VPERM2F128
        0x06 if v.pp == 1 =>
        {
            let pick = |which: u8| -> reg128 {
                let bit = which & 1 != 0;
                if which & 8 != 0 {
                    reg128 { u64: [0, 0] }
                }
                else if bit {
                    if which & 2 != 0 { s2hi } else { s1hi }
                }
                else if which & 2 != 0 {
                    s2lo
                }
                else {
                    s1lo
                }
            };
            let rlo = pick(imm & 0x0F);
            let rhi = pick((imm >> 4) & 0x0F);
            vex_set(dst, rlo, rhi, v.l);
        },
        // VROUNDPS/PD (08/09) and VROUNDSS/SD (0A/0B)
        0x08 | 0x09 | 0x0A | 0x0B if v.pp == 1 =>
        {
            if opcode == 0x08 || opcode == 0x0A {
                let lanes = if opcode == 0x08 { 4 } else { 1 };
                let mut rlo = s1lo;
                let mut rhi = s1hi;
                for i in 0..lanes {
                    rlo.f32[i] = crate::cpu::interp::simd_instr::round_apply(s2lo.f32[i] as f64, imm) as f32;
                }
                if opcode == 0x08 && v.l {
                    for i in 0..4 {
                        rhi.f32[i] = crate::cpu::interp::simd_instr::round_apply(s2hi.f32[i] as f64, imm) as f32;
                    }
                }
                vex_set(dst, rlo, rhi, v.l);
            }
            else {
                let lanes = if opcode == 0x09 { 2 } else { 1 };
                let mut rlo = s1lo;
                let mut rhi = s1hi;
                for i in 0..lanes {
                    rlo.f64[i] = crate::cpu::interp::simd_instr::round_apply(s2lo.f64[i], imm);
                }
                if opcode == 0x09 && v.l {
                    for i in 0..2 {
                        rhi.f64[i] = crate::cpu::interp::simd_instr::round_apply(s2hi.f64[i], imm);
                    }
                }
                vex_set(dst, rlo, rhi, v.l);
            }
        },
        // VBLENDPS/VBLENDPD/VPBLENDW
        0x0C | 0x0D | 0x0E if v.pp == 1 =>
        {
            let (lanes, bits) = match opcode {
                0x0C => (4, 32),
                0x0D => (2, 64),
                _ => (8, 16),
            };
            let rlo = crate::cpu::interp::simd_instr::blend_imm_apply(s1lo, s2lo, imm, lanes, bits);
            let rhi = if v.l {
                crate::cpu::interp::simd_instr::blend_imm_apply(s1hi, s2hi, imm, lanes, bits)
            }
            else {
                reg128 { u64: [0, 0] }
            };
            vex_set(dst, rlo, rhi, v.l);
        },
        // VPALIGNR
        0x0F if v.pp == 1 =>
        {
            let align = |a: reg128, b: reg128| -> reg128 {
                let mut temp = [0u8; 32];
                temp[..16].copy_from_slice(&b.u8);
                temp[16..].copy_from_slice(&a.u8);
                let mut r = reg128 { u64: [0, 0] };
                let shift = imm as usize & 0x1F;
                for i in 0..16 {
                    r.u8[i] = if shift + i < 32 { temp[shift + i] } else { 0 };
                }
                r
            };
            let rlo = align(s1lo, s2lo);
            let rhi = if v.l { align(s1hi, s2hi) } else { reg128 { u64: [0, 0] } };
            vex_set(dst, rlo, rhi, v.l);
        },
        // VINSERTF128
        0x18 if v.pp == 1 =>
        {
            let mut rlo = s1lo;
            let mut rhi = s1hi;
            if imm & 1 != 0 {
                rhi = s2lo;
            }
            else {
                rlo = s2lo;
            }
            vex_set(dst, rlo, rhi, true);
        },
        // VEXTRACTF128
        0x19 if v.pp == 1 =>
        {
            let value = if imm & 1 != 0 { s1hi } else { s1lo };
            vex_store(operand, value, reg128 { u64: [0, 0] }, false)?;
        },
        // VBLENDVPS/VBLENDVPD/VPBLENDVB (mask register in imm[7:4])
        0x4A | 0x4B | 0x4C if v.pp == 1 =>
        {
            let mask_reg = imm >> 4;
            let mask = xmm_get(mask_reg);
            let element_bytes = match opcode { 0x4A => 4, 0x4B => 8, _ => 1 };
            let rlo = crate::cpu::interp::simd_instr::blendv_apply(s1lo, s2lo, mask, element_bytes);
            let rhi = if v.l {
                let mhi = ymm_hi(mask_reg);
                crate::cpu::interp::simd_instr::blendv_apply(s1hi, s2hi, mhi, element_bytes)
            }
            else {
                reg128 { u64: [0, 0] }
            };
            vex_set(dst, rlo, rhi, v.l);
        },
        // VPERMILPS/PD with an immediate
        // VPERMQ (00) and VPERMPD (01), SDM Vol. 2. VPERMQ reads a 2-bit field
        // per destination qword, VPERMPD one field for all of them, and the
        // source is the whole vector -- so in the 256-bit form a lane can cross
        // the 128-bit boundary and the halves cannot be permuted separately.
        //
        // Out-of-range selectors are masked rather than rejected. All eight
        // immediate bits are live at 256 bits, so `vpermq $0x1b` -- reverse all
        // lanes, and what clang emits -- is legal; rejecting the unused high bits
        // once made it #UD. Masking cannot reject an encoding real hardware
        // accepts, which is the safer direction to be wrong in.
        0x00 | 0x01 if v.pp == 1 && v.w =>
        {
            let lanes = if v.l { 4 } else { 2 };
            let pick = |lo: reg128, hi: reg128, i: usize| -> u64 {
                match i {
                    0 => lo.u64[0],
                    1 => lo.u64[1],
                    2 => hi.u64[0],
                    _ => hi.u64[1],
                }
            };
            let mut out = [0u64; 4];
            if opcode == 0x01 {
                let sel = (imm & 3) as usize % lanes;
                for slot in out.iter_mut().take(lanes) {
                    *slot = pick(s2lo, s2hi, sel);
                }
            }
            else {
                for i in 0..lanes {
                    let sel = ((imm >> (i * 2)) & 3) as usize % lanes;
                    out[i] = pick(s2lo, s2hi, sel);
                }
            }
            let rlo = reg128 { u64: [out[0], out[1]] };
            let rhi = if v.l { reg128 { u64: [out[2], out[3]] } } else { reg128 { u64: [0, 0] } };
            vex_set(dst, rlo, rhi, v.l);
        },
        0x04 | 0x05 if v.pp == 1 =>
        {
            let perm = |data: reg128, f64: bool| -> reg128 {
                let mut r = data;
                if f64 {
                    r.f64[0] = data.f64[(imm & 1) as usize];
                    r.f64[1] = data.f64[((imm >> 1) & 1) as usize];
                }
                else {
                    for i in 0..4 {
                        r.f32[i] = data.f32[((imm >> (i * 2)) & 3) as usize];
                    }
                }
                r
            };
            let f64 = opcode == 0x05;
            let rlo = perm(s2lo, f64);
            let rhi = if v.l { perm(s2hi, f64) } else { reg128 { u64: [0, 0] } };
            vex_set(dst, rlo, rhi, v.l);
        },
        // VPEXTRB/W/D/Q and VEXTRACTPS (source is vvvv, destination is rm)
        0x14 | 0x15 | 0x16 | 0x17 if v.pp == 1 =>
        {
            let (value, size) = match opcode {
                0x14 => (s1lo.u8[imm as usize] as u64, OpSize::S8),
                0x15 => (s1lo.u16[imm as usize & 7] as u64, OpSize::S16),
                0x16 =>
                    if v.w {
                        (s1lo.u64[imm as usize & 1], OpSize::S64)
                    }
                    else {
                        (s1lo.u32[imm as usize & 3] as u64, OpSize::S32)
                    },
                _ => (s1lo.u32[imm as usize & 3] as u64, OpSize::S32),
            };
            match operand {
                Operand::Reg(r) => write_reg(r, size, true, value),
                Operand::Mem(address) =>
                {
                    let _ = size;
                    match opcode {
                        0x14 => mem_write(address, OpSize::S8, value)?,
                        0x15 => mem_write(address, OpSize::S16, value)?,
                        0x16 if v.w => mem_write(address, OpSize::S64, value)?,
                        _ => mem_write(address, OpSize::S32, value)?,
                    }
                },
            }
        },
        // VPINSRB/VINSERTPS/VPINSRD/Q (base is vvvv, source is rm)
        0x20 | 0x21 | 0x22 if v.pp == 1 =>
        {
            if opcode == 0x21 {
                vex_set(dst, crate::cpu::interp::simd_instr::insertps(s1lo, s2lo, imm), s1hi, v.l);
            }
            else {
                let size = if opcode == 0x20 {
                    OpSize::S8
                }
                else if v.w {
                    OpSize::S64
                }
                else {
                    OpSize::S32
                };
                let value = read_operand(operand, size, true)?;
                let mut rlo = s1lo;
                match size {
                    OpSize::S8 => rlo.u8[imm as usize] = value as u8,
                    OpSize::S32 => rlo.u32[imm as usize & 3] = value as u32,
                    _ => rlo.u64[imm as usize & 1] = value,
                }
                vex_set(dst, rlo, s1hi, v.l);
            }
        },
        // VDPPS/VDPPD/VMPSADBW
        0x40 | 0x41 | 0x42 if v.pp == 1 =>
        {
            let (rlo, rhi) = match opcode {
                0x40 => (
                    crate::cpu::interp::simd_instr::dpps(s1lo, s2lo, imm),
                    if v.l { crate::cpu::interp::simd_instr::dpps(s1hi, s2hi, imm) } else { reg128 { u64: [0, 0] } },
                ),
                0x41 => (
                    crate::cpu::interp::simd_instr::dppd(s1lo, s2lo, imm),
                    if v.l { crate::cpu::interp::simd_instr::dppd(s1hi, s2hi, imm) } else { reg128 { u64: [0, 0] } },
                ),
                _ => (
                    crate::cpu::interp::simd_instr::mpsadbw(s1lo, s2lo, imm),
                    if v.l { crate::cpu::interp::simd_instr::mpsadbw(s1hi, s2hi, imm) } else { reg128 { u64: [0, 0] } },
                ),
            };
            vex_set(dst, rlo, rhi, v.l);
        },
        // VPCMPESTRM/ESTRI (60/61) and VPCMPISTRM/ISTRI (62/63)
        0x60 | 0x61 | 0x62 | 0x63 if v.pp == 1 =>
        {
            let (la, lb) = if opcode == 0x60 || opcode == 0x61 {
                (
                    read_reg(RAX, OpSize::S32, false) as u32 as i32,
                    read_reg(RDX, OpSize::S32, false) as u32 as i32,
                )
            }
            else {
                (i32::MIN, i32::MIN)
            };
            let a = s1lo.u8;
            let b = s2lo.u8;
            let r = crate::cpu::interp::simd_instr::strcmp_apply(&a, &b, la, lb, imm);
            if opcode == 0x60 || opcode == 0x62 {
                xmm_set(0, r.mask);
            }
            write_reg(RCX, OpSize::S32, false, r.index as u64);
            *flags &= !((FLAG_CF | FLAG_ZF | FLAG_SF | FLAG_OF | FLAG_AF | FLAG_PF) as i32);
            if r.intres != 0 {
                *flags |= FLAG_CF as i32;
            }
            if r.zf {
                *flags |= FLAG_ZF as i32;
            }
            if r.sf {
                *flags |= FLAG_SF as i32;
            }
            if r.of {
                *flags |= FLAG_OF as i32;
            }
            *flags_changed = 0;
        },
        // VAESKEYGENASSIST
        0xDF if v.pp == 1 =>
        {
            let rlo = crate::cpu::interp::simd_instr::aeskeygenassist(s2lo, imm);
            let rhi = if v.l { crate::cpu::interp::simd_instr::aeskeygenassist(s2hi, imm) } else { reg128 { u64: [0, 0] } };
            vex_set(dst, rlo, rhi, v.l);
        },
        // VPCLMULQDQ
        0x44 if v.pp == 1 =>
        {
            let rlo = crate::cpu::interp::simd_instr::pclmulqdq(s1lo, s2lo, imm);
            let rhi = if v.l { crate::cpu::interp::simd_instr::pclmulqdq(s1hi, s2hi, imm) } else { reg128 { u64: [0, 0] } };
            vex_set(dst, rlo, rhi, v.l);
        },
        // VPBLENDD
        0x02 if v.pp == 1 =>
        {
            let rlo = crate::cpu::interp::simd_instr::blend_d(s1lo, s2lo, imm & 0x0F);
            let rhi = if v.l {
                crate::cpu::interp::simd_instr::blend_d(s1hi, s2hi, imm >> 4)
            }
            else {
                reg128 { u64: [0, 0] }
            };
            vex_set(dst, rlo, rhi, v.l);
        },
        // VINSERTI128
        0x38 if v.pp == 1 =>
        {
            let mut rlo = s1lo;
            let mut rhi = s1hi;
            if imm & 1 != 0 {
                rhi = s2lo;
            }
            else {
                rlo = s2lo;
            }
            vex_set(dst, rlo, rhi, true);
        },
        // VEXTRACTI128
        0x39 if v.pp == 1 =>
        {
            let value = if imm & 1 != 0 { s1hi } else { s1lo };
            vex_store(operand, value, reg128 { u64: [0, 0] }, false)?;
        },
        // VPERM2I128
        0x46 if v.pp == 1 =>
        {
            let pick = |which: u8| -> reg128 {
                if which & 8 != 0 {
                    reg128 { u64: [0, 0] }
                }
                else if which & 1 != 0 {
                    if which & 2 != 0 { s2hi } else { s1hi }
                }
                else if which & 2 != 0 {
                    s2lo
                }
                else {
                    s1lo
                }
            };
            let rlo = pick(imm & 0x0F);
            let rhi = pick((imm >> 4) & 0x0F);
            vex_set(dst, rlo, rhi, v.l);
        },
        _ => crate::cpu::core::trigger_ud(),
    }
    Ok(())
}

pub(crate) unsafe fn run_vex(first: u8, outer: &Prefixes) -> OrPageFault<()> {
    let mut v = decode_vex(first)?;
    v.segment = outer.segment;
    match v.map {
        1 => run_vex_0f(&v),
        2 => run_vex_0f38(&v),
        3 => run_vex_0f3a(&v),
        _ =>
        {
            crate::cpu::core::trigger_ud();
            Ok(())
        },
    }
}

#[inline(always)]
pub(crate) unsafe fn run_0f(pfx: &Prefixes) -> OrPageFault<()> {
    let opcode = fetch8()?;
    if INTERP64_OPCODE_STATS {
        INTERP64_OPCODE_0F[opcode as usize] = INTERP64_OPCODE_0F[opcode as usize].wrapping_add(1);
    }
    let size = pfx.operand_size();

    if run_sse(opcode, pfx)? {
        return Ok(());
    }

    match opcode {
        // ENDBR64 (F3 0F 1E FA) / ENDBR32 (F3 0F 1E FB): no effect
        0x1E if pfx.f3 => {
            let modrm = fetch8()?;
            if modrm != 0xFA && modrm != 0xFB {
                crate::cpu::core::trigger_ud();
            }
        },

        // LAR/LSL (0F 02/03); only the GDT is modelled, else ZF is cleared.
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/lar
        0x02 | 0x03 => {
            let (modrm, operand) = decode_operand(&pfx)?;
            let selector = read_operand(operand, OpSize::S16, pfx.has_rex())? as u16;
            let index = (selector >> 3) as u64;
            let in_ldt = selector >> 2 & 1 != 0;
            let descriptor = if selector & !7 != 0 && !in_ldt {
                mem_read(GDTR_BASE + index * 8, OpSize::S64)?
            }
            else {
                0
            };
            let present = descriptor >> 47 & 1 != 0;
            if present {
                let value = if opcode == 0x02 {
                    // access rights in bits 8-15, flags (AVL/L/D/B/G) in 20-23;
                    // a 64-bit destination takes them in the upper dword
                    let rights = (descriptor >> 40 & 0xFF) << 8 | (descriptor >> 52 & 0xF) << 20;
                    if size == OpSize::S64 { rights << 32 } else { rights }
                }
                else {
                    let mut limit = descriptor & 0xFFFF | (descriptor >> 32 & 0xF0000);
                    if descriptor >> 55 & 1 != 0 {
                        // page granularity: limit is scaled by 4 KiB
                        limit = (limit << 12) | 0xFFF;
                    }
                    limit
                };
                *flags &= !(FLAG_ZF as i32);
                *flags |= FLAG_ZF as i32;
                write_reg(modrm.reg, size, pfx.has_rex(), value & size.mask());
            }
            else {
                *flags &= !(FLAG_ZF as i32);
            }
        },

        // FEMMS (0F 0E): no-op, MMX state is not modelled
        0x0E => {},

        // SHLD/SHRD r/m, r, imm8 or CL.
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/shld
        0xA4 | 0xA5 | 0xAC | 0xAD => {
            let (modrm, operand) = decode_operand(&pfx)?;
            let bits = size.bits();
            let count = if opcode & 1 == 0 {
                fetch8()? as u32
            }
            else {
                read_reg(RCX, OpSize::S8, false) as u32
            } & 0x3F;
            let count = count % bits;
            let value = read_operand(operand, size, pfx.has_rex())? & size.mask();
            let source = read_reg(modrm.reg, size, pfx.has_rex()) & size.mask();
            let result = if count == 0 {
                value
            }
            else if opcode & 0x08 == 0 {
                (value << count | source >> (bits - count)) & size.mask()
            }
            else {
                (value >> count | source << (bits - count)) & size.mask()
            };
            if count != 0 {
                let carry = if opcode & 0x08 == 0 {
                    value >> (bits - count) & 1
                }
                else {
                    value >> (count - 1) & 1
                };
                *flags &= !(FLAG_CF as i32);
                *flags |= (carry as i32) * FLAG_CF as i32;
            }
            probe_operand_write(operand, size)?;
            write_operand(operand, size, pfx.has_rex(), result)?;
        },

        // Three-byte escapes
        0x38 => run_0f38(&pfx)?,
        0x3A => run_0f3a(&pfx)?,

        // EMMS (0F 77): MMX state is not modelled
        0x77 => {},

        // PUSH/POP FS (0F A0/A1) and GS (0F A8/A9)
        0xA0 => push_segment(4)?,
        0xA1 => pop_segment(4)?,
        0xA8 => push_segment(5)?,
        0xA9 => pop_segment(5)?,

        // MOVNTI (0F C3), a plain store here; cf. Intel SDM Vol. 2:
// https://www.felixcloutier.com/x86/movnti
        0xC3 => {
            let (modrm, operand) = decode_operand(&pfx)?;
            let value = read_reg64(modrm.reg as i32);
            probe_operand_write(operand, size)?;
            write_operand(operand, size, pfx.has_rex(), value)?;
        },

        // LSS/LFS/LGS (0F B2/B4/B5).
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/lss
        0xB2 => load_far(2, size, &pfx)?,
        0xB4 => load_far(4, size, &pfx)?,
        0xB5 => load_far(5, size, &pfx)?,

        // RDPMC (0F 33); cf. Intel SDM Vol. 2:
// https://www.felixcloutier.com/x86/rdpmc
        0x33 => {
            if *cpl != 0 && *cr.offset(4) & (1 << 8) == 0 {
                crate::cpu::core::trigger_gp(0);
                return Ok(());
            }
            let tsc = read_tsc();
            write_reg(RAX, OpSize::S32, false, tsc & 0xFFFF_FFFF);
            write_reg(RDX, OpSize::S32, false, tsc >> 32);
        },

        // SYSENTER/SYSEXIT (0F 34/35).
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/sysenter
        0x34 => {
            *sreg.offset(1) = *sysenter_cs as u16 & !3;
            *sreg.offset(2) = *sysenter_cs as u16 + 8;
            *cpl = 0;
            write_reg64(RSP as i32, *sysenter_esp as u32 as u64);
            *rip = *sysenter_eip as u32 as u64;
        },
        0x35 => {
            *sreg.offset(1) = *sysenter_cs as u16 + 16 | 3;
            *sreg.offset(2) = *sysenter_cs as u16 + 24 | 3;
            *cpl = 3;
            write_reg64(RSP as i32, read_reg64(RCX as i32));
            *rip = read_reg64(RDX as i32);
        },

        // MOV DR/TR (0F 21/23/24/26); cf. Intel SDM Vol. 2:
// https://www.felixcloutier.com/x86/mov-1
        0x24 | 0x26 => {
            let modrm = decode_modrm(&pfx)?;
            if modrm.mod_bits != 3 {
                crate::cpu::core::trigger_ud();
                return Ok(());
            }
            *dreg.offset((modrm.reg & 7) as isize) = read_reg64(modrm.rm as i32) as u32 as i32;
        },

        // PREFETCHT0/T1/T2/NTA (0F 18 /0../3) and PREFETCHW (0F 0D /0,/1):
        // no architectural effect; the operand is decoded and never faults.
        0x18 | 0x0D => {
            let _ = decode_operand(pfx)?;
        },

        // CMPXCHG8B/CMPXCHG16B (0F C7 /1), RDRAND (0F C7 /6)
        0xC7 => {
            let modrm = decode_modrm(pfx)?;
            match modrm.reg & 7 {
                1 if modrm.mod_bits != 3 => {
                    let operand = decode_operand_after_modrm(pfx, modrm, 0)?.1;
                    let Operand::Mem(address) = operand else {
                        crate::cpu::core::trigger_ud();
                        return Ok(());
                    };
                    let acc_lo = read_reg(RAX, OpSize::S64, false);
                    let acc_hi = read_reg(RDX, OpSize::S64, false);
                    if pfx.rex & prefix::REX_W != 0 {
                        // CMPXCHG16B: compare RDX:RAX with m128 and exchange
                        let lo = mem_read(address, OpSize::S64)?;
                        let hi = mem_read(address + 8, OpSize::S64)?;
                        if lo == acc_lo && hi == acc_hi {
                            set_zf(true);
                            mem_write(address, OpSize::S64, read_reg(RBX, OpSize::S64, false))?;
                            mem_write(address + 8, OpSize::S64, read_reg(RCX, OpSize::S64, false))?;
                        }
                        else {
                            set_zf(false);
                            write_reg(RAX, OpSize::S64, false, lo);
                            write_reg(RDX, OpSize::S64, false, hi);
                        }
                    }
                    else {
                        // CMPXCHG8B: compare EDX:EAX with m64 and exchange
                        let lo = mem_read(address, OpSize::S64)?;
                        if lo == (acc_lo & 0xFFFF_FFFF) | (acc_hi << 32) {
                            set_zf(true);
                            mem_write(address, OpSize::S32, read_reg(RBX, OpSize::S32, false))?;
                            mem_write(address + 4, OpSize::S32, read_reg(RCX, OpSize::S32, false))?;
                        }
                        else {
                            set_zf(false);
                            write_reg(RAX, OpSize::S32, false, lo & 0xFFFF_FFFF);
                            write_reg(RDX, OpSize::S32, false, lo >> 32);
                        }
                    }
                },
                6 | 7 if modrm.mod_bits == 3 => {
                    // RDRAND (0F C7 /6) / RDSEED (0F C7 /7): CF reports the
                    // result, OF/SF/ZF/AF/PF are all set to 0. Clearing only CF
                    // and OF left the other four visible to the caller.
                    let low = crate::cpu::core::js::get_rand_int() as u32 as u64;
                    let high = crate::cpu::core::js::get_rand_int() as u32 as u64;
                    let random = if size == OpSize::S64 { low | high << 32 } else { low };
                    write_reg(modrm.rm, size, pfx.has_rex(), random);
                    *flags &= !((FLAG_CF | FLAG_OF | FLAG_SF | FLAG_ZF | FLAG_AF
                                 | FLAG_PF) as i32);
                    *flags |= FLAG_CF as i32;
                    *flags_changed = 0;
                },
                _ => crate::cpu::core::trigger_ud(),
            }
        },

        // Jcc rel32 (0F 80-0F 8F)
        0x80..=0x8F => {
            let displacement = fetch32()? as i32 as i64;
            jcc(opcode - 0x80, displacement);
        },

        // CMOVcc r, r/m
        0x40..=0x4F => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = read_operand(operand, size, pfx.has_rex())?;
            if condition(opcode - 0x40) {
                write_reg(modrm.reg, size, pfx.has_rex(), value);
            }
        },

        // SETcc r/m8
        0x90..=0x9F => {
            let (_, operand) = decode_operand(pfx)?;
            write_operand(
                operand,
                OpSize::S8,
                pfx.has_rex(),
                condition(opcode - 0x90) as u64,
            )?;
        },

        // MOVZX r, r/m8 (0F B6) / r/m16 (0F B7)
        0xB6 | 0xB7 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src_size = if opcode == 0xB6 { OpSize::S8 } else { OpSize::S16 };
            let value = read_operand(operand, src_size, pfx.has_rex())?;
            write_reg(modrm.reg, size, pfx.has_rex(), value);
        },

        // MOVSX r, r/m8 (0F BE) / r/m16 (0F BF)
        0xBE | 0xBF => {
            let (modrm, operand) = decode_operand(pfx)?;
            let (src_size, sign) = if opcode == 0xBE {
                (OpSize::S8, 0x80u64)
            }
            else {
                (OpSize::S16, 0x8000u64)
            };
            let value = read_operand(operand, src_size, pfx.has_rex())?;
            let value = if value & sign != 0 {
                value | !src_size.mask()
            }
            else {
                value
            } & size.mask();
            write_reg(modrm.reg, size, pfx.has_rex(), value);
        },

        // IMUL r, r/m (0F AF)
        0xAF => {
            let (modrm, operand) = decode_operand(pfx)?;
            let lhs = read_reg(modrm.reg, size, pfx.has_rex());
            let rhs = read_operand(operand, size, pfx.has_rex())?;
            write_reg(modrm.reg, size, pfx.has_rex(), imul_value(lhs, rhs, size));
        },

        // NOP r/m (0F 1F)
        0x1F => {
            let _ = decode_operand(pfx)?;
        },

        // ENDBR64/ENDBR32
        0x1E => {
            match fetch8()? {
                0xFA | 0xFB => {},
                _ => crate::cpu::core::trigger_ud(),
            }
        },

        // MOV r64, CRn (0F 20) / MOV CRn, r64 (0F 22)
        0x20 | 0x22 => {
            if *cpl != 0 {
                crate::cpu::core::trigger_gp(0);
                return Ok(());
            }
            let modrm = decode_modrm(pfx)?;
            if modrm.mod_bits != 3 || modrm.reg > 7 {
                crate::cpu::core::trigger_ud();
                return Ok(());
            }
            if opcode == 0x20 {
                let value = match modrm.reg {
                    0 => *cr as u32 as u64,
                    2 => *cr2,
                    3 => *cr.offset(3) as u32 as u64,
                    4 => *cr.offset(4) as u32 as u64,
                    _ => {
                        crate::cpu::core::trigger_ud();
                        return Ok(());
                    },
                };
                write_reg(modrm.rm, OpSize::S64, true, value);
            }
            else if modrm.reg == 2 {
                let value = read_reg(modrm.rm, OpSize::S64, true);
                *cr2 = value;
                *cr.offset(2) = value as i32;
            }
            else {
                crate::cpu::interp::instructions_0f::instr_0F22(modrm.rm as i32, modrm.reg as i32);
            }
        },

        // MOV r64, DRn (0F 21) / MOV DRn, r64 (0F 23)
        0x21 | 0x23 => {
            if *cpl != 0 {
                crate::cpu::core::trigger_gp(0);
                return Ok(());
            }
            let modrm = decode_modrm(pfx)?;
            if modrm.mod_bits != 3 || modrm.reg > 7 {
                crate::cpu::core::trigger_ud();
                return Ok(());
            }
            if opcode == 0x21 {
                let value = *dreg.offset(modrm.reg as isize) as u32 as u64;
                write_reg(modrm.rm, OpSize::S64, true, value);
            }
            else {
                let value = read_reg(modrm.rm, OpSize::S64, true);
                *dreg.offset(modrm.reg as isize) = value as i32;
            }
        },

        // CPUID (0F A2)
        0xA2 => {
            crate::cpu::interp::instructions_0f::instr_0FA2();
        },

        // BSWAP r32/r64 (0F C8-0F CF)
        0xC8..=0xCF => {
            let r = opcode - 0xC8 | (pfx.rex & prefix::REX_B) << 3;
            let value = read_reg(r, size, pfx.has_rex());
            write_reg(r, size, pfx.has_rex(), crate::cpu::interp::misc_instr::bswap_value(value, size));
        },

        // Group 6: SLDT/STR/LLDT/LTR/VERR/VERW (0F 00)
        0x00 => {
            let (modrm, operand) = decode_operand(pfx)?;
            match modrm.reg & 7 {
                0 => { // SLDT
                    let value = *sreg.offset(7) as u64;
                    write_operand(operand, OpSize::S16, pfx.has_rex(), value)?;
                },
                1 => { // STR
                    let value = *sreg.offset(6) as u64;
                    write_operand(operand, OpSize::S16, pfx.has_rex(), value)?;
                },
                2 => { // LLDT
                    let value = read_operand(operand, OpSize::S16, pfx.has_rex())? as u16;
                    *sreg.offset(7) = value;
                },
                3 => { // LTR
                    let value = read_operand(operand, OpSize::S16, pfx.has_rex())? as u16;
                    *sreg.offset(6) = value;
                    load_tss(value);
                },
                4 | 5 => { // VERR / VERW
                    let value = read_operand(operand, OpSize::S16, pfx.has_rex())? as u16;
                    set_zf(selector_valid(value));
                },
                _ => crate::cpu::core::trigger_ud(),
            }
        },

        // Group 7: SGDT/SIDT/LGDT/LIDT/SMSW/LMSW/INVLPG and the mod=3 forms
        // (SWAPGS, RDTSCP, XGETBV, XSETBV).
        0x01 => {
            let modrm = decode_modrm(pfx)?;
            if modrm.mod_bits == 3 {
                match modrm.reg & 7 {
                    // SWAPGS (0F 01 F8)
                    7 if modrm.rm == 0 => {
                        let tmp = *gs_base;
                        *gs_base = *kernel_gs_base;
                        *kernel_gs_base = tmp;
                    },
                    // RDTSCP (0F 01 F9)
                    7 if modrm.rm == 1 => {
                        let tsc = read_tsc();
                        write_reg(RAX, OpSize::S32, false, tsc & 0xFFFF_FFFF);
                        write_reg(RDX, OpSize::S32, false, tsc >> 32);
                        write_reg(RCX, OpSize::S32, false, 0); // TSC_AUX
                    },
                    // XGETBV (0F 01 D0)
                    2 if modrm.rm == 0 => {
                        write_reg(RAX, OpSize::S32, false, XCR0 & 0xFFFF_FFFF);
                        write_reg(RDX, OpSize::S32, false, XCR0 >> 32);
                    },
                    // XSETBV (0F 01 D1)
                    2 if modrm.rm == 1 => {
                        if *cpl != 0 || read_reg(RCX, OpSize::S32, false) != 0 {
                            crate::cpu::core::trigger_gp(0);
                            return Ok(());
                        }
                        let value = read_reg(RAX, OpSize::S32, false) as u32 as u64
                            | (read_reg(RDX, OpSize::S32, false) as u32 as u64) << 32;
                        if value & 1 != 0 && value & !0x7 == 0 {
                            XCR0 = value;
                        }
                        else {
                            crate::cpu::core::trigger_gp(0);
                        }
                    },
                    _ => crate::cpu::core::trigger_ud(),
                }
            }
            else {
                let (_, operand) = decode_operand_after_modrm(pfx, modrm, 0)?;
                let Operand::Mem(address) = operand else {
                    unreachable!()
                };
                match modrm.reg & 7 {
                    // SGDT
                    0 => {
                        mem_write(address, OpSize::S16, *gdtr_size as u32 as u64)?;
                        let base = if GDTR_BASE != 0 { GDTR_BASE } else { *gdtr_offset as u32 as u64 };
                        mem_write(address + 2, OpSize::S64, base)?;
                    },
                    // SIDT
                    1 => {
                        mem_write(address, OpSize::S16, *idtr_size as u32 as u64)?;
                        let base = if IDTR_BASE != 0 { IDTR_BASE } else { *idtr_offset as u32 as u64 };
                        mem_write(address + 2, OpSize::S64, base)?;
                    },
                    // LGDT
                    2 => {
                        *gdtr_size = mem_read(address, OpSize::S16)? as i32;
                        let base = mem_read(address + 2, OpSize::S64)?;
                        *gdtr_offset = base as u32 as i32;
                        GDTR_BASE = base;
                    },
                    // LIDT
                    3 => {
                        *idtr_size = mem_read(address, OpSize::S16)? as i32;
                        let base = mem_read(address + 2, OpSize::S64)?;
                        *idtr_offset = base as u32 as i32;
                        IDTR_BASE = base;
                    },
                    // SMSW
                    4 => {
                        let msw = *cr as u32 & 0xFFFF;
                        mem_write(address, OpSize::S16, msw as u64)?;
                    },
                    // LMSW
                    6 => {
                        let value = mem_read(address, OpSize::S16)? as u32;
                        *cr = ((*cr as u32 & !0xFFFF) | value) as i32;
                    },
                    // INVLPG
                    7 => crate::cpu::core::full_clear_tlb(),
                    _ => crate::cpu::core::trigger_ud(),
                }
            }
        },

        // CLTS (0F 06)
        0x06 => *cr &= !(1 << 3),

        // INVD (0F 08) / WBINVD (0F 09): no caches to flush
        0x08 | 0x09 => {},

        // LFENCE/MFENCE/SFENCE and CLFLUSH (0F AE)
        0xAE => {
            let modrm = decode_modrm(pfx)?;
            if modrm.mod_bits == 3 {
                if pfx.f3 {
                    // RDFSBASE/RDGSBASE/WRFSBASE/WRGSBASE, if CR4.FSGSBASE
                    if *cr.offset(4) & (1 << 16) == 0 {
                        crate::cpu::core::trigger_ud();
                        return Ok(());
                    }
                    match modrm.reg & 7 {
                        0 => {
                            let value = *fs_base;
                            write_reg(modrm.rm, size, pfx.has_rex(), value);
                        },
                        1 => {
                            let value = *gs_base;
                            write_reg(modrm.rm, size, pfx.has_rex(), value);
                        },
                        2 => {
                            let value = read_reg(modrm.rm, size, pfx.has_rex());
                            *fs_base = value;
                        },
                        3 => {
                            let value = read_reg(modrm.rm, size, pfx.has_rex());
                            *gs_base = value;
                        },
                        _ => crate::cpu::core::trigger_ud(),
                    }
                }
                else {
                    match modrm.reg & 7 {
                        5 | 6 | 7 => {}, // LFENCE / MFENCE / SFENCE
                        _ => crate::cpu::core::trigger_ud(),
                    }
                }
            }
            else {
                let (modrm, operand) = decode_operand_after_modrm(pfx, modrm, 0)?;
                match modrm.reg & 7 {
                    // FXSAVE (0) / XSAVE (4): x87+SSE state only; XSAVE also
                    // records it in the XSTATE_BV header.
                    0 | 4 => {
                        let Operand::Mem(address) = operand else {
                            crate::cpu::core::trigger_ud();
                            return Ok(());
                        };
                        for i in 0..512u64 {
                            mem_write(address + i, OpSize::S8, 0)?;
                        }
                        mem_write(address, OpSize::S16, *fpu_control_word as u64)?;
                        mem_write(address + 2, OpSize::S16, fpu_load_status_word() as u64)?;
                        mem_write(address + 4, OpSize::S16, (!*fpu_stack_empty & 0xFF) as u64)?;
                        mem_write(address + 6, OpSize::S16, *fpu_opcode as u64)?;
                        mem_write(address + 8, OpSize::S32, *fpu_ip as u32 as u64)?;
                        mem_write(address + 12, OpSize::S16, *fpu_ip_selector as u64)?;
                        mem_write(address + 16, OpSize::S32, *fpu_dp as u32 as u64)?;
                        mem_write(address + 20, OpSize::S16, *fpu_dp_selector as u64)?;
                        mem_write(address + 24, OpSize::S32, *mxcsr as u32 as u64)?;
                        // MXCSR_MASK: bits of MXCSR that may be set. Zero makes
                        // Linux treat its MXCSR as invalid and WARN on every switch.
                        mem_write(address + 28, OpSize::S32, 0xFFFF)?;
                        // ST0..ST7 in stack order, as 80-bit extended values
                        for i in 0..8u64 {
                            let reg_index = (i as i32 + *fpu_stack_ptr as i32) & 7;
                            let st = *fpu_st.offset(reg_index as isize);
                            mem_write(address + 32 + i * 16, OpSize::S64, st.mantissa)?;
                            mem_write(address + 32 + i * 16 + 8, OpSize::S16, st.sign_exponent as u64)?;
                        }
                        // Long mode has all 16 XMM registers
                        for i in 0..16u64 {
                            let x = xmm_get(i as u8);
                            mem_write(address + 160 + i * 16, OpSize::S64, x.u64[0])?;
                            mem_write(address + 160 + i * 16 + 8, OpSize::S64, x.u64[1])?;
                        }
                        if modrm.reg & 7 == 4 {
                            // XSTATE_BV records mask & XCR0; everything selected
                            // is written in the standard format.
                            let mask = read_reg(RAX, OpSize::S32, false) as u32 as u64
                                | (read_reg(RDX, OpSize::S32, false) as u32 as u64) << 32;
                            let selected = mask & XCR0;
                            mem_write(address + 512, OpSize::S64, selected)?;
                            mem_write(address + 520, OpSize::S64, 0)?;
                            if selected & 4 != 0 {
                                // AVX upper halves start at offset 576.
                                for i in 0..16u64 {
                                    let hi = ymm_hi(i as u8);
                                    mem_write(address + XSAVE_YMM_OFFSET as u64 + i * 16, OpSize::S64, hi.u64[0])?;
                                    mem_write(address + XSAVE_YMM_OFFSET as u64 + i * 16 + 8, OpSize::S64, hi.u64[1])?;
                                }
                            }
                        }
                    },
                    // FXRSTOR (1) and XRSTOR (5)
                    1 | 5 => {
                        let Operand::Mem(address) = operand else {
                            crate::cpu::core::trigger_ud();
                            return Ok(());
                        };
                        // XRSTOR honours the EDX:EAX mask; it is not FXRSTOR
                        // with an AVX tail. Restoring the legacy region
                        // unconditionally, as this used to, invents x87 and SSE
                        // state for a caller that asked for neither.
                        let mut restore_legacy = true;
                        let mut initial_legacy = false;
                        let mut present_ymm = false;
                        let mut initial_ymm = false;
                        if modrm.reg & 7 == 5 {
                            let mask = read_reg(RAX, OpSize::S32, false) as u32 as u64
                                | (read_reg(RDX, OpSize::S32, false) as u32 as u64) << 32;
                            let xstate_bv = mem_read(address + 512, OpSize::S64)?;
                            // A component in the area that XCR0 does not have
                            // enabled is a #GP.
                            if xstate_bv & !XCR0 != 0 {
                                crate::cpu::core::trigger_gp(0);
                                return Ok(());
                            }
                            let requested = mask & XCR0;
                            restore_legacy = requested & 3 & xstate_bv != 0;
                            initial_legacy = requested & 3 & !xstate_bv != 0;
                            present_ymm = requested & 4 & xstate_bv != 0;
                            initial_ymm = requested & 4 & !xstate_bv != 0;
                        }
                        if initial_legacy {
                            // Initialize all eight x87 registers as empty.
                            set_control_word(0x37F);
                            fpu_set_status_word(0);
                            *fpu_stack_empty = 0xFF;
                            *mxcsr = 0x1F80;
                            for i in 0..16u64 {
                                xmm_set(i as u8, reg128 { u64: [0, 0] });
                            }
                        }
                        if !restore_legacy {
                            if present_ymm {
                                for i in 0..16u64 {
                                    let lo = mem_read(address + XSAVE_YMM_OFFSET as u64 + i * 16, OpSize::S64)?;
                                    let hi = mem_read(address + XSAVE_YMM_OFFSET as u64 + i * 16 + 8, OpSize::S64)?;
                                    ymm_set_hi(i as u8, reg128 { u64: [lo, hi] });
                                }
                            }
                            if initial_ymm {
                                for i in 0..16u64 {
                                    ymm_set_hi(i as u8, reg128 { u64: [0, 0] });
                                }
                            }
                            return Ok(());
                        }
                        set_control_word(mem_read(address, OpSize::S16)? as u16);
                        fpu_set_status_word(mem_read(address + 2, OpSize::S16)? as u16);
                        *fpu_stack_empty = !mem_read(address + 4, OpSize::S8)? as u8;
                        *fpu_opcode = mem_read(address + 6, OpSize::S16)? as i32;
                        *fpu_ip = mem_read(address + 8, OpSize::S32)? as i32;
                        *fpu_ip_selector = mem_read(address + 12, OpSize::S16)? as i32;
                        *fpu_dp = mem_read(address + 16, OpSize::S32)? as i32;
                        *fpu_dp_selector = mem_read(address + 20, OpSize::S16)? as i32;
                        *mxcsr = mem_read(address + 24, OpSize::S32)? as u32 as i32;
                        for i in 0..8u64 {
                            let reg_index = (i as i32 + *fpu_stack_ptr as i32) & 7;
                            let mantissa = mem_read(address + 32 + i * 16, OpSize::S64)?;
                            let sign_exponent = mem_read(address + 32 + i * 16 + 8, OpSize::S16)? as u16;
                            *fpu_st.offset(reg_index as isize) = F80 { mantissa, sign_exponent };
                        }
                        for i in 0..16u64 {
                            let low = mem_read(address + 160 + i * 16, OpSize::S64)?;
                            let high = mem_read(address + 160 + i * 16 + 8, OpSize::S64)?;
                            xmm_set(i as u8, reg128 { u64: [low, high] });
                        }
                        if !restore_legacy {
                            return Ok(());
                        }
                        if present_ymm {
                            for i in 0..16u64 {
                                let lo = mem_read(address + XSAVE_YMM_OFFSET as u64 + i * 16, OpSize::S64)?;
                                let hi = mem_read(address + XSAVE_YMM_OFFSET as u64 + i * 16 + 8, OpSize::S64)?;
                                ymm_set_hi(i as u8, reg128 { u64: [lo, hi] });
                            }
                        }
                        if initial_ymm {
                            for i in 0..16u64 {
                                ymm_set_hi(i as u8, reg128 { u64: [0, 0] });
                            }
                        }
                    },
                    // LDMXCSR
                    2 => {
                        let Operand::Mem(address) = operand else {
                            crate::cpu::core::trigger_ud();
                            return Ok(());
                        };
                        *mxcsr = mem_read(address, OpSize::S32)? as u32 as i32;
                    },
                    // STMXCSR
                    3 => {
                        let Operand::Mem(address) = operand else {
                            crate::cpu::core::trigger_ud();
                            return Ok(());
                        };
                        mem_write(address, OpSize::S32, *mxcsr as u32 as u64)?;
                    },
                    7 => {}, // CLFLUSH
                    _ => crate::cpu::core::trigger_ud(),
                }
            }
        },

        // SYSCALL (0F 05)
        0x05 => {
            if *efer & EFER_SCE == 0 {
                crate::cpu::core::trigger_ud();
                return Ok(());
            }
            write_reg64(RCX as i32, *rip);
            write_reg64(R11 as i32, *flags as u32 as u64);
            *flags &= !(*sfmask as u32) as i32;
            *rip = *lstar;
            let cs = (*star >> 32 & 0xFFFF) as u16;
            set_cpl_segments(cs, 0);
        },

        // SYSRET: CS = STAR[63:48]+16 (64-bit) or +0 (compat, RIP = ECX);
        // SS = STAR[63:48]+8, RPL 3.
        0x07 => {
            if *efer & EFER_SCE == 0 {
                crate::cpu::core::trigger_ud();
                return Ok(());
            }
            let target = read_reg64(RCX as i32);
            *flags = read_reg64(R11 as i32) as u32 as i32;
            let star_user = (*star >> 48 & 0xFFFF) as u16;
            let (cs, ss, new_rip) = if pfx.has_rex_w() {
                (star_user.wrapping_add(16), star_user.wrapping_add(8), target)
            }
            else {
                (star_user, star_user.wrapping_add(8), target & 0xFFFF_FFFF)
            };
            *sreg.offset(CS as isize) = cs | 3;
            *sreg.offset(SS as isize) = ss | 3;
            *segment_offsets.offset(CS as isize) = 0;
            *segment_offsets.offset(SS as isize) = 0;
            *segment_is_null.offset(CS as isize) = false;
            *segment_is_null.offset(SS as isize) = false;
            *cpl = 3;
            *rip = new_rip;
        },

        // WRMSR (0F 30)
        0x30 => {
            if *cpl != 0 {
                crate::cpu::core::trigger_gp(0);
                return Ok(());
            }
            let index = read_reg64(RCX as i32) as i32;
            let value = read_reg64(RAX as i32) as u32 as u64
                | (read_reg64(RDX as i32) as u32 as u64) << 32;
            if !msr_write(index, value) {
                crate::cpu::interp::instructions_0f::instr_0F30();
            }
        },

        // RDTSC (0F 31)
        0x31 => crate::cpu::interp::instructions_0f::instr_0F31(),

        // RDMSR (0F 32)
        0x32 => {
            if *cpl != 0 {
                crate::cpu::core::trigger_gp(0);
                return Ok(());
            }
            let index = read_reg64(RCX as i32) as i32;
            match msr_read(index) {
                Some(value) => {
                    write_reg(RAX, OpSize::S32, false, value & 0xFFFF_FFFF);
                    write_reg(RDX, OpSize::S32, false, value >> 32);
                },
                None => crate::cpu::interp::instructions_0f::instr_0F32(),
            }
        },

        // BT/BTS/BTR/BTC r/m, r (0F A3/AB/B3/BB)
        0xA3 | 0xAB | 0xB3 | 0xBB => {
            let (modrm, operand) = decode_operand(pfx)?;
            let index = read_reg(modrm.reg, size, pfx.has_rex());
            let op = match opcode {
                0xA3 => 0,
                0xAB => 1,
                0xB3 => 2,
                _ => 3,
            };
            bit_test(operand, pfx, size, index, op)?;
        },

        // BT/BTS/BTR/BTC r/m, imm8 (0F BA /4../7)
        0xBA => {
            let (modrm, operand) = decode_operand_with_trailing(pfx, 1)?;
            let index = fetch8()? as u64;
            let op = match modrm.reg & 7 {
                4 => 0,
                5 => 1,
                6 => 2,
                7 => 3,
                _ => {
                    crate::cpu::core::trigger_ud();
                    return Ok(());
                },
            };
            bit_test(operand, pfx, size, index, op)?;
        },

        // BSF/BSR r, r/m (0F BC/BD); TZCNT/LZCNT with F3 (BMI1)
        0xBC | 0xBD => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = read_operand(operand, size, pfx.has_rex())? & size.mask();
            if pfx.f3 {
                let width = if size == OpSize::S64 { 64 } else { 32 };
                let result = if src == 0 {
                    width as u64
                }
                else if opcode == 0xBC {
                    src.trailing_zeros() as u64
                }
                else {
                    (src.leading_zeros() - (64 - width)) as u64
                };
                *flags &= !((FLAG_CF | FLAG_ZF) as i32);
                if src == 0 {
                    *flags |= FLAG_CF as i32;
                }
                if result == 0 {
                    *flags |= FLAG_ZF as i32;
                }
                *flags_changed = 0;
                write_reg(modrm.reg, size, pfx.has_rex(), result);
            }
            else if src == 0 {
                set_zf(true);
            }
            else {
                set_zf(false);
                let result = if opcode == 0xBC {
                    crate::cpu::interp::misc_instr::bit_scan_forward(src).unwrap() as u64
                }
                else {
                    crate::cpu::interp::misc_instr::bit_scan_reverse(src).unwrap() as u64
                };
                write_reg(modrm.reg, size, pfx.has_rex(), result);
            }
        },

        // CMPXCHG r/m, r (0F B1) and CMPXCHG r/m8, r8 (0F B0, always 8-bit)
        0xB0 | 0xB1 => {
            let size = if opcode == 0xB0 { OpSize::S8 } else { size };
            let (modrm, operand) = decode_operand(pfx)?;
            let dst = read_operand(operand, size, pfx.has_rex())? & size.mask();
            let src = read_reg(modrm.reg, size, pfx.has_rex());
            let acc = read_reg(RAX, size, false) & size.mask();
            if acc == dst {
                set_zf(true);
                probe_operand_write(operand, size)?;
                write_operand(operand, size, pfx.has_rex(), src)?;
            }
            else {
                set_zf(false);
                write_reg(RAX, size, false, dst);
            }
        },

        // XADD r/m, r (0F C1)
        0xC1 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let dst = read_operand(operand, size, pfx.has_rex())?;
            let src = read_reg(modrm.reg, size, pfx.has_rex());
            let result = alu(AluOp::Add, dst, src, size);
            probe_operand_write(operand, size)?;
            write_reg(modrm.reg, size, pfx.has_rex(), dst);
            write_operand(operand, size, pfx.has_rex(), result)?;
        },

        _ => {
            dbg_log!("#ud interp64: 0f opcode {:02x}", opcode);
            crate::cpu::core::trigger_ud();
        },
    }

    Ok(())
}
