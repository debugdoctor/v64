//! Hand-written x86-64 long mode interpreter, independent of `interp32*.rs`.
//! Operand size 32/64 (REX.W)/16 (0x66); address size 64/32 (0x67); phys < 4 GiB.

#![allow(dead_code, unused_imports)]

use crate::cpu::core::{
    condition, io_port_read8, io_port_write8, read_reg64, read_tsc, reg128, test_privileges_for_io,
    translate_address_64, write_reg64, APIC_MEM_ADDRESS, CS, FLAG_CARRY, FLAG_INTERRUPT, SS,
};
use crate::cpu::interp::decode_cache::{DecEntry, DecodeHost, DecodedInstr, JStep};
use crate::cpu::global_pointers::*;
use crate::cpu::interp::misc_instr::sign_extend;
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

#[path = "interp64_0f.rs"]
pub mod interp64_0f;

pub(crate) use crate::cpu::interp::interp64_core::*;


pub(crate) unsafe fn x87_load_f32(address: u64) -> OrPageFault<F80> {
    Ok(f32_to_f80(mem_read(address, OpSize::S32)? as u32 as i32))
}

pub(crate) unsafe fn x87_load_f64(address: u64) -> OrPageFault<F80> {
    Ok(f64_to_f80(mem_read(address, OpSize::S64)?))
}

pub(crate) unsafe fn x87_load_m80(address: u64) -> OrPageFault<F80> {
    let mantissa = mem_read(address, OpSize::S64)?;
    let sign_exponent = mem_read(address + 8, OpSize::S16)? as u16;
    Ok(F80 { mantissa, sign_exponent })
}

pub(crate) unsafe fn x87_store_f32(address: u64, value: F80) -> OrPageFault<()> {
    mem_write(address, OpSize::S32, f80_to_f32(value) as u32 as u64)
}

pub(crate) unsafe fn x87_store_f64(address: u64, value: F80) -> OrPageFault<()> {
    mem_write(address, OpSize::S64, f80_to_f64(value))
}

pub(crate) unsafe fn x87_store_m80(address: u64, value: F80) -> OrPageFault<()> {
    mem_write(address, OpSize::S64, value.mantissa)?;
    mem_write(address + 8, OpSize::S16, value.sign_exponent as u64)
}

// FIST/FISTP round per the control word; FISTTP truncates.
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/fist
pub(crate) unsafe fn x87_store_int(address: u64, value: F80, size: OpSize, truncate: bool) -> OrPageFault<()> {
    let converted = match (size, truncate) {
        (OpSize::S16, false) => fpu_convert_to_i16(value) as u16 as u64,
        (OpSize::S16, true) => fpu_truncate_to_i16(value) as u16 as u64,
        (OpSize::S32, false) => fpu_convert_to_i32(value) as u32 as u64,
        (OpSize::S32, true) => fpu_truncate_to_i32(value) as u32 as u64,
        (_, false) => fpu_convert_to_i64(value) as u64,
        (_, true) => fpu_truncate_to_i64(value) as u64,
    };
    mem_write(address, size, converted)
}

pub(crate) unsafe fn x87_load_int(address: u64, size: OpSize) -> OrPageFault<F80> {
    let value = mem_read(address, size)? & size.mask();
    Ok(match size {
        OpSize::S16 => i32_to_f80(value as u16 as i16 as i32),
        OpSize::S32 => i32_to_f80(value as u32 as i32),
        _ => i64_to_f80(value as i64),
    })
}

// FNSTENV, in the protected-mode layouts the 32-bit implementation uses
// (see crate::cpu::interp::fpu::fpu_fstenv16/32), per Intel SDM Vol. 2: FNSTENV.
pub(crate) unsafe fn x87_store_env(address: u64, sixteen: bool) -> OrPageFault<()> {
    if sixteen {
        mem_write(address, OpSize::S16, *fpu_control_word as u64)?;
        mem_write(address + 2, OpSize::S16, fpu_load_status_word() as u64)?;
        mem_write(address + 4, OpSize::S16, fpu_load_tag_word() as u64)?;
        mem_write(address + 6, OpSize::S16, *fpu_ip as u32 as u64 & 0xFFFF)?;
        mem_write(address + 8, OpSize::S16, *fpu_ip_selector as u32 as u64 & 0xFFFF)?;
        mem_write(address + 10, OpSize::S16, *fpu_dp as u32 as u64 & 0xFFFF)?;
        mem_write(address + 12, OpSize::S16, *fpu_dp_selector as u32 as u64 & 0xFFFF)?;
    }
    else {
        mem_write(address, OpSize::S32, 0xFFFF_0000 | *fpu_control_word as u32 as u64)?;
        mem_write(address + 4, OpSize::S32, 0xFFFF_0000 | fpu_load_status_word() as u64)?;
        mem_write(address + 8, OpSize::S32, 0xFFFF_0000 | fpu_load_tag_word() as u64)?;
        mem_write(address + 12, OpSize::S32, *fpu_ip as u32 as u64)?;
        mem_write(address + 16, OpSize::S16, *fpu_ip_selector as u32 as u64 & 0xFFFF)?;
        mem_write(address + 18, OpSize::S16, *fpu_opcode as u32 as u64 & 0xFFFF)?;
        mem_write(address + 20, OpSize::S32, *fpu_dp as u32 as u64)?;
        mem_write(address + 24, OpSize::S32, 0xFFFF_0000 | *fpu_dp_selector as u32 as u64)?;
    }
    Ok(())
}

pub(crate) unsafe fn x87_load_env(address: u64, sixteen: bool) -> OrPageFault<()> {
    if sixteen {
        set_control_word(mem_read(address, OpSize::S16)? as u16);
        fpu_set_status_word(mem_read(address + 2, OpSize::S16)? as u16);
        fpu_set_tag_word(mem_read(address + 4, OpSize::S16)? as i32);
        *fpu_ip = mem_read(address + 6, OpSize::S16)? as i32;
        *fpu_ip_selector = mem_read(address + 8, OpSize::S16)? as i32;
        *fpu_dp = mem_read(address + 10, OpSize::S16)? as i32;
        *fpu_dp_selector = mem_read(address + 12, OpSize::S16)? as i32;
    }
    else {
        set_control_word(mem_read(address, OpSize::S32)? as u32 as u16);
        fpu_set_status_word(mem_read(address + 4, OpSize::S32)? as u32 as u16);
        fpu_set_tag_word(mem_read(address + 8, OpSize::S32)? as u32 as u16 as i32);
        *fpu_ip = mem_read(address + 12, OpSize::S32)? as u32 as i32;
        *fpu_ip_selector = mem_read(address + 16, OpSize::S16)? as i32;
        *fpu_opcode = mem_read(address + 18, OpSize::S16)? as i32;
        *fpu_dp = mem_read(address + 20, OpSize::S32)? as u32 as i32;
        *fpu_dp_selector = mem_read(address + 24, OpSize::S32)? as u32 as u16 as i32;
    }
    Ok(())
}

// FNSAVE/FRSTOR: environment, then eight 10-byte registers in physical order.
pub(crate) unsafe fn x87_save_state(address: u64, sixteen: bool) -> OrPageFault<()> {
    x87_store_env(address, sixteen)?;
    let mut offset = if sixteen { 14 } else { 28 };
    for i in 0..8u64 {
        let physical = (*fpu_stack_ptr as u64 + i) & 7;
        x87_store_m80(address + offset, *fpu_st.offset(physical as isize))?;
        offset += 10;
    }
    fpu_finit();
    Ok(())
}

pub(crate) unsafe fn x87_restore_state(address: u64, sixteen: bool) -> OrPageFault<()> {
    x87_load_env(address, sixteen)?;
    let mut offset = if sixteen { 14 } else { 28 };
    for i in 0..8u64 {
        let value = x87_load_m80(address + offset)?;
        let physical = (*fpu_stack_ptr as u64 + i) & 7;
        *fpu_st.offset(physical as isize) = value;
        offset += 10;
    }
    Ok(())
}

// FBLD/FBSTP: signed packed BCD, 9 digits plus a sign byte.
pub(crate) unsafe fn x87_load_bcd(address: u64) -> OrPageFault<F80> {
    let mut value: u64 = 0;
    for i in (0..9u64).rev() {
        let byte = mem_read(address + i, OpSize::S8)? & 0xFF;
        value = value.wrapping_mul(100).wrapping_add((byte >> 4) * 10 + (byte & 0xF));
    }
    let mut result = i64_to_f80(value as i64);
    if mem_read(address + 9, OpSize::S8)? & 0x80 != 0 {
        result.sign_exponent ^= 0x8000;
    }
    Ok(result)
}

pub(crate) unsafe fn x87_store_bcd(address: u64, value: F80) -> OrPageFault<()> {
    let converted = fpu_convert_to_i64(value);
    let mut magnitude = converted.unsigned_abs();
    for i in 0..9u64 {
        let low = (magnitude % 10) as u64;
        magnitude /= 10;
        let high = (magnitude % 10) as u64;
        magnitude /= 10;
        mem_write(address + i, OpSize::S8, (high << 4) | low)?;
    }
    let sign = if converted < 0 { 0x80 } else { 0 };
    mem_write(address + 9, OpSize::S8, sign)
}

pub(crate) unsafe fn x87(pfx: &Prefixes, opcode: u8) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let reg = (modrm.reg & 7) as i32;
    let st = (modrm.rm & 7) as i32;
    let sixteen = pfx.opsize_16;
    if let Operand::Reg(_) = operand {
        x87_reg(opcode, reg, st, sixteen);
        return Ok(());
    }
    let Operand::Mem(address) = operand else { unreachable!() };
    match (opcode, reg) {
        // D8: arithmetic with an m32real
        (0xD8, 0) => fpu_fadd(0, x87_load_f32(address)?),
        (0xD8, 1) => fpu_fmul(0, x87_load_f32(address)?),
        (0xD8, 2) => fpu_fcom(x87_load_f32(address)?),
        (0xD8, 3) => fpu_fcomp(x87_load_f32(address)?),
        (0xD8, 4) => fpu_fsub(0, x87_load_f32(address)?),
        (0xD8, 5) => fpu_fsubr(0, x87_load_f32(address)?),
        (0xD8, 6) => fpu_fdiv(0, x87_load_f32(address)?),
        (0xD8, 7) => fpu_fdivr(0, x87_load_f32(address)?),

        // D9: load/store m32real and the control word
        (0xD9, 0) => fpu_push(x87_load_f32(address)?),
        (0xD9, 2) => x87_store_f32(address, fpu_get_st0())?,
        (0xD9, 3) => {
            x87_store_f32(address, fpu_get_st0())?;
            fpu_pop();
        },
        (0xD9, 4) => x87_load_env(address, sixteen)?,
        // FLDCW: must go through set_control_word so the softfloat rounding
        // mode and precision follow the loaded control word (FISTP etc. read it)
        (0xD9, 5) => set_control_word(mem_read(address, OpSize::S16)? as u16),
        (0xD9, 6) => x87_store_env(address, sixteen)?,
        (0xD9, 7) => mem_write(address, OpSize::S16, *fpu_control_word as u64)?,

        // DA: arithmetic with an m32int (register forms are FCMOV)
        (0xDA, 0) => fpu_fadd(0, x87_load_int(address, OpSize::S32)?),
        (0xDA, 1) => fpu_fmul(0, x87_load_int(address, OpSize::S32)?),
        (0xDA, 2) => fpu_fcom(x87_load_int(address, OpSize::S32)?),
        (0xDA, 3) => fpu_fcomp(x87_load_int(address, OpSize::S32)?),
        (0xDA, 4) => fpu_fsub(0, x87_load_int(address, OpSize::S32)?),
        (0xDA, 5) => fpu_fsubr(0, x87_load_int(address, OpSize::S32)?),
        (0xDA, 6) => fpu_fdiv(0, x87_load_int(address, OpSize::S32)?),
        (0xDA, 7) => fpu_fdivr(0, x87_load_int(address, OpSize::S32)?),

        // DB: integer loads/stores, m80real
        (0xDB, 0) => fpu_push(x87_load_int(address, OpSize::S32)?),
        (0xDB, 1) => {
            x87_store_int(address, fpu_get_st0(), OpSize::S32, true)?;
            fpu_pop();
        },
        (0xDB, 2) => x87_store_int(address, fpu_get_st0(), OpSize::S32, false)?,
        (0xDB, 3) => {
            x87_store_int(address, fpu_get_st0(), OpSize::S32, false)?;
            fpu_pop();
        },
        (0xDB, 5) => fpu_push(x87_load_m80(address)?),
        (0xDB, 7) => {
            x87_store_m80(address, fpu_get_st0())?;
            fpu_pop();
        },

        // DC: arithmetic with an m64real
        (0xDC, 0) => fpu_fadd(0, x87_load_f64(address)?),
        (0xDC, 1) => fpu_fmul(0, x87_load_f64(address)?),
        (0xDC, 2) => fpu_fcom(x87_load_f64(address)?),
        (0xDC, 3) => fpu_fcomp(x87_load_f64(address)?),
        (0xDC, 4) => fpu_fsub(0, x87_load_f64(address)?),
        (0xDC, 5) => fpu_fsubr(0, x87_load_f64(address)?),
        (0xDC, 6) => fpu_fdiv(0, x87_load_f64(address)?),
        (0xDC, 7) => fpu_fdivr(0, x87_load_f64(address)?),

        // DD: m64real loads/stores, status word
        (0xDD, 0) => fpu_push(x87_load_f64(address)?),
        (0xDD, 1) => {
            x87_store_int(address, fpu_get_st0(), OpSize::S64, true)?;
            fpu_pop();
        },
        (0xDD, 2) => x87_store_f64(address, fpu_get_st0())?,
        (0xDD, 3) => {
            x87_store_f64(address, fpu_get_st0())?;
            fpu_pop();
        },
        (0xDD, 4) => x87_restore_state(address, sixteen)?,
        (0xDD, 6) => x87_save_state(address, sixteen)?,
        (0xDD, 7) => mem_write(address, OpSize::S16, *fpu_status_word as u64)?,

        // DE: arithmetic with an m16int
        (0xDE, 0) => fpu_fadd(0, x87_load_int(address, OpSize::S16)?),
        (0xDE, 1) => fpu_fmul(0, x87_load_int(address, OpSize::S16)?),
        (0xDE, 2) => fpu_fcom(x87_load_int(address, OpSize::S16)?),
        (0xDE, 3) => fpu_fcomp(x87_load_int(address, OpSize::S16)?),
        (0xDE, 4) => fpu_fsub(0, x87_load_int(address, OpSize::S16)?),
        (0xDE, 5) => fpu_fsubr(0, x87_load_int(address, OpSize::S16)?),
        (0xDE, 6) => fpu_fdiv(0, x87_load_int(address, OpSize::S16)?),
        (0xDE, 7) => fpu_fdivr(0, x87_load_int(address, OpSize::S16)?),

        // DF: integer loads/stores, all widths
        (0xDF, 0) => fpu_push(x87_load_int(address, OpSize::S16)?),
        (0xDF, 1) => {
            x87_store_int(address, fpu_get_st0(), OpSize::S16, true)?;
            fpu_pop();
        },
        (0xDF, 2) => x87_store_int(address, fpu_get_st0(), OpSize::S16, false)?,
        (0xDF, 3) => {
            x87_store_int(address, fpu_get_st0(), OpSize::S16, false)?;
            fpu_pop();
        },
        (0xDF, 4) => fpu_push(x87_load_bcd(address)?),
        (0xDF, 5) => fpu_push(x87_load_int(address, OpSize::S64)?),
        (0xDF, 6) => {
            x87_store_bcd(address, fpu_get_st0())?;
            fpu_pop();
        },
        (0xDF, 7) => {
            x87_store_int(address, fpu_get_st0(), OpSize::S64, false)?;
            fpu_pop();
        },

        // Invalid x87 encodings
        _ => {
            dbg_log!("#ud interp64: x87 opcode {:02x} reg {}", opcode, reg);
            crate::cpu::core::trigger_ud();
        },
    }
    Ok(())
}

pub(crate) unsafe fn x87_reg(opcode: u8, reg: i32, st: i32, sixteen: bool) {
    use crate::cpu::interp::instructions as x87_32;
    match (opcode, reg) {
        (0xD8, 0) => x87_32::instr_D8_0_reg(st),
        (0xD8, 1) => x87_32::instr_D8_1_reg(st),
        (0xD8, 2) => x87_32::instr_D8_2_reg(st),
        (0xD8, 3) => x87_32::instr_D8_3_reg(st),
        (0xD8, 4) => x87_32::instr_D8_4_reg(st),
        (0xD8, 5) => x87_32::instr_D8_5_reg(st),
        (0xD8, 6) => x87_32::instr_D8_6_reg(st),
        (0xD8, 7) => x87_32::instr_D8_7_reg(st),
        (0xD9, 0) => if sixteen { x87_32::instr16_D9_0_reg(st) } else { x87_32::instr32_D9_0_reg(st) },
        (0xD9, 1) => if sixteen { x87_32::instr16_D9_1_reg(st) } else { x87_32::instr32_D9_1_reg(st) },
        (0xD9, 2) => if sixteen { x87_32::instr16_D9_2_reg(st) } else { x87_32::instr32_D9_2_reg(st) },
        (0xD9, 3) => if sixteen { x87_32::instr16_D9_3_reg(st) } else { x87_32::instr32_D9_3_reg(st) },
        (0xD9, 4) => if sixteen { x87_32::instr16_D9_4_reg(st) } else { x87_32::instr32_D9_4_reg(st) },
        (0xD9, 5) => if sixteen { x87_32::instr16_D9_5_reg(st) } else { x87_32::instr32_D9_5_reg(st) },
        (0xD9, 6) => if sixteen { x87_32::instr16_D9_6_reg(st) } else { x87_32::instr32_D9_6_reg(st) },
        (0xD9, 7) => if sixteen { x87_32::instr16_D9_7_reg(st) } else { x87_32::instr32_D9_7_reg(st) },
        (0xDA, 0) => x87_32::instr_DA_0_reg(st),
        (0xDA, 1) => x87_32::instr_DA_1_reg(st),
        (0xDA, 2) => x87_32::instr_DA_2_reg(st),
        (0xDA, 3) => x87_32::instr_DA_3_reg(st),
        (0xDA, 4) => x87_32::instr_DA_4_reg(st),
        (0xDA, 5) => x87_32::instr_DA_5_reg(st),
        (0xDA, 6) => x87_32::instr_DA_6_reg(st),
        (0xDA, 7) => x87_32::instr_DA_7_reg(st),
        (0xDB, 0) => x87_32::instr_DB_0_reg(st),
        (0xDB, 1) => x87_32::instr_DB_1_reg(st),
        (0xDB, 2) => x87_32::instr_DB_2_reg(st),
        (0xDB, 3) => x87_32::instr_DB_3_reg(st),
        (0xDB, 4) => x87_32::instr_DB_4_reg(st),
        (0xDB, 5) => x87_32::instr_DB_5_reg(st),
        (0xDB, 6) => x87_32::instr_DB_6_reg(st),
        (0xDB, 7) => x87_32::instr_DB_7_reg(st),
        (0xDC, 0) => x87_32::instr_DC_0_reg(st),
        (0xDC, 1) => x87_32::instr_DC_1_reg(st),
        (0xDC, 2) => x87_32::instr_DC_2_reg(st),
        (0xDC, 3) => x87_32::instr_DC_3_reg(st),
        (0xDC, 4) => x87_32::instr_DC_4_reg(st),
        (0xDC, 5) => x87_32::instr_DC_5_reg(st),
        (0xDC, 6) => x87_32::instr_DC_6_reg(st),
        (0xDC, 7) => x87_32::instr_DC_7_reg(st),
        (0xDD, 0) => if sixteen { x87_32::instr16_DD_0_reg(st) } else { x87_32::instr32_DD_0_reg(st) },
        (0xDD, 1) => if sixteen { x87_32::instr16_DD_1_reg(st) } else { x87_32::instr32_DD_1_reg(st) },
        (0xDD, 2) => if sixteen { x87_32::instr16_DD_2_reg(st) } else { x87_32::instr32_DD_2_reg(st) },
        (0xDD, 3) => if sixteen { x87_32::instr16_DD_3_reg(st) } else { x87_32::instr32_DD_3_reg(st) },
        (0xDD, 4) => if sixteen { x87_32::instr16_DD_4_reg(st) } else { x87_32::instr32_DD_4_reg(st) },
        (0xDD, 5) => if sixteen { x87_32::instr16_DD_5_reg(st) } else { x87_32::instr32_DD_5_reg(st) },
        (0xDD, 6) => if sixteen { x87_32::instr16_DD_6_reg(st) } else { x87_32::instr32_DD_6_reg(st) },
        (0xDD, 7) => if sixteen { x87_32::instr16_DD_7_reg(st) } else { x87_32::instr32_DD_7_reg(st) },
        (0xDE, 0) => x87_32::instr_DE_0_reg(st),
        (0xDE, 1) => x87_32::instr_DE_1_reg(st),
        (0xDE, 2) => x87_32::instr_DE_2_reg(st),
        (0xDE, 3) => x87_32::instr_DE_3_reg(st),
        (0xDE, 4) => x87_32::instr_DE_4_reg(st),
        (0xDE, 5) => x87_32::instr_DE_5_reg(st),
        (0xDE, 6) => x87_32::instr_DE_6_reg(st),
        (0xDE, 7) => x87_32::instr_DE_7_reg(st),
        (0xDF, 0) => x87_32::instr_DF_0_reg(st),
        (0xDF, 1) => x87_32::instr_DF_1_reg(st),
        (0xDF, 2) => x87_32::instr_DF_2_reg(st),
        (0xDF, 3) => x87_32::instr_DF_3_reg(st),
        (0xDF, 4) => x87_32::instr_DF_4_reg(st),
        (0xDF, 5) => x87_32::instr_DF_5_reg(st),
        (0xDF, 6) => x87_32::instr_DF_6_reg(st),
        (0xDF, 7) => x87_32::instr_DF_7_reg(st),
        _ => crate::cpu::core::trigger_ud(),
    }
}

pub(crate) unsafe fn jcc(code: u8, displacement: i64) {
    if condition(code) {
        *rip = (*rip).wrapping_add(displacement as u64);
    }
}

pub unsafe fn run_one() {
    pstat(stat::INTERP64_INSTRUCTIONS);
    *previous_rip = *rip;
    let _ = run_one_inner();
}

// JIT bridge: execute the VEX/AVX instruction at `address` in the interpreter.
// The second argument is unused (keeps the existing wasm import signature).
#[no_mangle]
pub unsafe fn jit64_avx(address: u64, _unused: i32) {
    *rip = address;
    run_one();
}

// The 32-bit instruction_pointer view is only read at boundaries; sync there.
#[inline(always)]
pub unsafe fn sync_instruction_pointer() {
    *instruction_pointer = *rip as u32 as i32;
}

// Interpreter-only batch: run up to `max` instructions without returning to the
// main loop (JIT dispatch amortised). Stops on hlt or when handle_irqs ran.
#[no_mangle]
pub unsafe fn interp64_run(max: u32) -> u32 {
    let mut n = 0;
    while n < max && !*in_hlt {
        // Execute a whole pre-decoded block when possible.
        if let Some(used) = decode_cache_run() {
            n += used;
            continue;
        }
        let interrupts_were_enabled = *flags & FLAG_INTERRUPT != 0;
        run_one();
        *instruction_counter = (*instruction_counter).wrapping_add(1);
        n += 1;
        if crate::cpu::core::interp_step_irqs(interrupts_were_enabled) {
            break;
        }
    }
    sync_instruction_pointer();
    n
}

#[no_mangle]
pub unsafe fn interp64_run_one() {
    *rip = *instruction_pointer as u32 as u64;
    run_one();
    sync_instruction_pointer();
}

// ---- Decode cache ----------------------------------------------------------
// Reuse the JIT decoder to pre-decode a basic block into Instructions, cache it by
// virtual RIP, and execute the Instructions directly. SMC is handled by validating the
// cached bytes against live memory on each use (no write-path hook needed).


pub(crate) const DECODE_CACHE_MAX: usize = 1 << 13;
pub(crate) const DECODE_CACHE_MIN: usize = 1 << 11;

// Runtime state for the INTERP64_DECODE_CACHE config key. The generated
// `config::INTERP64_DECODE_CACHE` is the key's index, not a flag, so it cannot
// be tested directly -- see `set_cpu_config`, which routes the key here.
pub static mut DECODE_CACHE_ENABLED: bool = true;

pub unsafe fn interp64_set_decode_cache(enabled: u32) {
    DECODE_CACHE_ENABLED = enabled != 0;
}

pub(crate) static mut DECODE_CACHE: *mut Vec<DecEntry<JInstr>> = std::ptr::null_mut();

pub(crate) unsafe fn decode_cache() -> &'static mut Vec<DecEntry<JInstr>> {
    if DECODE_CACHE.is_null() {
        // The per-page write counters back validate_block's fast path, and the
        // decoded-page marks keep the JIT's inline stores off those pages.
        crate::cpu::core::page_write_version_init();
        crate::cpu::core::page_decoded_init();
        // Keep the table small for small guests (the wasm memory is sized from
        // memory_size), larger when there is room to avoid thrashing.
        let n = if *memory_size >= 64 * 1024 * 1024 {
            DECODE_CACHE_MAX
        }
        else {
            DECODE_CACHE_MIN
        };
        let mut v: Vec<DecEntry<JInstr>> = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(DecEntry::default());
        }
        DECODE_CACHE = Box::into_raw(Box::new(v));
    }
    &mut *DECODE_CACHE
}

// Drop every decoded block (used when the 64-bit JIT is re-enabled: its inline
// writes bypass the per-page counters that validate_block relies on).
pub(crate) unsafe fn clear_decode_cache() {
    if DECODE_CACHE.is_null() {
        return;
    }
    // Collect the pages that held a block: marks are only ever added for these,
    // so this is exactly the set that needs its TLB64_HAS_CODE bit recomputed.
    let mut pages: Vec<u32> = Vec::new();
    for entry in &mut *DECODE_CACHE {
        if entry.valid && !pages.contains(&entry.phys_page) {
            pages.push(entry.phys_page);
        }
        entry.valid = false;
    }
    crate::cpu::core::clear_decoded_page_marks();
    for page in pages {
        crate::cpu::core::tlb64_refresh_code_flag(page);
    }
}

#[inline(always)]
pub(crate) fn jopsize(w: u8) -> OpSize {
    match w {
        8 => OpSize::S8,
        16 => OpSize::S16,
        32 => OpSize::S32,
        _ => OpSize::S64,
    }
}

// Effective address of a decoded memory operand (mirrors the JIT's gen_effective_addr).
#[inline(always)]
pub(crate) unsafe fn jmem_addr(mem: &JMem) -> u64 {
    let mask32 = mem.addr_size == 32;
    let mut have = false;
    let mut addr: u64 = 0;
    if let Some(base) = mem.base {
        addr = read_reg64(base as i32);
        if mask32 { addr &= 0xFFFF_FFFF; }
        have = true;
    }
    if let Some(index) = mem.index {
        let mut v = read_reg64(index as i32);
        if mask32 { v &= 0xFFFF_FFFF; }
        if mem.scale > 0 { v = v.wrapping_mul(1u64 << mem.scale); }
        addr = if have { addr.wrapping_add(v) } else { v };
    }
    if mem.disp != 0 { addr = addr.wrapping_add(mem.disp as u64); }
    if mask32 { addr &= 0xFFFF_FFFF; }
    match mem.segment {
        Some(0x64) => addr = addr.wrapping_add(*fs_base),
        Some(0x65) => addr = addr.wrapping_add(*gs_base),
        _ => {},
    }
    addr
}

pub(crate) fn jextend(v: u64, src_width: u8, signed: bool) -> u64 {
    match (src_width, signed) {
        (8, true) => v as u8 as i8 as i64 as u64,
        (8, false) => v & 0xFF,
        (16, true) => v as u16 as i16 as i64 as u64,
        (16, false) => v & 0xFFFF,
        (32, true) => v as u32 as i32 as i64 as u64,
        (32, false) => v & 0xFFFF_FFFF,
        _ => v,
    }
}

pub(crate) fn jalu_op(op: JArithOp) -> AluOp {
    match op {
        JArithOp::Add => AluOp::Add,
        JArithOp::Adc => AluOp::Adc,
        JArithOp::Sub => AluOp::Sub,
        JArithOp::Sbb => AluOp::Sbb,
        JArithOp::Cmp => AluOp::Cmp,
        JArithOp::And => AluOp::And,
        JArithOp::Or => AluOp::Or,
        JArithOp::Xor => AluOp::Xor,
        JArithOp::Test => AluOp::And, // Test is handled separately
    }
}

// Conservative set: skip anything with a high-byte register (r >= 16), XMM,
// port I/O, CPUID/RDTSC, cmpxchg, bit-test, bswap and imul-with-memory.
pub(crate) fn jreg_ok(r: u8) -> bool { r < 16 }

pub(crate) fn instr_supported(i: &JInstr) -> bool {
    use JInstr::*;
    match *i {
        MovRegImm { r, .. } => jreg_ok(r),
        MovRegReg { dst, src, .. } => jreg_ok(dst) && jreg_ok(src),
        MovRegMem { dst, .. } => jreg_ok(dst),
        MovMemReg { src, .. } => jreg_ok(src),
        MovMemImm { .. } => true,
        MovExtendReg { dst, src, high8, .. } => !high8 && jreg_ok(dst) && jreg_ok(src),
        MovExtendMem { dst, .. } => jreg_ok(dst),
        AddRegReg { dst, src, .. } => jreg_ok(dst) && jreg_ok(src),
        AddRegImm { r, .. } => jreg_ok(r),
        ArithRegReg { dst, src, .. } => jreg_ok(dst) && jreg_ok(src),
        ArithRegImm { r, .. } => jreg_ok(r),
        ArithRegMem { dst, .. } => jreg_ok(dst),
        ArithMemReg { src, .. } => jreg_ok(src),
        ArithMemImm { .. } => true,
        IncDecReg { r, .. } => jreg_ok(r),
        IncDecMem { .. } => true,
        NotReg { r, .. } => jreg_ok(r),
        NotMem { .. } => true,
        NegReg { r, .. } => jreg_ok(r),
        NegMem { .. } => true,
        CmovRegReg { dst, src, .. } => jreg_ok(dst) && jreg_ok(src),
        CmovRegMem { dst, .. } => jreg_ok(dst),
        SetccReg { dst, high8, .. } => !high8 && jreg_ok(dst),
        SetccMem { .. } => true,
        ShiftReg { r, .. } => jreg_ok(r),
        ShiftRegCl { r, .. } => jreg_ok(r),
        ShiftMem { .. } => true,
        ShiftMemCl { .. } => true,
        XchgRegReg { a, b, .. } => jreg_ok(a) && jreg_ok(b),
        XchgMemReg { r, .. } => jreg_ok(r),
        Lea { dst, .. } => jreg_ok(dst),
        PushReg { r } => jreg_ok(r),
        PopReg { r } => jreg_ok(r),
        PushImm { .. } => true,
        Call { .. } | CallMem { .. } | Jmp { .. } | Jcc { .. } | Ret { .. } => true,
        CallReg { r, .. } => jreg_ok(r),
        JmpReg { r } => jreg_ok(r),
        JmpMem { .. } => true,
        Leave | Nop | Hlt | Cli | PushFlags => true,
        Avx { .. } => true,
        // SSE2 moves, xor, integer ops, shifts and shuffles (jexec handles them
        // through the shared `interp64_0f` helpers).
        XmmCopy { .. } | XmmLoad { .. } | XmmStore { .. } | XmmXor { .. } | XmmXorMem { .. } => true,
        XmmInt { .. } | XmmIntMem { .. } | XmmShiftImm { .. } | XmmShuf { .. } | XmmShufMem { .. } => true,
        ImulRegReg { .. } | ImulRegImm { .. } | ImulRegMem { .. } => true,
        BitTestReg { .. } | BitTestImm { .. } => true,
        Bswap { .. } => true,
        CmpxchgReg { .. } | CmpxchgMem { .. } => true,
        _ => false,
    }
}

#[inline(always)]
pub(crate) unsafe fn jexec(instr: &JInstr) -> OrPageFault<JStep> {
    use JInstr::*;
    match *instr {
        MovRegImm { r, value, width } => write_reg(r, jopsize(width), true, value),
        MovRegReg { dst, src, width } => {
            let v = read_reg(src, jopsize(width), true);
            write_reg(dst, jopsize(width), true, v);
        },
        MovRegMem { dst, mem, width } => {
            let a = jmem_addr(&mem);
            let v = mem_read(a, jopsize(width))?;
            write_reg(dst, jopsize(width), true, v);
        },
        MovMemReg { mem, src, width } => {
            let a = jmem_addr(&mem);
            let v = read_reg(src, jopsize(width), true);
            mem_write(a, jopsize(width), v)?;
        },
        MovMemImm { mem, value, width } => {
            let a = jmem_addr(&mem);
            mem_write(a, jopsize(width), value)?;
        },
        MovExtendReg { dst, src, src_width, dst_width, signed, .. } => {
            let v = jextend(read_reg(src, jopsize(src_width), true), src_width, signed);
            write_reg(dst, jopsize(dst_width), true, v);
        },
        MovExtendMem { dst, mem, src_width, dst_width, signed } => {
            let a = jmem_addr(&mem);
            let v = jextend(mem_read(a, jopsize(src_width))?, src_width, signed);
            write_reg(dst, jopsize(dst_width), true, v);
        },
        AddRegReg { dst, src, width } => {
            let sz = jopsize(width);
            let d = read_reg(dst, sz, true);
            let s = read_reg(src, sz, true);
            let r = alu(AluOp::Add, d, s, sz);
            write_reg(dst, sz, true, r);
        },
        AddRegImm { r, value, width } => {
            let sz = jopsize(width);
            let d = read_reg(r, sz, true);
            let res = alu(AluOp::Add, d, value, sz);
            write_reg(r, sz, true, res);
        },
        ArithRegReg { op, dst, src, width } => {
            let sz = jopsize(width);
            let d = read_reg(dst, sz, true);
            let s = read_reg(src, sz, true);
            jalu(op, dst, d, s, sz);
        },
        ArithRegImm { op, r, value, width } => {
            let sz = jopsize(width);
            let d = read_reg(r, sz, true);
            jalu(op, r, d, value, sz);
        },
        ArithRegMem { op, dst, mem, width } => {
            let sz = jopsize(width);
            let a = jmem_addr(&mem);
            let d = read_reg(dst, sz, true);
            let s = mem_read(a, sz)?;
            jalu(op, dst, d, s, sz);
        },
        ArithMemReg { op, mem, src, width } => {
            let sz = jopsize(width);
            let a = jmem_addr(&mem);
            let d = mem_read(a, sz)?;
            let s = read_reg(src, sz, true);
            if op == JArithOp::Cmp || op == JArithOp::Test {
                jalu(op, 0, d, s, sz);
            }
            else {
                probe_operand_write(Operand::Mem(a), sz)?;
                let res = alu(jalu_op(op), d, s, sz);
                mem_write(a, sz, res)?;
            }
        },
        ArithMemImm { op, mem, value, width } => {
            let sz = jopsize(width);
            let a = jmem_addr(&mem);
            let d = mem_read(a, sz)?;
            if op == JArithOp::Cmp || op == JArithOp::Test {
                jalu(op, 0, d, value, sz);
            }
            else {
                probe_operand_write(Operand::Mem(a), sz)?;
                let res = alu(jalu_op(op), d, value, sz);
                mem_write(a, sz, res)?;
            }
        },
        IncDecReg { r, width, decrement } => {
            let sz = jopsize(width);
            let carry = *flags & FLAG_CF as i32;
            let d = read_reg(r, sz, true);
            let res = alu(if decrement { AluOp::Sub } else { AluOp::Add }, d, 1, sz);
            *flags = (*flags & !(FLAG_CF as i32)) | carry;
            write_reg(r, sz, true, res);
        },
        IncDecMem { mem, width, decrement } => {
            let sz = jopsize(width);
            let carry = *flags & FLAG_CF as i32;
            let a = jmem_addr(&mem);
            probe_operand_write(Operand::Mem(a), sz)?;
            let d = mem_read(a, sz)?;
            let res = alu(if decrement { AluOp::Sub } else { AluOp::Add }, d, 1, sz);
            *flags = (*flags & !(FLAG_CF as i32)) | carry;
            mem_write(a, sz, res)?;
        },
        NotReg { r, width } => {
            let sz = jopsize(width);
            let v = read_reg(r, sz, true);
            write_reg(r, sz, true, !v);
        },
        NotMem { mem, width } => {
            let sz = jopsize(width);
            let a = jmem_addr(&mem);
            probe_operand_write(Operand::Mem(a), sz)?;
            let v = mem_read(a, sz)?;
            mem_write(a, sz, !v)?;
        },
        NegReg { r, width } => {
            let sz = jopsize(width);
            let v = read_reg(r, sz, true);
            let res = alu(AluOp::Sub, 0, v, sz);
            write_reg(r, sz, true, res);
        },
        NegMem { mem, width } => {
            let sz = jopsize(width);
            let a = jmem_addr(&mem);
            probe_operand_write(Operand::Mem(a), sz)?;
            let v = mem_read(a, sz)?;
            let res = alu(AluOp::Sub, 0, v, sz);
            mem_write(a, sz, res)?;
        },
        CmovRegReg { code, dst, src, width } => {
            if condition(code) {
                let v = read_reg(src, jopsize(width), true);
                write_reg(dst, jopsize(width), true, v);
            }
        },
        CmovRegMem { code, dst, mem, width } => {
            let sz = jopsize(width);
            let a = jmem_addr(&mem);
            let v = mem_read(a, sz)?;
            if condition(code) {
                write_reg(dst, sz, true, v);
            }
        },
        SetccReg { code, dst, .. } => {
            write_reg(dst, OpSize::S8, true, condition(code) as u64);
        },
        SetccMem { code, mem } => {
            let a = jmem_addr(&mem);
            mem_write(a, OpSize::S8, condition(code) as u64)?;
        },
        ShiftReg { kind, r, width, count } => {
            if count != 0 {
                let sz = jopsize(width);
                let v = read_reg(r, sz, true);
                let res = crate::cpu::jit::jit64::shift_apply(v, count as u64, width as u32 | (kind as u32) << 8);
                write_reg(r, sz, true, res);
            }
        },
        ShiftRegCl { kind, r, width } => {
            let sz = jopsize(width);
            let count = read_reg64(1);
            let v = read_reg(r, sz, true);
            let res = crate::cpu::jit::jit64::shift_apply(v, count, width as u32 | (kind as u32) << 8);
            write_reg(r, sz, true, res);
        },
        ShiftMem { kind, mem, width, count } => {
            if count != 0 {
                let sz = jopsize(width);
                let a = jmem_addr(&mem);
                probe_operand_write(Operand::Mem(a), sz)?;
                let v = mem_read(a, sz)?;
                let res = crate::cpu::jit::jit64::shift_apply(v, count as u64, width as u32 | (kind as u32) << 8);
                mem_write(a, sz, res)?;
            }
        },
        ShiftMemCl { kind, mem, width } => {
            let sz = jopsize(width);
            let count = read_reg64(1) & if width == 64 { 63 } else { 31 };
            if count != 0 {
                let a = jmem_addr(&mem);
                probe_operand_write(Operand::Mem(a), sz)?;
                let v = mem_read(a, sz)?;
                let res = crate::cpu::jit::jit64::shift_apply(v, count, width as u32 | (kind as u32) << 8);
                mem_write(a, sz, res)?;
            }
        },
        XchgRegReg { a, b, width } => {
            let sz = jopsize(width);
            let va = read_reg(a, sz, true);
            let vb = read_reg(b, sz, true);
            write_reg(a, sz, true, vb);
            write_reg(b, sz, true, va);
        },
        XchgMemReg { mem, r, width } => {
            let sz = jopsize(width);
            let a = jmem_addr(&mem);
            probe_operand_write(Operand::Mem(a), sz)?;
            let old_mem = mem_read(a, sz)?;
            let old_reg = read_reg(r, sz, true);
            mem_write(a, sz, old_reg)?;
            write_reg(r, sz, true, old_mem);
        },
        Lea { dst, mem, width } => {
            let a = jmem_addr(&mem);
            write_reg(dst, jopsize(width), true, a);
        },
        PushReg { r } => push64(read_reg64(r as i32))?,
        PopReg { r } => {
            let v = pop64()?;
            write_reg64(r as i32, v);
        },
        PushImm { value } => push64(value)?,
        Jmp { target } => {
            *rip = target;
            return Ok(JStep::Stop);
        },
        Jcc { code, target, fallthrough } => {
            *rip = if condition(code) { target } else { fallthrough };
            return Ok(JStep::Stop);
        },
        Call { target, return_address } => {
            push64(return_address)?;
            *rip = target;
            return Ok(JStep::Stop);
        },
        CallReg { r, return_address } => {
            let target = read_reg64(r as i32);
            push64(return_address)?;
            *rip = target;
            return Ok(JStep::Stop);
        },
        CallMem { mem, return_address } => {
            let a = jmem_addr(&mem);
            let target = mem_read(a, OpSize::S64)?;
            push64(return_address)?;
            *rip = target;
            return Ok(JStep::Stop);
        },
        JmpReg { r } => {
            *rip = read_reg64(r as i32);
            return Ok(JStep::Stop);
        },
        JmpMem { mem } => {
            let a = jmem_addr(&mem);
            *rip = mem_read(a, OpSize::S64)?;
            return Ok(JStep::Stop);
        },
        Ret { adjustment } => {
            let target = pop64()?;
            if adjustment != 0 {
                let rsp = read_reg64(RSP as i32);
                write_reg64(RSP as i32, rsp.wrapping_add(adjustment as u64));
            }
            *rip = target;
            return Ok(JStep::Stop);
        },
        Leave => {
            let rbp = read_reg64(RBP as i32);
            write_reg64(RSP as i32, rbp);
            let v = pop64()?;
            write_reg64(RBP as i32, v);
        },
        Nop => {},
        Cli => crate::cpu::jit::jit64::jit64_cli(),
        PushFlags => crate::cpu::jit::jit64::jit64_pushfq(),
        Hlt => {
            *in_hlt = true;
            return Ok(JStep::Stop);
        },
        // VEX/AVX: run the shared implementation at the instruction's RIP.
        Avx { rip: address } =>
        {
            *rip = address;
            *previous_rip = address;
            run_one_inner()?;
        },
        // SSE2 register/memory moves, xor, integer ops, shifts and shuffles.
        // The XMM file lives in emulated memory, so these go through the
        // shared helpers in `interp64_0f`.
        XmmLoad { dst, mem } => {
            let a = jmem_addr(&mem);
            self::interp64_0f::xmm_set(dst, self::interp64_0f::mem_read128(a)?);
        },
        XmmStore { mem, src } => {
            let a = jmem_addr(&mem);
            self::interp64_0f::mem_write128(a, self::interp64_0f::xmm_get(src))?;
        },
        XmmCopy { dst, src } => self::interp64_0f::xmm_set(dst, self::interp64_0f::xmm_get(src)),
        XmmXor { dst, src } => {
            let mut v = self::interp64_0f::xmm_get(dst);
            let s = self::interp64_0f::xmm_get(src);
            v.u64[0] ^= s.u64[0];
            v.u64[1] ^= s.u64[1];
            self::interp64_0f::xmm_set(dst, v);
        },
        XmmXorMem { dst, mem } => {
            let a = jmem_addr(&mem);
            let s = self::interp64_0f::mem_read128(a)?;
            let mut v = self::interp64_0f::xmm_get(dst);
            v.u64[0] ^= s.u64[0];
            v.u64[1] ^= s.u64[1];
            self::interp64_0f::xmm_set(dst, v);
        },
        XmmInt { op, dst, src } => {
            let s = self::interp64_0f::xmm_get(src);
            self::interp64_0f::jit64_sse_int(op as i32, s.u64[0], s.u64[1], dst as i32);
        },
        XmmIntMem { op, dst, mem } => {
            let a = jmem_addr(&mem);
            let s = self::interp64_0f::mem_read128(a)?;
            self::interp64_0f::jit64_sse_int(op as i32, s.u64[0], s.u64[1], dst as i32);
        },
        XmmShiftImm { op, group, count, dst } => {
            let v = self::interp64_0f::xmm_get(dst);
            self::interp64_0f::jit64_sse_shift_imm(
                op as i32 | (group as i32) << 8 | (count as i32) << 16,
                v.u64[0],
                v.u64[1],
                dst as i32,
            );
        },
        XmmShuf { control, dst, src } => {
            let s = self::interp64_0f::xmm_get(src);
            self::interp64_0f::jit64_sse_pshuf(control as i32, s.u64[0], s.u64[1], dst as i32);
        },
        XmmShufMem { control, dst, mem } => {
            let a = jmem_addr(&mem);
            let s = self::interp64_0f::mem_read128(a)?;
            self::interp64_0f::jit64_sse_pshuf(control as i32, s.u64[0], s.u64[1], dst as i32);
        },
        // Common integer 0F forms: IMUL, BT/BTS/BTR/BTC, BSWAP, CMPXCHG.
        ImulRegReg { dst, lhs, rhs, width } => {
            let size = jopsize(width);
            let l = read_reg(lhs, size, true);
            let r = read_reg(rhs, size, true);
            write_reg(dst, size, true, imul_value(l, r, size));
        },
        ImulRegImm { dst, src, value, width } => {
            let size = jopsize(width);
            let s = read_reg(src, size, true);
            write_reg(dst, size, true, imul_value(s, value, size));
        },
        ImulRegMem { dst, mem, value, width } => {
            let size = jopsize(width);
            let a = jmem_addr(&mem);
            let m = mem_read(a, size)?;
            let r = match value {
                Some(v) => imul_value(m, v, size),
                None => imul_value(read_reg(dst, size, true), m, size),
            };
            write_reg(dst, size, true, r);
        },
        Bswap { r, width } => {
            let size = jopsize(width);
            let v = read_reg(r, size, true);
            let value = if width == 64 { v.swap_bytes() } else { (v as u32).swap_bytes() as u64 };
            write_reg(r, size, true, value);
        },
        BitTestReg { r, index_r, op, width } => {
            let size = jopsize(width);
            let index = read_reg(index_r, size, true);
            bit_test(Operand::Reg(r), &Prefixes::default(), size, index, op)?;
        },
        BitTestImm { r, index, op, width } => {
            let size = jopsize(width);
            bit_test(Operand::Reg(r), &Prefixes::default(), size, index, op)?;
        },
        CmpxchgReg { dst, src, width } => {
            let size = jopsize(width);
            let d = read_reg(dst, size, true) & size.mask();
            let s = read_reg(src, size, true);
            let acc = read_reg(RAX, size, false) & size.mask();
            if acc == d {
                set_zf(true);
                write_reg(dst, size, true, s);
            }
            else {
                set_zf(false);
                write_reg(RAX, size, false, d);
            }
        },
        CmpxchgMem { mem, src, width } => {
            let size = jopsize(width);
            let a = jmem_addr(&mem);
            let d = mem_read(a, size)? & size.mask();
            let s = read_reg(src, size, true);
            let acc = read_reg(RAX, size, false) & size.mask();
            if acc == d {
                set_zf(true);
                mem_write(a, size, s)?;
            }
            else {
                set_zf(false);
                write_reg(RAX, size, false, d);
            }
        },
        _ => {},
    }
    Ok(JStep::Continue)
}

#[inline(always)]
pub(crate) unsafe fn jalu(op: JArithOp, dst_reg: u8, dst: u64, src: u64, size: OpSize) {
    if op == JArithOp::Test {
        set_logic_flags(dst & src & size.mask(), size);
        return;
    }
    let result = alu(jalu_op(op), dst, src, size);
    if op != JArithOp::Cmp {
        write_reg(dst_reg, size, true, result);
    }
}

// Compare the cached block bytes with live memory (SMC).
pub(crate) unsafe fn validate_block(start_rip: u64, bytes: &[u8]) -> bool {
    let phys = match crate::cpu::core::translate_address_64_no_side_effects(start_rip) {
        Ok(p) => p,
        Err(_) => return false,
    };
    let phys_base = phys & 0xFFFFF000;
    if memory::in_mapped_range(phys_base) {
        return false;
    }
    let off = (start_rip & 0xFFF) as usize;
    if off + bytes.len() > 0x1000 {
        return false;
    }
    let page = std::slice::from_raw_parts(memory::mem8.add(phys_base as usize) as *const u8, 0x1000);
    page[off..off + bytes.len()] == *bytes
}

pub(crate) unsafe fn decode_block_at(
    start_rip: u64,
) -> Option<(Vec<JInstr>, Vec<u64>, u64, Vec<u8>, u32)> {
    let phys = crate::cpu::core::translate_address_64_no_side_effects(start_rip).ok()?;
    let phys_base = phys & 0xFFFFF000;
    if memory::in_mapped_range(phys_base) {
        return None;
    }
    let off = (start_rip & 0xFFF) as usize;
    let page = std::slice::from_raw_parts(memory::mem8.add(phys_base as usize) as *const u8, 0x1000);
    let (instrs, rips, end_rip) = crate::cpu::jit::jit64::decode_block_parts(start_rip, &page[off..]).ok()?;
    if instrs.is_empty() || end_rip <= start_rip {
        return None;
    }
    let block_bytes = page[off..off + (end_rip - start_rip) as usize].to_vec();
    Some((instrs, rips, end_rip, block_bytes, phys_base >> 12))
}

impl DecodedInstr for JInstr {
    #[inline(always)]
    fn is_supported(&self) -> bool { instr_supported(self) }
    #[inline(always)]
    fn jcc(&self) -> Option<(u8, u64, u64)> {
        if let JInstr::Jcc { code, target, fallthrough } = *self {
            Some((code, target, fallthrough))
        }
        else {
            None
        }
    }
    #[inline(always)]
    fn exec(&self) -> OrPageFault<JStep> { unsafe { jexec(self) } }
}

struct Interp64;

impl DecodeHost for Interp64 {
    type Instr = JInstr;
    #[inline(always)]
    unsafe fn set_previous_ip(next: u64) { *previous_rip = next; }
    #[inline(always)]
    unsafe fn set_ip(next: u64) { *rip = next; }
    #[inline(always)]
    unsafe fn retire(n: u32) { *instruction_counter = (*instruction_counter).wrapping_add(n); }
    #[inline(always)]
    unsafe fn page_write_count(phys_page: u32) -> u32 {
        crate::cpu::core::page_write_count(phys_page)
    }
    unsafe fn validate_block(start_rip: u64, bytes: &[u8]) -> bool {
        validate_block(start_rip, bytes)
    }
    unsafe fn decode_block_at(
        start_rip: u64,
    ) -> Option<(Vec<JInstr>, Vec<u64>, u64, Vec<u8>, u32)> {
        decode_block_at(start_rip)
    }
    /// A block here is validated against the page's write counter, so the JIT
    /// must not write this page inline -- an inline store skips
    /// invalidate_physical_page, and the counter would not move.
    unsafe fn on_block_cached(phys_page: u32) {
        crate::cpu::core::page_decoded_init();
        crate::cpu::core::mark_page_decoded(phys_page);
        crate::cpu::core::tlb64_refresh_code_flag(phys_page);
    }
}

// Run one decoded block if possible. Returns the number of instructions run.
pub(crate) unsafe fn decode_cache_run() -> Option<u32> {
    if !DECODE_CACHE_ENABLED || crate::cpu::core::interrupt_shadow() != 0 {
        return None;
    }
    let cache = decode_cache();
    crate::cpu::interp::decode_cache::cache_run::<Interp64>(
        cache.as_mut_slice(),
        *rip,
        *cpl == 3,
        crate::cpu::core::tlb64_generation,
    )
}

// Point CR3 at a PML4, enable PAE and paging, and switch the main loop over to
// the 64-bit interpreter.
#[no_mangle]
pub unsafe fn enter_long_mode(cr3: u32) {
    // The 16/32-bit interpreter keeps arithmetic flags lazy. Long mode reads the
    // flags word directly, so resolve them once on entry.
    *flags = crate::cpu::core::get_eflags();
    *flags_changed = 0;
    *rip = *instruction_pointer as u32 as u64;
    *previous_rip = *rip;
    *cr.offset(3) = cr3 as i32; // CR3
    *cr.offset(4) |= crate::cpu::core::CR4_PAE; // CR4.PAE
    *cr.offset(4) |= crate::cpu::core::CR4_PSE; // CR4.PSE (2 MiB pages)
    *long_mode = true;
    *efer |= EFER_LME | EFER_LMA;
    *cr |= crate::cpu::core::CR0_PG; // CR0.PG
    // Linux assumes the LAPIC is already enabled and never writes APICBASE; v64
    // has no firmware, so enable it here or no IRQ is ever delivered.
    *apic_enabled = true;
    crate::cpu::core::full_clear_tlb();
}

/// Enter long mode at a kernel entry point, as a 64-bit boot loader would:
/// identity-mapped lower memory, CS = 0x10, SS = 0x18, RSI = boot_params.
#[no_mangle]
pub unsafe fn boot64(cr3: u32, entry: u64, boot_params: u64) {
    enter_long_mode(cr3);
    *rip = entry;
    *previous_rip = entry;
    *instruction_pointer = entry as u32 as i32;
    write_reg64(RSI as i32, boot_params);
    write_reg64(RSP as i32, 0x9F000);
    set_cpl_segments(0x10, 0);
}

// --- AVX (VEX-encoded) -----------------------------------------------------

pub(crate) unsafe fn run_one_inner() -> OrPageFault<()> {
    let (pfx, opcode) = decode_prefixes()?;
    if INTERP64_OPCODE_STATS {
        INTERP64_OPCODE[opcode as usize] = INTERP64_OPCODE[opcode as usize].wrapping_add(1);
    }

    if opcode == 0x0F {
        return self::interp64_0f::run_0f(&pfx);
    }

    if opcode == 0xC4 || opcode == 0xC5 {
        return self::interp64_0f::run_vex(opcode, &pfx);
    }

    let size = pfx.operand_size();

    match opcode {
        // NOP (0x90) / XCHG r, rAX (0x90 with REX.B, 0x91-0x97)
        0x90..=0x97 => {
            let r = (opcode - 0x90) | (pfx.rex & prefix::REX_B) << 3;
            if r != 0 {
                let a = read_reg(RAX, size, pfx.has_rex());
                let b = read_reg(r, size, pfx.has_rex());
                write_reg(RAX, size, pfx.has_rex(), b);
                write_reg(r, size, pfx.has_rex(), a);
            }
        },

        // HLT
        0xF4 => {
            *in_hlt = true;
        },

        // MOV r/m8, r8  /  MOV r/m, r
        0x88 | 0x89 => {
            let (modrm, operand) = decode_operand(&pfx)?;
            let size = if opcode == 0x88 { OpSize::S8 } else { size };
            let value = read_reg(modrm.reg, size, pfx.has_rex());
            write_operand(operand, size, pfx.has_rex(), value)?;
        },

        // MOV r8, r/m8  /  MOV r, r/m
        0x8A | 0x8B => {
            let (modrm, operand) = decode_operand(&pfx)?;
            let size = if opcode == 0x8A { OpSize::S8 } else { size };
            let value = read_operand(operand, size, pfx.has_rex())?;
            write_reg(modrm.reg, size, pfx.has_rex(), value);
        },

        // XCHG r/m, r
        0x86 | 0x87 => {
            let (modrm, operand) = decode_operand(&pfx)?;
            let size = if opcode == 0x86 { OpSize::S8 } else { size };
            let left = read_operand(operand, size, pfx.has_rex())?;
            let right = read_reg(modrm.reg, size, pfx.has_rex());
            write_operand(operand, size, pfx.has_rex(), right)?;
            write_reg(modrm.reg, size, pfx.has_rex(), left);
        },

        // MOV r/m, imm (0xC6: r/m8, imm8; 0xC7: r/m, imm16/32)
        0xC6 | 0xC7 => {
            let size = if opcode == 0xC6 { OpSize::S8 } else { size };
            let trailing = if size == OpSize::S8 { 1 } else if size == OpSize::S16 { 2 } else { 4 };
            let (_, operand) = decode_operand_with_trailing(&pfx, trailing)?;
            let value = fetch_imm(size)?;
            write_operand(operand, size, pfx.has_rex(), value)?;
        },

        // MOV r, imm (0xB0-0xB7: r8, imm8; 0xB8-0xBF: r, imm)
        0xB0..=0xB7 => {
            let r = (opcode - 0xB0) | (pfx.rex & prefix::REX_B) << 3;
            let value = fetch8()? as u64;
            write_reg(r, OpSize::S8, pfx.has_rex(), value);
        },
        0xB8..=0xBF => {
            let r = (opcode - 0xB8) | (pfx.rex & prefix::REX_B) << 3;
            let value = if size == OpSize::S64 {
                fetch64()?
            }
            else {
                fetch_imm(size)?
            };
            write_reg(r, size, pfx.has_rex(), value);
        },

        // LEA r, m (0x8D)
        0x8D => {
            let modrm = decode_modrm(&pfx)?;
            if modrm.mod_bits == 3 {
                crate::cpu::core::trigger_ud();
                return Ok(());
            }
            let address = modrm_effective_address(&modrm, &pfx)?;
            write_reg(modrm.reg, size, pfx.has_rex(), address);
        },

        // ALU r/m, r  and  r, r/m
        0x00 | 0x01 | 0x02 | 0x03 | 0x08 | 0x09 | 0x0A | 0x0B | 0x10 | 0x11 | 0x12 | 0x13
        | 0x18 | 0x19 | 0x1A | 0x1B | 0x20 | 0x21 | 0x22 | 0x23 | 0x28 | 0x29 | 0x2A
        | 0x2B | 0x30 | 0x31 | 0x32 | 0x33 | 0x38 | 0x39 | 0x3A | 0x3B => {
            let size = if opcode & 1 == 0 { OpSize::S8 } else { size };
            alu_rm_r_rm(alu_op_of(opcode), opcode, &pfx, size)?;
        },

        // ALU accumulator, imm (0x04/0x0C/...: AL, imm8; 0x05/0x0D/...: rAX, imm)
        0x04 | 0x0C | 0x14 | 0x1C | 0x24 | 0x2C | 0x34 | 0x3C => {
            let imm = fetch8()? as u64;
            alu_acc_imm(alu_op_of(opcode), OpSize::S8, imm)?;
        },
        0x05 | 0x0D | 0x15 | 0x1D | 0x25 | 0x2D | 0x35 | 0x3D => {
            let imm = fetch_imm(size)?;
            alu_acc_imm(alu_op_of(opcode), size, imm)?;
        },

        // ALU r/m, imm
        0x80 => alu_group1(opcode, &pfx, OpSize::S8)?,
        0x81 => alu_group1(opcode, &pfx, size)?,
        0x83 => alu_group1(opcode, &pfx, size)?,

        // TEST
        0x84 => test_rm_r(&pfx, OpSize::S8)?,
        0x85 => test_rm_r(&pfx, size)?,
        0xA8 => {
            let a = read_reg(RAX, OpSize::S8, false);
            let imm = fetch8()? as u64;
            set_logic_flags(a & imm, OpSize::S8);
        },
        0xA9 => {
            let a = read_reg(RAX, size, false);
            let imm = fetch_imm(size)?;
            set_logic_flags(a & imm & size.mask(), size);
        },

        // Group 3: TEST/NOT/NEG/MUL/IMUL/DIV/IDIV (F6/F7)
        0xF6 | 0xF7 => group3(opcode, &pfx, size)?,

        // Group 2: ROL/ROR/RCL/RCR/SHL/SHR/SAR (C0/C1/D0/D1/D2/D3)
        0xC0 | 0xC1 | 0xD0 | 0xD1 | 0xD2 | 0xD3 => group2(opcode, &pfx, size)?,

        // PUSH/POP r64 (0x50-0x5F)
        0x50..=0x57 => {
            let r = (opcode - 0x50) | (pfx.rex & prefix::REX_B) << 3;
            push64(read_reg64(r as i32))?;
        },
        0x58..=0x5F => {
            let r = (opcode - 0x58) | (pfx.rex & prefix::REX_B) << 3;
            let value = pop64()?;
            write_reg64(r as i32, value);
        },

        // MOVSXD r64, r/m32 (REX.W); without it AMD's r32 form is accepted.
        0x63 => {
            let (modrm, operand) = decode_operand(&pfx)?;
            let value = read_operand(operand, OpSize::S32, pfx.has_rex())?;
            if size == OpSize::S64 {
                write_reg64(modrm.reg as i32, value as u32 as i32 as i64 as u64);
            }
            else {
                write_reg(modrm.reg, OpSize::S32, pfx.has_rex(), value);
            }
        },

        // IMUL r, r/m, imm32/imm8
        0x69 | 0x6B => {
            let trailing = if opcode == 0x6B { 1 } else if size == OpSize::S16 { 2 } else { 4 };
            let (modrm, operand) = decode_operand_with_trailing(&pfx, trailing)?;
            let source = read_operand(operand, size, pfx.has_rex())?;
            let immediate = if opcode == 0x6B {
                fetch_imm8_signed(size)?
            }
            else {
                fetch_imm(size)?
            };
            write_reg(
                modrm.reg,
                size,
                pfx.has_rex(),
                imul_value(source, immediate, size),
            );
        },
        // PUSH imm (0x68: imm32, 0x6A: imm8)
        0x68 => {
            let value = fetch32()? as i32 as i64 as u64;
            push64(value)?;
        },
        0x6A => {
            let value = fetch8()? as i8 as i64 as u64;
            push64(value)?;
        },

        // Jcc rel8 (0x70-0x7F)
        0x70..=0x7F => {
            let displacement = fetch8()? as i8 as i64;
            jcc(opcode - 0x70, displacement);
        },

        // IN AL, imm8 (E4) / IN AX|EAX, imm8 (E5)
        0xE4 | 0xE5 => {
            let port = fetch8()? as i32;
            let (op, width) = if opcode == 0xE4 {
                (OpSize::S8, 1)
            }
            else if size == OpSize::S16 {
                (OpSize::S16, 2)
            }
            else {
                (OpSize::S32, 4)
            };
            if test_privileges_for_io(port, width) {
                let value = match width {
                    1 => io_port_read8(port) as u64,
                    2 => crate::cpu::core::io_port_read16(port) as u16 as u64,
                    _ => crate::cpu::core::io_port_read32(port) as u32 as u64,
                };
                write_reg(RAX, op, false, value);
            }
        },

        // OUT imm8, AL (E6) / OUT imm8, AX|EAX (E7)
        0xE6 | 0xE7 => {
            let port = fetch8()? as i32;
            let (op, width) = if opcode == 0xE6 {
                (OpSize::S8, 1)
            }
            else if size == OpSize::S16 {
                (OpSize::S16, 2)
            }
            else {
                (OpSize::S32, 4)
            };
            if test_privileges_for_io(port, width) {
                match width {
                    1 => io_port_write8(port, read_reg(RAX, op, false) as i32),
                    2 => crate::cpu::core::io_port_write16(port, read_reg(RAX, op, false) as i32),
                    _ => crate::cpu::core::io_port_write32(port, read_reg(RAX, op, false) as i32),
                }
            }
        },

        // IN AL, DX (EC) / IN AX|EAX, DX (ED)
        0xEC | 0xED => {
            let port = read_reg(RDX, OpSize::S16, false) as i32;
            let (op, width) = if opcode == 0xEC {
                (OpSize::S8, 1)
            }
            else if size == OpSize::S16 {
                (OpSize::S16, 2)
            }
            else {
                (OpSize::S32, 4)
            };
            if test_privileges_for_io(port, width) {
                let value = match width {
                    1 => io_port_read8(port) as u64,
                    2 => crate::cpu::core::io_port_read16(port) as u16 as u64,
                    _ => crate::cpu::core::io_port_read32(port) as u32 as u64,
                };
                write_reg(RAX, op, false, value);
            }
        },

        // OUT DX, AL (EE) / OUT DX, AX|EAX (EF)
        0xEE | 0xEF => {
            let port = read_reg(RDX, OpSize::S16, false) as i32;
            let (op, width) = if opcode == 0xEE {
                (OpSize::S8, 1)
            }
            else if size == OpSize::S16 {
                (OpSize::S16, 2)
            }
            else {
                (OpSize::S32, 4)
            };
            if test_privileges_for_io(port, width) {
                match width {
                    1 => io_port_write8(port, read_reg(RAX, op, false) as i32),
                    2 => crate::cpu::core::io_port_write16(port, read_reg(RAX, op, false) as i32),
                    _ => crate::cpu::core::io_port_write32(port, read_reg(RAX, op, false) as i32),
                }
            }
        },

        // JMP rel8/rel32
        0xEB => {
            let displacement = fetch8()? as i8 as i64;
            *rip = (*rip).wrapping_add(displacement as u64);
        },
        0xE9 => {
            let displacement = fetch32()? as i32 as i64;
            *rip = (*rip).wrapping_add(displacement as u64);
        },

        // LOOPNE/LOOPE/LOOP/JRCXZ (0xE0-0xE3); the counter is RCX or ECX.
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/loop
        0xE0 | 0xE1 | 0xE2 | 0xE3 => {
            let displacement = fetch8()? as i8 as i64;
            let counter_mask = if pfx.addrsize_32 { 0xFFFF_FFFFu64 } else { u64::MAX };
            let count = read_reg64(RCX as i32) & counter_mask;
            let taken = if opcode == 0xE3 {
                count == 0
            }
            else {
                let count = count.wrapping_sub(1) & counter_mask;
                write_reg64(RCX as i32, count);
                let zf = *flags as u32 & FLAG_ZF != 0;
                match opcode {
                    0xE0 => count != 0 && !zf,
                    0xE1 => count != 0 && zf,
                    _ => count != 0,
                }
            };
            if taken {
                *rip = (*rip).wrapping_add(displacement as u64);
            }
        },

        // POP r/m64 (8F /0)
        0x8F => {
            let (modrm, operand) = decode_operand(&pfx)?;
            if modrm.reg & 7 != 0 {
                crate::cpu::core::trigger_ud();
            }
            else {
                let value = pop64()?;
                probe_operand_write(operand, OpSize::S64)?;
                write_operand(operand, OpSize::S64, pfx.has_rex(), value)?;
            }
        },

        // INC/DEC r/m8 (FE /0 and /1)
        0xFE => {
            let (modrm, operand) = decode_operand(&pfx)?;
            match modrm.reg & 7 {
                group @ (0 | 1) => {
                    let carry = *flags & FLAG_CARRY;
                    let value = read_operand(operand, OpSize::S8, pfx.has_rex())?;
                    probe_operand_write(operand, OpSize::S8)?;
                    let result = alu(
                        if group == 0 { AluOp::Add } else { AluOp::Sub },
                        value,
                        1,
                        OpSize::S8,
                    );
                    *flags = *flags & !FLAG_CARRY | carry;
                    write_operand(operand, OpSize::S8, pfx.has_rex(), result)?;
                },
                _ => crate::cpu::core::trigger_ud(),
            }
        },

        // CALL/JMP r/m64 (FF /2 and /4)
        0xFF => {
            let (modrm, operand) = decode_operand(&pfx)?;
            match modrm.reg & 7 {
                group @ (0 | 1) => {
                    let carry = *flags & FLAG_CARRY;
                    let value = read_operand(operand, size, pfx.has_rex())?;
                    probe_operand_write(operand, size)?;
                    let result = alu(
                        if group == 0 { AluOp::Add } else { AluOp::Sub },
                        value,
                        1,
                        size,
                    );
                    *flags = *flags & !FLAG_CARRY | carry;
                    write_operand(operand, size, pfx.has_rex(), result)?;
                },
                2 => {
                    let target = read_operand(operand, OpSize::S64, pfx.has_rex())?;
                    let return_address = *rip;
                    push64(return_address)?;
                    *rip = target;
                },
                4 => *rip = read_operand(operand, OpSize::S64, pfx.has_rex())?,
                // jmp m16:64. The indirect far jump firmware uses to land in a
                // 64-bit code segment; long mode has no direct form.
                5 => {
                    let Operand::Mem(address) = operand else {
                        crate::cpu::core::trigger_ud();
                        return Ok(());
                    };
                    let target = mem_read(address, OpSize::S64)?;
                    let selector = mem_read(address.wrapping_add(8), OpSize::S16)? as u16;
                    set_cpl_segments(selector, selector as u8 & 3);
                    *rip = target;
                },
                // push r/m64
                6 => {
                    let value = read_operand(operand, OpSize::S64, pfx.has_rex())?;
                    push64(value)?;
                },
                _ => crate::cpu::core::trigger_ud(),
            }
        },

        // CALL rel32 (0xE8)
        0xE8 => {
            let displacement = fetch32()? as i32 as i64;
            let return_address = *rip;
            push64(return_address)?;
            *rip = (*rip).wrapping_add(displacement as u64);
        },

        // RET (0xC3) / RET imm16 (0xC2)
        0xC3 => {
            *rip = pop64()?;
        },
        0xC2 => {
            let adjustment = fetch16()? as u64;
            *rip = pop64()?;
            let rsp = read_reg64(RSP as i32).wrapping_add(adjustment);
            write_reg64(RSP as i32, rsp);
        },

        // RETF (0xCB) / RETF imm16 (0xCA): far return. In long mode the return
        // address and the CS selector are popped as 64-bit values.
        0xCB | 0xCA => {
            let adjustment = if opcode == 0xCA { fetch16()? as u64 } else { 0 };
            // A far return pops 64-bit RIP/CS in 64-bit code, 32/16-bit only in
            // compatibility mode. cf. Intel SDM Vol. 2, RET.
            let compat = *is_32 && size != OpSize::S64;
            let (new_rip, new_cs) = if compat
            {
                (pop_size(size)? & size.mask(), pop_size(size)? & size.mask())
            }
            else
            {
                (pop64()?, pop64()?)
            };
            *sreg.offset(1) = new_cs as u16; // CS
            *cpl = new_cs as u8 & 3;
            *rip = new_rip;
            let rsp = read_reg64(RSP as i32).wrapping_add(adjustment);
            write_reg64(RSP as i32, rsp);
        },

        // MOVS (A4/A5), STOS (AA/AB), LODS (AC/AD), INS (6C/6D), OUTS (6E/6F).
        // F3 repeats.
        0x6C | 0x6D | 0x6E | 0x6F | 0xA4 | 0xA5 | 0xAA | 0xAB | 0xAC | 0xAD => {
            let elem_size = if opcode & 1 == 0 { OpSize::S8 } else { size };
            let bytes = (elem_size.bits() / 8) as u64;
            let delta = if *flags & (1 << 10) != 0 { -(bytes as i64) } else { bytes as i64 };

            let rep = pfx.f3;
            let mut count = if rep { read_reg64(RCX as i32) } else { 1 };

            if rep && count > 0 && (opcode & 0xFE == 0xA4 || opcode & 0xFE == 0xAA) {
                rep_movs_stos_fast(opcode, elem_size, delta, pfx.has_rex(), &mut count)?;
            }

            while count > 0 {
                let rsi = read_reg64(RSI as i32);
                let rdi = read_reg64(RDI as i32);

                match opcode & 0xFE {
                    // MOVS
                    0xA4 => {
                        let value = mem_read(rsi, elem_size)?;
                        mem_write(rdi, elem_size, value)?;
                        write_reg64(RSI as i32, rsi.wrapping_add(delta as u64));
                        write_reg64(RDI as i32, rdi.wrapping_add(delta as u64));
                    },
                    // STOS
                    0xAA => {
                        let value = read_reg(RAX, elem_size, pfx.has_rex());
                        mem_write(rdi, elem_size, value)?;
                        write_reg64(RDI as i32, rdi.wrapping_add(delta as u64));
                    },
                    // INS: read the port (DX) into [rdi]
                    0x6C => {
                        let port = read_reg(RDX, OpSize::S16, false) as i32;
                        let width = (elem_size.bits() / 8) as i32;
                        if test_privileges_for_io(port, width) {
                            let value = match width {
                                1 => io_port_read8(port) as u64,
                                2 => crate::cpu::core::io_port_read16(port) as u16 as u64,
                                _ => crate::cpu::core::io_port_read32(port) as u32 as u64,
                            };
                            mem_write(rdi, elem_size, value)?;
                        }
                        write_reg64(RDI as i32, rdi.wrapping_add(delta as u64));
                    },
                    // OUTS: write [rsi] to the port (DX)
                    0x6E => {
                        let port = read_reg(RDX, OpSize::S16, false) as i32;
                        let width = (elem_size.bits() / 8) as i32;
                        let value = mem_read(rsi, elem_size)?;
                        if test_privileges_for_io(port, width) {
                            match width {
                                1 => io_port_write8(port, value as i32),
                                2 => crate::cpu::core::io_port_write16(port, value as i32),
                                _ => crate::cpu::core::io_port_write32(port, value as i32),
                            }
                        }
                        write_reg64(RSI as i32, rsi.wrapping_add(delta as u64));
                    },
                    // LODS
                    _ => {
                        let value = mem_read(rsi, elem_size)?;
                        write_reg(RAX, elem_size, pfx.has_rex(), value);
                        write_reg64(RSI as i32, rsi.wrapping_add(delta as u64));
                    },
                }

                if rep {
                    count -= 1;
                    write_reg64(RCX as i32, count);
                }
                else {
                    count = 0;
                }
            }
        },

        // CMPS (0xA6/0xA7) and SCAS (0xAE/0xAF). F3 repeats while equal,
        // F2 repeats while not equal.
        0xA6 | 0xA7 | 0xAE | 0xAF => {
            let elem_size = if opcode & 1 == 0 { OpSize::S8 } else { size };
            let bytes = (elem_size.bits() / 8) as u64;
            let delta = if *flags & (1 << 10) != 0 { -(bytes as i64) } else { bytes as i64 };

            let rep = pfx.f3 || pfx.f2;
            let mut count = if rep { read_reg64(RCX as i32) } else { 1 };

            while count > 0 {
                let rsi = read_reg64(RSI as i32);
                let rdi = read_reg64(RDI as i32);

                if opcode & 0xFE == 0xA6 {
                    // CMPS: [rsi] - [rdi]
                    let a = mem_read(rsi, elem_size)?;
                    let b = mem_read(rdi, elem_size)?;
                    alu(AluOp::Cmp, a, b, elem_size);
                    write_reg64(RSI as i32, rsi.wrapping_add(delta as u64));
                }
                else {
                    // SCAS: accumulator - [rdi]
                    let a = read_reg(RAX, elem_size, pfx.has_rex());
                    let b = mem_read(rdi, elem_size)?;
                    alu(AluOp::Cmp, a, b, elem_size);
                }
                write_reg64(RDI as i32, rdi.wrapping_add(delta as u64));

                count -= 1;
                if rep {
                    write_reg64(RCX as i32, count);
                    let zf = *flags & FLAG_ZF as i32 != 0;
                    if zf != pfx.f3 {
                        break;
                    }
                }
                else {
                    count = 0;
                }
            }
        },

        // ENTER imm16, imm8; cf. Intel SDM Vol. 2:
// https://www.felixcloutier.com/x86/enter
        0xC8 => {
            let alloc = fetch16()? as u64;
            let nesting = fetch8()? & 0x1F;
            let frame = (size.bits() / 8) as u64;
            let mut rsp = read_reg64(RSP as i32);
            let rbp = read_reg64(RBP as i32);
            rsp = rsp.wrapping_sub(frame);
            mem_write(rsp, size, rbp & size.mask())?;
            let frame_temp = rsp;
            if nesting > 0 {
                let mut walk = rbp;
                for _ in 1..nesting {
                    walk = walk.wrapping_sub(frame);
                    let value = mem_read(walk, size)? & size.mask();
                    rsp = rsp.wrapping_sub(frame);
                    mem_write(rsp, size, value)?;
                }
                rsp = rsp.wrapping_sub(frame);
                mem_write(rsp, size, frame_temp)?;
            }
            write_reg64(RBP as i32, frame_temp);
            write_reg64(RSP as i32, rsp.wrapping_sub(alloc));
        },

        // LEAVE: mov rsp, rbp; pop rbp
        0xC9 => {
            write_reg64(RSP as i32, read_reg64(RBP as i32));
            let value = pop64()?;
            write_reg64(RBP as i32, value);
        },

        // XLAT (0xD7); cf. Intel SDM Vol. 2:
// https://www.felixcloutier.com/x86/xlat
        0xD7 => {
            let mut base = read_reg64((RBX | (pfx.rex & prefix::REX_B) << 3) as i32);
            if pfx.addrsize_32 {
                base &= 0xFFFF_FFFF;
            }
            let offset = read_reg(RAX, OpSize::S8, false);
            let value = mem_read(base.wrapping_add(offset), OpSize::S8)?;
            write_reg(RAX, OpSize::S8, false, value);
        },

        // x87 (D8-DF). The register forms reuse the 32-bit implementations;
        // the memory forms are done here so addresses are not truncated.
        0xD8..=0xDF => {
            x87(&pfx, opcode)?;
        },

        // IRETQ (0xCF)
        0xCF => {
            let new_rip = pop64()?;
            let cs = pop64()?;
            let rflags = pop64()?;
            let rsp = pop64()?;
            let ss = pop64()?;
            *sreg.offset(1) = cs as u16; // CS
            *sreg.offset(2) = ss as u16; // SS
            *cpl = cs as u8 & 3;
            *flags = rflags as u32 as i32;
            write_reg64(RSP as i32, rsp);
            *rip = new_rip;
        },

        // INT3 (0xCC) / INT imm8 (0xCD)
        0xCC => crate::cpu::core::call_interrupt_vector64(3, None),
        0xCD => {
            let vector = fetch8()? as i32;
            crate::cpu::core::call_interrupt_vector64(vector, None);
        },

        // CLI (0xFA) / STI (0xFB)
        0xFA => crate::cpu::interp::instructions::instr_FA(),
        0xFB => {
            let was_enabled = *flags & FLAG_INTERRUPT != 0;
            if !crate::cpu::interp::instructions::instr_FB_without_fault() {
                crate::cpu::core::trigger_gp(0);
            }
            else if !was_enabled {
                crate::cpu::core::set_sti_shadow();
            }
        },

        // CMC (0xF5) / CLC (0xF8) / STC (0xF9) / CLD (0xFC) / STD (0xFD)
        0xF5 => *flags ^= FLAG_CF as i32,
        0xF8 => *flags &= !(FLAG_CF as i32),
        0xF9 => *flags |= FLAG_CF as i32,
        0xFC => *flags &= !(1 << 10), // FLAG_DIRECTION
        0xFD => *flags |= 1 << 10,

        // MOV r/m16, Sreg (0x8C)
        0x8C => {
            let (modrm, operand) = decode_operand(&pfx)?;
            let value = *sreg.offset((modrm.reg & 7) as isize) as u64;
            write_operand(operand, OpSize::S16, pfx.has_rex(), value)?;
        },

        // MOV Sreg, r/m16 (0x8E)
        0x8E => {
            let (modrm, operand) = decode_operand(&pfx)?;
            let seg = modrm.reg & 7;
            if seg == 1 || seg >= 6 {
                // CS cannot be loaded directly; LDTR/TR use LLDT/LTR
                crate::cpu::core::trigger_ud();
                return Ok(());
            }
            let value = read_operand(operand, OpSize::S16, pfx.has_rex())? as u16;
            *sreg.offset(seg as isize) = value;
        },

        // CBW/CWDE/CDQE (0x98)
        0x98 => {
            let value = match size {
                OpSize::S16 => read_reg(RAX, OpSize::S8, false) as u8 as i8 as i64 as u64,
                OpSize::S32 => read_reg(RAX, OpSize::S16, false) as u16 as i16 as i64 as u64,
                OpSize::S64 => read_reg(RAX, OpSize::S32, false) as u32 as i32 as i64 as u64,
                OpSize::S8 => 0,
            };
            write_reg(RAX, size, false, value);
        },

        // CWD/CDQ/CQO (0x99): DX/EDX/RDX = sign of AX/EAX/RAX
        0x99 => {
            let value = read_reg(RAX, size, false);
            let extended = if value & size.sign_bit() != 0 {
                size.mask()
            }
            else {
                0
            };
            match size {
                OpSize::S16 => write_reg(RDX, OpSize::S16, false, extended),
                OpSize::S32 => write_reg(RDX, OpSize::S32, false, extended),
                OpSize::S64 => write_reg(RDX, OpSize::S64, false, extended),
                OpSize::S8 => {},
            }
        },

        // PUSHFQ (0x9C) / POPFQ (0x9D)
        0x9C => {
            let value = *flags as u32 as u64;
            push64(value)?;
        },
        0x9D => {
            let value = pop64()?;
            *flags = value as u32 as i32;
            *flags_changed = 0;
        },

        // FWAIT (0x9B): no pending x87 exceptions are modelled
        0x9B => {},

        // SAHF (0x9E) / LAHF (0x9F): AH <-> SF ZF 0 AF 0 PF 1 CF.
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/lahf
        0x9E => {
            let mask = FLAG_CF | FLAG_PF | FLAG_AF | FLAG_ZF | FLAG_SF;
            let ah = (read_reg64(RAX as i32) >> 8) as u32 & 0xFF;
            *flags = ((*flags as u32 & !mask) | (ah & mask)) as i32;
            *flags_changed = 0;
        },
        0x9F => {
            let mask = FLAG_CF | FLAG_PF | FLAG_AF | FLAG_ZF | FLAG_SF;
            let ah = (*flags as u32 & mask) | 0x02; // bit 1 reads as 1
            let rax = read_reg64(RAX as i32);
            write_reg64(RAX as i32, (rax & !0xFF00) | (ah as u64) << 8);
        },

        // MOV AL/eAX, moffs (0xA0/A1), MOV moffs, AL/eAX (0xA2/A3)
        0xA0 | 0xA1 | 0xA2 | 0xA3 => {
            let address = if pfx.addrsize_32 { fetch32()? as u64 } else { fetch64()? };
            let acc_size = if opcode & 1 == 0 { OpSize::S8 } else { size };
            if opcode & 2 == 0 {
                let value = mem_read(address, acc_size)? & acc_size.mask();
                write_reg(RAX, acc_size, false, value);
            }
            else {
                let value = read_reg(RAX, acc_size, false);
                mem_write(address, acc_size, value)?;
            }
        },

        _ => {
            dbg_log!("#ud interp64: opcode {:02x}", opcode);
            crate::cpu::core::trigger_ud();
        },
    }

    Ok(())
}

