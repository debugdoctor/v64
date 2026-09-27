//! Hand-written x86-64 long mode interpreter, independent of `gen/interpreter*.rs`.
//! Operand size: 32, 64 with REX.W, 16 with 0x66. Address size: 64, 32 with 0x67.
//! Virtual addresses are 64-bit; physical addresses stay below 4 GiB (wasm32).
//! http://www.sandpile.org/x86/opra.htm

#![allow(dead_code)]

use crate::cpu::cpu::{
    io_port_read8, io_port_write8, read_reg64, read_tsc, reg128, test_privileges_for_io, translate_address_64,
    write_reg64, APIC_MEM_ADDRESS, CR4_TSD, CS, FLAG_CARRY, FLAG_INTERRUPT, SS,
};
use crate::cpu::global_pointers::*;
use crate::cpu::fpu::{
    f32_to_f80, f64_to_f80, f80_to_f32, f80_to_f64, fpu_convert_to_i16, fpu_convert_to_i32,
    fpu_convert_to_i64, fpu_fadd, fpu_fcom, fpu_fcomp, fpu_fdiv, fpu_fdivr, fpu_fmul, fpu_pop,
    fpu_push, fpu_fsub, fpu_fsubr, fpu_get_st0, fpu_truncate_to_i16, fpu_truncate_to_i32,
    fpu_finit, fpu_load_status_word, fpu_load_tag_word, fpu_set_status_word, fpu_set_tag_word,
    fpu_truncate_to_i64, i32_to_f80, i64_to_f80, set_control_word,
};
use crate::softfloat::F80;
use crate::cpu::memory;
use crate::paging::OrPageFault;
use crate::prefix;

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
const MSR_GS_BASE: i32 = 0xC0000101u32 as i32;
const MSR_KERNEL_GS_BASE: i32 = 0xC0000102u32 as i32;

const EFER_SCE: u64 = 1;
const EFER_LME: u64 = 1 << 8;
const EFER_LMA: u64 = 1 << 10;

// RFLAGS bits
const FLAG_CF: u32 = 1;
const FLAG_PF: u32 = 1 << 2;
const FLAG_AF: u32 = 1 << 4;
const FLAG_ZF: u32 = 1 << 6;
const FLAG_SF: u32 = 1 << 7;
const FLAG_OF: u32 = 1 << 11;

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum OpSize {
    S8,
    S16,
    S32,
    S64,
}

impl OpSize {
    fn bits(self) -> u32 {
        match self {
            OpSize::S8 => 8,
            OpSize::S16 => 16,
            OpSize::S32 => 32,
            OpSize::S64 => 64,
        }
    }

    fn mask(self) -> u64 {
        match self {
            OpSize::S8 => 0xFF,
            OpSize::S16 => 0xFFFF,
            OpSize::S32 => 0xFFFF_FFFF,
            OpSize::S64 => u64::MAX,
        }
    }

    fn sign_bit(self) -> u64 {
        match self {
            OpSize::S8 => 0x80,
            OpSize::S16 => 0x8000,
            OpSize::S32 => 0x8000_0000,
            OpSize::S64 => 0x8000_0000_0000_0000,
        }
    }
}

#[derive(Copy, Clone, Default)]
struct Prefixes {
    opsize_16: bool,
    addrsize_32: bool,
    segment: u8, // 0 = none, else a segment register index (FS/GS)
    rex: u8,
    f3: bool, // REP/SSE/FSGSBASE prefix
    f2: bool, // SSE prefix
    p66: bool, // 0x66 prefix (SSE2 for 0F opcodes)
}

impl Prefixes {
    fn has_rex(&self) -> bool { self.rex != 0 }

    fn operand_size(&self) -> OpSize {
        if self.rex & prefix::REX_W != 0 {
            OpSize::S64
        }
        else if self.opsize_16 {
            OpSize::S16
        }
        else {
            OpSize::S32
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Modrm {
    mod_bits: u8,
    reg: u8,
    rm: u8,
}

unsafe fn fetch8() -> OrPageFault<u8> {
    let phys = translate_address_64(*rip, false, *cpl == 3)?;
    let value = memory::read8(phys) as u8;
    *rip = (*rip).wrapping_add(1);
    Ok(value)
}

unsafe fn fetch16() -> OrPageFault<u16> {
    let phys = translate_address_64(*rip, false, *cpl == 3)?;
    let value = memory::read16(phys) as u16;
    *rip = (*rip).wrapping_add(2);
    Ok(value)
}

unsafe fn fetch32() -> OrPageFault<u32> {
    let phys = translate_address_64(*rip, false, *cpl == 3)?;
    let value = memory::read32s(phys) as u32;
    *rip = (*rip).wrapping_add(4);
    Ok(value)
}

unsafe fn fetch64() -> OrPageFault<u64> {
    let low = fetch32()? as u64;
    let high = fetch32()? as u64;
    Ok(low | (high << 32))
}

unsafe fn decode_prefixes() -> OrPageFault<(Prefixes, u8)> {
    let mut pfx = Prefixes::default();

    loop {
        let byte = fetch8()?;
        match byte {
            0x66 => { pfx.opsize_16 = true; pfx.p66 = true; },
            0x67 => pfx.addrsize_32 = true,
            0xF0 => {}, // LOCK: ignored
            0xF2 => pfx.f2 = true,
            0xF3 => pfx.f3 = true,
            0x26 | 0x2E | 0x36 | 0x3E => {}, // null prefix in long mode
            0x64 => pfx.segment = 4, // FS
            0x65 => pfx.segment = 5, // GS
            0x40..=0x4F => pfx.rex = byte,
            opcode => return Ok((pfx, opcode)),
        }
    }
}

unsafe fn decode_modrm(pfx: &Prefixes) -> OrPageFault<Modrm> {
    let byte = fetch8()?;
    Ok(Modrm {
        mod_bits: byte >> 6,
        // REX.R/REX.B extend the 3-bit field by 8
        reg: (byte >> 3 & 7) | (pfx.rex & prefix::REX_R) << 1,
        rm: (byte & 7) | (pfx.rex & prefix::REX_B) << 3,
    })
}

// mod != 3
unsafe fn modrm_effective_address(modrm: &Modrm, pfx: &Prefixes) -> OrPageFault<u64> {
    dbg_assert!(modrm.mod_bits != 3);

    let mut address: u64;
    let rm_low = modrm.rm & 7;

    if rm_low == 4 {
        // SIB byte
        let sib = fetch8()?;
        let scale = 1u64 << (sib >> 6);
        let index_low = sib >> 3 & 7;
        let base_low = sib & 7;

        // index == 4 with no REX.X means "no index"
        let index_value = if index_low == 4 && pfx.rex & prefix::REX_X == 0 {
            0
        }
        else {
            let index = (index_low | (pfx.rex & prefix::REX_X) << 2) as i32;
            read_reg64(index)
        };

        if base_low == 5 && modrm.mod_bits == 0 {
            // no base, disp32
            address = fetch32()? as i32 as i64 as u64;
        }
        else {
            let base = (base_low | (pfx.rex & prefix::REX_B) << 3) as i32;
            address = read_reg64(base);
        }
        address = address.wrapping_add(index_value.wrapping_mul(scale));
    }
    else if rm_low == 5 && modrm.mod_bits == 0 {
        // RIP-relative
        let displacement = fetch32()? as i32 as i64 as u64;
        address = (*rip).wrapping_add(displacement);
    }
    else {
        address = read_reg64(modrm.rm as i32);
    }

    match modrm.mod_bits {
        0 => {},
        1 => address = address.wrapping_add(fetch8()? as i8 as i64 as u64),
        2 => address = address.wrapping_add(fetch32()? as i32 as i64 as u64),
        _ => unreachable!(),
    }

    if pfx.segment == 4 {
        address = address.wrapping_add(*fs_base);
    }
    else if pfx.segment == 5 {
        address = address.wrapping_add(*gs_base);
    }

    Ok(address)
}

unsafe fn set_cpl_segments(cs: u16, new_cpl: u8) {
    // long mode: code/data bases are 0, SS = CS + 8
    *sreg.offset(CS as isize) = cs;
    *sreg.offset(SS as isize) = cs.wrapping_add(8);
    *segment_offsets.offset(CS as isize) = 0;
    *segment_offsets.offset(SS as isize) = 0;
    *segment_is_null.offset(CS as isize) = false;
    *segment_is_null.offset(SS as isize) = false;
    *cpl = new_cpl;
}

unsafe fn msr_read(index: i32) -> Option<u64> {
    match index {
        MSR_EFER => Some(*efer),
        MSR_STAR => Some(*star),
        MSR_LSTAR => Some(*lstar),
        MSR_SFMASK => Some(*sfmask),
        MSR_FS_BASE => Some(*fs_base),
        MSR_GS_BASE => Some(*gs_base),
        MSR_KERNEL_GS_BASE => Some(*kernel_gs_base),
        // IA32_APIC_BASE: the base is fixed, only the enable bit changes
        0x1B => Some(APIC_MEM_ADDRESS as u64 | if *apic_enabled { 0x800 } else { 0 }),
        _ => None,
    }
}

unsafe fn msr_write(index: i32, value: u64) -> bool {
    match index {
        MSR_EFER => *efer = value | EFER_LMA,
        MSR_STAR => *star = value,
        MSR_LSTAR => *lstar = value,
        MSR_SFMASK => *sfmask = value,
        MSR_FS_BASE => *fs_base = value,
        MSR_GS_BASE => *gs_base = value,
        MSR_KERNEL_GS_BASE => *kernel_gs_base = value,
        _ => return false,
    }
    true
}

unsafe fn set_zf(value: bool) {
    let mut f = *flags as u32;
    f &= !FLAG_ZF;
    if value {
        f |= FLAG_ZF;
    }
    *flags = f as i32;
    *flags_changed = 0;
}

// Load TR from its GDT descriptor so the IST path can find the TSS.
unsafe fn load_tss(selector: u16) {
    let index = (selector >> 3) as u32;
    let base = (*gdtr_offset as u32).wrapping_add(index * 8);
    let low = memory::read32s(base) as u32;
    let high = memory::read32s(base + 4) as u32;
    let seg_base = (low >> 16 & 0xFFFF) | (high & 0xFF) << 16 | (high >> 24 & 0xFF) << 24;
    let limit = (low & 0xFFFF) | (high >> 16 & 0xF) << 16;
    *segment_offsets.offset(6) = seg_base as i32;
    *segment_limits.offset(6) = limit;
    *tss_size_32 = false;
}

unsafe fn selector_valid(selector: u16) -> bool {
    if selector & !7 == 0 {
        return false;
    }
    let index = (selector >> 3) as u32;
    index * 8 + 7 <= *gdtr_size as u32
}

unsafe fn read_reg8(r: u8, has_rex: bool) -> u64 {
    if !has_rex && r >= 4 && r < 8 {
        // AH/CH/DH/BH
        read_reg64((r - 4) as i32) >> 8 & 0xFF
    }
    else {
        read_reg64(r as i32) & 0xFF
    }
}

unsafe fn write_reg8(r: u8, has_rex: bool, value: u64) {
    if !has_rex && r >= 4 && r < 8 {
        // AH/CH/DH/BH: high byte of r0-r3
        let old = read_reg64((r - 4) as i32);
        write_reg64((r - 4) as i32, old & !0xFF00 | (value & 0xFF) << 8);
    }
    else {
        let old = read_reg64(r as i32);
        write_reg64(r as i32, old & !0xFF | value & 0xFF);
    }
}

unsafe fn read_reg(r: u8, size: OpSize, has_rex: bool) -> u64 {
    match size {
        OpSize::S8 => read_reg8(r, has_rex),
        OpSize::S16 => read_reg64(r as i32) & 0xFFFF,
        OpSize::S32 => read_reg64(r as i32) & 0xFFFF_FFFF,
        OpSize::S64 => read_reg64(r as i32),
    }
}

unsafe fn write_reg(r: u8, size: OpSize, has_rex: bool, value: u64) {
    match size {
        OpSize::S8 => write_reg8(r, has_rex, value),
        // 32-bit writes zero-extend
        OpSize::S32 => write_reg64(r as i32, value & 0xFFFF_FFFF),
        // 16-bit writes preserve the upper bits
        OpSize::S16 => {
            let old = read_reg64(r as i32);
            write_reg64(r as i32, old & !0xFFFF | value & 0xFFFF);
        },
        OpSize::S64 => write_reg64(r as i32, value),
    }
}

unsafe fn mem_read(address: u64, size: OpSize) -> OrPageFault<u64> {
    let bytes = (size.bits() / 8) as usize;
    let mut physical = [0u32; 8];
    for offset in 0..bytes {
        physical[offset] = translate_address_64(address + offset as u64, false, *cpl == 3)?;
    }
    if (address as usize & 0xFFF) + bytes <= 0x1000 {
        return Ok(match size {
            OpSize::S8 => memory::read8(physical[0]) as u8 as u64,
            OpSize::S16 => memory::read16(physical[0]) as u16 as u64,
            OpSize::S32 => memory::read32s(physical[0]) as u32 as u64,
            OpSize::S64 => {
                memory::read32s(physical[0]) as u32 as u64
                    | (memory::read32s(physical[0].wrapping_add(4)) as u32 as u64) << 32
            },
        });
    }
    let mut value = 0;
    for offset in 0..bytes {
        value |= (memory::read8(physical[offset]) as u8 as u64) << (offset * 8);
    }
    Ok(value)
}

unsafe fn mem_write(address: u64, size: OpSize, value: u64) -> OrPageFault<()> {
    let bytes = (size.bits() / 8) as usize;
    let mut physical = [0u32; 8];
    for offset in 0..bytes {
        physical[offset] = translate_address_64(address + offset as u64, true, *cpl == 3)?;
    }
    if (address as usize & 0xFFF) + bytes <= 0x1000 {
        match size {
            OpSize::S8 => memory::write8(physical[0], value as i32),
            OpSize::S16 => memory::write16(physical[0], value as i32),
            OpSize::S32 => memory::write32(physical[0], value as i32),
            OpSize::S64 => {
                memory::write32(physical[0], value as i32);
                memory::write32(physical[0] + 4, (value >> 32) as i32);
            },
        }
        return Ok(());
    }
    for offset in 0..bytes {
        memory::write8(physical[offset], (value >> (offset * 8)) as u8 as i32);
    }
    Ok(())
}

unsafe fn mem_probe_write(address: u64, size: OpSize) -> OrPageFault<()> {
    for offset in 0..size.bits() / 8 {
        translate_address_64(address + offset as u64, true, *cpl == 3)?;
    }
    Ok(())
}

#[derive(Copy, Clone)]
enum Operand {
    Reg(u8),
    Mem(u64),
}

unsafe fn probe_operand_write(operand: Operand, size: OpSize) -> OrPageFault<()> {
    if let Operand::Mem(address) = operand {
        mem_probe_write(address, size)?;
    }
    Ok(())
}

unsafe fn decode_operand_after_modrm(
    pfx: &Prefixes,
    modrm: Modrm,
    trailing_bytes: u64,
) -> OrPageFault<(Modrm, Operand)> {
    let operand = if modrm.mod_bits == 3 {
        Operand::Reg(modrm.rm)
    }
    else {
        let mut address = modrm_effective_address(&modrm, pfx)?;
        if modrm.mod_bits == 0 && modrm.rm & 7 == 5 {
            address = address.wrapping_add(trailing_bytes);
        }
        Operand::Mem(address)
    };
    Ok((modrm, operand))
}

unsafe fn decode_operand_with_trailing(
    pfx: &Prefixes,
    trailing_bytes: u64,
) -> OrPageFault<(Modrm, Operand)> {
    let modrm = decode_modrm(pfx)?;
    decode_operand_after_modrm(pfx, modrm, trailing_bytes)
}

unsafe fn decode_operand(pfx: &Prefixes) -> OrPageFault<(Modrm, Operand)> {
    decode_operand_with_trailing(pfx, 0)
}

unsafe fn read_operand(operand: Operand, size: OpSize, has_rex: bool) -> OrPageFault<u64> {
    match operand {
        Operand::Reg(r) => Ok(read_reg(r, size, has_rex)),
        Operand::Mem(address) => mem_read(address, size),
    }
}

unsafe fn write_operand(
    operand: Operand,
    size: OpSize,
    has_rex: bool,
    value: u64,
) -> OrPageFault<()> {
    match operand {
        Operand::Reg(r) => {
            write_reg(r, size, has_rex, value);
            Ok(())
        },
        Operand::Mem(address) => mem_write(address, size, value),
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum AluOp {
    Add,
    Adc,
    Or,
    And,
    Sub,
    Sbb,
    Xor,
    Cmp,
}

// ---- SSE / SSE2 ----

unsafe fn xmm_get(r: u8) -> reg128 { *reg_xmm.offset(r as isize) }
unsafe fn xmm_set(r: u8, value: reg128) { *reg_xmm.offset(r as isize) = value; }

unsafe fn mem_read128(address: u64) -> OrPageFault<reg128> {
    let low = mem_read(address, OpSize::S64)?;
    let high = mem_read(address + 8, OpSize::S64)?;
    Ok(reg128 { u64: [low, high] })
}

unsafe fn mem_write128(address: u64, value: reg128) -> OrPageFault<()> {
    mem_write(address, OpSize::S64, value.u64[0])?;
    mem_write(address + 8, OpSize::S64, value.u64[1])?;
    Ok(())
}

unsafe fn sse_read(operand: Operand) -> OrPageFault<reg128> {
    match operand {
        Operand::Reg(r) => Ok(xmm_get(r)),
        Operand::Mem(address) => mem_read128(address),
    }
}

// Returns true if `opcode` was an SSE instruction.
unsafe fn run_sse(opcode: u8, pfx: &Prefixes) -> OrPageFault<bool> {
    let size = pfx.operand_size();
    match opcode {
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
                        crate::cpu::cpu::trigger_ud();
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
                crate::cpu::cpu::trigger_ud();
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
                        crate::cpu::cpu::trigger_ud();
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
                crate::cpu::cpu::trigger_ud();
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
            crate::cpu::cpu::transition_fpu_to_mmx();
            crate::cpu::cpu::write_mmx_reg64(modrm.reg as i32, value);
        },

        // MMX MOVD r/m32, mm (0F 7E, no 66)
        0x7E if !pfx.p66 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = crate::cpu::cpu::read_mmx64s(modrm.reg as i32);
            write_operand(operand, OpSize::S32, pfx.has_rex(), value)?;
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

        // MOVMSKPS r32, xmm (0F 50)
        0x50 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let value = sse_read(operand)?;
            let mut mask = 0u64;
            for i in 0..4 {
                if value.u32[i] & 0x8000_0000 != 0 {
                    mask |= 1 << i;
                }
            }
            write_reg(modrm.reg, OpSize::S32, pfx.has_rex(), mask);
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
        0x2C if pfx.f3 => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = sse_read(operand)?;
            let value = src.f32[0] as i64 as u64;
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

        // Group 14 shifts: PSRD (0x72 /2), PSLD (0x72 /6)
        0x72 if pfx.p66 => {
            let (modrm, operand) = decode_operand_with_trailing(pfx, 1)?;
            let count = fetch8()? as u32 & 31;
            let value = sse_read(operand)?;
            let mut result = value;
            match modrm.reg & 7 {
                2 => {
                    for i in 0..4 { result.u32[i] >>= count; }
                },
                6 => {
                    for i in 0..4 { result.u32[i] = result.u32[i].wrapping_shl(count); }
                },
                _ => {
                    crate::cpu::cpu::trigger_ud();
                    return Ok(true);
                },
            }
            match operand {
                Operand::Reg(r) => xmm_set(r, result),
                Operand::Mem(address) => mem_write128(address, result)?,
            }
        },

        _ => return Ok(false),
    }
    Ok(true)
}

fn alu_op_of(opcode: u8) -> AluOp {
    match opcode & 0x38 {
        0x00 => AluOp::Add,
        0x08 => AluOp::Or,
        0x10 => AluOp::Adc,
        0x18 => AluOp::Sbb,
        0x20 => AluOp::And,
        0x28 => AluOp::Sub,
        0x30 => AluOp::Xor,
        0x38 => AluOp::Cmp,
        _ => unreachable!(),
    }
}

fn parity_even(value: u64) -> bool { (value as u8).count_ones() % 2 == 0 }

unsafe fn set_szp_flags(result: u64, size: OpSize) {
    let mut f = *flags as u32;
    f &= !(FLAG_ZF | FLAG_SF | FLAG_PF);

    if result & size.mask() == 0 {
        f |= FLAG_ZF;
    }
    if result & size.sign_bit() != 0 {
        f |= FLAG_SF;
    }
    if parity_even(result) {
        f |= FLAG_PF;
    }
    *flags = f as i32;
    *flags_changed = 0;
}

unsafe fn set_add_flags(result: u64, dst: u64, src: u64, size: OpSize) {
    set_szp_flags(result, size);
    let mut f = *flags as u32;
    f &= !(FLAG_CF | FLAG_OF | FLAG_AF);

    if (dst as u128 + src as u128) >> size.bits() & 1 != 0 {
        f |= FLAG_CF;
    }
    if (dst ^ src ^ result) & 0x10 != 0 {
        f |= FLAG_AF;
    }
    if (dst ^ result) & (src ^ result) & size.sign_bit() != 0 {
        f |= FLAG_OF;
    }
    *flags = f as i32;
    *flags_changed = 0;
}

unsafe fn set_sub_flags(result: u64, dst: u64, src: u64, size: OpSize) {
    set_szp_flags(result, size);
    let mut f = *flags as u32;
    f &= !(FLAG_CF | FLAG_OF | FLAG_AF);

    if dst & size.mask() < src & size.mask() {
        f |= FLAG_CF;
    }
    if (dst ^ src ^ result) & 0x10 != 0 {
        f |= FLAG_AF;
    }
    if (dst ^ src) & (dst ^ result) & size.sign_bit() != 0 {
        f |= FLAG_OF;
    }
    *flags = f as i32;
    *flags_changed = 0;
}

unsafe fn set_logic_flags(result: u64, size: OpSize) {
    set_szp_flags(result, size);
    let mut f = *flags as u32;
    f &= !(FLAG_CF | FLAG_OF);
    *flags = f as i32;
    *flags_changed = 0;
}

unsafe fn alu(op: AluOp, dst: u64, src: u64, size: OpSize) -> u64 {
    let mask = size.mask();
    match op {
        AluOp::Add => {
            let result = dst.wrapping_add(src) & mask;
            set_add_flags(result, dst, src, size);
            result
        },
        AluOp::Adc | AluOp::Sbb => adc_sbb_value(dst, src, size, op == AluOp::Sbb),
        AluOp::Sub | AluOp::Cmp => {
            let result = dst.wrapping_sub(src) & mask;
            set_sub_flags(result, dst, src, size);
            result
        },
        AluOp::And => {
            let result = dst & src & mask;
            set_logic_flags(result, size);
            result
        },
        AluOp::Or => {
            let result = (dst | src) & mask;
            set_logic_flags(result, size);
            result
        },
        AluOp::Xor => {
            let result = (dst ^ src) & mask;
            set_logic_flags(result, size);
            result
        },
    }
}

fn sign_extend(value: u64, size: OpSize) -> i128 {
    match size {
        OpSize::S8 => value as u8 as i8 as i128,
        OpSize::S16 => value as u16 as i16 as i128,
        OpSize::S32 => value as u32 as i32 as i128,
        OpSize::S64 => value as i64 as i128,
    }
}

unsafe fn adc_sbb_value(dst: u64, src: u64, size: OpSize, subtract: bool) -> u64 {
    let mask = size.mask();
    let dst = dst & mask;
    let src = src & mask;
    let signed_dst = sign_extend(dst, size);
    let signed_src = sign_extend(src, size);
    let carry = (*flags as u32 & FLAG_CF) as u64;
    let (result, carry_out, overflow, adjust) = if subtract {
        let rhs = src as u128 + carry as u128;
        let result = dst.wrapping_sub(src).wrapping_sub(carry) & mask;
        let signed = signed_dst - signed_src - carry as i128;
        let min = -(1i128 << (size.bits() - 1));
        let max = (1i128 << (size.bits() - 1)) - 1;
        (result, (dst as u128) < rhs, signed < min || signed > max, (dst & 0xF) < (src & 0xF) + carry)
    }
    else {
        let sum = dst as u128 + src as u128 + carry as u128;
        let result = sum as u64 & mask;
        let signed = signed_dst + signed_src + carry as i128;
        let min = -(1i128 << (size.bits() - 1));
        let max = (1i128 << (size.bits() - 1)) - 1;
        (result, sum > mask as u128, signed < min || signed > max, (dst & 0xF) + (src & 0xF) + carry > 0xF)
    };
    set_szp_flags(result, size);
    let mut f = *flags as u32 & !(FLAG_CF | FLAG_AF | FLAG_OF);
    if carry_out { f |= FLAG_CF; }
    if adjust { f |= FLAG_AF; }
    if overflow { f |= FLAG_OF; }
    *flags = f as i32;
    result
}

unsafe fn imul_value(lhs: u64, rhs: u64, size: OpSize) -> u64 {
    let product = sign_extend(lhs, size) * sign_extend(rhs, size);
    let result = product as u64 & size.mask();
    let extended = sign_extend(result, size);
    *flags &= !(FLAG_CF | FLAG_OF) as i32;
    if product != extended {
        *flags |= (FLAG_CF | FLAG_OF) as i32;
    }
    result
}

// accumulator (AL/AX/EAX/RAX) with an immediate
fn sext(value: u64, size: OpSize) -> i128 {
    match size {
        OpSize::S8 => value as u8 as i8 as i128,
        OpSize::S16 => value as u16 as i16 as i128,
        OpSize::S32 => value as u32 as i32 as i128,
        OpSize::S64 => value as i64 as i128,
    }
}

unsafe fn write_quot_rem(quotient: u64, remainder: u64, size: OpSize) {
    match size {
        OpSize::S8 => {
            write_reg8(0, false, quotient & 0xFF);
            write_reg8(4, false, remainder & 0xFF);
        },
        OpSize::S16 => {
            write_reg(RAX, OpSize::S16, false, quotient);
            write_reg(RDX, OpSize::S16, false, remainder);
        },
        OpSize::S32 => {
            write_reg(RAX, OpSize::S32, false, quotient);
            write_reg(RDX, OpSize::S32, false, remainder);
        },
        OpSize::S64 => {
            write_reg(RAX, OpSize::S64, false, quotient);
            write_reg(RDX, OpSize::S64, false, remainder);
        },
    }
}

// One-operand MUL/IMUL: the widening product goes to AX/DX:AX/EDX:EAX/RDX:RAX.
unsafe fn mul_wide(acc: u64, src: u64, size: OpSize, signed: bool) {
    let (product, overflow) = if signed {
        let sp = sext(acc, size) * sext(src, size);
        let bits = sp as u128;
        let low = bits as u64 & size.mask();
        (bits, sp != sext(low, size))
    }
    else {
        let up = (acc as u128) * (src as u128);
        (up, (up >> size.bits()) != 0)
    };

    match size {
        OpSize::S8 => write_reg(RAX, OpSize::S16, false, product as u16 as u64),
        OpSize::S16 => {
            write_reg(RAX, OpSize::S16, false, product as u16 as u64);
            write_reg(RDX, OpSize::S16, false, (product >> 16) as u16 as u64);
        },
        OpSize::S32 => {
            write_reg(RAX, OpSize::S32, false, product as u32 as u64);
            write_reg(RDX, OpSize::S32, false, (product >> 32) as u32 as u64);
        },
        OpSize::S64 => {
            write_reg(RAX, OpSize::S64, false, product as u64);
            write_reg(RDX, OpSize::S64, false, (product >> 64) as u64);
        },
    }

    *flags &= !(FLAG_CF | FLAG_OF) as i32;
    if overflow {
        *flags |= (FLAG_CF | FLAG_OF) as i32;
    }
    *flags_changed = 0;
}

// One-operand DIV/IDIV. #DE on a zero divisor or a quotient that does not fit.
unsafe fn div_wide(src: u64, size: OpSize, signed: bool) {
    let divisor = src & size.mask();
    if divisor == 0 {
        crate::cpu::cpu::trigger_de();
        return;
    }

    if signed
    {
        let dividend = match size {
            OpSize::S8 => read_reg(RAX, OpSize::S16, false) as u16 as i16 as i128,
            OpSize::S16 => {
                (((read_reg(RDX, OpSize::S16, false) as u16 as u32) << 16
                    | read_reg(RAX, OpSize::S16, false) as u16 as u32) as u32 as i32) as i128
            },
            OpSize::S32 => {
                (((read_reg(RDX, OpSize::S32, false) as u32 as u64) << 32
                    | read_reg(RAX, OpSize::S32, false) as u32 as u64) as u64 as i64) as i128
            },
            OpSize::S64 => {
                ((read_reg64(RDX as i32) as u128) << 64 | read_reg64(RAX as i32) as u128) as i128
            },
        };
        let signed_divisor = sext(divisor, size);
        let quotient = dividend / signed_divisor;
        let remainder = dividend % signed_divisor;
        let (min, max) = match size {
            OpSize::S8 => (-128i128, 127i128),
            OpSize::S16 => (-32768, 32767),
            OpSize::S32 => (-2147483648, 2147483647),
            OpSize::S64 => (i64::MIN as i128, i64::MAX as i128),
        };
        if quotient < min || quotient > max {
            crate::cpu::cpu::trigger_de();
            return;
        }
        write_quot_rem(quotient as u64, remainder as u64, size);
    }
    else
    {
        let dividend = match size {
            OpSize::S8 => read_reg(RAX, OpSize::S16, false) as u128,
            OpSize::S16 => {
                ((read_reg(RDX, OpSize::S16, false) as u16 as u32) << 16
                    | read_reg(RAX, OpSize::S16, false) as u16 as u32) as u128
            },
            OpSize::S32 => {
                ((read_reg(RDX, OpSize::S32, false) as u32 as u64) << 32
                    | read_reg(RAX, OpSize::S32, false) as u32 as u64) as u128
            },
            OpSize::S64 => {
                (read_reg64(RDX as i32) as u128) << 64 | read_reg64(RAX as i32) as u128
            },
        };
        let quotient = dividend / divisor as u128;
        let remainder = dividend % divisor as u128;
        if quotient > size.mask() as u128 {
            crate::cpu::cpu::trigger_de();
            return;
        }
        write_quot_rem(quotient as u64, remainder as u64, size);
    }
}

// F6/F7 group 3: TEST/NOT/NEG/MUL/IMUL/DIV/IDIV.
unsafe fn group3(opcode: u8, pfx: &Prefixes, operand_size: OpSize) -> OrPageFault<()> {
    let is8 = opcode == 0xF6;
    let size = if is8 { OpSize::S8 } else { operand_size };
    let has_rex = pfx.has_rex();
    let modrm = decode_modrm(pfx)?;
    let trailing = if modrm.reg & 7 == 0 {
        if size == OpSize::S8 {
            1
        }
        else if size == OpSize::S16 {
            2
        }
        else {
            4
        }
    }
    else {
        0
    };
    let (modrm, operand) = decode_operand_after_modrm(pfx, modrm, trailing)?;

    match modrm.reg & 7 {
        0 => {
            let value = read_operand(operand, size, has_rex)?;
            let immediate = fetch_imm(size)?;
            set_logic_flags(value & immediate & size.mask(), size);
        },
        2 => {
            let value = read_operand(operand, size, has_rex)?;
            probe_operand_write(operand, size)?;
            write_operand(operand, size, has_rex, !value & size.mask())?;
        },
        3 => {
            let value = read_operand(operand, size, has_rex)?;
            probe_operand_write(operand, size)?;
            let result = alu(AluOp::Sub, 0, value, size);
            write_operand(operand, size, has_rex, result)?;
        },
        4 | 5 => {
            let src = read_operand(operand, size, has_rex)? & size.mask();
            let acc = read_reg(RAX, size, false) & size.mask();
            mul_wide(acc, src, size, modrm.reg & 7 == 5);
        },
        6 | 7 => {
            let src = read_operand(operand, size, has_rex)? & size.mask();
            div_wide(src, size, modrm.reg & 7 == 7);
        },
        _ => crate::cpu::cpu::trigger_ud(),
    }
    Ok(())
}

// C0/C1/D0/D1/D2/D3 group 2: ROL/ROR/RCL/RCR/SHL/SHR/SAR.
unsafe fn group2(opcode: u8, pfx: &Prefixes, operand_size: OpSize) -> OrPageFault<()> {
    let is8 = opcode == 0xC0 || opcode == 0xD0 || opcode == 0xD2;
    let size = if is8 { OpSize::S8 } else { operand_size };
    let has_rex = pfx.has_rex();
    let trailing = if opcode == 0xC0 || opcode == 0xC1 { 1 } else { 0 };
    let (modrm, operand) = decode_operand_with_trailing(pfx, trailing)?;
    let count = match opcode {
        0xD0 | 0xD1 => 1,
        0xC0 | 0xC1 => fetch8()? as u32,
        _ => read_reg64(RCX as i32) as u32 & 0xFF,
    } & if size == OpSize::S64 { 63 } else { 31 };

    if count == 0 {
        return Ok(());
    }

    let value = read_operand(operand, size, has_rex)? & size.mask();
    probe_operand_write(operand, size)?;
    let bits = size.bits();
    let sign = size.sign_bit();

    let (result, cf, of) = match modrm.reg & 7 {
        0 => {
            // ROL
            let result = ((value << count) | (value >> (bits - count))) & size.mask();
            let cf = value >> (bits - count) & 1 != 0;
            (result, cf, count == 1 && (result & sign != 0) ^ cf)
        },
        1 => {
            // ROR
            let result = ((value >> count) | (value << (bits - count))) & size.mask();
            let cf = result & sign != 0;
            (result, cf, count == 1 && (result & sign != 0) ^ (result & (sign >> 1) != 0))
        },
        2 | 3 => {
            // RCL/RCR through CF
            let mut result = value;
            let mut carry = *flags & FLAG_CF as i32 != 0;
            for _ in 0..count {
                if modrm.reg & 7 == 2 {
                    let next = result & sign != 0;
                    result = ((result << 1) & size.mask()) | if carry { 1 } else { 0 };
                    carry = next;
                }
                else {
                    let next = result & 1 != 0;
                    result = (result >> 1) | if carry { sign } else { 0 };
                    carry = next;
                }
            }
            (result, carry, false)
        },
        4 => {
            // SHL
            let result = value << count & size.mask();
            let cf = value >> (bits - count) & 1 != 0;
            (result, cf, count == 1 && (result & sign != 0) ^ cf)
        },
        5 => {
            // SHR
            let result = value >> count;
            let cf = value >> (count - 1) & 1 != 0;
            (result, cf, count == 1 && value & sign != 0)
        },
        7 => {
            // SAR
            let signed = (value ^ sign).wrapping_sub(sign);
            let result = ((signed as i64) >> count) as u64 & size.mask();
            (result, value >> (count - 1) & 1 != 0, false)
        },
        _ => {
            crate::cpu::cpu::trigger_ud();
            return Ok(());
        },
    };

    set_logic_flags(result, size);
    *flags &= !(FLAG_CF | FLAG_OF) as i32;
    if cf {
        *flags |= FLAG_CF as i32;
    }
    if of {
        *flags |= FLAG_OF as i32;
    }
    write_operand(operand, size, has_rex, result)?;
    Ok(())
}

// BT/BTS/BTR/BTC. `op`: 0 = test, 1 = set, 2 = reset, 3 = complement.
unsafe fn bit_test(
    operand: Operand,
    pfx: &Prefixes,
    size: OpSize,
    index: u64,
    op: u8,
) -> OrPageFault<()> {
    let has_rex = pfx.has_rex();
    // the index spans the whole string: bt [mem],100 addresses dword 3.
    let operand = if let Operand::Mem(address) = operand {
        let bits = size.bits() as u64;
        Operand::Mem(address.wrapping_add(index / bits * (bits / 8)))
    }
    else {
        operand
    };
    let value = read_operand(operand, size, has_rex)? & size.mask();
    let bit = index as u32 & (size.bits() - 1);
    let old = value >> bit & 1 != 0;

    *flags &= !FLAG_CF as i32;
    if old {
        *flags |= FLAG_CF as i32;
    }
    *flags_changed = 0;

    if op != 0 {
        let result = match op {
            1 => value | 1u64 << bit,
            2 => value & !(1u64 << bit),
            _ => value ^ 1u64 << bit,
        } & size.mask();
        probe_operand_write(operand, size)?;
        write_operand(operand, size, has_rex, result)?;
    }
    Ok(())
}

unsafe fn alu_acc_imm(op: AluOp, size: OpSize, imm: u64) -> OrPageFault<()> {
    let has_rex = false; // no REX on the accumulator immediate forms
    let dst = read_reg(RAX, size, has_rex);
    let result = alu(op, dst, imm, size);
    if op != AluOp::Cmp {
        write_reg(RAX, size, has_rex, result);
    }
    Ok(())
}

unsafe fn alu_rm_r_rm(op: AluOp, opcode: u8, pfx: &Prefixes, size: OpSize) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let has_rex = pfx.has_rex();
    if opcode & 0x02 == 0 {
        // r/m, r
        let src = read_reg(modrm.reg, size, has_rex);
        let dst = read_operand(operand, size, has_rex)?;
        if op != AluOp::Cmp {
            probe_operand_write(operand, size)?;
        }
        let result = alu(op, dst, src, size);
        if op != AluOp::Cmp {
            write_operand(operand, size, has_rex, result)?;
        }
    }
    else {
        // r, r/m
        let src = read_operand(operand, size, has_rex)?;
        let dst = read_reg(modrm.reg, size, has_rex);
        let result = alu(op, dst, src, size);
        if op != AluOp::Cmp {
            write_reg(modrm.reg, size, has_rex, result);
        }
    }
    Ok(())
}

unsafe fn fetch_imm(size: OpSize) -> OrPageFault<u64> {
    Ok(match size {
        OpSize::S8 => fetch8()? as u64,
        OpSize::S16 => fetch16()? as u64,
        // imm32 is sign-extended to 64 bits in long mode
        OpSize::S32 => fetch32()? as u64,
        OpSize::S64 => fetch32()? as i32 as i64 as u64,
    })
}

unsafe fn fetch_imm8_signed(size: OpSize) -> OrPageFault<u64> {
    Ok(fetch8()? as i8 as i64 as u64 & size.mask())
}

unsafe fn alu_group1(opcode: u8, pfx: &Prefixes, size: OpSize) -> OrPageFault<()> {
    let trailing = if opcode == 0x80 || opcode == 0x83 || size == OpSize::S8 {
        1
    }
    else if size == OpSize::S16 {
        2
    }
    else {
        4
    };
    let (modrm, operand) = decode_operand_with_trailing(pfx, trailing)?;
    let op = match modrm.reg & 7 {
        0 => AluOp::Add,
        1 => AluOp::Or,
        2 => AluOp::Adc,
        3 => AluOp::Sbb,
        4 => AluOp::And,
        5 => AluOp::Sub,
        6 => AluOp::Xor,
        7 => AluOp::Cmp,
        _ => {
            crate::cpu::cpu::trigger_ud();
            return Ok(());
        },
    };
    let has_rex = pfx.has_rex();
    let dst = read_operand(operand, size, has_rex)?;
    // The immediate follows the ModRM/SIB/displacement bytes
    let imm = if opcode == 0x83 {
        fetch_imm8_signed(size)?
    }
    else {
        fetch_imm(size)?
    };
    if op != AluOp::Cmp {
        probe_operand_write(operand, size)?;
    }
    let result = alu(op, dst, imm, size);
    if op != AluOp::Cmp {
        write_operand(operand, size, has_rex, result)?;
    }
    Ok(())
}

unsafe fn test_rm_r(pfx: &Prefixes, size: OpSize) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let a = read_operand(operand, size, pfx.has_rex())?;
    let b = read_reg(modrm.reg, size, pfx.has_rex());
    set_logic_flags(a & b & size.mask(), size);
    Ok(())
}

unsafe fn push64(value: u64) -> OrPageFault<()> {
    let rsp = read_reg64(RSP as i32).wrapping_sub(8);
    mem_write(rsp, OpSize::S64, value)?;
    write_reg64(RSP as i32, rsp);
    Ok(())
}

unsafe fn pop64() -> OrPageFault<u64> {
    let rsp = read_reg64(RSP as i32);
    let value = mem_read(rsp, OpSize::S64)?;
    write_reg64(RSP as i32, rsp.wrapping_add(8));
    Ok(value)
}

unsafe fn condition(code: u8) -> bool {
    let f = *flags as u32;
    let cf = f & FLAG_CF != 0;
    let zf = f & FLAG_ZF != 0;
    let sf = f & FLAG_SF != 0;
    let of = f & FLAG_OF != 0;
    let pf = f & FLAG_PF != 0;

    match code {
        0x0 => of,              // O
        0x1 => !of,             // NO
        0x2 => cf,              // B/C
        0x3 => !cf,             // AE/NC
        0x4 => zf,              // E/Z
        0x5 => !zf,             // NE/NZ
        0x6 => cf || zf,        // BE
        0x7 => !cf && !zf,      // A
        0x8 => sf,              // S
        0x9 => !sf,             // NS
        0xA => pf,              // P
        0xB => !pf,             // NP
        0xC => sf != of,        // L
        0xD => sf == of,        // GE
        0xE => zf || sf != of,  // LE
        0xF => !zf && sf == of, // G
        _ => unreachable!(),
    }
}

// --- x87 ------------------------------------------------------------------

unsafe fn x87_load_f32(address: u64) -> OrPageFault<F80> {
    Ok(f32_to_f80(mem_read(address, OpSize::S32)? as u32 as i32))
}

unsafe fn x87_load_f64(address: u64) -> OrPageFault<F80> {
    Ok(f64_to_f80(mem_read(address, OpSize::S64)?))
}

unsafe fn x87_load_m80(address: u64) -> OrPageFault<F80> {
    let mantissa = mem_read(address, OpSize::S64)?;
    let sign_exponent = mem_read(address + 8, OpSize::S16)? as u16;
    Ok(F80 { mantissa, sign_exponent })
}

unsafe fn x87_store_f32(address: u64, value: F80) -> OrPageFault<()> {
    mem_write(address, OpSize::S32, f80_to_f32(value) as u32 as u64)
}

unsafe fn x87_store_f64(address: u64, value: F80) -> OrPageFault<()> {
    mem_write(address, OpSize::S64, f80_to_f64(value))
}

unsafe fn x87_store_m80(address: u64, value: F80) -> OrPageFault<()> {
    mem_write(address, OpSize::S64, value.mantissa)?;
    mem_write(address + 8, OpSize::S16, value.sign_exponent as u64)
}

// FIST/FISTP round per the control word; FISTTP truncates.
unsafe fn x87_store_int(address: u64, value: F80, size: OpSize, truncate: bool) -> OrPageFault<()> {
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

unsafe fn x87_load_int(address: u64, size: OpSize) -> OrPageFault<F80> {
    let value = mem_read(address, size)? & size.mask();
    Ok(match size {
        OpSize::S16 => i32_to_f80(value as u16 as i16 as i32),
        OpSize::S32 => i32_to_f80(value as u32 as i32),
        _ => i64_to_f80(value as i64),
    })
}

// FNSTENV, in the protected-mode layouts the 32-bit implementation uses.
unsafe fn x87_store_env(address: u64, sixteen: bool) -> OrPageFault<()> {
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

unsafe fn x87_load_env(address: u64, sixteen: bool) -> OrPageFault<()> {
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
unsafe fn x87_save_state(address: u64, sixteen: bool) -> OrPageFault<()> {
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

unsafe fn x87_restore_state(address: u64, sixteen: bool) -> OrPageFault<()> {
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
unsafe fn x87_load_bcd(address: u64) -> OrPageFault<F80> {
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

unsafe fn x87_store_bcd(address: u64, value: F80) -> OrPageFault<()> {
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

unsafe fn x87(pfx: &Prefixes, opcode: u8) -> OrPageFault<()> {
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
        (0xD9, 5) => *fpu_control_word = mem_read(address, OpSize::S16)? as u16,
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
            crate::cpu::cpu::trigger_ud();
        },
    }
    Ok(())
}

unsafe fn x87_reg(opcode: u8, reg: i32, st: i32, sixteen: bool) {
    use crate::cpu::instructions as x87_32;
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
        _ => crate::cpu::cpu::trigger_ud(),
    }
}

unsafe fn jcc(code: u8, displacement: i64) {
    if condition(code) {
        *rip = (*rip).wrapping_add(displacement as u64);
    }
}

pub unsafe fn run_one() {
    *previous_rip = *rip;
    let _ = run_one_inner();
    // keep the 32-bit view in sync
    *instruction_pointer = *rip as u32 as i32;
}

#[no_mangle]
pub unsafe fn interp64_run_one() {
    *rip = *instruction_pointer as u32 as u64;
    run_one();
}

// Point CR3 at a PML4, enable PAE and paging, and switch the main loop over to
// the 64-bit interpreter.
#[no_mangle]
pub unsafe fn enter_long_mode(cr3: u32) {
    // The 16/32-bit interpreter keeps arithmetic flags lazy. Long mode reads the
    // flags word directly, so resolve them once on entry.
    *flags = crate::cpu::cpu::get_eflags();
    *flags_changed = 0;
    *rip = *instruction_pointer as u32 as u64;
    *previous_rip = *rip;
    *cr.offset(3) = cr3 as i32; // CR3
    *cr.offset(4) |= crate::cpu::cpu::CR4_PAE; // CR4.PAE
    *cr.offset(4) |= crate::cpu::cpu::CR4_PSE; // CR4.PSE (2 MiB pages)
    *long_mode = true;
    *efer |= EFER_LME | EFER_LMA;
    *cr |= crate::cpu::cpu::CR0_PG; // CR0.PG
    crate::cpu::cpu::full_clear_tlb();
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

unsafe fn run_one_inner() -> OrPageFault<()> {
    let (pfx, opcode) = decode_prefixes()?;

    if opcode == 0x0F {
        return run_0f(&pfx);
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
                crate::cpu::cpu::trigger_ud();
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
                    2 => crate::cpu::cpu::io_port_read16(port) as u16 as u64,
                    _ => crate::cpu::cpu::io_port_read32(port) as u32 as u64,
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
                    2 => crate::cpu::cpu::io_port_write16(port, read_reg(RAX, op, false) as i32),
                    _ => crate::cpu::cpu::io_port_write32(port, read_reg(RAX, op, false) as i32),
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
                    2 => crate::cpu::cpu::io_port_read16(port) as u16 as u64,
                    _ => crate::cpu::cpu::io_port_read32(port) as u32 as u64,
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
                    2 => crate::cpu::cpu::io_port_write16(port, read_reg(RAX, op, false) as i32),
                    _ => crate::cpu::cpu::io_port_write32(port, read_reg(RAX, op, false) as i32),
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
                crate::cpu::cpu::trigger_ud();
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
                _ => crate::cpu::cpu::trigger_ud(),
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
                        crate::cpu::cpu::trigger_ud();
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
                _ => crate::cpu::cpu::trigger_ud(),
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
            let new_rip = pop64()?;
            let new_cs = pop64()?;
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
                                2 => crate::cpu::cpu::io_port_read16(port) as u16 as u64,
                                _ => crate::cpu::cpu::io_port_read32(port) as u32 as u64,
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
                                2 => crate::cpu::cpu::io_port_write16(port, value as i32),
                                _ => crate::cpu::cpu::io_port_write32(port, value as i32),
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

        // ENTER imm16, imm8: build a stack frame
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

        // XLAT (0xD7): AL = [RBX + AL] (BX with a 32-bit address size)
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
        0xCC => crate::cpu::cpu::call_interrupt_vector64(3, None),
        0xCD => {
            let vector = fetch8()? as i32;
            crate::cpu::cpu::call_interrupt_vector64(vector, None);
        },

        // CLI (0xFA) / STI (0xFB)
        0xFA => *flags &= !FLAG_INTERRUPT,
        0xFB => *flags |= FLAG_INTERRUPT,

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
                crate::cpu::cpu::trigger_ud();
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

        // SAHF (0x9E) / LAHF (0x9F): AH <-> SF ZF 0 AF 0 PF 1 CF
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

        // MOVSB (0xA4) / REP MOVSB
        0xA4 => {
            let count = if pfx.f3 { read_reg64(RCX as i32) } else { 1 };
            let direction = *flags & (1 << 10) != 0;
            for _ in 0..count {
                let source = read_reg64(RSI as i32);
                let value = mem_read(source, OpSize::S8)?;
                let destination = read_reg64(RDI as i32);
                mem_write(destination, OpSize::S8, value)?;
                if direction {
                    write_reg64(RSI as i32, source.wrapping_sub(1));
                    write_reg64(RDI as i32, destination.wrapping_sub(1));
                }
                else {
                    write_reg64(RSI as i32, source.wrapping_add(1));
                    write_reg64(RDI as i32, destination.wrapping_add(1));
                }
            }
            if pfx.f3 {
                write_reg64(RCX as i32, 0);
            }
        },

        _ => {
            dbg_log!("#ud interp64: opcode {:02x}", opcode);
            crate::cpu::cpu::trigger_ud();
        },
    }

    Ok(())
}

unsafe fn run_0f(pfx: &Prefixes) -> OrPageFault<()> {
    let opcode = fetch8()?;
    let size = pfx.operand_size();

    if run_sse(opcode, pfx)? {
        return Ok(());
    }

    match opcode {
        // ENDBR64 (F3 0F 1E FA) / ENDBR32 (F3 0F 1E FB): no effect
        0x1E if pfx.f3 => {
            let modrm = fetch8()?;
            if modrm != 0xFA && modrm != 0xFB {
                crate::cpu::cpu::trigger_ud();
            }
        },

        // LAR/LSL (0F 02/03); only the GDT is modelled, else ZF is cleared.
        0x02 | 0x03 => {
            let (modrm, operand) = decode_operand(&pfx)?;
            let selector = read_operand(operand, OpSize::S16, pfx.has_rex())? as u16;
            let index = (selector >> 3) as u64;
            let in_ldt = selector >> 2 & 1 != 0;
            let descriptor = if selector & !7 != 0 && !in_ldt {
                mem_read(*crate::cpu::global_pointers::gdtr_base + index * 8, OpSize::S64)?
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

        // PREFETCHT0/T1/T2/NTA (0F 18 /0../3) and PREFETCHW (0F 0D /0,/1).
        // No architectural effect: only the operand has to be decoded, and
        // prefetches never fault.
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
                        crate::cpu::cpu::trigger_ud();
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
                6 if modrm.mod_bits == 3 => {
                    // RDRAND r
                    let random = crate::cpu::cpu::js::get_rand_int() as u32 as u64;
                    write_reg(modrm.rm, size, pfx.has_rex(), random);
                    *flags &= !(FLAG_CF as i32 | FLAG_OF as i32);
                    *flags |= FLAG_CF as i32;
                    *flags_changed = 0;
                },
                _ => crate::cpu::cpu::trigger_ud(),
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
                _ => crate::cpu::cpu::trigger_ud(),
            }
        },

        // MOV r64, CRn (0F 20) / MOV CRn, r64 (0F 22)
        0x20 | 0x22 => {
            if *cpl != 0 {
                crate::cpu::cpu::trigger_gp(0);
                return Ok(());
            }
            let modrm = decode_modrm(pfx)?;
            if modrm.mod_bits != 3 || modrm.reg > 7 {
                crate::cpu::cpu::trigger_ud();
                return Ok(());
            }
            if opcode == 0x20 {
                let value = match modrm.reg {
                    0 => *cr as u32 as u64,
                    2 => *cr2,
                    3 => *cr.offset(3) as u32 as u64,
                    4 => *cr.offset(4) as u32 as u64,
                    _ => {
                        crate::cpu::cpu::trigger_ud();
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
                crate::cpu::instructions_0f::instr_0F22(modrm.rm as i32, modrm.reg as i32);
            }
        },

        // MOV r64, DRn (0F 21) / MOV DRn, r64 (0F 23)
        0x21 | 0x23 => {
            if *cpl != 0 {
                crate::cpu::cpu::trigger_gp(0);
                return Ok(());
            }
            let modrm = decode_modrm(pfx)?;
            if modrm.mod_bits != 3 || modrm.reg > 7 {
                crate::cpu::cpu::trigger_ud();
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

        // SHLD/SHRD r/m, r, imm8 (0F A4/0F AC) or CL (0F A5/0F AD)
        0xA4 | 0xA5 | 0xAC | 0xAD => {
            let shld = opcode == 0xA4 || opcode == 0xA5;
            let trailing = if opcode == 0xA4 || opcode == 0xAC { 1 } else { 0 };
            let (modrm, operand) = decode_operand_with_trailing(pfx, trailing)?;
            let size = if size == OpSize::S8 { OpSize::S32 } else { size };
            let has_rex = pfx.has_rex();
            let count = if trailing != 0 {
                fetch8()? as u32
            }
            else {
                read_reg64(RCX as i32) as u32 & 0xFF
            } & if size == OpSize::S64 { 63 } else { 31 };
            if count == 0 {
                return Ok(());
            }
            let dst = read_operand(operand, size, has_rex)? & size.mask();
            let src = read_reg(modrm.reg, size, has_rex) & size.mask();
            probe_operand_write(operand, size)?;
            let bits = size.bits();
            let sign = size.sign_bit();
            let (result, cf) = if shld {
                let result = (dst << count | src >> (bits - count)) & size.mask();
                (result, dst >> (bits - count) & 1 != 0)
            }
            else {
                let result = (dst >> count | src << (bits - count)) & size.mask();
                (result, dst >> (count - 1) & 1 != 0)
            };
            set_logic_flags(result, size);
            *flags &= !(FLAG_CF | FLAG_OF) as i32;
            if cf {
                *flags |= FLAG_CF as i32;
            }
            // OF is only defined for a count of one
            if count == 1
                && if shld {
                    (result & sign != 0) ^ cf
                }
                else {
                    (dst & sign != 0) ^ cf
                }
            {
                *flags |= FLAG_OF as i32;
            }
            write_operand(operand, size, has_rex, result)?;
        },

        // CPUID (0F A2)
        0xA2 => {
            crate::cpu::instructions_0f::instr_0FA2();
        },

        // BSWAP r32/r64 (0F C8-0F CF)
        0xC8..=0xCF => {
            let r = opcode - 0xC8 | (pfx.rex & prefix::REX_B) << 3;
            let value = read_reg(r, size, pfx.has_rex());
            write_reg(
                r,
                size,
                pfx.has_rex(),
                if size == OpSize::S64 { value.swap_bytes() } else { (value as u32).swap_bytes() as u64 },
            );
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
                _ => crate::cpu::cpu::trigger_ud(),
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
                        write_reg(RAX, OpSize::S32, false, 1); // XCR0.x87
                        write_reg(RDX, OpSize::S32, false, 0);
                    },
                    // XSETBV (0F 01 D1): accepted, XCR0 stays at x87
                    2 if modrm.rm == 1 => {},
                    _ => crate::cpu::cpu::trigger_ud(),
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
                        let base = if *gdtr_base != 0 { *gdtr_base } else { *gdtr_offset as u32 as u64 };
                        mem_write(address + 2, OpSize::S64, base)?;
                    },
                    // SIDT
                    1 => {
                        mem_write(address, OpSize::S16, *idtr_size as u32 as u64)?;
                        let base = if *idtr_base != 0 { *idtr_base } else { *idtr_offset as u32 as u64 };
                        mem_write(address + 2, OpSize::S64, base)?;
                    },
                    // LGDT
                    2 => {
                        *gdtr_size = mem_read(address, OpSize::S16)? as i32;
                        let base = mem_read(address + 2, OpSize::S64)?;
                        *gdtr_offset = base as u32 as i32;
                        *gdtr_base = base;
                    },
                    // LIDT
                    3 => {
                        *idtr_size = mem_read(address, OpSize::S16)? as i32;
                        let base = mem_read(address + 2, OpSize::S64)?;
                        *idtr_offset = base as u32 as i32;
                        *idtr_base = base;
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
                    7 => crate::cpu::cpu::full_clear_tlb(),
                    _ => crate::cpu::cpu::trigger_ud(),
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
                        crate::cpu::cpu::trigger_ud();
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
                        _ => crate::cpu::cpu::trigger_ud(),
                    }
                }
                else {
                    match modrm.reg & 7 {
                        5 | 6 | 7 => {}, // LFENCE / MFENCE / SFENCE
                        _ => crate::cpu::cpu::trigger_ud(),
                    }
                }
            }
            else {
                let (modrm, operand) = decode_operand_after_modrm(pfx, modrm, 0)?;
                match modrm.reg & 7 {
                    // FXSAVE (0) and XSAVE (4). Only x87 and SSE state is
                    // supported, whose layout is the same as FXSAVE's; XSAVE
                    // additionally records them in the XSTATE_BV header.
                    0 | 4 => {
                        let Operand::Mem(address) = operand else {
                            crate::cpu::cpu::trigger_ud();
                            return Ok(());
                        };
                        for i in 0..512u64 {
                            mem_write(address + i, OpSize::S8, 0)?;
                        }
                        mem_write(address, OpSize::S16, *fpu_control_word as u64)?;
                        mem_write(address + 2, OpSize::S16, *fpu_status_word as u64)?;
                        mem_write(address + 24, OpSize::S32, *mxcsr as u32 as u64)?;
                        // MXCSR_MASK: the bits of MXCSR that may be set. Real CPUs
                        // report 0xffff; leaving it zero makes the kernel (which
                        // derives mxcsr_feature_mask from it) consider its own
                        // MXCSR value invalid and WARN on every context switch.
                        mem_write(address + 28, OpSize::S32, 0xFFFF)?;
                        for i in 0..8u64 {
                            let x = xmm_get(i as u8);
                            mem_write(address + 160 + i * 16, OpSize::S64, x.u64[0])?;
                            mem_write(address + 160 + i * 16 + 8, OpSize::S64, x.u64[1])?;
                        }
                        if modrm.reg & 7 == 4 {
                            // XSTATE_BV: x87 (bit 0) and SSE (bit 1) were saved
                            mem_write(address + 512, OpSize::S64, 3)?;
                            mem_write(address + 520, OpSize::S64, 0)?;
                        }
                    },
                    // FXRSTOR (1) and XRSTOR (5)
                    1 | 5 => {
                        let Operand::Mem(address) = operand else {
                            crate::cpu::cpu::trigger_ud();
                            return Ok(());
                        };
                        *fpu_control_word = mem_read(address, OpSize::S16)? as u16;
                        *fpu_status_word = mem_read(address + 2, OpSize::S16)? as u16;
                        *mxcsr = mem_read(address + 24, OpSize::S32)? as u32 as i32;
                        for i in 0..8u64 {
                            let low = mem_read(address + 160 + i * 16, OpSize::S64)?;
                            let high = mem_read(address + 160 + i * 16 + 8, OpSize::S64)?;
                            xmm_set(i as u8, reg128 { u64: [low, high] });
                        }
                    },
                    // LDMXCSR
                    2 => {
                        let Operand::Mem(address) = operand else {
                            crate::cpu::cpu::trigger_ud();
                            return Ok(());
                        };
                        *mxcsr = mem_read(address, OpSize::S32)? as u32 as i32;
                    },
                    // STMXCSR
                    3 => {
                        let Operand::Mem(address) = operand else {
                            crate::cpu::cpu::trigger_ud();
                            return Ok(());
                        };
                        mem_write(address, OpSize::S32, *mxcsr as u32 as u64)?;
                    },
                    7 => {}, // CLFLUSH
                    _ => crate::cpu::cpu::trigger_ud(),
                }
            }
        },

        // SYSCALL (0F 05)
        0x05 => {
            if *efer & EFER_SCE == 0 {
                crate::cpu::cpu::trigger_ud();
                return Ok(());
            }
            write_reg64(RCX as i32, *rip);
            write_reg64(R11 as i32, *flags as u32 as u64);
            *flags &= !(*sfmask as u32) as i32;
            *rip = *lstar;
            let cs = (*star >> 32 & 0xFFFF) as u16;
            set_cpl_segments(cs, 0);
        },

        // SYSRET (0F 07)
        0x07 => {
            if *efer & EFER_SCE == 0 {
                crate::cpu::cpu::trigger_ud();
                return Ok(());
            }
            let target = read_reg64(RCX as i32);
            *flags = read_reg64(R11 as i32) as u32 as i32;
            let cs = (*star >> 48 & 0xFFFF) as u16 | 3;
            set_cpl_segments(cs, 3);
            *rip = target;
        },

        // WRMSR (0F 30)
        0x30 => {
            if *cpl != 0 {
                crate::cpu::cpu::trigger_gp(0);
                return Ok(());
            }
            let index = read_reg64(RCX as i32) as i32;
            let value = read_reg64(RAX as i32) as u32 as u64
                | (read_reg64(RDX as i32) as u32 as u64) << 32;
            if !msr_write(index, value) {
                crate::cpu::instructions_0f::instr_0F30();
            }
        },

        // RDTSC (0F 31)
        0x31 => {
            if *cpl == 0 || *cr.offset(4) & CR4_TSD == 0 {
                let tsc = read_tsc();
                write_reg(RAX, OpSize::S32, false, tsc & 0xFFFF_FFFF);
                write_reg(RDX, OpSize::S32, false, tsc >> 32);
            }
            else {
                crate::cpu::cpu::trigger_gp(0);
            }
        },

        // RDMSR (0F 32)
        0x32 => {
            if *cpl != 0 {
                crate::cpu::cpu::trigger_gp(0);
                return Ok(());
            }
            let index = read_reg64(RCX as i32) as i32;
            match msr_read(index) {
                Some(value) => {
                    write_reg(RAX, OpSize::S32, false, value & 0xFFFF_FFFF);
                    write_reg(RDX, OpSize::S32, false, value >> 32);
                },
                None => crate::cpu::instructions_0f::instr_0F32(),
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
                    crate::cpu::cpu::trigger_ud();
                    return Ok(());
                },
            };
            bit_test(operand, pfx, size, index, op)?;
        },

        // BSF/BSR r, r/m (0F BC/BD)
        0xBC | 0xBD => {
            let (modrm, operand) = decode_operand(pfx)?;
            let src = read_operand(operand, size, pfx.has_rex())? & size.mask();
            if src == 0 {
                set_zf(true);
            }
            else {
                set_zf(false);
                let result = if opcode == 0xBC {
                    src.trailing_zeros() as u64
                }
                else {
                    63 - src.leading_zeros() as u64
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
            crate::cpu::cpu::trigger_ud();
        },
    }

    Ok(())
}
