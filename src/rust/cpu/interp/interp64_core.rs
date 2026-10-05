//! Shared long-mode interpreter core: prefix/ModRM decode, the operand and
//! address model, memory and register access, flags and the integer ALU. Used
//! by both `interp64` (base opcodes) and `interp64_0f` (0F/SSE/AVX).
//! Operand size 32/64 (REX.W)/16 (0x66); address size 64/32 (0x67); phys < 4 GiB.

#![allow(dead_code, unused_imports)]

use crate::cpu::core::{
    io_port_read8, io_port_write8, read_reg64, read_tsc, reg128, test_privileges_for_io, translate_address_64,
    write_reg64, APIC_MEM_ADDRESS, CS, FLAG_CARRY, FLAG_INTERRUPT, SS,
};
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

pub use crate::cpu::decode::operand::{Modrm, Operand, Prefixes};

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

// XCR0: x87 (bit 0), SSE (bit 1) and AVX (bit 2) are supported. Linux clears
// the AVX feature early if XCR0 lacks it, so advertise it enabled from reset.
pub(crate) static mut XCR0: u64 = 7;

pub unsafe fn get_xcr0() -> u64 { XCR0 }

pub unsafe fn set_xcr0(value: u64) { XCR0 = value; }

// RFLAGS bits
pub(crate) const FLAG_CF: u32 = 1;
pub(crate) const FLAG_PF: u32 = 1 << 2;
pub(crate) const FLAG_AF: u32 = 1 << 4;
pub(crate) const FLAG_ZF: u32 = 1 << 6;
pub(crate) const FLAG_SF: u32 = 1 << 7;
pub(crate) const FLAG_OF: u32 = 1 << 11;



// Same-page accesses translate once. Runtime A/B switch.
pub static mut INTERP64_MEM_SINGLE: bool = true;
// Interpreter-only tight batch loop (see interp64_run). Runtime A/B switch.
pub static mut INTERP64_BATCH: bool = true;

// Diagnostic: interpreted opcode histogram (what still falls back to interp).
#[no_mangle]
pub static mut INTERP64_OPCODE: [u32; 256] = [0; 256];
#[no_mangle]
pub static mut INTERP64_OPCODE_0F: [u32; 256] = [0; 256];
// Off by default so the per-instruction counter costs nothing.
pub static mut INTERP64_OPCODE_STATS: bool = false;
pub unsafe fn interp64_set_opcode_stats(enabled: u32) { INTERP64_OPCODE_STATS = enabled != 0; }

#[no_mangle]
pub unsafe fn interp64_opcode_count(op: u32) -> u32 {
    if op < 256 { INTERP64_OPCODE[op as usize] } else { 0 }
}

#[no_mangle]
pub unsafe fn interp64_opcode0f_count(op: u32) -> u32 {
    if op < 256 { INTERP64_OPCODE_0F[op as usize] } else { 0 }
}

pub unsafe fn interp64_set_mem_single(enabled: u32) { INTERP64_MEM_SINGLE = enabled != 0; }

pub unsafe fn interp64_set_batch(enabled: u32) { INTERP64_BATCH = enabled != 0; }

// Answers for its own key range; `crate::config` chains the subsystems.
pub static mut INTERP64_DECODE_CACHE: bool = true;

pub unsafe fn interp64_set_decode_cache(enabled: u32) { INTERP64_DECODE_CACHE = enabled != 0; }

pub unsafe fn set_config(index: u32, value: u32) -> bool {
    use crate::config as cfg;

    match index {
        cfg::INTERP64_BATCH => interp64_set_batch(value),
        cfg::INTERP64_MEM_SINGLE => interp64_set_mem_single(value),
        cfg::INTERP64_DECODE_CACHE => interp64_set_decode_cache(value),
        cfg::INTERP64_OPCODE_STATS => interp64_set_opcode_stats(value),
        _ => return false,
    }

    true
}

pub unsafe fn get_config(index: u32) -> Option<u32> {
    use crate::config as cfg;

    Some(match index {
        cfg::INTERP64_BATCH => INTERP64_BATCH as u32,
        cfg::INTERP64_MEM_SINGLE => INTERP64_MEM_SINGLE as u32,
        cfg::INTERP64_DECODE_CACHE => INTERP64_DECODE_CACHE as u32,
        cfg::INTERP64_OPCODE_STATS => INTERP64_OPCODE_STATS as u32,
        _ => return None,
    })
}

#[inline(always)]
pub(crate) unsafe fn fetch_phys(address: u64) -> OrPageFault<u32> {
    pstat(stat::INTERP64_FETCH_TRANSLATIONS);
    translate_address_64(address, false, *cpl == 3)
}

#[inline(always)]
pub(crate) unsafe fn fetch8() -> OrPageFault<u8> {
    pstat(stat::INTERP64_FETCH_BYTES);
    let address = *rip;
    let value = memory::read8(fetch_phys(address)?) as u8;
    *rip = address.wrapping_add(1);
    Ok(value)
}

// Profiling counters, compiled out unless the `profiler` feature is enabled.
#[cfg(feature = "profiler")]
#[inline(always)]
pub(crate) fn pstat(s: crate::profiler::stat) { crate::profiler::stat_increment(s); }
#[cfg(not(feature = "profiler"))]
#[inline(always)]
pub(crate) fn pstat(_s: crate::profiler::stat) {}

pub(crate) unsafe fn fetch16() -> OrPageFault<u16> {
    let address = *rip;
    // Don't fetch past the page end: the next physical page is not the next
    // virtual one. cf. Intel SDM Vol. 1, 3.2.1 (words don't straddle pages).
    if (address as usize & 0xFFF) + 2 <= 0x1000 {
        pstat(stat::INTERP64_FETCH_BYTES);
        pstat(stat::INTERP64_FETCH_BYTES);
        let phys = fetch_phys(address)?;
        let value = memory::read16(phys) as u16;
        *rip = address.wrapping_add(2);
        Ok(value)
    }
    else {
        let low = fetch8()? as u16;
        let high = fetch8()? as u16;
        Ok(low | high << 8)
    }
}

pub(crate) unsafe fn fetch32() -> OrPageFault<u32> {
    let address = *rip;
    if (address as usize & 0xFFF) + 4 <= 0x1000 {
        for _ in 0..4 { pstat(stat::INTERP64_FETCH_BYTES); }
        let phys = fetch_phys(address)?;
        let value = memory::read32s(phys) as u32;
        *rip = address.wrapping_add(4);
        Ok(value)
    }
    else {
        let low = fetch16()? as u32;
        let high = fetch16()? as u32;
        Ok(low | high << 16)
    }
}

pub(crate) unsafe fn fetch64() -> OrPageFault<u64> {
    let low = fetch32()? as u64;
    let high = fetch32()? as u64;
    Ok(low | (high << 32))
}

pub(crate) unsafe fn decode_prefixes() -> OrPageFault<(Prefixes, u8)> {
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

#[inline(always)]
pub(crate) unsafe fn decode_modrm(pfx: &Prefixes) -> OrPageFault<Modrm> {
    let byte = fetch8()?;
    Ok(Modrm {
        mod_bits: byte >> 6,
        // REX.R/REX.B extend the 3-bit field by 8
        reg: (byte >> 3 & 7) | (pfx.rex & prefix::REX_R) << 1,
        rm: (byte & 7) | (pfx.rex & prefix::REX_B) << 3,
    })
}

// mod != 3
#[inline(always)]
pub(crate) unsafe fn modrm_effective_address(modrm: &Modrm, pfx: &Prefixes) -> OrPageFault<u64> {
    dbg_assert!(modrm.mod_bits != 3);

    let mut base = 0u64;
    let mut index = 0u64;
    let mut scale = 0u8;
    let mut disp = 0u64;
    let rm_low = modrm.rm & 7;

    if rm_low == 4 {
        // SIB byte
        let sib = fetch8()?;
        scale = sib >> 6;
        let index_low = sib >> 3 & 7;
        let base_low = sib & 7;

        // index == 4 with no REX.X means "no index"
        if !(index_low == 4 && pfx.rex & prefix::REX_X == 0) {
            let idx = (index_low | (pfx.rex & prefix::REX_X) << 2) as i32;
            index = read_reg64(idx);
        }

        if base_low == 5 && modrm.mod_bits == 0 {
            // no base, disp32
            disp = fetch32()? as i32 as i64 as u64;
        }
        else {
            let b = (base_low | (pfx.rex & prefix::REX_B) << 3) as i32;
            base = read_reg64(b);
        }
    }
    else if rm_low == 5 && modrm.mod_bits == 0 {
        // RIP-relative: the base is rip after the disp32 (the caller adds the
        // remaining instruction bytes via trailing_bytes).
        disp = fetch32()? as i32 as i64 as u64;
        base = *rip;
    }
    else {
        base = read_reg64(modrm.rm as i32);
    }

    match modrm.mod_bits {
        0 => {},
        1 => disp = disp.wrapping_add(fetch8()? as i8 as i64 as u64),
        2 => disp = disp.wrapping_add(fetch32()? as i32 as i64 as u64),
        _ => unreachable!(),
    }

    let seg_base = if pfx.segment == 4 {
        *fs_base
    }
    else if pfx.segment == 5 {
        *gs_base
    }
    else {
        0
    };

    // 0x67 selects a 32-bit address: the base/index/displacement sum wraps at
    // 32 bits (the segment base is still added outside the wrap).
    let addr_size = if pfx.addrsize_32 { 32 } else { 64 };
    Ok(crate::cpu::decode::address::effective_address(
        base, index, scale, disp, seg_base, addr_size,
    ))
}

pub(crate) unsafe fn set_cpl_segments(cs: u16, new_cpl: u8) {
    // long mode: code/data bases are 0, SS = CS + 8
    *sreg.offset(CS as isize) = cs;
    *sreg.offset(SS as isize) = cs.wrapping_add(8);
    *segment_offsets.offset(CS as isize) = 0;
    *segment_offsets.offset(SS as isize) = 0;
    *segment_is_null.offset(CS as isize) = false;
    *segment_is_null.offset(SS as isize) = false;
    *cpl = new_cpl;
}

pub(crate) unsafe fn msr_read(index: i32) -> Option<u64> {
    match index {
        MSR_EFER => Some(*efer),
        MSR_STAR => Some(*star),
        MSR_LSTAR => Some(*lstar),
        MSR_SFMASK => Some(*sfmask),
        MSR_FS_BASE => Some(*fs_base),
        MSR_GS_BASE => Some(*gs_base),
        MSR_KERNEL_GS_BASE => Some(*kernel_gs_base),
        // IA32_APIC_BASE: the base is fixed; the enable bit changes and this
        // single CPU is the bootstrap processor (bit 8).
        0x1B => Some(
            APIC_MEM_ADDRESS as u64 | 0x100 | if *apic_enabled { 0x800 } else { 0 }
        ),
        _ => None,
    }
}

pub(crate) unsafe fn msr_write(index: i32, value: u64) -> bool {
    match index {
        MSR_EFER => {
            *efer = value | EFER_LMA
        },
        MSR_STAR => *star = value,
        MSR_LSTAR => *lstar = value,
        MSR_SFMASK => *sfmask = value,
        MSR_FS_BASE => *fs_base = value,
        MSR_GS_BASE => *gs_base = value,
        MSR_KERNEL_GS_BASE => *kernel_gs_base = value,
        // IA32_TSC_DEADLINE: arms the local APIC's TSC-deadline one-shot timer
        MSR_TSC_DEADLINE => crate::devices::apic::set_tsc_deadline(value),
        _ => return false,
    }
    true
}

pub(crate) unsafe fn set_zf(value: bool) {
    let mut f = *flags as u32;
    f &= !FLAG_ZF;
    if value {
        f |= FLAG_ZF;
    }
    *flags = f as i32;
    *flags_changed = 0;
}

// Load TR from its GDT descriptor so the IST path can find the TSS.
pub(crate) unsafe fn load_tss(selector: u16) {
    let index = (selector >> 3) as u32;
    // The GDT can sit above 4 GiB (KPTI cpu_entry_area), so translate the full base.
    let base = if GDTR_BASE != 0
    {
        match crate::cpu::core::translate_address_system_read64(GDTR_BASE + index as u64 * 8)
        {
            Ok(phys) => phys,
            Err(_) => return,
        }
    }
    else
    {
        (*gdtr_offset as u32).wrapping_add(index * 8)
    };
    let low = memory::read32s(base) as u32;
    let high = memory::read32s(base + 4) as u32;
    // base[63:32] lives in the descriptor's upper doubleword; the kernel puts
    // the TSS in the KPTI cpu_entry_area, so the base does not fit in 32 bits.
    let seg_base = ((low >> 16 & 0xFFFF) as u64)
        | (((high & 0xFF) as u64) << 16)
        | (((high >> 24 & 0xFF) as u64) << 24)
        | ((memory::read32s(base + 8) as u32 as u64) << 32);
    let limit = (low & 0xFFFF) | (high >> 16 & 0xF) << 16;
    *segment_offsets.offset(6) = seg_base as i32;
    *segment_limits.offset(6) = limit;
    *tss_size_32 = false;
    TSS_BASE = seg_base;
}

pub(crate) unsafe fn selector_valid(selector: u16) -> bool {
    if selector & !7 == 0 {
        return false;
    }
    let index = (selector >> 3) as u32;
    index * 8 + 7 <= *gdtr_size as u32
}

#[inline(always)]
pub(crate) unsafe fn read_reg8(r: u8, has_rex: bool) -> u64 {
    if !has_rex && r >= 4 && r < 8 {
        // AH/CH/DH/BH
        read_reg64((r - 4) as i32) >> 8 & 0xFF
    }
    else {
        read_reg64(r as i32) & 0xFF
    }
}

#[inline(always)]
pub(crate) unsafe fn write_reg8(r: u8, has_rex: bool, value: u64) {
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

#[inline(always)]
pub(crate) unsafe fn read_reg(r: u8, size: OpSize, has_rex: bool) -> u64 {
    match size {
        OpSize::S8 => read_reg8(r, has_rex),
        OpSize::S16 => read_reg64(r as i32) & 0xFFFF,
        OpSize::S32 => read_reg64(r as i32) & 0xFFFF_FFFF,
        OpSize::S64 => read_reg64(r as i32),
    }
}

#[inline(always)]
pub(crate) unsafe fn write_reg(r: u8, size: OpSize, has_rex: bool, value: u64) {
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

#[inline(always)]
pub(crate) unsafe fn mem_read(address: u64, size: OpSize) -> OrPageFault<u64> {
    let bytes = (size.bits() / 8) as usize;
    pstat(stat::INTERP64_MEM_READS);
    // A same-page access maps contiguously, so one translation is enough.
    if INTERP64_MEM_SINGLE && (address as usize & 0xFFF) + bytes <= 0x1000 {
        pstat(stat::INTERP64_MEM_TRANSLATIONS);
        let phys = translate_address_64(address, false, *cpl == 3)?;
        return Ok(match size {
            OpSize::S8 => memory::read8(phys) as u8 as u64,
            OpSize::S16 => memory::read16(phys) as u16 as u64,
            OpSize::S32 => memory::read32s(phys) as u32 as u64,
            OpSize::S64 => memory::read64s(phys) as u64,
        });
    }
    let mut physical = [0u32; 8];
    for offset in 0..bytes {
        pstat(stat::INTERP64_MEM_TRANSLATIONS);
        physical[offset] = translate_address_64(address + offset as u64, false, *cpl == 3)?;
    }
    let mut value = 0;
    for offset in 0..bytes {
        value |= (memory::read8(physical[offset]) as u8 as u64) << (offset * 8);
    }
    Ok(value)
}

#[inline(always)]
pub(crate) unsafe fn mem_write(address: u64, size: OpSize, value: u64) -> OrPageFault<()> {
    let bytes = (size.bits() / 8) as usize;
    pstat(stat::INTERP64_MEM_WRITES);
    if INTERP64_MEM_SINGLE && (address as usize & 0xFFF) + bytes <= 0x1000 {
        pstat(stat::INTERP64_MEM_TRANSLATIONS);
        let phys = translate_address_64(address, true, *cpl == 3)?;
        match size {
            OpSize::S8 => memory::write8(phys, value as i32),
            OpSize::S16 => memory::write16(phys, value as i32),
            OpSize::S32 => memory::write32(phys, value as i32),
            OpSize::S64 => memory::write64(phys, value),
        }
        return Ok(());
    }
    // Cross-page: translate every byte first, then write, so a fault on the
    // second page cannot leave the first one partly written.
    let mut physical = [0u32; 8];
    for offset in 0..bytes {
        pstat(stat::INTERP64_MEM_TRANSLATIONS);
        physical[offset] = translate_address_64(address + offset as u64, true, *cpl == 3)?;
    }
    for offset in 0..bytes {
        memory::write8(physical[offset], (value >> (offset * 8)) as u8 as i32);
    }
    Ok(())
}

// Elements of this size that fit in the current page, moving in `direction`.
pub(crate) fn count_until_end_of_page64(direction: i32, bytes: u64, addr: u32) -> u64 {
    if direction == 1 {
        (0x1000 - (addr & 0xFFF) as u64) / bytes
    }
    else {
        ((addr & 0xFFF) as u64) / bytes + 1
    }
}

// REP MOVS/STOS fast path: move/fill a whole page of plain RAM at once. Returns
// Ok(true) when done, Ok(false) for MMIO, page crossings, overlap or bad addresses.
pub(crate) unsafe fn rep_movs_stos_fast(
    opcode: u8,
    elem_size: OpSize,
    delta: i64,
    has_rex: bool,
    count: &mut u64,
) -> OrPageFault<bool> {
    let bytes = (elem_size.bits() / 8) as u64;
    let is_movs = opcode & 0xFE == 0xA4;
    let direction = if delta > 0 { 1 } else { -1 };

    while *count > 0 {
        let rsi = read_reg64(RSI as i32);
        let rdi = read_reg64(RDI as i32);

        let phys_dst = translate_address_64(rdi, true, *cpl == 3)?;
        if memory::in_mapped_range(phys_dst) {
            return Ok(false);
        }
        let phys_src = if is_movs {
            let phys = translate_address_64(rsi, false, *cpl == 3)?;
            if memory::in_mapped_range(phys) {
                return Ok(false);
            }
            phys
        }
        else {
            0
        };
        if (phys_dst & 0xFFF) as u64 + bytes > 0x1000
            || is_movs && (phys_src & 0xFFF) as u64 + bytes > 0x1000
        {
            return Ok(false);
        }

        let mut n = u64::min(*count, count_until_end_of_page64(direction, bytes, phys_dst));
        if is_movs {
            n = u64::min(n, count_until_end_of_page64(direction, bytes, phys_src));
        }
        if n == 0 {
            return Ok(false);
        }

        if is_movs {
            // A forward copy that overlaps must not be batched (memmove differs
            // from the element-by-element x86 semantics).
            let len = (n * bytes) as u32;
            let overlap = if phys_src < phys_dst {
                phys_dst - phys_src < len && direction == 1
            }
            else if phys_src > phys_dst {
                phys_src - phys_dst < len && direction == -1
            }
            else {
                false
            };
            if overlap {
                return Ok(false);
            }
        }

        let mut ps = phys_src;
        let mut pd = phys_dst;
        if direction == -1 {
            ps = ps.wrapping_sub(((n - 1) * bytes) as u32);
            pd = pd.wrapping_sub(((n - 1) * bytes) as u32);
        }

        jit::jit_dirty_page(Page::page_of(pd));
        if is_movs {
            memory::memcpy_no_mmap_or_dirty_check(ps, pd, (n * bytes) as u32);
        }
        else {
            let value = read_reg(RAX, elem_size, has_rex);
            match bytes {
                1 => memory::memset_no_mmap_or_dirty_check(pd, value as u8, n as u32),
                2 =>
                {
                    for i in 0..n as u32 {
                        memory::write16_no_mmap_or_dirty_check(pd + i * 2, value as i32);
                    }
                },
                4 =>
                {
                    for i in 0..n as u32 {
                        memory::write32_no_mmap_or_dirty_check(pd + i * 4, value as i32);
                    }
                },
                _ =>
                {
                    for i in 0..n as u32 {
                        memory::write64_no_mmap_or_dirty_check(pd + i * 8, value);
                    }
                },
            }
        }

        let step = (n as i64).wrapping_mul(delta) as u64;
        if is_movs {
            write_reg64(RSI as i32, rsi.wrapping_add(step));
        }
        write_reg64(RDI as i32, rdi.wrapping_add(step));
        *count -= n;
        write_reg64(RCX as i32, *count);
    }
    Ok(true)
}

pub(crate) unsafe fn mem_probe_write(address: u64, size: OpSize) -> OrPageFault<()> {
    for offset in 0..size.bits() / 8 {
        translate_address_64(address + offset as u64, true, *cpl == 3)?;
    }
    Ok(())
}

#[inline(always)]
pub(crate) unsafe fn probe_operand_write(operand: Operand, size: OpSize) -> OrPageFault<()> {
    if let Operand::Mem(address) = operand {
        mem_probe_write(address, size)?;
    }
    Ok(())
}

#[inline(always)]
pub(crate) unsafe fn decode_operand_after_modrm(
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
        // 0x67: in long mode the address is computed in 32 bits and
        // zero-extended. (The decode cache already does this in `jmem_addr`.)
        if pfx.addrsize_32 {
            address &= 0xFFFF_FFFF;
        }
        Operand::Mem(address)
    };
    Ok((modrm, operand))
}

#[inline(always)]
pub(crate) unsafe fn decode_operand_with_trailing(
    pfx: &Prefixes,
    trailing_bytes: u64,
) -> OrPageFault<(Modrm, Operand)> {
    let modrm = decode_modrm(pfx)?;
    decode_operand_after_modrm(pfx, modrm, trailing_bytes)
}

#[inline(always)]
pub(crate) unsafe fn decode_operand(pfx: &Prefixes) -> OrPageFault<(Modrm, Operand)> {
    decode_operand_with_trailing(pfx, 0)
}

#[inline(always)]
pub(crate) unsafe fn read_operand(operand: Operand, size: OpSize, has_rex: bool) -> OrPageFault<u64> {
    match operand {
        Operand::Reg(r) => Ok(read_reg(r, size, has_rex)),
        Operand::Mem(address) => mem_read(address, size),
    }
}

#[inline(always)]
pub(crate) unsafe fn write_operand(
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

#[derive(Copy, Clone, PartialEq)]
pub(crate) enum AluOp {
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

pub(crate) unsafe fn segment_base(selector: u16) -> u64 {
    let index = (selector >> 3) as u64;
    if selector & !7 == 0 || selector >> 2 & 1 != 0 {
        return 0;
    }
    let descriptor = match mem_read(GDTR_BASE + index * 8, OpSize::S64) {
        Ok(value) => value,
        Err(_) => return 0,
    };
    (descriptor >> 16 & 0xFFFF) | (descriptor >> 32 & 0xFF) << 16 | (descriptor >> 56 & 0xFF) << 24
}

// LSS/LFS/LGS: load a far pointer (offset then selector) into a register and
// the matching segment.
pub(crate) unsafe fn load_far(segment: i32, size: OpSize, pfx: &Prefixes) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let Operand::Mem(address) = operand else {
        crate::cpu::core::trigger_ud();
        return Ok(());
    };
    let bytes = (size.bits() / 8) as u64;
    let offset = mem_read(address, size)? & size.mask();
    let selector = mem_read(address + bytes, OpSize::S16)? as u16;
    if selector & !7 == 0 {
        crate::cpu::core::trigger_ud();
        return Ok(());
    }
    write_reg(modrm.reg, size, pfx.has_rex(), offset);
    *sreg.offset(segment as isize) = selector;
    // In long mode FS/GS base comes from the MSR (WRFSBASE/arch_prctl), not the
    // descriptor. cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/pop
    if !*long_mode
    {
        let base = segment_base(selector);
        match segment {
            4 => *fs_base = base,
            5 => *gs_base = base,
            _ => {},
        }
    }
    Ok(())
}

// PUSH/POP FS and GS (0F A0/A1/A8/A9).
// cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/push
pub(crate) unsafe fn push_segment(segment: i32) -> OrPageFault<()> {
    let value = *sreg.offset(segment as isize) as u64;
    push64(value)
}

pub(crate) unsafe fn pop_segment(segment: i32) -> OrPageFault<()> {
    let selector = pop64()? as u16;
    *sreg.offset(segment as isize) = selector;
    // In long mode FS/GS base comes from the MSR (WRFSBASE/arch_prctl), not the
    // descriptor. cf. Intel SDM Vol. 2: https://www.felixcloutier.com/x86/pop
    if !*long_mode
    {
        let base = segment_base(selector);
        match segment {
            4 => *fs_base = base,
            5 => *gs_base = base,
            _ => {},
        }
    }
    Ok(())
}

// --- SSE/SSE2/SSSE3 and x87 ---
// Semantics: Intel SDM Vol. 2 (AMD64 APM cross-check); x87 via the v86 interpreter.

// SSE arithmetic, PS/PD/SS/SD forms.
// cf. Intel SDM Vol. 2 (https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html)

pub(crate) fn alu_op_of(opcode: u8) -> AluOp {
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

#[inline(always)]
pub(crate) fn parity_even(value: u64) -> bool { (value as u8).count_ones() % 2 == 0 }

#[inline(always)]
pub(crate) unsafe fn set_szp_flags(result: u64, size: OpSize) {
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

#[inline(always)]
pub(crate) unsafe fn set_add_flags(result: u64, dst: u64, src: u64, size: OpSize) {
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

#[inline(always)]
pub(crate) unsafe fn set_sub_flags(result: u64, dst: u64, src: u64, size: OpSize) {
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

#[inline(always)]
pub(crate) unsafe fn set_logic_flags(result: u64, size: OpSize) {
    set_szp_flags(result, size);
    let mut f = *flags as u32;
    f &= !(FLAG_CF | FLAG_OF);
    *flags = f as i32;
    *flags_changed = 0;
}

#[inline(always)]
pub(crate) unsafe fn alu(op: AluOp, dst: u64, src: u64, size: OpSize) -> u64 {
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

pub(crate) unsafe fn adc_sbb_value(dst: u64, src: u64, size: OpSize, subtract: bool) -> u64 {
    let mask = size.mask();
    let dst = dst & mask;
    let src = src & mask;
    let signed_dst = sign_extend(dst, size.bits());
    let signed_src = sign_extend(src, size.bits());
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

pub(crate) unsafe fn imul_value(lhs: u64, rhs: u64, size: OpSize) -> u64 {
    let (product, overflow) = crate::cpu::interp::misc_instr::mul_wide(lhs, rhs, size, true);
    *flags &= !(FLAG_CF | FLAG_OF) as i32;
    if overflow {
        *flags |= (FLAG_CF | FLAG_OF) as i32;
    }
    product as u64 & size.mask()
}

// accumulator (AL/AX/EAX/RAX) with an immediate
pub(crate) unsafe fn write_quot_rem(quotient: u64, remainder: u64, size: OpSize) {
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
pub(crate) unsafe fn mul_wide(acc: u64, src: u64, size: OpSize, signed: bool) {
    let (product, overflow) = crate::cpu::interp::misc_instr::mul_wide(acc, src, size, signed);

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
pub(crate) unsafe fn div_wide(src: u64, size: OpSize, signed: bool) {
    let dividend: u128 = match size {
        OpSize::S8 => read_reg(RAX, OpSize::S16, false) as u16 as u128,
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

    match crate::cpu::interp::misc_instr::div_wide(dividend, src, size, signed) {
        None => crate::cpu::core::trigger_de(),
        Some((quotient, remainder)) => write_quot_rem(quotient, remainder, size),
    }
}

// F6/F7 group 3: TEST/NOT/NEG/MUL/IMUL/DIV/IDIV.
#[inline(always)]
pub(crate) unsafe fn group3(opcode: u8, pfx: &Prefixes, operand_size: OpSize) -> OrPageFault<()> {
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
        _ => crate::cpu::core::trigger_ud(),
    }
    Ok(())
}

// C0/C1/D0/D1/D2/D3 group 2: ROL/ROR/RCL/RCR/SHL/SHR/SAR.
#[inline(always)]
pub(crate) unsafe fn group2(opcode: u8, pfx: &Prefixes, operand_size: OpSize) -> OrPageFault<()> {
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
            crate::cpu::core::trigger_ud();
            return Ok(());
        },
    };

    // Rotates (groups 0-3) leave SF/ZF/PF; shifts (4-7) also set them.
    if modrm.reg & 7 >= 4 {
        set_logic_flags(result, size);
    }
    else {
        *flags_changed = 0;
    }
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
pub(crate) unsafe fn bit_test(
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

#[inline(always)]
pub(crate) unsafe fn alu_acc_imm(op: AluOp, size: OpSize, imm: u64) -> OrPageFault<()> {
    let has_rex = false; // no REX on the accumulator immediate forms
    let dst = read_reg(RAX, size, has_rex);
    let result = alu(op, dst, imm, size);
    if op != AluOp::Cmp {
        write_reg(RAX, size, has_rex, result);
    }
    Ok(())
}

#[inline(always)]
pub(crate) unsafe fn alu_rm_r_rm(op: AluOp, opcode: u8, pfx: &Prefixes, size: OpSize) -> OrPageFault<()> {
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

#[inline(always)]
pub(crate) unsafe fn fetch_imm(size: OpSize) -> OrPageFault<u64> {
    Ok(match size {
        OpSize::S8 => fetch8()? as u64,
        OpSize::S16 => fetch16()? as u64,
        // imm32 is sign-extended to 64 bits in long mode
        OpSize::S32 => fetch32()? as u64,
        OpSize::S64 => fetch32()? as i32 as i64 as u64,
    })
}

#[inline(always)]
pub(crate) unsafe fn fetch_imm8_signed(size: OpSize) -> OrPageFault<u64> {
    Ok(fetch8()? as i8 as i64 as u64 & size.mask())
}

#[inline(always)]
pub(crate) unsafe fn alu_group1(opcode: u8, pfx: &Prefixes, size: OpSize) -> OrPageFault<()> {
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
            crate::cpu::core::trigger_ud();
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

#[inline(always)]
pub(crate) unsafe fn test_rm_r(pfx: &Prefixes, size: OpSize) -> OrPageFault<()> {
    let (modrm, operand) = decode_operand(pfx)?;
    let a = read_operand(operand, size, pfx.has_rex())?;
    let b = read_reg(modrm.reg, size, pfx.has_rex());
    set_logic_flags(a & b & size.mask(), size);
    Ok(())
}

pub(crate) unsafe fn push64(value: u64) -> OrPageFault<()> {
    let rsp = read_reg64(RSP as i32).wrapping_sub(8);
    mem_write(rsp, OpSize::S64, value)?;
    write_reg64(RSP as i32, rsp);
    Ok(())
}

pub(crate) unsafe fn pop64() -> OrPageFault<u64> {
    let rsp = read_reg64(RSP as i32);
    let value = mem_read(rsp, OpSize::S64)?;
    write_reg64(RSP as i32, rsp.wrapping_add(8));
    Ok(value)
}

pub(crate) unsafe fn pop_size(size: OpSize) -> OrPageFault<u64> {
    let rsp = read_reg64(RSP as i32);
    let value = mem_read(rsp, size)?;
    write_reg64(RSP as i32, rsp.wrapping_add(size.bits() as u64 / 8));
    Ok(value)
}

// --- x87 ------------------------------------------------------------------
