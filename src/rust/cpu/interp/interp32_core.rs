//! Decode-cache support for the 16/32-bit interpreter: the protected-mode
//! counterpart of the same block in `interp64.rs`.
//!
//! Blocks are decoded by `jit32::decode_block_parts` (pure) into the shared
//! `decode_cache::Instruction`; this file owns the paging side
//! (`decode_block_at`, `validate_block`, the cache table) and the executor
//! (`jexec32`) that runs a block in protected mode. `interp32_run` calls
//! `decode_cache_run` and falls back to `run_instruction` when a block cannot be
//! cached.
//!
//! Flat segmentation (DS/SS/CS bases zero) is required, because `jmem_addr32`
//! adds no segment base; FS/GS overrides still contribute theirs.

use crate::cpu::core::{CS, FS, GS, SS};
use crate::cpu::interp::decode_cache::{
    ArithOp as JArithOp, DecEntry, DecodeHost, DecodedInstr, Instruction as JInstr, JStep,
    Mem as JMem, ShiftKind,
};
use crate::cpu::global_pointers::*;
use crate::memory;
use crate::paging::OrPageFault;

pub(crate) static mut DECODE_CACHE32: *mut Vec<DecEntry<Instr32>> = std::ptr::null_mut();

const DECODE_CACHE_MAX: usize = 1 << 13;
const DECODE_CACHE_MIN: usize = 1 << 11;

#[derive(Copy, Clone)]
pub struct Instr32(pub JInstr);

impl DecodedInstr for Instr32 {
    #[inline(always)]
    fn is_supported(&self) -> bool {
        use JInstr::*;
        // Guards against a decoder change emitting something jexec32 can't run:
        // exec_decoded would otherwise skip it silently.
        matches!(
            self.0,
            MovRegImm { .. }
                | MovRegReg { .. }
                | MovRegMem { .. }
                | MovMemReg { .. }
                | MovMemImm { .. }
                | ArithRegReg { .. }
                | ArithRegImm { .. }
                | ArithRegMem { .. }
                | ArithMemReg { .. }
                | ArithMemImm { .. }
                | IncDecReg { .. }
                | IncDecMem { .. }
                | NotReg { .. }
                | NotMem { .. }
                | NegReg { .. }
                | NegMem { .. }
                | CmovRegReg { .. }
                | CmovRegMem { .. }
                | SetccReg { .. }
                | SetccMem { .. }
                | ShiftReg { .. }
                | ShiftRegCl { .. }
                | ShiftMem { .. }
                | ShiftMemCl { .. }
                | Lea { .. }
                | PushReg { .. }
                | PopReg { .. }
                | Call { .. }
                | CallReg { .. }
                | CallMem { .. }
                | Jmp { .. }
                | JmpReg { .. }
                | JmpMem { .. }
                | Ret { .. }
                | Leave
                | XchgRegReg { .. }
                | Nop
                | Hlt
                | Jcc { .. }
        )
    }
    #[inline(always)]
    fn jcc(&self) -> Option<(u8, u64, u64)> {
        if let JInstr::Jcc { code, target, fallthrough } = self.0 {
            unsafe { materialize_condition_flags() };
            Some((code, target, fallthrough))
        }
        else {
            None
        }
    }
    #[inline(always)]
    fn exec(&self) -> OrPageFault<JStep> { unsafe { jexec32(&self.0) } }
}

pub struct Interp32;

impl DecodeHost for Interp32 {
    type Instr = Instr32;
    #[inline(always)]
    unsafe fn set_previous_ip(next: u64) { *previous_ip = next as i32; }
    #[inline(always)]
    unsafe fn set_ip(next: u64) { *instruction_pointer = next as i32; }
    // Instruction accounting happens once per batch in interp32_run, so that
    // *instruction_counter advances exactly as it did before the cache existed
    // and nothing observes a partially-updated counter mid-batch.
    #[inline(always)]
    unsafe fn retire(n: u32) {
        *instruction_counter = (*instruction_counter).wrapping_add(n);
    }
    #[inline(always)]
    unsafe fn page_write_count(phys_page: u32) -> u32 {
        crate::cpu::core::page_write_count(phys_page)
    }
    unsafe fn validate_block(start_rip: u64, bytes: &[u8]) -> bool {
        validate_block(start_rip, bytes)
    }
    unsafe fn decode_block_at(
        start_rip: u64,
    ) -> Option<(Vec<Instr32>, Vec<u64>, u64, Vec<u8>, u32)> {
        decode_block_at(start_rip)
    }
}


// --- engine ----------------------------------------------------------------

// The 16/32-bit interpreter computes arithmetic flags lazily: arith.rs records
// last_op1/last_result/flags_changed and the `test_*` getters recompute on
// demand. The shared decode-cache loop evaluates Jcc via `condition()`, which
// reads `flags` directly, so materialize the affected flag bits first.
#[inline(always)]
unsafe fn materialize_condition_flags() {
    use crate::cpu::core::{
        FLAG_ADJUST, FLAG_CARRY, FLAG_OVERFLOW, FLAG_PARITY, FLAG_SIGN, FLAG_ZERO,
    };
    use crate::cpu::interp::misc_instr::{getaf, getcf, getof, getpf, getsf, getzf};
    const CC: i32 = FLAG_CARRY | FLAG_ZERO | FLAG_SIGN | FLAG_OVERFLOW | FLAG_PARITY | FLAG_ADJUST;
    *flags = *flags & !CC
        | (getcf() as i32) * FLAG_CARRY
        | (getzf() as i32) * FLAG_ZERO
        | (getsf() as i32) * FLAG_SIGN
        | (getof() as i32) * FLAG_OVERFLOW
        | (getpf() as i32) * FLAG_PARITY
        | (getaf() as i32) * FLAG_ADJUST;
    *flags_changed = *flags_changed & !CC;
}

#[inline(always)]
unsafe fn rd(r: u8, width: u8) -> i32 {
    match width {
        8 => crate::cpu::core::read_reg8(r as i32),
        16 => crate::cpu::core::read_reg16(r as i32),
        _ => crate::cpu::core::read_reg32(r as i32),
    }
}

#[inline(always)]
unsafe fn wr(r: u8, width: u8, value: i32) {
    match width {
        8 => crate::cpu::core::write_reg8(r as i32, value),
        16 => crate::cpu::core::write_reg16(r as i32, value),
        _ => crate::cpu::core::write_reg32(r as i32, value),
    }
}

/// 32-bit paging translation plus dirty-page bookkeeping, like the generated
/// interpreter's safe_read* / safe_write*.
#[inline(always)]
unsafe fn rm(addr: u64, width: u8) -> OrPageFault<i32> {
    let a = addr as u32 as i32;
    match width {
        8 => crate::cpu::core::safe_read8(a),
        16 => crate::cpu::core::safe_read16(a),
        _ => crate::cpu::core::safe_read32s(a),
    }
}

#[inline(always)]
unsafe fn wm(addr: u64, width: u8, value: i32) -> OrPageFault<()> {
    let a = addr as u32 as i32;
    match width {
        8 => crate::cpu::core::safe_write8(a, value),
        16 => crate::cpu::core::safe_write16(a, value),
        _ => crate::cpu::core::safe_write32(a, value),
    }
}

#[inline(always)]
fn wmask(width: u8) -> i32 {
    match width {
        8 => 0xFF,
        16 => 0xFFFF,
        _ => -1,
    }
}

macro_rules! alu_by_width {
    ($width:expr, $a:expr, $b:expr, $f8:ident, $f16:ident, $f32:ident) => {
        match $width {
            8 => crate::cpu::interp::arith::$f8($a, $b),
            16 => crate::cpu::interp::arith::$f16($a, $b),
            _ => crate::cpu::interp::arith::$f32($a, $b),
        }
    };
}

/// dst = dst op src, with the 32-bit interpreter's lazy flag bookkeeping.
/// Cmp and Test only set flags.
#[inline(always)]
unsafe fn jalu32(op: JArithOp, dst: u8, width: u8, d: i32, s: i32) {
    let m = wmask(width);
    let d = d & m;
    let s = s & m;
    match op {
        JArithOp::Add => wr(dst, width, alu_by_width!(width, d, s, add8, add16, add32)),
        JArithOp::Or => wr(dst, width, alu_by_width!(width, d, s, or8, or16, or32)),
        JArithOp::Adc => wr(dst, width, alu_by_width!(width, d, s, adc8, adc16, adc32)),
        JArithOp::Sbb => wr(dst, width, alu_by_width!(width, d, s, sbb8, sbb16, sbb32)),
        JArithOp::And => wr(dst, width, alu_by_width!(width, d, s, and8, and16, and32)),
        JArithOp::Sub => wr(dst, width, alu_by_width!(width, d, s, sub8, sub16, sub32)),
        JArithOp::Xor => wr(dst, width, alu_by_width!(width, d, s, xor8, xor16, xor32)),
        JArithOp::Cmp => {
            alu_by_width!(width, d, s, cmp8, cmp16, cmp32);
        },
        JArithOp::Test => match width {
            8 => crate::cpu::interp::arith::test8(d, s),
            16 => crate::cpu::interp::arith::test16(d, s),
            _ => crate::cpu::interp::arith::test32(d, s),
        },
    }
}

/// r/m = r/m op src. Cmp and Test only set flags.
#[inline(always)]
unsafe fn jalu32_mem(op: JArithOp, addr: u64, width: u8, d: i32, s: i32) -> OrPageFault<()> {
    let m = wmask(width);
    let d = d & m;
    let s = s & m;
    match op {
        JArithOp::Add => {
            let r = alu_by_width!(width, d, s, add8, add16, add32);
            wm(addr, width, r)
        },
        JArithOp::Or => {
            let r = alu_by_width!(width, d, s, or8, or16, or32);
            wm(addr, width, r)
        },
        JArithOp::Adc => {
            let r = alu_by_width!(width, d, s, adc8, adc16, adc32);
            wm(addr, width, r)
        },
        JArithOp::Sbb => {
            let r = alu_by_width!(width, d, s, sbb8, sbb16, sbb32);
            wm(addr, width, r)
        },
        JArithOp::And => {
            let r = alu_by_width!(width, d, s, and8, and16, and32);
            wm(addr, width, r)
        },
        JArithOp::Sub => {
            let r = alu_by_width!(width, d, s, sub8, sub16, sub32);
            wm(addr, width, r)
        },
        JArithOp::Xor => {
            let r = alu_by_width!(width, d, s, xor8, xor16, xor32);
            wm(addr, width, r)
        },
        JArithOp::Cmp => {
            alu_by_width!(width, d, s, cmp8, cmp16, cmp32);
            Ok(())
        },
        JArithOp::Test => {
            match width {
                8 => crate::cpu::interp::arith::test8(d, s),
                16 => crate::cpu::interp::arith::test16(d, s),
                _ => crate::cpu::interp::arith::test32(d, s),
            }
            Ok(())
        },
    }
}

#[inline(always)]
unsafe fn shift32(kind: ShiftKind, width: u8, v: i32, count: i32) -> i32 {
    let count = count & 31;
    match kind {
        ShiftKind::Shl => alu_by_width!(width, v & wmask(width), count, shl8, shl16, shl32),
        ShiftKind::Shr => alu_by_width!(width, v & wmask(width), count, shr8, shr16, shr32),
        ShiftKind::Sar => alu_by_width!(width, v & wmask(width), count, sar8, sar16, sar32),
        ShiftKind::Rol => alu_by_width!(width, v & wmask(width), count, rol8, rol16, rol32),
        ShiftKind::Ror => alu_by_width!(width, v & wmask(width), count, ror8, ror16, ror32),
    }
}

/// 32-bit effective address. Only valid under flat segmentation (guaranteed by
/// the caller); FS/GS overrides still add their base. A `67` prefix selects the
/// 16-bit form, which reads the low halves of the base/index registers and wraps
/// the sum at 64 KiB before the segment base is added.
unsafe fn jmem_addr32(mem: &JMem) -> u64 {
    let a16 = mem.addr_size == 16;
    let mut addr: u64 = 0;
    if let Some(base) = mem.base {
        addr = if a16 {
            crate::cpu::core::read_reg16(base as i32) as u16 as u64
        }
        else {
            crate::cpu::core::read_reg32(base as i32) as u32 as u64
        };
    }
    if let Some(index) = mem.index {
        let mut v = if a16 {
            crate::cpu::core::read_reg16(index as i32) as u16 as u64
        }
        else {
            crate::cpu::core::read_reg32(index as i32) as u32 as u64
        };
        if mem.scale > 0 {
            v = v.wrapping_mul(1u64 << mem.scale);
        }
        addr = addr.wrapping_add(v);
    }
    if mem.disp != 0 {
        addr = addr.wrapping_add(mem.disp as u64);
    }
    addr &= if a16 { 0xFFFF } else { 0xFFFF_FFFF };
    match mem.segment {
        Some(0x64) => addr = addr.wrapping_add(
            *segment_offsets.offset(FS as isize) as u32 as u64
        ),
        Some(0x65) => addr = addr.wrapping_add(
            *segment_offsets.offset(GS as isize) as u32 as u64
        ),
        _ => {},
    }
    addr
}

#[inline(always)]
unsafe fn cond32(code: u8) -> bool {
    materialize_condition_flags();
    crate::cpu::core::condition(code)
}

pub(crate) unsafe fn jexec32(instr: &JInstr) -> OrPageFault<JStep> {
    use crate::cpu::interp::misc_instr::{adjust_stack_reg, pop32s, push32};
    use JInstr::*;
    match *instr {
        MovRegImm { r, value, width } => wr(r, width, value as i32),
        MovRegReg { dst, src, width } => {
            let v = rd(src, width);
            wr(dst, width, v);
        },
        MovRegMem { dst, mem, width } => {
            let v = rm(jmem_addr32(&mem), width)?;
            wr(dst, width, v);
        },
        MovMemReg { mem, src, width } => {
            let v = rd(src, width);
            wm(jmem_addr32(&mem), width, v)?;
        },
        MovMemImm { mem, value, width } => {
            wm(jmem_addr32(&mem), width, value as i32)?;
        },
        MovExtendReg { dst, src, src_width, dst_width, signed, .. } => {
            let v = rd(src, src_width);
            let v = if signed {
                match src_width {
                    8 => v as u8 as i8 as i32,
                    16 => v as u16 as i16 as i32,
                    _ => v,
                }
            }
            else {
                v & wmask(src_width)
            };
            wr(dst, dst_width, v);
        },
        MovExtendMem { dst, mem, src_width, dst_width, signed } => {
            let v = rm(jmem_addr32(&mem), src_width)?;
            let v = if signed {
                match src_width {
                    8 => v as u8 as i8 as i32,
                    16 => v as u16 as i16 as i32,
                    _ => v,
                }
            }
            else {
                v & wmask(src_width)
            };
            wr(dst, dst_width, v);
        },
        ArithRegReg { op, dst, src, width } => {
            let d = rd(dst, width);
            let s = rd(src, width);
            jalu32(op, dst, width, d, s);
        },
        ArithRegImm { op, r, value, width } => {
            let d = rd(r, width);
            jalu32(op, r, width, d, value as i32);
        },
        ArithRegMem { op, dst, mem, width } => {
            let d = rd(dst, width);
            let s = rm(jmem_addr32(&mem), width)?;
            jalu32(op, dst, width, d, s);
        },
        ArithMemReg { op, mem, src, width } => {
            let a = jmem_addr32(&mem);
            let d = rm(a, width)?;
            let s = rd(src, width);
            jalu32_mem(op, a, width, d, s)?;
        },
        ArithMemImm { op, mem, value, width } => {
            let a = jmem_addr32(&mem);
            let d = rm(a, width)?;
            jalu32_mem(op, a, width, d, value as i32)?;
        },
        IncDecReg { r, width, decrement } => {
            use crate::cpu::interp::arith::{dec16, dec32, dec8, inc16, inc32, inc8};
            let v = rd(r, width);
            let r2 = match (width, decrement) {
                (8, false) => inc8(v),
                (8, true) => dec8(v),
                (16, false) => inc16(v),
                (16, true) => dec16(v),
                (_, false) => inc32(v),
                (_, true) => dec32(v),
            };
            wr(r, width, r2);
        },
        IncDecMem { mem, width, decrement } => {
            use crate::cpu::interp::arith::{dec16, dec32, dec8, inc16, inc32, inc8};
            let a = jmem_addr32(&mem);
            let v = rm(a, width)?;
            let r2 = match (width, decrement) {
                (8, false) => inc8(v),
                (8, true) => dec8(v),
                (16, false) => inc16(v),
                (16, true) => dec16(v),
                (_, false) => inc32(v),
                (_, true) => dec32(v),
            };
            wm(a, width, r2)?;
        },
        NotReg { r, width } => {
            let v = rd(r, width);
            wr(r, width, !v & wmask(width));
        },
        NotMem { mem, width } => {
            let a = jmem_addr32(&mem);
            let v = rm(a, width)?;
            // Mask to the operand width: safe_write16/32 assert their value fits.
            wm(a, width, !v & wmask(width))?;
        },
        NegReg { r, width } => {
            use crate::cpu::interp::arith::{neg16, neg32, neg8};
            let v = rd(r, width);
            let r2 = match width {
                8 => neg8(v),
                16 => neg16(v),
                _ => neg32(v),
            };
            wr(r, width, r2);
        },
        NegMem { mem, width } => {
            use crate::cpu::interp::arith::{neg16, neg32, neg8};
            let a = jmem_addr32(&mem);
            let v = rm(a, width)?;
            let r2 = match width {
                8 => neg8(v),
                16 => neg16(v),
                _ => neg32(v),
            };
            wm(a, width, r2)?;
        },
        CmovRegReg { code, dst, src, width } => {
            if cond32(code) {
                let v = rd(src, width);
                wr(dst, width, v);
            }
        },
        CmovRegMem { code, dst, mem, width } => {
            let a = jmem_addr32(&mem);
            let v = rm(a, width)?;
            if cond32(code) {
                wr(dst, width, v);
            }
        },
        SetccReg { code, dst, .. } => {
            crate::cpu::core::write_reg8(dst as i32, cond32(code) as i32);
        },
        SetccMem { code, mem } => {
            let a = jmem_addr32(&mem);
            wm(a, 8, cond32(code) as i32)?;
        },
        ShiftReg { kind, r, width, count } => {
            let v = shift32(kind, width, rd(r, width), count as i32);
            wr(r, width, v);
        },
        ShiftRegCl { kind, r, width } => {
            let count = crate::cpu::core::read_reg8(1);
            let v = shift32(kind, width, rd(r, width), count);
            wr(r, width, v);
        },
        ShiftMem { kind, mem, width, count } => {
            let a = jmem_addr32(&mem);
            let v = shift32(kind, width, rm(a, width)?, count as i32);
            wm(a, width, v)?;
        },
        ShiftMemCl { kind, mem, width } => {
            let a = jmem_addr32(&mem);
            let count = crate::cpu::core::read_reg8(1);
            let v = shift32(kind, width, rm(a, width)?, count);
            wm(a, width, v)?;
        },
        Lea { dst, mem, width } => {
            let a = jmem_addr32(&mem) as u32 as i32;
            wr(dst, width, a);
        },
        PushReg { r } => push32(crate::cpu::core::read_reg32(r as i32))?,
        PopReg { r } => {
            let v = pop32s()?;
            crate::cpu::core::write_reg32(r as i32, v);
        },
        PushImm { value } => push32(value as i32)?,
        Call { target, return_address } => {
            push32(return_address as u32 as i32)?;
            *instruction_pointer = target as u32 as i32;
            return Ok(JStep::Stop);
        },
        CallReg { r, return_address } => {
            push32(return_address as u32 as i32)?;
            *instruction_pointer = crate::cpu::core::read_reg32(r as i32);
            return Ok(JStep::Stop);
        },
        CallMem { mem, return_address } => {
            let target = rm(jmem_addr32(&mem), 32)?;
            push32(return_address as u32 as i32)?;
            *instruction_pointer = target;
            return Ok(JStep::Stop);
        },
        Jmp { target } => {
            *instruction_pointer = target as u32 as i32;
            return Ok(JStep::Stop);
        },
        JmpReg { r } => {
            *instruction_pointer = crate::cpu::core::read_reg32(r as i32);
            return Ok(JStep::Stop);
        },
        JmpMem { mem } => {
            *instruction_pointer = rm(jmem_addr32(&mem), 32)?;
            return Ok(JStep::Stop);
        },
        Ret { adjustment } => {
            let ip = pop32s()?;
            *instruction_pointer = (*segment_offsets.offset(CS as isize)).wrapping_add(ip);
            if adjustment != 0 {
                adjust_stack_reg(adjustment as i32);
            }
            return Ok(JStep::Stop);
        },
        Leave => {
            // leave == mov esp, ebp; pop ebp: the stack pointer comes *from*
            // EBP, it is not incremented.
            let old_bp =
                if *stack_size_32 { crate::cpu::core::read_reg32(5) } else { crate::cpu::core::read_reg16(5) };
            let new_bp = crate::cpu::core::safe_read32s(
                (*segment_offsets.offset(SS as isize)).wrapping_add(old_bp),
            )?;
            crate::cpu::core::set_stack_reg(old_bp.wrapping_add(4));
            crate::cpu::core::write_reg32(5, new_bp);
        },
        XchgRegReg { a, b, width } => {
            let va = rd(a, width);
            let vb = rd(b, width);
            wr(a, width, vb);
            wr(b, width, va);
        },
        Nop => {},
        Hlt => {
            if *cpl != 0 {
                crate::cpu::core::trigger_gp(0);
                return Ok(JStep::Stop);
            }
            *in_hlt = true;
            return Ok(JStep::Stop);
        },
        _ => return Err(()),
    }
    Ok(JStep::Continue)
}

// --- cache -----------------------------------------------------------------

pub(crate) unsafe fn decode_cache() -> &'static mut Vec<DecEntry<Instr32>> {
    if DECODE_CACHE32.is_null() {
        crate::cpu::core::page_write_version_init();
        let n = if *memory_size >= 64 * 1024 * 1024 {
            DECODE_CACHE_MAX
        }
        else {
            DECODE_CACHE_MIN
        };
        let mut v: Vec<DecEntry<Instr32>> = Vec::with_capacity(n);
        for _ in 0..n {
            v.push(DecEntry::default());
        }
        DECODE_CACHE32 = Box::into_raw(Box::new(v));
    }
    &mut *DECODE_CACHE32
}

pub(crate) unsafe fn clear_decode_cache() {
    if DECODE_CACHE32.is_null() {
        return;
    }
    for entry in &mut *DECODE_CACHE32 {
        entry.valid = false;
    }
}

// Translate a 32-bit virtual address with the current CR3/segments.
unsafe fn translate_read(vaddr: u32) -> Option<u32> {
    crate::cpu::core::translate_address_read_no_side_effects(vaddr as i32).ok()
}

unsafe fn validate_block(start_rip: u64, bytes: &[u8]) -> bool {
    let phys = match translate_read(start_rip as u32) {
        Some(p) => p,
        None => return false,
    };
    let phys_base = phys & 0xFFFFF000;
    if memory::in_mapped_range(phys_base) {
        return false;
    }
    let off = (start_rip & 0xFFF) as usize;
    if off + bytes.len() > 0x1000 {
        return false;
    }
    let page =
        std::slice::from_raw_parts(memory::mem8.add(phys_base as usize) as *const u8, 0x1000);
    page[off..off + bytes.len()] == *bytes
}

unsafe fn decode_block_at(start_rip: u64) -> Option<(Vec<Instr32>, Vec<u64>, u64, Vec<u8>, u32)> {
    // interp32_run also serves real mode and 16-bit protected mode, where the
    // ModRM/SIB forms and the default operand size differ. Decode 32-bit only.
    if !*is_32 {
        return None;
    }
    // The effective address omits the DS/SS/CS base, so only cache blocks while
    // segmentation is flat (the common case; FS/GS overrides still work).
    if !crate::cpu::core::has_flat_segmentation() {
        return None;
    }
    let phys = translate_read(start_rip as u32)?;
    let phys_base = phys & 0xFFFFF000;
    if memory::in_mapped_range(phys_base) {
        return None;
    }
    let off = (start_rip & 0xFFF) as usize;
    let page =
        std::slice::from_raw_parts(memory::mem8.add(phys_base as usize) as *const u8, 0x1000);
    let bytes = &page[off..off + (0x1000 - off).min(page.len() - off)];
    let (instrs, rips, end_rip) = crate::cpu::interp::jit32::decode_block_parts(start_rip, bytes).ok()?;
    if instrs.is_empty() || end_rip <= start_rip {
        return None;
    }
    let len = (end_rip - start_rip) as usize;
    let block_bytes = bytes[..len].to_vec();
    Some((instrs.into_iter().map(Instr32).collect(), rips, end_rip, block_bytes, phys_base >> 12))
}

// Lets the differential tests run a guest with and without the cache in one
// process and compare the results.
pub static mut DISABLE_32_DECODE_CACHE: bool = false;

#[no_mangle]
pub fn set_dbg_disable_32_decode_cache(value: u32) {
    unsafe { DISABLE_32_DECODE_CACHE = value != 0; }
}

// Test hook: how many blocks the cache executed. The differential tests assert
// this grows, so a harness that stops reaching the cache fails loudly instead of
// passing vacuously. Only compiled into debug builds.
#[cfg(debug_assertions)]
static mut DECODE_CACHE_HITS: u32 = 0;

#[cfg(debug_assertions)]
#[no_mangle]
pub fn dbg_decode_cache_hits() -> u32 { unsafe { DECODE_CACHE_HITS } }

pub(crate) unsafe fn decode_cache_run() -> Option<u32> {
    if unsafe { DISABLE_32_DECODE_CACHE } {
        return None;
    }
    if crate::cpu::core::interrupt_shadow() != 0 {
        return None;
    }
    let start_rip = *instruction_pointer as u32 as u64;
    let cache = decode_cache();
    let r = crate::cpu::interp::decode_cache::cache_run::<Interp32>(
        cache.as_mut_slice(),
        start_rip,
        *cpl == 3,
        crate::cpu::core::tlb64_generation,
    );
    #[cfg(debug_assertions)]
    if r.is_some() {
        DECODE_CACHE_HITS = DECODE_CACHE_HITS.wrapping_add(1);
    }
    r
}
