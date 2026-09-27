#![allow(non_upper_case_globals)]

use crate::cpu::cpu::reg128;
use crate::softfloat::F80;
use crate::state_flags::CachedStateFlags;

pub const reg8: *mut u8 = 64 as *mut u8;
pub const reg16: *mut u16 = 64 as *mut u16;
pub const reg32: *mut i32 = 64 as *mut i32;

pub const reg_high: *mut u32 = 128 as *mut u32; // r0-r7 high 32 bits

pub const reg64_ext: *mut u64 = 160 as *mut u64; // r8-r15

pub const rex: *mut u8 = 224 as *mut u8; // current REX prefix, 0 if none

pub const long_mode: *mut bool = 225 as *mut bool; // EFER.LMA

// long mode instruction pointer; virtual, may be above 4 GiB
pub const rip: *mut u64 = 232 as *mut u64;
pub const previous_rip: *mut u64 = 240 as *mut u64;

pub const exception_in_progress: *mut u8 = 248 as *mut u8; // 0 = none, 1 = delivering, 2 = delivering #DF

pub const last_op_size: *mut i32 = 96 as *mut i32;
pub const flags_changed: *mut i32 = 100 as *mut i32;
pub const last_op1: *mut i32 = 104 as *mut i32;
pub const state_flags: *mut CachedStateFlags = 108 as *mut CachedStateFlags;
pub const last_result: *mut i32 = 112 as *mut i32;
pub const flags: *mut i32 = 120 as *mut i32;

pub const segment_access_bytes: *mut u8 = 512 as *mut u8; // TODO: reorder below segment_limits

pub const apic_enabled: *mut bool = 548 as *mut bool;
pub const acpi_enabled: *mut bool = 552 as *mut bool;

pub const instruction_pointer: *mut i32 = 556 as *mut i32;
pub const previous_ip: *mut i32 = 560 as *mut i32;
pub const idtr_size: *mut i32 = 564 as *mut i32;
pub const idtr_offset: *mut i32 = 568 as *mut i32;
pub const gdtr_size: *mut i32 = 572 as *mut i32;
pub const gdtr_offset: *mut i32 = 576 as *mut i32;
pub const cr: *mut i32 = 580 as *mut i32;
pub const cpl: *mut u8 = 612 as *mut u8;
pub const in_hlt: *mut bool = 616 as *mut bool;
pub const last_virt_eip: *mut i32 = 620 as *mut i32;
pub const eip_phys: *mut i32 = 624 as *mut i32;

pub const sysenter_cs: *mut i32 = 636 as *mut i32;
pub const sysenter_esp: *mut i32 = 640 as *mut i32;
pub const sysenter_eip: *mut i32 = 644 as *mut i32;
pub const prefixes: *mut u8 = 648 as *mut u8;
pub const instruction_counter: *mut u32 = 664 as *mut u32;
pub const sreg: *mut u16 = 668 as *mut u16;
pub const dreg: *mut i32 = 684 as *mut i32;

// filled in by svga_fill_pixel_buffer, read by javacsript for optimised putImageData calls
pub const svga_dirty_bitmap_min_offset: *mut u32 = 716 as *mut u32;
pub const svga_dirty_bitmap_max_offset: *mut u32 = 720 as *mut u32;

pub const segment_is_null: *mut bool = 724 as *mut bool;
pub const segment_offsets: *mut i32 = 736 as *mut i32;
pub const segment_limits: *mut u32 = 768 as *mut u32;

pub const protected_mode: *mut bool = 800 as *mut bool;
pub const is_32: *mut bool = 804 as *mut bool;
pub const stack_size_32: *mut bool = 808 as *mut bool;
pub const memory_size: *mut u32 = 812 as *mut u32;
pub const fpu_stack_empty: *mut u8 = 816 as *mut u8;
pub const mxcsr: *mut i32 = 824 as *mut i32;

pub const reg_xmm: *mut reg128 = 832 as *mut reg128;

// xmm8-15 exist only in long mode. reg_xmm holds only 8 entries; from entry 8
// on it would run into current_tsc/reg_pdpte/fpu state/cr2/efer/star, so the
// high half gets its own storage. Must stay 16-byte aligned: reg128 is 16 bytes
// and an unaligned dereference trips Rust's misalignment check.
#[repr(align(16))]
struct XmmHigh([u64; 16]);

static mut XMM_HIGH: XmmHigh = XmmHigh([0; 16]);

#[inline]
pub unsafe fn xmm_ptr(r: i32) -> *mut reg128 {
    if r < 8 {
        reg_xmm.offset(r as isize)
    }
    else {
        (std::ptr::addr_of_mut!(XMM_HIGH.0) as *mut u64)
            .add((r as usize - 8) * 2) as *mut reg128
    }
}
pub const current_tsc: *mut u64 = 960 as *mut u64;

pub const reg_pdpte: *mut u64 = 968 as *mut u64; // 4 64-bit entries

pub const fpu_stack_ptr: *mut u8 = 1032 as *mut u8;
pub const fpu_control_word: *mut u16 = 1036 as *mut u16;
pub const fpu_status_word: *mut u16 = 1040 as *mut u16;
pub const fpu_opcode: *mut i32 = 1044 as *mut i32;
pub const fpu_ip: *mut i32 = 1048 as *mut i32;
pub const fpu_ip_selector: *mut i32 = 1052 as *mut i32;
pub const fpu_dp: *mut i32 = 1056 as *mut i32;
pub const fpu_dp_selector: *mut i32 = 1060 as *mut i32;

pub const cr2: *mut u64 = 1064 as *mut u64; // full CR2; cr[2] is the low 32 bits

// long mode MSRs
pub const efer: *mut u64 = 1072 as *mut u64;
pub const star: *mut u64 = 1080 as *mut u64;
pub const lstar: *mut u64 = 1088 as *mut u64;
pub const sfmask: *mut u64 = 1096 as *mut u64;
pub const fs_base: *mut u64 = 1104 as *mut u64;
pub const gs_base: *mut u64 = 1112 as *mut u64;
pub const kernel_gs_base: *mut u64 = 1120 as *mut u64;

pub const tss_size_32: *mut bool = 1128 as *mut bool;

// Full 64-bit TSS base in long mode: a 64-bit system descriptor carries
// base[63:32] in its upper doubleword, and the kernel maps the TSS into the
// KPTI cpu_entry_area (above 4 GiB). segment_offsets[TR] keeps the low 32 bits.
pub static mut TSS_BASE: u64 = 0;

// Full 64-bit IDT/GDT base in long mode; the *_offset fields keep the low 32 bits.
// Statics, not zero-page offsets: fpu_st (1152, 8x16-byte F80) covers
// [1152, 1280), so the old 1240/1248 slots sat inside st(5)/st(6) and x87
// writes clobbered the bases, breaking later interrupt delivery.
pub static mut IDTR_BASE: u64 = 0;
pub static mut GDTR_BASE: u64 = 0;

pub const sse_scratch_register: *mut reg128 = 1136 as *mut reg128;

pub const fpu_st: *mut F80 = 1152 as *mut F80;

pub fn get_reg32_offset(r: u32) -> u32 {
    dbg_assert!(r < 16);
    if r < 8 {
        (unsafe { reg32.offset(r as isize) }) as u32
    }
    else {
        (unsafe { reg64_ext.offset((r - 8) as isize) }) as u32
    }
}

pub fn get_reg_mmx_offset(r: u32) -> u32 {
    dbg_assert!(r < 8);
    (unsafe { fpu_st.offset(r as isize) }) as u32
}

pub fn get_reg_xmm_offset(r: u32) -> u32 {
    dbg_assert!(r < 8);
    (unsafe { reg_xmm.offset(r as isize) }) as u32
}

// Also for xmm8-15, which live in XMM_HIGH (long mode).
pub fn get_reg_xmm_addr(r: u8) -> u32 {
    unsafe { xmm_ptr(r as i32) as u32 }
}

pub fn get_sreg_offset(s: u32) -> u32 {
    dbg_assert!(s < 6);
    (unsafe { sreg.offset(s as isize) }) as u32
}

pub fn get_seg_offset(s: u32) -> u32 {
    dbg_assert!(s < 8);
    (unsafe { segment_offsets.offset(s as isize) }) as u32
}

pub fn get_segment_is_null_offset(s: u32) -> u32 {
    dbg_assert!(s < 8);
    (unsafe { segment_is_null.offset(s as isize) }) as u32
}

pub fn get_creg_offset(i: u32) -> u32 {
    dbg_assert!(i < 8);
    (unsafe { cr.offset(i as isize) }) as u32
}
