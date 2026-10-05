#![allow(non_snake_case)]

unsafe fn undefined_instruction() {
    dbg_assert!(false, "Undefined instructions");
    trigger_ud()
}
unsafe fn unimplemented_sse() {
    dbg_assert!(false, "Unimplemented SSE instruction");
    trigger_ud()
}

use crate::config;
use std::sync::Mutex;

// MSRs whose values are kept outside the shared state layout. The values are
// only needed within a session, like the PIC/APIC/HPET state.
struct MsrExtra {
    pat: u64,
    mtrr_def_type: u64,
    tsc_aux: u64,
    sysenter_esp: u64,
    sysenter_eip: u64,
}

static MSR_EXTRA: Mutex<MsrExtra> = Mutex::new(MsrExtra {
    pat: 0x0007_0406_0007_0406,
    mtrr_def_type: 0,
    tsc_aux: 0,
    sysenter_esp: 0,
    sysenter_eip: 0,
});

fn msr_extra() -> std::sync::MutexGuard<'static, MsrExtra> { MSR_EXTRA.try_lock().unwrap() }

use crate::cpu::interp::arith::{
    bsf16, bsf32, bsr16, bsr32, bt_mem, bt_reg, btc_mem, btc_reg, btr_mem, btr_reg, bts_mem,
    bts_reg, cmpxchg16, cmpxchg32, cmpxchg8, popcnt, shld16, shld32, shrd16, shrd32, xadd16,
    xadd32, xadd8,
};
use crate::cpu::interp::arith::{
    imul_reg16, imul_reg32, saturate_sd_to_sb, saturate_sd_to_sw, saturate_sd_to_ub,
    saturate_sw_to_sb, saturate_sw_to_ub, saturate_ud_to_ub, saturate_uw,
};
use crate::cpu::core::*;
use crate::cpu::interp::fpu::fpu_set_tag_word;
use crate::cpu::global_pointers::*;
use crate::cpu::interp::misc_instr::{
    adjust_stack_reg, bswap, cmovcc16, cmovcc32, fxrstor, fxsave, get_stack_pointer, jmpcc16,
    jmpcc32, push16, push32_sreg, setcc_mem, setcc_reg, test_b, test_be, test_l, test_le, test_o,
    test_p, test_s, test_z,
};
use crate::cpu::interp::misc_instr::{lar, lsl, verr, verw};
use crate::cpu::interp::misc_instr::{lss16, lss32};
use crate::cpu::interp::sse_instr::*;

#[no_mangle]
pub unsafe fn instr16_0F00_0_mem(addr: i32) {
    // sldt
    if !*protected_mode || vm86_mode() {
        trigger_ud();
        return;
    }
    return_on_pagefault!(safe_write16(addr, *sreg.offset(LDTR as isize) as i32));
}
#[no_mangle]
pub unsafe fn instr32_0F00_0_mem(addr: i32) { instr16_0F00_0_mem(addr) }
#[no_mangle]
pub unsafe fn instr16_0F00_0_reg(r: i32) {
    if !*protected_mode || vm86_mode() {
        trigger_ud();
        return;
    }
    write_reg16(r, *sreg.offset(LDTR as isize) as i32);
}
#[no_mangle]
pub unsafe fn instr32_0F00_0_reg(r: i32) {
    if !*protected_mode || vm86_mode() {
        trigger_ud();
        return;
    }
    write_reg32(r, *sreg.offset(LDTR as isize) as i32);
}

#[no_mangle]
pub unsafe fn instr16_0F00_1_mem(addr: i32) {
    // str
    if !*protected_mode || vm86_mode() {
        trigger_ud();
        return;
    }
    return_on_pagefault!(safe_write16(addr, *sreg.offset(TR as isize) as i32));
}
#[no_mangle]
pub unsafe fn instr32_0F00_1_mem(addr: i32) { instr16_0F00_1_mem(addr) }
#[no_mangle]
pub unsafe fn instr16_0F00_1_reg(r: i32) {
    if !*protected_mode || vm86_mode() {
        trigger_ud();
        return;
    }
    write_reg16(r, *sreg.offset(TR as isize) as i32);
}
#[no_mangle]
pub unsafe fn instr32_0F00_1_reg(r: i32) {
    if !*protected_mode || vm86_mode() {
        trigger_ud();
        return;
    }
    write_reg32(r, *sreg.offset(TR as isize) as i32);
}

#[no_mangle]
pub unsafe fn instr16_0F00_2_mem(addr: i32) {
    // lldt
    if !*protected_mode || vm86_mode() {
        trigger_ud();
    }
    else if 0 != *cpl {
        trigger_gp(0);
    }
    else {
        return_on_pagefault!(load_ldt(return_on_pagefault!(safe_read16(addr))));
    };
}
#[no_mangle]
pub unsafe fn instr32_0F00_2_mem(addr: i32) { instr16_0F00_2_mem(addr) }
#[no_mangle]
pub unsafe fn instr16_0F00_2_reg(r: i32) {
    if !*protected_mode || vm86_mode() {
        trigger_ud();
    }
    else if 0 != *cpl {
        trigger_gp(0);
    }
    else {
        return_on_pagefault!(load_ldt(read_reg16(r)));
    };
}
#[no_mangle]
pub unsafe fn instr32_0F00_2_reg(r: i32) { instr16_0F00_2_reg(r) }

#[no_mangle]
pub unsafe fn instr16_0F00_3_mem(addr: i32) {
    // ltr
    if !*protected_mode || vm86_mode() {
        trigger_ud();
    }
    else if 0 != *cpl {
        trigger_gp(0);
    }
    else {
        load_tr(return_on_pagefault!(safe_read16(addr)));
    };
}
#[no_mangle]
pub unsafe fn instr32_0F00_3_mem(addr: i32) { instr16_0F00_3_mem(addr); }
#[no_mangle]
pub unsafe fn instr16_0F00_3_reg(r: i32) {
    if !*protected_mode || vm86_mode() {
        trigger_ud();
    }
    else if 0 != *cpl {
        trigger_gp(0);
    }
    else {
        load_tr(read_reg16(r));
    };
}
#[no_mangle]
pub unsafe fn instr32_0F00_3_reg(r: i32) { instr16_0F00_3_reg(r) }

#[no_mangle]
pub unsafe fn instr16_0F00_4_mem(addr: i32) {
    if !*protected_mode || vm86_mode() {
        dbg_log!("verr #ud");
        trigger_ud();
        return;
    }
    verr(return_on_pagefault!(safe_read16(addr)));
}
#[no_mangle]
pub unsafe fn instr32_0F00_4_mem(addr: i32) { instr16_0F00_4_mem(addr) }
#[no_mangle]
pub unsafe fn instr16_0F00_4_reg(r: i32) {
    if !*protected_mode || vm86_mode() {
        dbg_log!("verr #ud");
        trigger_ud();
        return;
    }
    verr(read_reg16(r));
}
#[no_mangle]
pub unsafe fn instr32_0F00_4_reg(r: i32) { instr16_0F00_4_reg(r) }
#[no_mangle]
pub unsafe fn instr16_0F00_5_mem(addr: i32) {
    if !*protected_mode || vm86_mode() {
        dbg_log!("verw #ud");
        trigger_ud();
        return;
    }
    verw(return_on_pagefault!(safe_read16(addr)));
}
#[no_mangle]
pub unsafe fn instr32_0F00_5_mem(addr: i32) { instr16_0F00_5_mem(addr) }
#[no_mangle]
pub unsafe fn instr16_0F00_5_reg(r: i32) {
    if !*protected_mode || vm86_mode() {
        dbg_log!("verw #ud");
        trigger_ud();
        return;
    }
    verw(read_reg16(r));
}
#[no_mangle]
pub unsafe fn instr32_0F00_5_reg(r: i32) { instr16_0F00_5_reg(r) }

#[no_mangle]
pub unsafe fn instr16_0F01_0_reg(_r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr32_0F01_0_reg(_r: i32) { trigger_ud(); }

unsafe fn sgdt(addr: i32, mask: i32) {
    return_on_pagefault!(writable_or_pagefault(addr, 6));
    safe_write16(addr, *gdtr_size).unwrap();
    safe_write32(addr + 2, *gdtr_offset & mask).unwrap();
}
#[no_mangle]
pub unsafe fn instr16_0F01_0_mem(addr: i32) { sgdt(addr, 0xFFFFFF) }
#[no_mangle]
pub unsafe fn instr32_0F01_0_mem(addr: i32) { sgdt(addr, -1) }

#[no_mangle]
pub unsafe fn instr16_0F01_1_reg(_r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr32_0F01_1_reg(_r: i32) { trigger_ud(); }

unsafe fn sidt(addr: i32, mask: i32) {
    return_on_pagefault!(writable_or_pagefault(addr, 6));
    safe_write16(addr, *idtr_size).unwrap();
    safe_write32(addr + 2, *idtr_offset & mask).unwrap();
}
#[no_mangle]
pub unsafe fn instr16_0F01_1_mem(addr: i32) { sidt(addr, 0xFFFFFF) }
#[no_mangle]
pub unsafe fn instr32_0F01_1_mem(addr: i32) { sidt(addr, -1) }

// XGETBV (0F 01 D0) / XSETBV (0F 01 D1): extended control register XCR0. Linux
// probes OSXSAVE via CPUID and writes XCR0 here before using XSAVE/AVX, so a
// missing implementation #UDs early in boot. cf. Intel SDM Vol. 2, XGETBV/XSETBV.
#[no_mangle]
pub unsafe fn instr16_0F01_2_reg(r: i32) { instr32_0F01_2_reg(r); }
#[no_mangle]
pub unsafe fn instr32_0F01_2_reg(r: i32) {
    match r & 7 {
        // XGETBV
        0 => {
            let xcr0 = crate::cpu::interp::interp64::get_xcr0();
            write_reg32(EAX, xcr0 as u32 as i32);
            write_reg32(EDX, (xcr0 >> 32) as u32 as i32);
        },
        // XSETBV
        1 => {
            if *cpl != 0 || read_reg32(ECX) != 0 {
                trigger_gp(0);
                return;
            }
            let value = read_reg32(EAX) as u32 as u64 | (read_reg32(EDX) as u32 as u64) << 32;
            if value & 1 != 0 && value & !0x7 == 0 {
                crate::cpu::interp::interp64::set_xcr0(value);
            }
            else {
                trigger_gp(0);
            }
        },
        _ => trigger_ud(),
    }
}

unsafe fn lgdt(addr: i32, mask: i32) {
    if 0 != *cpl {
        trigger_gp(0);
        return;
    }
    let size = return_on_pagefault!(safe_read16(addr));
    let offset = return_on_pagefault!(safe_read32s(addr + 2));
    *gdtr_size = size;
    *gdtr_offset = offset & mask;
    // Keep the 64-bit base in sync for descriptor lookups. cf. SDM Vol. 2, LGDT.
    GDTR_BASE = (offset & mask) as u32 as u64;
}
#[no_mangle]
pub unsafe fn instr16_0F01_2_mem(addr: i32) { lgdt(addr, 0xFFFFFF); }
#[no_mangle]
pub unsafe fn instr32_0F01_2_mem(addr: i32) { lgdt(addr, -1); }

#[no_mangle]
pub unsafe fn instr16_0F01_3_reg(_r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr32_0F01_3_reg(_r: i32) { trigger_ud(); }

unsafe fn lidt(addr: i32, mask: i32) {
    if 0 != *cpl {
        trigger_gp(0);
        return;
    }
    let size = return_on_pagefault!(safe_read16(addr));
    let offset = return_on_pagefault!(safe_read32s(addr + 2));
    *idtr_size = size;
    *idtr_offset = offset & mask;
    // See lgdt. cf. SDM Vol. 2, LIDT.
    IDTR_BASE = (offset & mask) as u32 as u64;
}
#[no_mangle]
pub unsafe fn instr16_0F01_3_mem(addr: i32) { lidt(addr, 0xFFFFFF); }
#[no_mangle]
pub unsafe fn instr32_0F01_3_mem(addr: i32) { lidt(addr, -1); }

#[no_mangle]
pub unsafe fn instr16_0F01_4_reg(r: i32) {
    // smsw
    write_reg16(r, *cr);
}
#[no_mangle]
pub unsafe fn instr32_0F01_4_reg(r: i32) { write_reg32(r, *cr); }
#[no_mangle]
pub unsafe fn instr16_0F01_4_mem(addr: i32) {
    return_on_pagefault!(safe_write16(addr, *cr & 0xFFFF));
}
#[no_mangle]
pub unsafe fn instr32_0F01_4_mem(addr: i32) {
    return_on_pagefault!(safe_write16(addr, *cr & 0xFFFF));
}

#[no_mangle]
pub unsafe fn lmsw(mut new_cr0: i32) {
    new_cr0 = *cr & !15 | new_cr0 & 15;
    if *protected_mode {
        // lmsw cannot be used to switch back
        new_cr0 |= CR0_PE
    }
    set_cr0(new_cr0);
}
#[no_mangle]
pub unsafe fn instr16_0F01_6_reg(r: i32) {
    if 0 != *cpl {
        trigger_gp(0);
        return;
    }
    lmsw(read_reg16(r));
}
#[no_mangle]
pub unsafe fn instr32_0F01_6_reg(r: i32) { instr16_0F01_6_reg(r); }
#[no_mangle]
pub unsafe fn instr16_0F01_6_mem(addr: i32) {
    if 0 != *cpl {
        trigger_gp(0);
        return;
    }
    lmsw(return_on_pagefault!(safe_read16(addr)));
}
#[no_mangle]
pub unsafe fn instr32_0F01_6_mem(addr: i32) { instr16_0F01_6_mem(addr) }

#[no_mangle]
pub unsafe fn instr16_0F01_7_reg(_r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr32_0F01_7_reg(_r: i32) { trigger_ud(); }

#[no_mangle]
pub unsafe fn instr16_0F01_7_mem(addr: i32) {
    // invlpg
    if 0 != *cpl {
        trigger_gp(0);
        return;
    }
    invlpg(addr);
}
#[no_mangle]
pub unsafe fn instr32_0F01_7_mem(addr: i32) { instr16_0F01_7_mem(addr) }

#[no_mangle]
pub unsafe fn instr16_0F02_mem(addr: i32, r: i32) {
    if !*protected_mode || vm86_mode() {
        dbg_log!("lar #ud");
        trigger_ud();
        return;
    }
    write_reg16(
        r,
        lar(return_on_pagefault!(safe_read16(addr)), read_reg16(r)),
    );
}
#[no_mangle]
pub unsafe fn instr16_0F02_reg(r1: i32, r: i32) {
    if !*protected_mode || vm86_mode() {
        dbg_log!("lar #ud");
        trigger_ud();
        return;
    }
    write_reg16(r, lar(read_reg16(r1), read_reg16(r)));
}
#[no_mangle]
pub unsafe fn instr32_0F02_mem(addr: i32, r: i32) {
    if !*protected_mode || vm86_mode() {
        dbg_log!("lar #ud");
        trigger_ud();
        return;
    }
    write_reg32(
        r,
        lar(return_on_pagefault!(safe_read16(addr)), read_reg32(r)),
    );
}
#[no_mangle]
pub unsafe fn instr32_0F02_reg(r1: i32, r: i32) {
    if !*protected_mode || vm86_mode() {
        dbg_log!("lar #ud");
        trigger_ud();
        return;
    }
    write_reg32(r, lar(read_reg16(r1), read_reg32(r)));
}
#[no_mangle]
pub unsafe fn instr16_0F03_mem(addr: i32, r: i32) {
    if !*protected_mode || vm86_mode() {
        dbg_log!("lsl #ud");
        trigger_ud();
        return;
    }
    write_reg16(
        r,
        lsl(return_on_pagefault!(safe_read16(addr)), read_reg16(r)),
    );
}
#[no_mangle]
pub unsafe fn instr16_0F03_reg(r1: i32, r: i32) {
    if !*protected_mode || vm86_mode() {
        dbg_log!("lsl #ud");
        trigger_ud();
        return;
    }
    write_reg16(r, lsl(read_reg16(r1), read_reg16(r)));
}
#[no_mangle]
pub unsafe fn instr32_0F03_mem(addr: i32, r: i32) {
    if !*protected_mode || vm86_mode() {
        dbg_log!("lsl #ud");
        trigger_ud();
        return;
    }
    write_reg32(
        r,
        lsl(return_on_pagefault!(safe_read16(addr)), read_reg32(r)),
    );
}
#[no_mangle]
pub unsafe fn instr32_0F03_reg(r1: i32, r: i32) {
    if !*protected_mode || vm86_mode() {
        dbg_log!("lsl #ud");
        trigger_ud();
        return;
    }
    write_reg32(r, lsl(read_reg16(r1), read_reg32(r)));
}
#[no_mangle]
pub unsafe fn instr_0F04() { undefined_instruction(); }
#[no_mangle]
pub unsafe fn instr_0F05() { undefined_instruction(); }
#[no_mangle]
pub unsafe fn instr_0F06() {
    // clts
    if 0 != *cpl {
        dbg_log!("clts #gp");
        trigger_gp(0);
    }
    else {
        if false {
            dbg_log!("clts");
        }
        *cr &= !CR0_TS;
    };
}
#[no_mangle]
pub unsafe fn instr_0F07() { undefined_instruction(); }
#[no_mangle]
pub unsafe fn instr_0F08() {
    // invd
    undefined_instruction();
}
#[no_mangle]
pub unsafe fn instr_0F09() {
    if 0 != *cpl {
        dbg_log!("wbinvd #gp");
        trigger_gp(0);
    }
    else {
        // wbinvd
    };
}
#[no_mangle]
pub unsafe fn instr_0F0A() { undefined_instruction(); }
pub unsafe fn instr_0F0B() {
    // UD2
    trigger_ud();
}
#[no_mangle]
pub unsafe fn instr_0F0C() { undefined_instruction(); }
#[no_mangle]
pub unsafe fn instr_0F0D() {
    // nop
    undefined_instruction();
}
#[no_mangle]
pub unsafe fn instr_0F0E() { undefined_instruction(); }
#[no_mangle]
pub unsafe fn instr_0F0F() { undefined_instruction(); }

pub unsafe fn instr_0F10(source: reg128, r: i32) {
    // movups xmm, xmm/m128
    mov_rm_r128(source, r);
}
pub unsafe fn instr_0F10_reg(r1: i32, r2: i32) { instr_0F10(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F10_mem(addr: i32, r: i32) {
    instr_0F10(return_on_pagefault!(safe_read128s(addr)), r);
}
pub unsafe fn instr_F30F10_reg(r1: i32, r2: i32) {
    // movss xmm, xmm/m32
    let data = read_xmm128s(r1);
    write_xmm32(r2, data.u32[0] as i32);
}
pub unsafe fn instr_F30F10_mem(addr: i32, r: i32) {
    // movss xmm, xmm/m32
    let data = return_on_pagefault!(safe_read32s(addr));
    write_xmm128(r, data, 0, 0, 0);
}
pub unsafe fn instr_660F10(source: reg128, r: i32) {
    // movupd xmm, xmm/m128
    mov_rm_r128(source, r);
}
pub unsafe fn instr_660F10_reg(r1: i32, r2: i32) { instr_660F10(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F10_mem(addr: i32, r: i32) {
    instr_660F10(return_on_pagefault!(safe_read128s(addr)), r);
}
pub unsafe fn instr_F20F10_reg(r1: i32, r2: i32) {
    // movsd xmm, xmm/m64
    let data = read_xmm128s(r1);
    write_xmm64(r2, data.u64[0]);
}
pub unsafe fn instr_F20F10_mem(addr: i32, r: i32) {
    // movsd xmm, xmm/m64
    let data = return_on_pagefault!(safe_read64s(addr));
    write_xmm128_2(r, data, 0);
}
pub unsafe fn instr_0F11_reg(r1: i32, r2: i32) {
    // movups xmm/m128, xmm
    mov_r_r128(r1, r2);
}
pub unsafe fn instr_0F11_mem(addr: i32, r: i32) {
    // movups xmm/m128, xmm
    mov_r_m128(addr, r);
}
pub unsafe fn instr_F30F11_reg(rm_dest: i32, reg_src: i32) {
    // movss xmm/m32, xmm
    let data = read_xmm128s(reg_src);
    write_xmm32(rm_dest, data.u32[0] as i32);
}
pub unsafe fn instr_F30F11_mem(addr: i32, r: i32) {
    // movss xmm/m32, xmm
    let data = read_xmm128s(r);
    return_on_pagefault!(safe_write32(addr, data.u32[0] as i32));
}
pub unsafe fn instr_660F11_reg(r1: i32, r2: i32) {
    // movupd xmm/m128, xmm
    mov_r_r128(r1, r2);
}
pub unsafe fn instr_660F11_mem(addr: i32, r: i32) {
    // movupd xmm/m128, xmm
    mov_r_m128(addr, r);
}
pub unsafe fn instr_F20F11_reg(r1: i32, r2: i32) {
    // movsd xmm/m64, xmm
    let data = read_xmm128s(r2);
    write_xmm64(r1, data.u64[0]);
}
pub unsafe fn instr_F20F11_mem(addr: i32, r: i32) {
    // movsd xmm/m64, xmm
    let data = read_xmm64s(r);
    return_on_pagefault!(safe_write64(addr, data));
}
pub unsafe fn instr_0F12_mem(addr: i32, r: i32) {
    // movlps xmm, m64
    let data = return_on_pagefault!(safe_read64s(addr));
    write_xmm64(r, data);
}
pub unsafe fn instr_0F12_reg(r1: i32, r2: i32) {
    // movhlps xmm, xmm
    let data = read_xmm128s(r1);
    write_xmm64(r2, data.u64[1]);
}
pub unsafe fn instr_660F12_reg(_r1: i32, _r: i32) { trigger_ud(); }
pub unsafe fn instr_660F12_mem(addr: i32, r: i32) {
    // movlpd xmm, m64
    let data = return_on_pagefault!(safe_read64s(addr));
    write_xmm64(r, data);
}
#[no_mangle]
pub unsafe fn instr_F20F12(source: u64, r: i32) {
    // movddup xmm1, xmm2/m64
    write_xmm_reg128(
        r,
        reg128 {
            u64: [source, source],
        },
    );
}
pub unsafe fn instr_F20F12_reg(r1: i32, r2: i32) { instr_F20F12(read_xmm64s(r1), r2); }
pub unsafe fn instr_F20F12_mem(addr: i32, r: i32) {
    instr_F20F12(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F30F12(source: reg128, r: i32) {
    // movsldup xmm1, xmm2/m128
    write_xmm_reg128(
        r,
        reg128 {
            u32: [source.u32[0], source.u32[0], source.u32[2], source.u32[2]],
        },
    );
}
pub unsafe fn instr_F30F12_reg(r1: i32, r2: i32) { instr_F30F12(read_xmm128s(r1), r2); }
pub unsafe fn instr_F30F12_mem(addr: i32, r: i32) {
    instr_F30F12(return_on_pagefault!(safe_read128s(addr)), r);
}
pub unsafe fn instr_0F13_mem(addr: i32, r: i32) {
    // movlps m64, xmm
    movl_r128_m64(addr, r);
}
pub unsafe fn instr_0F13_reg(_r1: i32, _r2: i32) { trigger_ud(); }
pub unsafe fn instr_660F13_reg(_r1: i32, _r: i32) { trigger_ud(); }
pub unsafe fn instr_660F13_mem(addr: i32, r: i32) {
    // movlpd xmm/m64, xmm
    movl_r128_m64(addr, r);
}

#[no_mangle]
pub unsafe fn instr_0F14(source: u64, r: i32) {
    // unpcklps xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm64s(r);
    write_xmm128(
        r,
        destination as i32,
        source as i32,
        (destination >> 32) as i32,
        (source >> 32) as i32,
    );
}
pub unsafe fn instr_0F14_reg(r1: i32, r2: i32) { instr_0F14(read_xmm64s(r1), r2); }
pub unsafe fn instr_0F14_mem(addr: i32, r: i32) {
    instr_0F14(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F14(source: u64, r: i32) {
    // unpcklpd xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm64s(r);
    write_xmm128(
        r,
        destination as i32,
        (destination >> 32) as i32,
        source as i32,
        (source >> 32) as i32,
    );
}
pub unsafe fn instr_660F14_reg(r1: i32, r2: i32) { instr_660F14(read_xmm64s(r1), r2); }
pub unsafe fn instr_660F14_mem(addr: i32, r: i32) {
    instr_660F14(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F15(source: reg128, r: i32) {
    // unpckhps xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    write_xmm128(
        r,
        destination.u32[2] as i32,
        source.u32[2] as i32,
        destination.u32[3] as i32,
        source.u32[3] as i32,
    );
}
pub unsafe fn instr_0F15_reg(r1: i32, r2: i32) { instr_0F15(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F15_mem(addr: i32, r: i32) {
    instr_0F15(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F15(source: reg128, r: i32) {
    // unpckhpd xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    write_xmm128(
        r,
        destination.u32[2] as i32,
        destination.u32[3] as i32,
        source.u32[2] as i32,
        source.u32[3] as i32,
    );
}
pub unsafe fn instr_660F15_reg(r1: i32, r2: i32) { instr_660F15(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F15_mem(addr: i32, r: i32) {
    instr_660F15(return_on_pagefault!(safe_read128s(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_0F16(source: u64, r: i32) { (*xmm_ptr(r)).u64[1] = source; }
pub unsafe fn instr_0F16_mem(addr: i32, r: i32) {
    // movhps xmm, m64
    instr_0F16(return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr_0F16_reg(r1: i32, r2: i32) {
    // movlhps xmm, xmm
    instr_0F16(read_xmm64s(r1), r2);
}
pub unsafe fn instr_660F16_mem(addr: i32, r: i32) {
    // movhpd xmm, m64
    instr_0F16(return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr_660F16_reg(_r1: i32, _r2: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_F30F16(source: reg128, r: i32) {
    // movshdup xmm1, xmm2/m128
    write_xmm_reg128(
        r,
        reg128 {
            u32: [source.u32[1], source.u32[1], source.u32[3], source.u32[3]],
        },
    );
}
pub unsafe fn instr_F30F16_reg(r1: i32, r2: i32) { instr_F30F16(read_xmm128s(r1), r2); }
pub unsafe fn instr_F30F16_mem(addr: i32, r: i32) {
    instr_F30F16(return_on_pagefault!(safe_read128s(addr)), r);
}
pub unsafe fn instr_0F17_mem(addr: i32, r: i32) {
    // movhps m64, xmm
    movh_r128_m64(addr, r);
}
pub unsafe fn instr_0F17_reg(_r1: i32, _r2: i32) { trigger_ud(); }
pub unsafe fn instr_660F17_mem(addr: i32, r: i32) {
    // movhpd m64, xmm
    movh_r128_m64(addr, r);
}
pub unsafe fn instr_660F17_reg(_r1: i32, _r2: i32) { trigger_ud(); }

pub unsafe fn instr_0F18_reg(_r1: i32, _r2: i32) {
    // reserved nop
}
pub unsafe fn instr_0F18_mem(_addr: i32, _r: i32) {
    // prefetch
    // nop for us
}

pub unsafe fn instr_0F19_reg(_r1: i32, _r2: i32) {}
pub unsafe fn instr_0F19_mem(_addr: i32, _r: i32) {}

#[no_mangle]
pub unsafe fn instr_0F1A() { undefined_instruction(); }
#[no_mangle]
pub unsafe fn instr_0F1B() { undefined_instruction(); }

pub unsafe fn instr_0F1C_reg(_r1: i32, _r2: i32) {}
pub unsafe fn instr_0F1C_mem(_addr: i32, _r: i32) {}
pub unsafe fn instr_0F1D_reg(_r1: i32, _r2: i32) {}
pub unsafe fn instr_0F1D_mem(_addr: i32, _r: i32) {}
pub unsafe fn instr_0F1E_reg(_r1: i32, _r2: i32) {}
pub unsafe fn instr_0F1E_mem(_addr: i32, _r: i32) {}
pub unsafe fn instr_0F1F_reg(_r1: i32, _r2: i32) {}
pub unsafe fn instr_0F1F_mem(_addr: i32, _r: i32) {}

#[no_mangle]
pub unsafe fn instr_0F20(r: i32, creg: i32) {
    if 0 != *cpl {
        trigger_gp(0);
        return;
    }

    match creg {
        0 => {
            write_reg32(r, *cr);
        },
        2 => {
            write_reg32(r, *cr.offset(2));
        },
        3 => {
            write_reg32(r, *cr.offset(3));
        },
        4 => {
            write_reg32(r, *cr.offset(4));
        },
        _ => {
            dbg_log!("{}", creg);
            undefined_instruction();
        },
    }
}
#[no_mangle]
pub unsafe fn instr_0F21(r: i32, mut dreg_index: i32) {
    if 0 != *cpl {
        trigger_gp(0);
        return;
    }

    if dreg_index == 4 || dreg_index == 5 {
        if 0 != *cr.offset(4) & CR4_DE {
            dbg_log!("#ud mov dreg 4/5 with cr4.DE set");
            trigger_ud();
            return;
        }
        else {
            // DR4 and DR5 refer to DR6 and DR7 respectively
            dreg_index += 2
        }
    }
    write_reg32(r, *dreg.offset(dreg_index as isize));

    if false {
        dbg_log!(
            "read dr{}: {:x}",
            dreg_index,
            *dreg.offset(dreg_index as isize)
        );
    }
}
#[no_mangle]
pub unsafe fn instr_0F22(r: i32, creg: i32) {
    if 0 != *cpl {
        trigger_gp(0);
        return;
    }

    let data = read_reg32(r);
    // mov cr, addr
    match creg {
        0 => {
            if false {
                dbg_log!("cr0 <- {:x}", data);
            }
            set_cr0(data);
        },
        2 => {
            dbg_log!("cr2 <- {:x}", data);
            *cr.offset(2) = data;
            *cr2 = data as u32 as u64;
        },
        3 => set_cr3(data),
        4 => {
            let data64 = read_reg64(r);
            if data64 >> 32 != 0 {
                // bits 32..63 of CR4 are reserved
                trigger_gp(0);
                return;
            }
            let data = data64 as u32 as i32;
            dbg_log!("cr4 <- {:x}", data);
            // Reserved: 15, 19, 23..31, and LA57 (12), which is not supported.
            if 0 != data as u32 & ((1 << 12 | 1 << 15 | 1 << 19) as u32 | 0xFF80_0000) {
                trigger_gp(0);
                return;
            }
            if 0 != (*cr.offset(4) ^ data) & (CR4_PGE | CR4_PSE | CR4_PAE) {
                full_clear_tlb();
            }
            // load_pdpte is the 32-bit PAE path; in long mode CR3 is a PML4.
            if !*long_mode
                && data & CR4_PAE != 0
                && 0 != (*cr.offset(4) ^ data) & (CR4_PGE | CR4_PSE | CR4_SMEP)
            {
                load_pdpte(*cr.offset(3));
            }
            *cr.offset(4) = data;
        },
        _ => {
            dbg_log!("{}", creg);
            undefined_instruction();
        },
    }
}
#[no_mangle]
pub unsafe fn instr_0F23(r: i32, mut dreg_index: i32) {
    if 0 != *cpl {
        trigger_gp(0);
        return;
    }

    if dreg_index == 4 || dreg_index == 5 {
        if 0 != *cr.offset(4) & CR4_DE {
            dbg_log!("#ud mov dreg 4/5 with cr4.DE set");
            trigger_ud();
            return;
        }
        else {
            // DR4 and DR5 refer to DR6 and DR7 respectively
            dreg_index += 2
        }
    }
    *dreg.offset(dreg_index as isize) = read_reg32(r);
    if false {
        dbg_log!(
            "write dr{}: {:x}",
            dreg_index,
            *dreg.offset(dreg_index as isize)
        );
    }
}
#[no_mangle]
pub unsafe fn instr_0F24() { undefined_instruction(); }
#[no_mangle]
pub unsafe fn instr_0F25() { undefined_instruction(); }
#[no_mangle]
pub unsafe fn instr_0F26() { undefined_instruction(); }
#[no_mangle]
pub unsafe fn instr_0F27() { undefined_instruction(); }

pub unsafe fn instr_0F28(source: reg128, r: i32) {
    // movaps xmm, xmm/m128
    // XXX: Aligned read or #gp
    mov_rm_r128(source, r);
}
pub unsafe fn instr_0F28_reg(r1: i32, r2: i32) { instr_0F28(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F28_mem(addr: i32, r: i32) {
    instr_0F28(return_on_pagefault!(safe_read128s(addr)), r);
}
pub unsafe fn instr_660F28(source: reg128, r: i32) {
    // movapd xmm, xmm/m128
    // XXX: Aligned read or #gp
    // Note: Same as movdqa (660F6F)
    mov_rm_r128(source, r);
}
pub unsafe fn instr_660F28_reg(r1: i32, r2: i32) { instr_660F28(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F28_mem(addr: i32, r: i32) {
    instr_660F28(return_on_pagefault!(safe_read128s(addr)), r);
}
pub unsafe fn instr_0F29_mem(addr: i32, r: i32) {
    // movaps m128, xmm
    let data = read_xmm128s(r);
    // XXX: Aligned write or #gp
    return_on_pagefault!(safe_write128(addr, data));
}
pub unsafe fn instr_0F29_reg(r1: i32, r2: i32) {
    // movaps xmm, xmm
    mov_r_r128(r1, r2);
}
pub unsafe fn instr_660F29_mem(addr: i32, r: i32) {
    // movapd m128, xmm
    let data = read_xmm128s(r);
    // XXX: Aligned write or #gp
    return_on_pagefault!(safe_write128(addr, data));
}
pub unsafe fn instr_660F29_reg(r1: i32, r2: i32) {
    // movapd xmm, xmm
    mov_r_r128(r1, r2);
}

#[no_mangle]
pub unsafe fn instr_0F2A(source: u64, r: i32) {
    // cvtpi2ps xmm, mm/m64
    // Note: Casts here can fail
    // XXX: Should round according to round control
    let source: [i32; 2] = std::mem::transmute(source);
    let result = [source[0] as f32, source[1] as f32];
    write_xmm64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F2A_reg(r1: i32, r2: i32) { instr_0F2A(read_mmx64s(r1), r2); }
pub unsafe fn instr_0F2A_mem(addr: i32, r: i32) {
    instr_0F2A(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F2A(source: u64, r: i32) {
    // cvtpi2pd xmm, xmm/m64
    // These casts can't fail
    let source: [i32; 2] = std::mem::transmute(source);
    let result = reg128 {
        f64: [source[0] as f64, source[1] as f64],
    };
    write_xmm_reg128(r, result);
}
#[no_mangle]
pub unsafe fn instr_660F2A_reg(r1: i32, r2: i32) {
    instr_660F2A(read_mmx64s(r1), r2);
    transition_fpu_to_mmx();
}
pub unsafe fn instr_660F2A_mem(addr: i32, r: i32) {
    instr_660F2A(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F20F2A(source: i32, r: i32) {
    // cvtsi2sd xmm, r32/m32
    // This cast can't fail
    write_xmm_f64(r, source as f64);
}
pub unsafe fn instr_F20F2A_reg(r1: i32, r2: i32) { instr_F20F2A(read_reg32(r1), r2); }
pub unsafe fn instr_F20F2A_mem(addr: i32, r: i32) {
    instr_F20F2A(return_on_pagefault!(safe_read32s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F30F2A(source: i32, r: i32) {
    // cvtsi2ss xmm, r/m32
    // Note: This cast can fail
    // XXX: Should round according to round control
    let result = source as f32;
    write_xmm_f32(r, result);
}
pub unsafe fn instr_F30F2A_reg(r1: i32, r2: i32) { instr_F30F2A(read_reg32(r1), r2); }
pub unsafe fn instr_F30F2A_mem(addr: i32, r: i32) {
    instr_F30F2A(return_on_pagefault!(safe_read32s(addr)), r);
}

pub unsafe fn instr_0F2B_reg(_r1: i32, _r2: i32) { trigger_ud(); }
pub unsafe fn instr_0F2B_mem(addr: i32, r: i32) {
    // movntps m128, xmm
    // XXX: Aligned write or #gp
    mov_r_m128(addr, r);
}
pub unsafe fn instr_660F2B_reg(_r1: i32, _r2: i32) { trigger_ud(); }
pub unsafe fn instr_660F2B_mem(addr: i32, r: i32) {
    // movntpd m128, xmm
    // XXX: Aligned write or #gp
    mov_r_m128(addr, r);
}

pub unsafe fn instr_0F2C(source: u64, r: i32) {
    // cvttps2pi mm, xmm/m64
    let low = f32::from_bits(source as u32);
    let high = f32::from_bits((source >> 32) as u32);
    write_mmx_reg64(
        r,
        sse_convert_with_truncation_f32_to_i32(low) as u32 as u64
            | (sse_convert_with_truncation_f32_to_i32(high) as u32 as u64) << 32,
    );
    transition_fpu_to_mmx();
}
#[no_mangle]
pub unsafe fn instr_0F2C_mem(addr: i32, r: i32) {
    instr_0F2C(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F2C_reg(r1: i32, r2: i32) { instr_0F2C(read_xmm64s(r1), r2); }

pub unsafe fn instr_660F2C(source: reg128, r: i32) {
    // cvttpd2pi mm, xmm/m128
    write_mmx_reg64(
        r,
        sse_convert_with_truncation_f64_to_i32(source.f64[0]) as u32 as u64
            | (sse_convert_with_truncation_f64_to_i32(source.f64[1]) as u32 as u64) << 32,
    );
    transition_fpu_to_mmx();
}
#[no_mangle]
pub unsafe fn instr_660F2C_mem(addr: i32, r: i32) {
    instr_660F2C(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F2C_reg(r1: i32, r2: i32) { instr_660F2C(read_xmm128s(r1), r2); }

pub unsafe fn instr_F20F2C(source: u64, r: i32) {
    // cvttsd2si r32, xmm/m64
    let source = f64::from_bits(source);
    write_reg32(r, sse_convert_with_truncation_f64_to_i32(source));
}
#[no_mangle]
pub unsafe fn instr_F20F2C_reg(r1: i32, r2: i32) { instr_F20F2C(read_xmm64s(r1), r2); }
#[no_mangle]
pub unsafe fn instr_F20F2C_mem(addr: i32, r: i32) {
    instr_F20F2C(return_on_pagefault!(safe_read64s(addr)), r);
}

pub unsafe fn instr_F30F2C(source: f32, r: i32) {
    // cvttss2si
    write_reg32(r, sse_convert_with_truncation_f32_to_i32(source));
}
#[no_mangle]
pub unsafe fn instr_F30F2C_mem(addr: i32, r: i32) {
    instr_F30F2C(return_on_pagefault!(safe_read_f32(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F30F2C_reg(r1: i32, r2: i32) { instr_F30F2C(read_xmm_f32(r1), r2); }

pub unsafe fn instr_0F2D(source: u64, r: i32) {
    // cvtps2pi mm, xmm/m64
    let source: [f32; 2] = std::mem::transmute(source);
    let result = [
        sse_convert_f32_to_i32(source[0]),
        sse_convert_f32_to_i32(source[1]),
    ];
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
#[no_mangle]
pub unsafe fn instr_0F2D_reg(r1: i32, r2: i32) { instr_0F2D(read_xmm64s(r1), r2); }
#[no_mangle]
pub unsafe fn instr_0F2D_mem(addr: i32, r: i32) {
    instr_0F2D(return_on_pagefault!(safe_read64s(addr)), r);
}

pub unsafe fn instr_660F2D(source: reg128, r: i32) {
    // cvtpd2pi mm, xmm/m128
    let result = [
        sse_convert_f64_to_i32(source.f64[0]),
        sse_convert_f64_to_i32(source.f64[1]),
    ];
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
#[no_mangle]
pub unsafe fn instr_660F2D_reg(r1: i32, r2: i32) { instr_660F2D(read_xmm128s(r1), r2); }
#[no_mangle]
pub unsafe fn instr_660F2D_mem(addr: i32, r: i32) {
    instr_660F2D(return_on_pagefault!(safe_read128s(addr)), r);
}
pub unsafe fn instr_F20F2D(source: u64, r: i32) {
    // cvtsd2si r32, xmm/m64
    write_reg32(r, sse_convert_f64_to_i32(f64::from_bits(source)));
}
pub unsafe fn instr_F20F2D_reg(r1: i32, r2: i32) { instr_F20F2D(read_xmm64s(r1), r2); }
pub unsafe fn instr_F20F2D_mem(addr: i32, r: i32) {
    instr_F20F2D(return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr_F30F2D(source: f32, r: i32) {
    // cvtss2si r32, xmm1/m32
    write_reg32(r, sse_convert_f32_to_i32(source));
}
pub unsafe fn instr_F30F2D_reg(r1: i32, r2: i32) { instr_F30F2D(read_xmm_f32(r1), r2); }
pub unsafe fn instr_F30F2D_mem(addr: i32, r: i32) {
    instr_F30F2D(return_on_pagefault!(safe_read_f32(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_0F2E(source: f32, r: i32) {
    // ucomiss xmm1, xmm2/m32
    let destination = read_xmm_f32(r);
    *flags_changed = 0;
    *flags &= !FLAGS_ALL;
    if destination == source {
        *flags |= FLAG_ZERO
    }
    else if destination < source {
        *flags |= FLAG_CARRY
    }
    else if destination > source {
        // all flags cleared
    }
    else {
        // TODO: Signal on SNaN
        *flags |= FLAG_ZERO | FLAG_PARITY | FLAG_CARRY
    }
}
pub unsafe fn instr_0F2E_reg(r1: i32, r2: i32) { instr_0F2E(read_xmm_f32(r1), r2) }
pub unsafe fn instr_0F2E_mem(addr: i32, r: i32) {
    instr_0F2E(return_on_pagefault!(safe_read_f32(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_660F2E(source: u64, r: i32) {
    // ucomisd xmm1, xmm2/m64
    let destination = f64::from_bits(read_xmm64s(r));
    let source = f64::from_bits(source);
    *flags_changed = 0;
    *flags &= !FLAGS_ALL;
    if destination == source {
        *flags |= FLAG_ZERO
    }
    else if destination < source {
        *flags |= FLAG_CARRY
    }
    else if destination > source {
        // all flags cleared
    }
    else {
        // TODO: Signal on SNaN
        *flags |= FLAG_ZERO | FLAG_PARITY | FLAG_CARRY
    }
}
pub unsafe fn instr_660F2E_reg(r1: i32, r: i32) { instr_660F2E(read_xmm64s(r1), r); }
pub unsafe fn instr_660F2E_mem(addr: i32, r: i32) {
    instr_660F2E(return_on_pagefault!(safe_read64s(addr)), r)
}

#[no_mangle]
pub unsafe fn instr_0F2F(source: f32, r: i32) {
    // comiss xmm1, xmm2/m32
    let destination = read_xmm_f32(r);
    *flags_changed = 0;
    *flags &= !FLAGS_ALL;
    if destination == source {
        *flags |= FLAG_ZERO
    }
    else if destination < source {
        *flags |= FLAG_CARRY
    }
    else if destination > source {
        // all flags cleared
    }
    else {
        // TODO: Signal on SNaN or QNaN
        *flags |= FLAG_ZERO | FLAG_PARITY | FLAG_CARRY
    }
}
pub unsafe fn instr_0F2F_reg(r1: i32, r2: i32) { instr_0F2F(read_xmm_f32(r1), r2) }
pub unsafe fn instr_0F2F_mem(addr: i32, r: i32) {
    instr_0F2F(return_on_pagefault!(safe_read_f32(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_660F2F(source: u64, r: i32) {
    // comisd xmm1, xmm2/m64
    let destination = f64::from_bits(read_xmm64s(r));
    let source = f64::from_bits(source);
    *flags_changed = 0;
    *flags &= !FLAGS_ALL;
    if destination == source {
        *flags |= FLAG_ZERO
    }
    else if destination < source {
        *flags |= FLAG_CARRY
    }
    else if destination > source {
        // all flags cleared
    }
    else {
        // TODO: Signal on SNaN or QNaN
        *flags |= FLAG_ZERO | FLAG_PARITY | FLAG_CARRY
    }
}
pub unsafe fn instr_660F2F_reg(r1: i32, r: i32) { instr_660F2F(read_xmm64s(r1), r); }
pub unsafe fn instr_660F2F_mem(addr: i32, r: i32) {
    instr_660F2F(return_on_pagefault!(safe_read64s(addr)), r)
}

#[no_mangle]
pub unsafe fn instr_0F30() {
    // wrmsr - write maschine specific register
    if 0 != *cpl {
        trigger_gp(0);
        return;
    }

    let index = read_reg32(ECX);
    let low = read_reg32(EAX);
    let high = read_reg32(EDX);

    if index != IA32_SYSENTER_ESP {
        dbg_log!("wrmsr ecx={:x} data={:x}:{:x}", index, high, low);
    }

    match index {
        IA32_SYSENTER_CS => *sysenter_cs = low & 0xFFFF,
        IA32_SYSENTER_EIP => {
            let value = (high as u32 as u64) << 32 | low as u32 as u64;
            msr_extra().sysenter_eip = value;
            *sysenter_eip = low;
        },
        IA32_SYSENTER_ESP => {
            let value = (high as u32 as u64) << 32 | low as u32 as u64;
            msr_extra().sysenter_esp = value;
            *sysenter_esp = low;
        },
        IA32_FEAT_CTL => {}, // linux 5.x
        MSR_TEST_CTRL => {}, // linux 5.x
        IA32_APIC_BASE => {
            dbg_assert!(
                high == 0,
                "Changing APIC address (high 32 bits) not supported"
            );
            let address = low & !(IA32_APIC_BASE_BSP | IA32_APIC_BASE_EXTD | IA32_APIC_BASE_EN);
            dbg_assert!(
                (address == 0 && !*acpi_enabled) // windows me
                || address == APIC_MEM_ADDRESS as i32,
                "Changing APIC address not supported"
            );
            dbg_assert!(low & IA32_APIC_BASE_EXTD == 0, "x2apic not supported");
            *apic_enabled = low & IA32_APIC_BASE_EN == IA32_APIC_BASE_EN
        },
        IA32_TIME_STAMP_COUNTER => set_tsc(low as u32, high as u32),
        IA32_BIOS_UPDT_TRIG => {}, // windows xp
        IA32_BIOS_SIGN_ID => {},
        MISC_FEATURE_ENABLES => {
            // Linux 4, see: https://patchwork.kernel.org/patch/9528279/
        },
        IA32_MISC_ENABLE => {
            // Enable Misc. Processor Features
        },
        IA32_MCG_CAP => {}, // netbsd
        IA32_KERNEL_GS_BASE => {
            // Only used in 64 bit mode (by SWAPGS), but set by kvm-unit-test
            dbg_log!("GS Base written");
        },
        x if x == 0xC0000080u32 as i32 => {
            // EFER. LME is armed here; long mode starts when paging is enabled.
            let value = (high as u32 as u64) << 32 | low as u32 as u64;
            *efer = value & !(1 << 10);
            if *cr & CR0_PG != 0 && value & (1 << 8) != 0 {
                *long_mode = true;
                *efer |= 1 << 10;
            }
        },
        IA32_PERFEVTSEL0 | IA32_PERFEVTSEL1 => {}, // linux/9legacy
        IA32_PMC0 | IA32_PMC1 => {},               // linux
        IA32_PAT => {
            msr_extra().pat = (high as u32 as u64) << 32 | low as u32 as u64;
        },
        IA32_MTRR_DEF_TYPE => {
            msr_extra().mtrr_def_type = (high as u32 as u64) << 32 | low as u32 as u64;
        },
        IA32_TSC_AUX => {
            msr_extra().tsc_aux = (high as u32 as u64) << 32 | low as u32 as u64;
        },
        IA32_SPEC_CTRL => {},      // linux 5.19
        IA32_TSX_CTRL => {},       // linux 5.19
        MSR_TSX_FORCE_ABORT => {}, // linux 5.19
        IA32_MCU_OPT_CTRL => {},   // linux 5.19
        MSR_AMD64_LS_CFG => {},    // linux 5.19
        MSR_AMD64_DE_CFG => {},    // linux 6.1
        _ => {
            dbg_log!("Unknown msr: {:x}", index);
            trigger_gp(0);
        },
    }
}

#[no_mangle]
pub unsafe fn instr_0F31() {
    // rdtsc - read timestamp counter
    if 0 == *cpl || 0 == *cr.offset(4) & CR4_TSD {
        let tsc = read_tsc();
        write_reg32(EAX, tsc as i32);
        write_reg32(EDX, (tsc >> 32) as i32);
        if false {
            dbg_log!("rdtsc  edx:eax={:x}:{:x}", read_reg32(EDX), read_reg32(EAX));
        }
    }
    else {
        trigger_gp(0);
    };
}

#[no_mangle]
pub unsafe fn instr_0F32() {
    // rdmsr - read maschine specific register
    if 0 != *cpl {
        trigger_gp(0);
        return;
    }

    let index = read_reg32(ECX);
    dbg_log!("rdmsr ecx={:x}", index);

    let mut low = 0;
    let mut high = 0;

    match index {
        x if x == 0xC0000080u32 as i32 => {
            // IA32_EFER. cf. Intel SDM Vol. 4: https://www.felixcloutier.com/x86/rdmsr
            let value = *efer;
            low = value as i32;
            high = (value >> 32) as i32;
        },
        IA32_SYSENTER_CS => low = *sysenter_cs,
        IA32_SYSENTER_EIP => {
            let value = msr_extra().sysenter_eip;
            low = value as i32;
            high = (value >> 32) as i32;
        },
        IA32_SYSENTER_ESP => {
            let value = msr_extra().sysenter_esp;
            low = value as i32;
            high = (value >> 32) as i32;
        },
        IA32_TIME_STAMP_COUNTER => {
            let tsc = read_tsc();
            low = tsc as i32;
            high = (tsc >> 32) as i32
        },
        IA32_FEAT_CTL => {}, // linux 5.x
        MSR_TEST_CTRL => {}, // linux 5.x
        IA32_PLATFORM_ID => {},
        IA32_APIC_BASE => {
            if *acpi_enabled {
                low = APIC_MEM_ADDRESS as i32;
                if *apic_enabled {
                    low |= IA32_APIC_BASE_EN
                }
            }
        },
        IA32_BIOS_SIGN_ID => {},
        MSR_PLATFORM_INFO => low = 1 << 8,
        MISC_FEATURE_ENABLES => {},
        IA32_MISC_ENABLE => {
            // Enable Misc. Processor Features
            low = 1 << 0; // fast string
        },
        IA32_RTIT_CTL => {}, // linux4
        MSR_SMI_COUNT => {},
        IA32_MCG_CAP => {},                        // netbsd
        IA32_PERFEVTSEL0 | IA32_PERFEVTSEL1 => {}, // linux/9legacy
        IA32_PMC0 | IA32_PMC1 => {},               // linux
        IA32_PAT => {
            let value = msr_extra().pat;
            low = value as i32;
            high = (value >> 32) as i32;
        },
        IA32_MTRR_DEF_TYPE => {
            let value = msr_extra().mtrr_def_type;
            low = value as i32;
            high = (value >> 32) as i32;
        },
        IA32_TSC_AUX => {
            let value = msr_extra().tsc_aux;
            low = value as i32;
            high = (value >> 32) as i32;
        },
        MSR_PKG_C2_RESIDENCY => {},
        IA32_SPEC_CTRL => {},      // linux 5.19
        IA32_TSX_CTRL => {},       // linux 5.19
        MSR_TSX_FORCE_ABORT => {}, // linux 5.19
        IA32_MCU_OPT_CTRL => {},   // linux 5.19
        MSR_AMD64_LS_CFG => {},    // linux 5.19
        MSR_AMD64_DE_CFG => {},    // linux 6.1
        _ => {
            dbg_log!("Unknown msr: {:x}", index);
            trigger_gp(0);
        },
    }

    write_reg32(EAX, low);
    write_reg32(EDX, high);
}
#[no_mangle]
pub unsafe fn instr_0F33() {
    // rdpmc
    undefined_instruction();
}
#[no_mangle]
pub unsafe fn instr_0F34() {
    // sysenter
    let seg = *sysenter_cs & 0xFFFC;
    if !*protected_mode || seg == 0 {
        trigger_gp(0);
        return;
    }
    else {
        *flags &= !FLAG_VM & !FLAG_INTERRUPT;
        *instruction_pointer = *sysenter_eip;
        write_reg32(ESP, *sysenter_esp);
        *sreg.offset(CS as isize) = seg as u16;
        *segment_is_null.offset(CS as isize) = false;
        *segment_limits.offset(CS as isize) = -1i32 as u32;
        *segment_offsets.offset(CS as isize) = 0;
        *segment_access_bytes.offset(CS as isize) = 0x80 | (0 << 5) | 0x10 | 0x08 | 0x02; // P dpl0 S E RW
        update_cs_size(true);
        *cpl = 0;
        cpl_changed();
        *sreg.offset(SS as isize) = (seg + 8) as u16;
        *segment_is_null.offset(SS as isize) = false;
        *segment_limits.offset(SS as isize) = -1i32 as u32;
        *segment_offsets.offset(SS as isize) = 0;
        *segment_access_bytes.offset(SS as isize) = 0x80 | (0 << 5) | 0x10 | 0x02; // P dpl0 S RW
        *stack_size_32 = true;
        update_state_flags();
        return;
    };
}
#[no_mangle]
pub unsafe fn instr_0F35() {
    // sysexit
    let seg = *sysenter_cs & 0xFFFC;
    if !*protected_mode || 0 != *cpl || seg == 0 {
        trigger_gp(0);
        return;
    }
    else {
        *instruction_pointer = read_reg32(EDX);
        write_reg32(ESP, read_reg32(ECX));
        *sreg.offset(CS as isize) = (seg + 16 | 3) as u16;
        *segment_is_null.offset(CS as isize) = false;
        *segment_limits.offset(CS as isize) = -1i32 as u32;
        *segment_offsets.offset(CS as isize) = 0;
        *segment_access_bytes.offset(CS as isize) = 0x80 | (3 << 5) | 0x10 | 0x08 | 0x02; // P dpl3 S E RW
        update_cs_size(true);
        *cpl = 3;
        cpl_changed();
        *sreg.offset(SS as isize) = (seg + 24 | 3) as u16;
        *segment_is_null.offset(SS as isize) = false;
        *segment_limits.offset(SS as isize) = -1i32 as u32;
        *segment_offsets.offset(SS as isize) = 0;
        *segment_access_bytes.offset(SS as isize) = 0x80 | (3 << 5) | 0x10 | 0x02; // P dpl3 S RW
        *stack_size_32 = true;
        update_state_flags();
        return;
    };
}
#[no_mangle]
pub unsafe fn instr_0F36() { undefined_instruction(); }
#[no_mangle]
pub unsafe fn instr_0F37() {
    // getsec
    undefined_instruction();
}
unsafe fn read_xmm_operand(modrm_byte: i32) -> crate::paging::OrPageFault<reg128> {
    if modrm_byte < 0xC0 {
        let addr = modrm_resolve(modrm_byte)?.addr() as i32;
        safe_read128s(addr)
    }
    else {
        Ok(read_xmm128s(modrm_byte & 7))
    }
}

#[no_mangle]
pub unsafe fn instr_0F38() {
    let opcode = return_on_pagefault!(read_imm8());
    let prefixes_ = *prefixes;
    let p66 = prefixes_ & crate::cpu::decode::prefix::PREFIX_66 != 0;
    let f2 = prefixes_ & crate::cpu::decode::prefix::PREFIX_F2 != 0;
    let f3 = prefixes_ & crate::cpu::decode::prefix::PREFIX_F3 != 0;

    // MOVBE r, m / MOVBE m, r (no 66/F2/F3)
    if !p66 && !f2 && !f3 && (opcode == 0xF0 || opcode == 0xF1) {
        let modrm_byte = return_on_pagefault!(read_imm8());
        if modrm_byte >= 0xC0 {
            trigger_ud();
            return;
        }
        let addr = return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32;
        let r = modrm_byte >> 3 & 7;
        if is_osize_32() {
            if opcode == 0xF0 {
                write_reg32(r, return_on_pagefault!(safe_read32s(addr)).swap_bytes());
            }
            else {
                return_on_pagefault!(safe_write32(addr, read_reg32(r).swap_bytes()));
            }
        }
        else if opcode == 0xF0 {
            write_reg16(r, return_on_pagefault!(safe_read16(addr)).swap_bytes() as i32);
        }
        else {
            return_on_pagefault!(safe_write16(addr, read_reg16(r).swap_bytes() as i32));
        }
        return;
    }

    // ADCX (66) / ADOX (F3): add with carry, updating only one flag
    if (p66 || f3) && opcode == 0xF6 {
        let modrm_byte = return_on_pagefault!(read_imm8());
        let r = modrm_byte >> 3 & 7;
        let mask = if is_osize_32() { 0xFFFF_FFFFu32 } else { 0xFFFF };
        let source = if modrm_byte < 0xC0 {
            let addr = return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32;
            if is_osize_32() {
                return_on_pagefault!(safe_read32s(addr)) as u32
            }
            else {
                return_on_pagefault!(safe_read16(addr)) as u32
            }
        }
        else if is_osize_32() {
            read_reg32(modrm_byte & 7) as u32
        }
        else {
            read_reg16(modrm_byte & 7) as u32
        };
        let destination = if is_osize_32() { read_reg32(r) as u32 } else { read_reg16(r) as u32 };
        let carry = if p66 { *flags & FLAG_CARRY } else { *flags & FLAG_OVERFLOW } != 0;
        let (sum, first) = destination.overflowing_add(source & mask);
        let (sum, second) = sum.overflowing_add(carry as u32);
        if is_osize_32() {
            write_reg32(r, sum as i32);
        }
        else {
            write_reg16(r, sum as i32 & 0xFFFF);
        }
        let carry_out = (first || second) as i32;
        if p66 {
            *flags = (*flags & !FLAG_CARRY) | carry_out * FLAG_CARRY;
        }
        else {
            *flags = (*flags & !FLAG_OVERFLOW) | carry_out * FLAG_OVERFLOW;
        }
        *flags_changed = 0;
        return;
    }

    // CRC32 r32, r/m8/16/32 (F2 0F 38 F0/F1)
    if f2 && (opcode == 0xF0 || opcode == 0xF1) {
        let modrm_byte = return_on_pagefault!(read_imm8());
        let r = modrm_byte >> 3 & 7;
        let (value, bytes) = if modrm_byte < 0xC0 {
            let addr = return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32;
            if opcode == 0xF0 {
                (return_on_pagefault!(safe_read8(addr)) as u8 as u64, 1usize)
            }
            else if is_osize_32() {
                (return_on_pagefault!(safe_read32s(addr)) as u32 as u64, 4)
            }
            else {
                (return_on_pagefault!(safe_read16(addr)) as u16 as u64, 2)
            }
        }
        else if opcode == 0xF0 {
            (read_reg8(modrm_byte & 7) as u8 as u64, 1)
        }
        else if is_osize_32() {
            (read_reg32(modrm_byte & 7) as u32 as u64, 4)
        }
        else {
            (read_reg16(modrm_byte & 7) as u16 as u64, 2)
        };
        let crc = read_reg32(r) as u32;
        write_reg32(r, crate::cpu::interp::simd_instr::crc32(crc, value, bytes) as i32);
        return;
    }

    if !p66 {
        trigger_ud();
        return;
    }

    if !task_switch_test_mmx() {
        return;
    }

    let modrm_byte = return_on_pagefault!(read_imm8());
    let dst = modrm_byte >> 3 & 7;
    let is_mem = modrm_byte < 0xC0;

    match opcode {
        // PBLENDVB / BLENDVPS / BLENDVPD use XMM0 as the mask.
        0x10 | 0x14 | 0x15 => {
            let source = return_on_pagefault!(read_xmm_operand(modrm_byte));
            let mask = read_xmm128s(0);
            write_xmm_reg128(dst, crate::cpu::interp::simd_instr::blendv_apply(read_xmm128s(dst), source, mask));
        },
        // PTEST
        0x17 => {
            let source = return_on_pagefault!(read_xmm_operand(modrm_byte));
            let (and, andn) = crate::cpu::interp::simd_instr::ptest(read_xmm128s(dst), source);
            *flags &= !(FLAG_CARRY | FLAG_ZERO);
            if and == 0 {
                *flags |= FLAG_ZERO as i32;
            }
            if andn == 0 {
                *flags |= FLAG_CARRY as i32;
            }
            *flags_changed = 0;
        },
        // MOVNTDQA
        0x2A => {
            if !is_mem {
                trigger_ud();
                return;
            }
            let addr = return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32;
            write_xmm_reg128(dst, return_on_pagefault!(safe_read128s(addr)));
        },
        // AES-NI
        0xDB => write_xmm_reg128(dst, crate::cpu::interp::simd_instr::aesimc(return_on_pagefault!(read_xmm_operand(modrm_byte)))),
        0xDC | 0xDD => {
            let source = return_on_pagefault!(read_xmm_operand(modrm_byte));
            let last = opcode == 0xDD;
            write_xmm_reg128(dst, crate::cpu::interp::simd_instr::aesenc(read_xmm128s(dst), source, last));
        },
        0xDE | 0xDF => {
            let source = return_on_pagefault!(read_xmm_operand(modrm_byte));
            let last = opcode == 0xDF;
            write_xmm_reg128(dst, crate::cpu::interp::simd_instr::aesdec(read_xmm128s(dst), source, last));
        },
        _ => {
            let source = return_on_pagefault!(read_xmm_operand(modrm_byte));
            let current = read_xmm128s(dst);
            if let Some(r) = crate::cpu::interp::simd_instr::sse4_38_apply(opcode as u8, current, source) {
                write_xmm_reg128(dst, r);
            }
            else if let Some(r) = crate::cpu::interp::simd_instr::ssse3_apply(opcode as u8, current, source) {
                write_xmm_reg128(dst, r);
            }
            else {
                trigger_ud();
            }
        },
    }
}
#[no_mangle]
pub unsafe fn instr_0F39() { unimplemented_sse(); }
#[no_mangle]
pub unsafe fn instr_0F3A() {
    let opcode = return_on_pagefault!(read_imm8());
    if *prefixes & crate::cpu::decode::prefix::PREFIX_66 == 0 {
        trigger_ud();
        return;
    }
    if !task_switch_test_mmx() {
        return;
    }
    let modrm_byte = return_on_pagefault!(read_imm8());
    let dst = modrm_byte >> 3 & 7;
    let is_mem = modrm_byte < 0xC0;
    let imm = return_on_pagefault!(read_imm8()) as u8;
    let current = read_xmm128s(dst);

    match opcode {
        // ROUNDPS/PD/SS/SD
        0x08 | 0x09 | 0x0A | 0x0B => {
            let source = return_on_pagefault!(read_xmm_operand(modrm_byte));
            let mut result = current;
            if opcode == 0x08 || opcode == 0x0A {
                let lanes = if opcode == 0x08 { 4 } else { 1 };
                for i in 0..lanes {
                    result.f32[i] = crate::cpu::interp::simd_instr::round_apply(source.f32[i] as f64, imm) as f32;
                }
            }
            else {
                let lanes = if opcode == 0x09 { 2 } else { 1 };
                for i in 0..lanes {
                    result.f64[i] = crate::cpu::interp::simd_instr::round_apply(source.f64[i], imm);
                }
            }
            write_xmm_reg128(dst, result);
        },
        // BLENDPS/BLENDPD/PBLENDW
        0x0C | 0x0D | 0x0E => {
            let source = return_on_pagefault!(read_xmm_operand(modrm_byte));
            let (lanes, bits) = match opcode {
                0x0C => (4, 32),
                0x0D => (2, 64),
                _ => (8, 16),
            };
            write_xmm_reg128(dst, crate::cpu::interp::simd_instr::blend_imm_apply(current, source, imm, lanes, bits));
        },
        // PALIGNR
        0x0F => {
            let source = return_on_pagefault!(read_xmm_operand(modrm_byte));
            let mut temp = [0u8; 32];
            temp[..16].copy_from_slice(&source.u8);
            temp[16..].copy_from_slice(&current.u8);
            let mut result = reg128 { u64: [0, 0] };
            let shift = imm as usize & 0x1F;
            for i in 0..16 {
                result.u8[i] = if shift + i < 32 { temp[shift + i] } else { 0 };
            }
            write_xmm_reg128(dst, result);
        },
        // PEXTRB/PEXTRW/PEXTRD
        0x14 | 0x15 | 0x16 => {
            let (value, bits) = match opcode {
                0x14 => (current.u8[imm as usize] as i32, 8),
                0x15 => (current.u16[imm as usize & 7] as i32, 16),
                _ => (current.u32[imm as usize & 3] as i32, 32),
            };
            if is_mem {
                let addr = return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32;
                match bits {
                    8 => return_on_pagefault!(safe_write8(addr, value)),
                    16 => return_on_pagefault!(safe_write16(addr, value)),
                    _ => return_on_pagefault!(safe_write32(addr, value)),
                }
            }
            else {
                let r = modrm_byte & 7;
                match bits {
                    8 => write_reg8(r, value),
                    16 => write_reg16(r, value),
                    _ => write_reg32(r, value),
                }
            }
        },
        // EXTRACTPS
        0x17 => {
            let value = crate::cpu::interp::simd_instr::extractps(current, imm) as i32;
            if is_mem {
                let addr = return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32;
                return_on_pagefault!(safe_write32(addr, value));
            }
            else {
                write_reg32(modrm_byte & 7, value);
            }
        },
        // PINSRB/PINSRD and INSERTPS
        0x20 | 0x21 | 0x22 => {
            if opcode == 0x21 {
                let source = return_on_pagefault!(read_xmm_operand(modrm_byte));
                write_xmm_reg128(dst, crate::cpu::interp::simd_instr::insertps(current, source, imm));
            }
            else {
                let value = if is_mem {
                    let addr = return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32;
                    if opcode == 0x20 {
                        return_on_pagefault!(safe_read8(addr)) as u32
                    }
                    else {
                        return_on_pagefault!(safe_read32s(addr)) as u32
                    }
                }
                else if opcode == 0x20 {
                    read_reg8(modrm_byte & 7) as u8 as u32
                }
                else {
                    read_reg32(modrm_byte & 7) as u32
                };
                let mut result = current;
                if opcode == 0x20 {
                    result.u8[imm as usize] = value as u8;
                }
                else {
                    result.u32[imm as usize & 3] = value;
                }
                write_xmm_reg128(dst, result);
            }
        },
        // DPPS/DPPD
        0x40 | 0x41 => {
            let source = return_on_pagefault!(read_xmm_operand(modrm_byte));
            let result = if opcode == 0x40 {
                crate::cpu::interp::simd_instr::dpps(current, source, imm)
            }
            else {
                crate::cpu::interp::simd_instr::dppd(current, source, imm)
            };
            write_xmm_reg128(dst, result);
        },
        // MPSADBW
        0x42 => {
            let source = return_on_pagefault!(read_xmm_operand(modrm_byte));
            write_xmm_reg128(dst, crate::cpu::interp::simd_instr::mpsadbw(current, source, imm));
        },
        // PCLMULQDQ
        0x44 => {
            let source = return_on_pagefault!(read_xmm_operand(modrm_byte));
            write_xmm_reg128(dst, crate::cpu::interp::simd_instr::pclmulqdq(current, source, imm));
        },
        // PCMPESTRM/PCMPESTRI (explicit lengths in EAX/EDX)
        0x60 | 0x61 | 0x62 | 0x63 => {
            let source = return_on_pagefault!(read_xmm_operand(modrm_byte));
            let (la, lb) = if opcode == 0x60 || opcode == 0x61 {
                (read_reg32(0), read_reg32(2))
            }
            else {
                (i32::MIN, i32::MIN)
            };
            let a = current.u8;
            let b = source.u8;
            let r = crate::cpu::interp::simd_instr::strcmp_apply(&a, &b, la, lb, imm);
            if opcode == 0x60 || opcode == 0x62 {
                write_xmm_reg128(0, r.mask);
            }
            write_reg32(1, r.index as i32);
            *flags &= !((FLAG_CARRY | FLAG_ZERO | FLAG_SIGN | FLAG_OVERFLOW | FLAG_ADJUST | FLAG_PARITY) as i32);
            if r.intres != 0 {
                *flags |= FLAG_CARRY as i32;
            }
            if r.zf {
                *flags |= FLAG_ZERO as i32;
            }
            if r.sf {
                *flags |= FLAG_SIGN as i32;
            }
            if r.of {
                *flags |= FLAG_OVERFLOW as i32;
            }
            *flags_changed = 0;
        },
        // AESKEYGENASSIST
        0xDF => {
            let source = return_on_pagefault!(read_xmm_operand(modrm_byte));
            write_xmm_reg128(dst, crate::cpu::interp::simd_instr::aeskeygenassist(source, imm));
        },
        _ => trigger_ud(),
    }
}
#[no_mangle]
pub unsafe fn instr_0F3B() { unimplemented_sse(); }
#[no_mangle]
pub unsafe fn instr_0F3C() { unimplemented_sse(); }
#[no_mangle]
pub unsafe fn instr_0F3D() { unimplemented_sse(); }
#[no_mangle]
pub unsafe fn instr_0F3E() { unimplemented_sse(); }
#[no_mangle]
pub unsafe fn instr_0F3F() { unimplemented_sse(); }

pub unsafe fn instr16_0F40_mem(addr: i32, r: i32) {
    cmovcc16(test_o(), return_on_pagefault!(safe_read16(addr)), r);
}
pub unsafe fn instr16_0F40_reg(r1: i32, r: i32) { cmovcc16(test_o(), read_reg16(r1), r); }
pub unsafe fn instr32_0F40_mem(addr: i32, r: i32) {
    cmovcc32(test_o(), return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr32_0F40_reg(r1: i32, r: i32) { cmovcc32(test_o(), read_reg32(r1), r); }
pub unsafe fn instr16_0F41_mem(addr: i32, r: i32) {
    cmovcc16(!test_o(), return_on_pagefault!(safe_read16(addr)), r);
}
pub unsafe fn instr16_0F41_reg(r1: i32, r: i32) { cmovcc16(!test_o(), read_reg16(r1), r); }
pub unsafe fn instr32_0F41_mem(addr: i32, r: i32) {
    cmovcc32(!test_o(), return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr32_0F41_reg(r1: i32, r: i32) { cmovcc32(!test_o(), read_reg32(r1), r); }
pub unsafe fn instr16_0F42_mem(addr: i32, r: i32) {
    cmovcc16(test_b(), return_on_pagefault!(safe_read16(addr)), r);
}
pub unsafe fn instr16_0F42_reg(r1: i32, r: i32) { cmovcc16(test_b(), read_reg16(r1), r); }
pub unsafe fn instr32_0F42_mem(addr: i32, r: i32) {
    cmovcc32(test_b(), return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr32_0F42_reg(r1: i32, r: i32) { cmovcc32(test_b(), read_reg32(r1), r); }
pub unsafe fn instr16_0F43_mem(addr: i32, r: i32) {
    cmovcc16(!test_b(), return_on_pagefault!(safe_read16(addr)), r);
}
pub unsafe fn instr16_0F43_reg(r1: i32, r: i32) { cmovcc16(!test_b(), read_reg16(r1), r); }
pub unsafe fn instr32_0F43_mem(addr: i32, r: i32) {
    cmovcc32(!test_b(), return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr32_0F43_reg(r1: i32, r: i32) { cmovcc32(!test_b(), read_reg32(r1), r); }
pub unsafe fn instr16_0F44_mem(addr: i32, r: i32) {
    cmovcc16(test_z(), return_on_pagefault!(safe_read16(addr)), r);
}
pub unsafe fn instr16_0F44_reg(r1: i32, r: i32) { cmovcc16(test_z(), read_reg16(r1), r); }
pub unsafe fn instr32_0F44_mem(addr: i32, r: i32) {
    cmovcc32(test_z(), return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr32_0F44_reg(r1: i32, r: i32) { cmovcc32(test_z(), read_reg32(r1), r); }
pub unsafe fn instr16_0F45_mem(addr: i32, r: i32) {
    cmovcc16(!test_z(), return_on_pagefault!(safe_read16(addr)), r);
}
pub unsafe fn instr16_0F45_reg(r1: i32, r: i32) { cmovcc16(!test_z(), read_reg16(r1), r); }
pub unsafe fn instr32_0F45_mem(addr: i32, r: i32) {
    cmovcc32(!test_z(), return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr32_0F45_reg(r1: i32, r: i32) { cmovcc32(!test_z(), read_reg32(r1), r); }
pub unsafe fn instr16_0F46_mem(addr: i32, r: i32) {
    cmovcc16(test_be(), return_on_pagefault!(safe_read16(addr)), r);
}
pub unsafe fn instr16_0F46_reg(r1: i32, r: i32) { cmovcc16(test_be(), read_reg16(r1), r); }
pub unsafe fn instr32_0F46_mem(addr: i32, r: i32) {
    cmovcc32(test_be(), return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr32_0F46_reg(r1: i32, r: i32) { cmovcc32(test_be(), read_reg32(r1), r); }
pub unsafe fn instr16_0F47_mem(addr: i32, r: i32) {
    cmovcc16(!test_be(), return_on_pagefault!(safe_read16(addr)), r);
}
pub unsafe fn instr16_0F47_reg(r1: i32, r: i32) { cmovcc16(!test_be(), read_reg16(r1), r); }
pub unsafe fn instr32_0F47_mem(addr: i32, r: i32) {
    cmovcc32(!test_be(), return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr32_0F47_reg(r1: i32, r: i32) { cmovcc32(!test_be(), read_reg32(r1), r); }
pub unsafe fn instr16_0F48_mem(addr: i32, r: i32) {
    cmovcc16(test_s(), return_on_pagefault!(safe_read16(addr)), r);
}
pub unsafe fn instr16_0F48_reg(r1: i32, r: i32) { cmovcc16(test_s(), read_reg16(r1), r); }
pub unsafe fn instr32_0F48_mem(addr: i32, r: i32) {
    cmovcc32(test_s(), return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr32_0F48_reg(r1: i32, r: i32) { cmovcc32(test_s(), read_reg32(r1), r); }
pub unsafe fn instr16_0F49_mem(addr: i32, r: i32) {
    cmovcc16(!test_s(), return_on_pagefault!(safe_read16(addr)), r);
}
pub unsafe fn instr16_0F49_reg(r1: i32, r: i32) { cmovcc16(!test_s(), read_reg16(r1), r); }
pub unsafe fn instr32_0F49_mem(addr: i32, r: i32) {
    cmovcc32(!test_s(), return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr32_0F49_reg(r1: i32, r: i32) { cmovcc32(!test_s(), read_reg32(r1), r); }
pub unsafe fn instr16_0F4A_mem(addr: i32, r: i32) {
    cmovcc16(test_p(), return_on_pagefault!(safe_read16(addr)), r);
}
pub unsafe fn instr16_0F4A_reg(r1: i32, r: i32) { cmovcc16(test_p(), read_reg16(r1), r); }
pub unsafe fn instr32_0F4A_mem(addr: i32, r: i32) {
    cmovcc32(test_p(), return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr32_0F4A_reg(r1: i32, r: i32) { cmovcc32(test_p(), read_reg32(r1), r); }
pub unsafe fn instr16_0F4B_mem(addr: i32, r: i32) {
    cmovcc16(!test_p(), return_on_pagefault!(safe_read16(addr)), r);
}
pub unsafe fn instr16_0F4B_reg(r1: i32, r: i32) { cmovcc16(!test_p(), read_reg16(r1), r); }
pub unsafe fn instr32_0F4B_mem(addr: i32, r: i32) {
    cmovcc32(!test_p(), return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr32_0F4B_reg(r1: i32, r: i32) { cmovcc32(!test_p(), read_reg32(r1), r); }
pub unsafe fn instr16_0F4C_mem(addr: i32, r: i32) {
    cmovcc16(test_l(), return_on_pagefault!(safe_read16(addr)), r);
}
pub unsafe fn instr16_0F4C_reg(r1: i32, r: i32) { cmovcc16(test_l(), read_reg16(r1), r); }
pub unsafe fn instr32_0F4C_mem(addr: i32, r: i32) {
    cmovcc32(test_l(), return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr32_0F4C_reg(r1: i32, r: i32) { cmovcc32(test_l(), read_reg32(r1), r); }
pub unsafe fn instr16_0F4D_mem(addr: i32, r: i32) {
    cmovcc16(!test_l(), return_on_pagefault!(safe_read16(addr)), r);
}
pub unsafe fn instr16_0F4D_reg(r1: i32, r: i32) { cmovcc16(!test_l(), read_reg16(r1), r); }
pub unsafe fn instr32_0F4D_mem(addr: i32, r: i32) {
    cmovcc32(!test_l(), return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr32_0F4D_reg(r1: i32, r: i32) { cmovcc32(!test_l(), read_reg32(r1), r); }
pub unsafe fn instr16_0F4E_mem(addr: i32, r: i32) {
    cmovcc16(test_le(), return_on_pagefault!(safe_read16(addr)), r);
}
pub unsafe fn instr16_0F4E_reg(r1: i32, r: i32) { cmovcc16(test_le(), read_reg16(r1), r); }
pub unsafe fn instr32_0F4E_mem(addr: i32, r: i32) {
    cmovcc32(test_le(), return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr32_0F4E_reg(r1: i32, r: i32) { cmovcc32(test_le(), read_reg32(r1), r); }
pub unsafe fn instr16_0F4F_mem(addr: i32, r: i32) {
    cmovcc16(!test_le(), return_on_pagefault!(safe_read16(addr)), r);
}
pub unsafe fn instr16_0F4F_reg(r1: i32, r: i32) { cmovcc16(!test_le(), read_reg16(r1), r); }
pub unsafe fn instr32_0F4F_mem(addr: i32, r: i32) {
    cmovcc32(!test_le(), return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr32_0F4F_reg(r1: i32, r: i32) { cmovcc32(!test_le(), read_reg32(r1), r); }

#[no_mangle]
pub unsafe fn instr_0F50_reg(r1: i32, r2: i32) {
    // movmskps r, xmm
    let source = read_xmm128s(r1);
    let data = (source.u32[0] >> 31
        | source.u32[1] >> 31 << 1
        | source.u32[2] >> 31 << 2
        | source.u32[3] >> 31 << 3) as i32;
    write_reg32(r2, data);
}
#[no_mangle]
pub unsafe fn instr_0F50_mem(_addr: i32, _r1: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_660F50_reg(r1: i32, r2: i32) {
    // movmskpd r, xmm
    let source = read_xmm128s(r1);
    let data = (source.u32[1] >> 31 | source.u32[3] >> 31 << 1) as i32;
    write_reg32(r2, data);
}
#[no_mangle]
pub unsafe fn instr_660F50_mem(_addr: i32, _r1: i32) { trigger_ud(); }

#[no_mangle]
pub unsafe fn instr_0F51(source: reg128, r: i32) {
    // sqrtps xmm, xmm/mem128
    // XXX: Should round according to round control
    let result = reg128 {
        f32: [
            source.f32[0].sqrt(),
            source.f32[1].sqrt(),
            source.f32[2].sqrt(),
            source.f32[3].sqrt(),
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_0F51_reg(r1: i32, r2: i32) { instr_0F51(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F51_mem(addr: i32, r: i32) {
    instr_0F51(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F51(source: reg128, r: i32) {
    // sqrtpd xmm, xmm/mem128
    // XXX: Should round according to round control
    let result = reg128 {
        f64: [source.f64[0].sqrt(), source.f64[1].sqrt()],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F51_reg(r1: i32, r2: i32) { instr_660F51(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F51_mem(addr: i32, r: i32) {
    instr_660F51(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F20F51(source: u64, r: i32) {
    // sqrtsd xmm, xmm/mem64
    // XXX: Should round according to round control
    write_xmm_f64(r, f64::from_bits(source).sqrt());
}
pub unsafe fn instr_F20F51_reg(r1: i32, r2: i32) { instr_F20F51(read_xmm64s(r1), r2); }
pub unsafe fn instr_F20F51_mem(addr: i32, r: i32) {
    instr_F20F51(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F30F51(source: f32, r: i32) {
    // sqrtss xmm, xmm/mem32
    // XXX: Should round according to round control
    write_xmm_f32(r, source.sqrt());
}
pub unsafe fn instr_F30F51_reg(r1: i32, r2: i32) { instr_F30F51(read_xmm_f32(r1), r2); }
pub unsafe fn instr_F30F51_mem(addr: i32, r: i32) {
    instr_F30F51(return_on_pagefault!(safe_read_f32(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_0F52(source: reg128, r: i32) {
    // rcpps xmm1, xmm2/m128
    let result = reg128 {
        f32: [
            1.0 / source.f32[0].sqrt(),
            1.0 / source.f32[1].sqrt(),
            1.0 / source.f32[2].sqrt(),
            1.0 / source.f32[3].sqrt(),
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_0F52_reg(r1: i32, r2: i32) { instr_0F52(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F52_mem(addr: i32, r: i32) {
    instr_0F52(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F30F52(source: f32, r: i32) {
    // rsqrtss xmm1, xmm2/m32
    write_xmm_f32(r, 1.0 / source.sqrt());
}
pub unsafe fn instr_F30F52_reg(r1: i32, r2: i32) { instr_F30F52(read_xmm_f32(r1), r2); }
pub unsafe fn instr_F30F52_mem(addr: i32, r: i32) {
    instr_F30F52(return_on_pagefault!(safe_read_f32(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_0F53(source: reg128, r: i32) {
    // rcpps xmm, xmm/m128
    let result = reg128 {
        f32: [
            1.0 / source.f32[0],
            1.0 / source.f32[1],
            1.0 / source.f32[2],
            1.0 / source.f32[3],
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_0F53_reg(r1: i32, r2: i32) { instr_0F53(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F53_mem(addr: i32, r: i32) {
    instr_0F53(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F30F53(source: f32, r: i32) {
    // rcpss xmm, xmm/m32
    write_xmm_f32(r, 1.0 / source);
}
pub unsafe fn instr_F30F53_reg(r1: i32, r2: i32) { instr_F30F53(read_xmm_f32(r1), r2); }
pub unsafe fn instr_F30F53_mem(addr: i32, r: i32) {
    instr_F30F53(return_on_pagefault!(safe_read_f32(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_0F54(source: reg128, r: i32) {
    // andps xmm, xmm/mem128
    // XXX: Aligned access or #gp
    pand_r128(source, r);
}
pub unsafe fn instr_0F54_reg(r1: i32, r2: i32) { instr_0F54(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F54_mem(addr: i32, r: i32) {
    instr_0F54(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F54(source: reg128, r: i32) {
    // andpd xmm, xmm/mem128
    // XXX: Aligned access or #gp
    pand_r128(source, r);
}
pub unsafe fn instr_660F54_reg(r1: i32, r2: i32) { instr_660F54(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F54_mem(addr: i32, r: i32) {
    instr_660F54(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F55(source: reg128, r: i32) {
    // andnps xmm, xmm/mem128
    // XXX: Aligned access or #gp
    pandn_r128(source, r);
}
pub unsafe fn instr_0F55_reg(r1: i32, r2: i32) { instr_0F55(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F55_mem(addr: i32, r: i32) {
    instr_0F55(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F55(source: reg128, r: i32) {
    // andnpd xmm, xmm/mem128
    // XXX: Aligned access or #gp
    pandn_r128(source, r);
}
pub unsafe fn instr_660F55_reg(r1: i32, r2: i32) { instr_660F55(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F55_mem(addr: i32, r: i32) {
    instr_660F55(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F56(source: reg128, r: i32) {
    // orps xmm, xmm/mem128
    // XXX: Aligned access or #gp
    por_r128(source, r);
}
pub unsafe fn instr_0F56_reg(r1: i32, r2: i32) { instr_0F56(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F56_mem(addr: i32, r: i32) {
    instr_0F56(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F56(source: reg128, r: i32) {
    // orpd xmm, xmm/mem128
    // XXX: Aligned access or #gp
    por_r128(source, r);
}
pub unsafe fn instr_660F56_reg(r1: i32, r2: i32) { instr_660F56(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F56_mem(addr: i32, r: i32) {
    instr_660F56(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F57(source: reg128, r: i32) {
    // xorps xmm, xmm/mem128
    // XXX: Aligned access or #gp
    pxor_r128(source, r);
}
pub unsafe fn instr_0F57_reg(r1: i32, r2: i32) { instr_0F57(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F57_mem(addr: i32, r: i32) {
    instr_0F57(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F57(source: reg128, r: i32) {
    // xorpd xmm, xmm/mem128
    // XXX: Aligned access or #gp
    pxor_r128(source, r);
}
pub unsafe fn instr_660F57_reg(r1: i32, r2: i32) { instr_660F57(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F57_mem(addr: i32, r: i32) {
    instr_660F57(return_on_pagefault!(safe_read128s(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_0F58(source: reg128, r: i32) {
    // addps xmm, xmm/mem128
    let destination = read_xmm128s(r);
    let result = reg128 {
        f32: [
            source.f32[0] + destination.f32[0],
            source.f32[1] + destination.f32[1],
            source.f32[2] + destination.f32[2],
            source.f32[3] + destination.f32[3],
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_0F58_reg(r1: i32, r2: i32) { instr_0F58(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F58_mem(addr: i32, r: i32) {
    instr_0F58(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F58(source: reg128, r: i32) {
    // addpd xmm, xmm/mem128
    let destination = read_xmm128s(r);
    let result = reg128 {
        f64: [
            source.f64[0] + destination.f64[0],
            source.f64[1] + destination.f64[1],
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F58_reg(r1: i32, r2: i32) { instr_660F58(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F58_mem(addr: i32, r: i32) {
    instr_660F58(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F20F58(source: u64, r: i32) {
    // addsd xmm, xmm/mem64
    let destination = read_xmm64s(r);
    write_xmm_f64(r, f64::from_bits(source) + f64::from_bits(destination));
}
pub unsafe fn instr_F20F58_reg(r1: i32, r2: i32) { instr_F20F58(read_xmm64s(r1), r2); }
pub unsafe fn instr_F20F58_mem(addr: i32, r: i32) {
    instr_F20F58(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F30F58(source: f32, r: i32) {
    // addss xmm, xmm/mem32
    let destination = read_xmm_f32(r);
    let result = source + destination;
    write_xmm_f32(r, result);
}
pub unsafe fn instr_F30F58_reg(r1: i32, r2: i32) { instr_F30F58(read_xmm_f32(r1), r2); }
pub unsafe fn instr_F30F58_mem(addr: i32, r: i32) {
    instr_F30F58(return_on_pagefault!(safe_read_f32(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_0F59(source: reg128, r: i32) {
    // mulps xmm, xmm/mem128
    let destination = read_xmm128s(r);
    let result = reg128 {
        f32: [
            source.f32[0] * destination.f32[0],
            source.f32[1] * destination.f32[1],
            source.f32[2] * destination.f32[2],
            source.f32[3] * destination.f32[3],
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_0F59_reg(r1: i32, r2: i32) { instr_0F59(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F59_mem(addr: i32, r: i32) {
    instr_0F59(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F59(source: reg128, r: i32) {
    // mulpd xmm, xmm/mem128
    let destination = read_xmm128s(r);
    let result = reg128 {
        f64: [
            source.f64[0] * destination.f64[0],
            source.f64[1] * destination.f64[1],
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F59_reg(r1: i32, r2: i32) { instr_660F59(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F59_mem(addr: i32, r: i32) {
    instr_660F59(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F20F59(source: u64, r: i32) {
    // mulsd xmm, xmm/mem64
    let destination = read_xmm64s(r);
    write_xmm_f64(r, f64::from_bits(source) * f64::from_bits(destination));
}
pub unsafe fn instr_F20F59_reg(r1: i32, r2: i32) { instr_F20F59(read_xmm64s(r1), r2); }
pub unsafe fn instr_F20F59_mem(addr: i32, r: i32) {
    instr_F20F59(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F30F59(source: f32, r: i32) {
    // mulss xmm, xmm/mem32
    let destination = read_xmm_f32(r);
    let result = source * destination;
    write_xmm_f32(r, result);
}
pub unsafe fn instr_F30F59_reg(r1: i32, r2: i32) { instr_F30F59(read_xmm_f32(r1), r2); }
pub unsafe fn instr_F30F59_mem(addr: i32, r: i32) {
    instr_F30F59(return_on_pagefault!(safe_read_f32(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_0F5A(source: u64, r: i32) {
    // cvtps2pd xmm1, xmm2/m64
    let source: [f32; 2] = std::mem::transmute(source);
    let result = reg128 {
        f64: [source[0] as f64, source[1] as f64],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_0F5A_reg(r1: i32, r2: i32) { instr_0F5A(read_xmm64s(r1), r2); }
pub unsafe fn instr_0F5A_mem(addr: i32, r: i32) {
    instr_0F5A(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F5A(source: reg128, r: i32) {
    // cvtpd2ps xmm1, xmm2/m128
    let result = reg128 {
        // XXX: These conversions are lossy and should round according to the round control
        f32: [source.f64[0] as f32, source.f64[1] as f32, 0., 0.],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F5A_reg(r1: i32, r2: i32) { instr_660F5A(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F5A_mem(addr: i32, r: i32) {
    instr_660F5A(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F20F5A(source: u64, r: i32) {
    // cvtsd2ss xmm1, xmm2/m64
    // XXX: This conversions is lossy and should round according to the round control
    write_xmm_f32(r, f64::from_bits(source) as f32);
}
pub unsafe fn instr_F20F5A_reg(r1: i32, r2: i32) { instr_F20F5A(read_xmm64s(r1), r2); }
pub unsafe fn instr_F20F5A_mem(addr: i32, r: i32) {
    instr_F20F5A(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F30F5A(source: f32, r: i32) {
    // cvtss2sd xmm1, xmm2/m32
    write_xmm_f64(r, source as f64);
}
pub unsafe fn instr_F30F5A_reg(r1: i32, r2: i32) { instr_F30F5A(read_xmm_f32(r1), r2); }
pub unsafe fn instr_F30F5A_mem(addr: i32, r: i32) {
    instr_F30F5A(return_on_pagefault!(safe_read_f32(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_0F5B(source: reg128, r: i32) {
    // cvtdq2ps xmm1, xmm2/m128
    // XXX: Should round according to round control
    let result = reg128 {
        f32: [
            // XXX: Precision exception
            source.i32[0] as f32,
            source.i32[1] as f32,
            source.i32[2] as f32,
            source.i32[3] as f32,
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_0F5B_reg(r1: i32, r2: i32) { instr_0F5B(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F5B_mem(addr: i32, r: i32) {
    instr_0F5B(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F5B(source: reg128, r: i32) {
    // cvtps2dq xmm1, xmm2/m128
    let result = reg128 {
        i32: [
            // XXX: Precision exception
            sse_convert_f32_to_i32(source.f32[0]),
            sse_convert_f32_to_i32(source.f32[1]),
            sse_convert_f32_to_i32(source.f32[2]),
            sse_convert_f32_to_i32(source.f32[3]),
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F5B_reg(r1: i32, r2: i32) { instr_660F5B(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F5B_mem(addr: i32, r: i32) {
    instr_660F5B(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F30F5B(source: reg128, r: i32) {
    // cvttps2dq xmm1, xmm2/m128
    let result = reg128 {
        i32: [
            sse_convert_with_truncation_f32_to_i32(source.f32[0]),
            sse_convert_with_truncation_f32_to_i32(source.f32[1]),
            sse_convert_with_truncation_f32_to_i32(source.f32[2]),
            sse_convert_with_truncation_f32_to_i32(source.f32[3]),
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_F30F5B_reg(r1: i32, r2: i32) { instr_F30F5B(read_xmm128s(r1), r2); }
pub unsafe fn instr_F30F5B_mem(addr: i32, r: i32) {
    instr_F30F5B(return_on_pagefault!(safe_read128s(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_0F5C(source: reg128, r: i32) {
    // subps xmm, xmm/mem128
    let destination = read_xmm128s(r);
    let result = reg128 {
        f32: [
            destination.f32[0] - source.f32[0],
            destination.f32[1] - source.f32[1],
            destination.f32[2] - source.f32[2],
            destination.f32[3] - source.f32[3],
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_0F5C_reg(r1: i32, r2: i32) { instr_0F5C(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F5C_mem(addr: i32, r: i32) {
    instr_0F5C(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F5C(source: reg128, r: i32) {
    // subpd xmm, xmm/mem128
    let destination = read_xmm128s(r);
    let result = reg128 {
        f64: [
            destination.f64[0] - source.f64[0],
            destination.f64[1] - source.f64[1],
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F5C_reg(r1: i32, r2: i32) { instr_660F5C(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F5C_mem(addr: i32, r: i32) {
    instr_660F5C(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F20F5C(source: u64, r: i32) {
    // subsd xmm, xmm/mem64
    let destination = read_xmm64s(r);
    write_xmm_f64(r, f64::from_bits(destination) - f64::from_bits(source));
}
pub unsafe fn instr_F20F5C_reg(r1: i32, r2: i32) { instr_F20F5C(read_xmm64s(r1), r2); }
pub unsafe fn instr_F20F5C_mem(addr: i32, r: i32) {
    instr_F20F5C(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F30F5C(source: f32, r: i32) {
    // subss xmm, xmm/mem32
    let destination = read_xmm_f32(r);
    let result = destination - source;
    write_xmm_f32(r, result);
}
pub unsafe fn instr_F30F5C_reg(r1: i32, r2: i32) { instr_F30F5C(read_xmm_f32(r1), r2); }
pub unsafe fn instr_F30F5C_mem(addr: i32, r: i32) {
    instr_F30F5C(return_on_pagefault!(safe_read_f32(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F5D(source: reg128, r: i32) {
    // minps xmm, xmm/mem128
    let destination = read_xmm128s(r);
    let result = reg128 {
        f32: [
            sse_min(destination.f32[0] as f64, source.f32[0] as f64) as f32,
            sse_min(destination.f32[1] as f64, source.f32[1] as f64) as f32,
            sse_min(destination.f32[2] as f64, source.f32[2] as f64) as f32,
            sse_min(destination.f32[3] as f64, source.f32[3] as f64) as f32,
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_0F5D_reg(r1: i32, r2: i32) { instr_0F5D(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F5D_mem(addr: i32, r: i32) {
    instr_0F5D(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F5D(source: reg128, r: i32) {
    // minpd xmm, xmm/mem128
    let destination = read_xmm128s(r);
    let result = reg128 {
        f64: [
            sse_min(destination.f64[0], source.f64[0]),
            sse_min(destination.f64[1], source.f64[1]),
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F5D_reg(r1: i32, r2: i32) { instr_660F5D(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F5D_mem(addr: i32, r: i32) {
    instr_660F5D(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F20F5D(source: u64, r: i32) {
    // minsd xmm, xmm/mem64
    let destination = read_xmm64s(r);
    write_xmm_f64(
        r,
        sse_min(f64::from_bits(destination), f64::from_bits(source)),
    );
}
pub unsafe fn instr_F20F5D_reg(r1: i32, r2: i32) { instr_F20F5D(read_xmm64s(r1), r2); }
pub unsafe fn instr_F20F5D_mem(addr: i32, r: i32) {
    instr_F20F5D(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F30F5D(source: f32, r: i32) {
    // minss xmm, xmm/mem32
    let destination = read_xmm_f32(r);
    let result = sse_min(destination as f64, source as f64) as f32;
    write_xmm_f32(r, result);
}
pub unsafe fn instr_F30F5D_reg(r1: i32, r2: i32) { instr_F30F5D(read_xmm_f32(r1), r2); }
pub unsafe fn instr_F30F5D_mem(addr: i32, r: i32) {
    instr_F30F5D(return_on_pagefault!(safe_read_f32(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F5E(source: reg128, r: i32) {
    // divps xmm, xmm/mem128
    let destination = read_xmm128s(r);
    let result = reg128 {
        f32: [
            destination.f32[0] / source.f32[0],
            destination.f32[1] / source.f32[1],
            destination.f32[2] / source.f32[2],
            destination.f32[3] / source.f32[3],
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_0F5E_reg(r1: i32, r2: i32) { instr_0F5E(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F5E_mem(addr: i32, r: i32) {
    instr_0F5E(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F5E(source: reg128, r: i32) {
    // divpd xmm, xmm/mem128
    let destination = read_xmm128s(r);
    let result = reg128 {
        f64: [
            destination.f64[0] / source.f64[0],
            destination.f64[1] / source.f64[1],
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F5E_reg(r1: i32, r2: i32) { instr_660F5E(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F5E_mem(addr: i32, r: i32) {
    instr_660F5E(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F20F5E(source: u64, r: i32) {
    // divsd xmm, xmm/mem64
    let destination = read_xmm64s(r);
    write_xmm_f64(r, f64::from_bits(destination) / f64::from_bits(source));
}
pub unsafe fn instr_F20F5E_reg(r1: i32, r2: i32) { instr_F20F5E(read_xmm64s(r1), r2); }
pub unsafe fn instr_F20F5E_mem(addr: i32, r: i32) {
    instr_F20F5E(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F30F5E(source: f32, r: i32) {
    // divss xmm, xmm/mem32
    let destination = read_xmm_f32(r);
    let result = destination / source;
    write_xmm_f32(r, result);
}
pub unsafe fn instr_F30F5E_reg(r1: i32, r2: i32) { instr_F30F5E(read_xmm_f32(r1), r2); }
pub unsafe fn instr_F30F5E_mem(addr: i32, r: i32) {
    instr_F30F5E(return_on_pagefault!(safe_read_f32(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F5F(source: reg128, r: i32) {
    // maxps xmm, xmm/mem128
    let destination = read_xmm128s(r);
    let result = reg128 {
        f32: [
            sse_max(destination.f32[0] as f64, source.f32[0] as f64) as f32,
            sse_max(destination.f32[1] as f64, source.f32[1] as f64) as f32,
            sse_max(destination.f32[2] as f64, source.f32[2] as f64) as f32,
            sse_max(destination.f32[3] as f64, source.f32[3] as f64) as f32,
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_0F5F_reg(r1: i32, r2: i32) { instr_0F5F(read_xmm128s(r1), r2); }
pub unsafe fn instr_0F5F_mem(addr: i32, r: i32) {
    instr_0F5F(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F5F(source: reg128, r: i32) {
    // maxpd xmm, xmm/mem128
    let destination = read_xmm128s(r);
    let result = reg128 {
        f64: [
            sse_max(destination.f64[0], source.f64[0]),
            sse_max(destination.f64[1], source.f64[1]),
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F5F_reg(r1: i32, r2: i32) { instr_660F5F(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F5F_mem(addr: i32, r: i32) {
    instr_660F5F(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F20F5F(source: u64, r: i32) {
    // maxsd xmm, xmm/mem64
    let destination = read_xmm64s(r);
    write_xmm_f64(
        r,
        sse_max(f64::from_bits(destination), f64::from_bits(source)),
    );
}
pub unsafe fn instr_F20F5F_reg(r1: i32, r2: i32) { instr_F20F5F(read_xmm64s(r1), r2); }
pub unsafe fn instr_F20F5F_mem(addr: i32, r: i32) {
    instr_F20F5F(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F30F5F(source: f32, r: i32) {
    // maxss xmm, xmm/mem32
    let destination = read_xmm_f32(r);
    let result = sse_max(destination as f64, source as f64) as f32;
    write_xmm_f32(r, result);
}
pub unsafe fn instr_F30F5F_reg(r1: i32, r2: i32) { instr_F30F5F(read_xmm_f32(r1), r2); }
pub unsafe fn instr_F30F5F_mem(addr: i32, r: i32) {
    instr_F30F5F(return_on_pagefault!(safe_read_f32(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_0F60(source: i32, r: i32) {
    // punpcklbw mm, mm/m32
    let destination: [u8; 8] = u64::to_le_bytes(read_mmx64s(r));
    let source: [u8; 4] = i32::to_le_bytes(source);
    let mut result = [0; 8];
    for i in 0..4 {
        result[2 * i + 0] = destination[i];
        result[2 * i + 1] = source[i];
    }
    write_mmx_reg64(r, u64::from_le_bytes(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F60_reg(r1: i32, r2: i32) { instr_0F60(read_mmx32s(r1), r2); }
pub unsafe fn instr_0F60_mem(addr: i32, r: i32) {
    instr_0F60(return_on_pagefault!(safe_read32s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F60(source: reg128, r: i32) {
    // punpcklbw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination: [u8; 8] = u64::to_le_bytes(read_xmm64s(r));
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.u8[2 * i + 0] = destination[i];
        result.u8[2 * i + 1] = source.u8[i];
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F60_reg(r1: i32, r2: i32) { instr_660F60(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F60_mem(addr: i32, r: i32) {
    instr_660F60(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F61(source: i32, r: i32) {
    // punpcklwd mm, mm/m32
    let destination: [u16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [u16; 2] = std::mem::transmute(source);
    let mut result = [0; 4];
    for i in 0..2 {
        result[2 * i + 0] = destination[i];
        result[2 * i + 1] = source[i];
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F61_reg(r1: i32, r2: i32) { instr_0F61(read_mmx32s(r1), r2); }
pub unsafe fn instr_0F61_mem(addr: i32, r: i32) {
    instr_0F61(return_on_pagefault!(safe_read32s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F61(source: reg128, r: i32) {
    // punpcklwd xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination: [u16; 4] = std::mem::transmute(read_xmm64s(r));
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..4 {
        result.u16[2 * i + 0] = destination[i];
        result.u16[2 * i + 1] = source.u16[i];
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F61_reg(r1: i32, r2: i32) { instr_660F61(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F61_mem(addr: i32, r: i32) {
    instr_660F61(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F62(source: i32, r: i32) {
    // punpckldq mm, mm/m32
    let destination = read_mmx64s(r);
    write_mmx_reg64(
        r,
        (destination & 0xFFFF_FFFF) | (source as u32 as u64) << 32,
    );
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F62_reg(r1: i32, r2: i32) { instr_0F62(read_mmx32s(r1), r2); }
pub unsafe fn instr_0F62_mem(addr: i32, r: i32) {
    instr_0F62(return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr_660F62(source: reg128, r: i32) {
    // punpckldq xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    write_xmm128(
        r,
        destination.u32[0] as i32,
        source.u32[0] as i32,
        destination.u32[1] as i32,
        source.u32[1] as i32,
    );
}
pub unsafe fn instr_660F62_reg(r1: i32, r2: i32) { instr_660F62(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F62_mem(addr: i32, r: i32) {
    instr_660F62(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F63(source: u64, r: i32) {
    // packsswb mm, mm/m64
    let destination: [u16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [u16; 4] = std::mem::transmute(source);
    let mut result: [u8; 8] = [0; 8];
    for i in 0..4 {
        result[i + 0] = saturate_sw_to_sb(destination[i] as i32);
        result[i + 4] = saturate_sw_to_sb(source[i] as i32);
    }
    write_mmx_reg64(r, u64::from_le_bytes(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F63_reg(r1: i32, r2: i32) { instr_0F63(read_mmx64s(r1), r2); }
pub unsafe fn instr_0F63_mem(addr: i32, r: i32) {
    instr_0F63(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F63(source: reg128, r: i32) {
    // packsswb xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.u8[i + 0] = saturate_sw_to_sb(destination.u16[i] as i32);
        result.u8[i + 8] = saturate_sw_to_sb(source.u16[i] as i32);
    }
    write_xmm_reg128(r, result)
}
pub unsafe fn instr_660F63_reg(r1: i32, r2: i32) { instr_660F63(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F63_mem(addr: i32, r: i32) {
    instr_660F63(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F64(source: u64, r: i32) {
    // pcmpgtb mm, mm/m64
    let destination: [i8; 8] = std::mem::transmute(read_mmx64s(r));
    let source: [i8; 8] = std::mem::transmute(source);
    let mut result: [u8; 8] = [0; 8];
    for i in 0..8 {
        result[i] = if destination[i] > source[i] { 255 } else { 0 };
    }
    write_mmx_reg64(r, u64::from_le_bytes(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F64_reg(r1: i32, r2: i32) { instr_0F64(read_mmx64s(r1), r2); }
pub unsafe fn instr_0F64_mem(addr: i32, r: i32) {
    instr_0F64(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F64(source: reg128, r: i32) {
    // pcmpgtb xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..16 {
        result.u8[i] = if destination.i8[i] as i32 > source.i8[i] as i32 { 255 } else { 0 };
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F64_reg(r1: i32, r2: i32) { instr_660F64(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F64_mem(addr: i32, r: i32) {
    instr_660F64(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F65(source: u64, r: i32) {
    // pcmpgtw mm, mm/m64
    let destination: [i16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [i16; 4] = std::mem::transmute(source);
    let mut result: [u16; 4] = [0; 4];
    for i in 0..4 {
        result[i] = if destination[i] > source[i] { 0xFFFF } else { 0 }
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F65_reg(r1: i32, r2: i32) { instr_0F65(read_mmx64s(r1), r2); }
pub unsafe fn instr_0F65_mem(addr: i32, r: i32) {
    instr_0F65(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F65(source: reg128, r: i32) {
    // pcmpgtw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.u16[i] = if destination.i16[i] > source.i16[i] { 0xFFFF } else { 0 };
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F65_reg(r1: i32, r2: i32) { instr_660F65(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F65_mem(addr: i32, r: i32) {
    instr_660F65(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F66(source: u64, r: i32) {
    // pcmpgtd mm, mm/m64
    let destination: [i32; 2] = std::mem::transmute(read_mmx64s(r));
    let source: [i32; 2] = std::mem::transmute(source);
    let mut result = [0; 2];
    for i in 0..2 {
        result[i] = if destination[i] > source[i] { -1 } else { 0 }
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F66_reg(r1: i32, r2: i32) { instr_0F66(read_mmx64s(r1), r2); }
pub unsafe fn instr_0F66_mem(addr: i32, r: i32) {
    instr_0F66(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F66(source: reg128, r: i32) {
    // pcmpgtd xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    write_xmm128(
        r,
        if destination.i32[0] > source.i32[0] { -1 } else { 0 },
        if destination.i32[1] > source.i32[1] { -1 } else { 0 },
        if destination.i32[2] > source.i32[2] { -1 } else { 0 },
        if destination.i32[3] > source.i32[3] { -1 } else { 0 },
    );
}
pub unsafe fn instr_660F66_reg(r1: i32, r2: i32) { instr_660F66(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F66_mem(addr: i32, r: i32) {
    instr_660F66(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F67(source: u64, r: i32) {
    // packuswb mm, mm/m64
    let destination: [u16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [u16; 4] = std::mem::transmute(source);
    let mut result = [0; 8];
    for i in 0..4 {
        result[i + 0] = saturate_sw_to_ub(destination[i]);
        result[i + 4] = saturate_sw_to_ub(source[i]);
    }
    write_mmx_reg64(r, u64::from_le_bytes(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F67_reg(r1: i32, r2: i32) { instr_0F67(read_mmx64s(r1), r2); }
pub unsafe fn instr_0F67_mem(addr: i32, r: i32) {
    instr_0F67(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F67(source: reg128, r: i32) {
    // packuswb xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.u8[i + 0] = saturate_sw_to_ub(destination.u16[i]);
        result.u8[i + 8] = saturate_sw_to_ub(source.u16[i]);
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F67_reg(r1: i32, r2: i32) { instr_660F67(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F67_mem(addr: i32, r: i32) {
    instr_660F67(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F68(source: u64, r: i32) {
    // punpckhbw mm, mm/m64
    let destination: [u8; 8] = u64::to_le_bytes(read_mmx64s(r));
    let source: [u8; 8] = u64::to_le_bytes(source);
    let mut result: [u8; 8] = [0; 8];
    for i in 0..4 {
        result[2 * i + 0] = destination[i + 4];
        result[2 * i + 1] = source[i + 4];
    }
    write_mmx_reg64(r, u64::from_le_bytes(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F68_reg(r1: i32, r2: i32) { instr_0F68(read_mmx64s(r1), r2); }
pub unsafe fn instr_0F68_mem(addr: i32, r: i32) {
    instr_0F68(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F68(source: reg128, r: i32) {
    // punpckhbw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.u8[2 * i + 0] = destination.u8[i + 8];
        result.u8[2 * i + 1] = source.u8[i + 8];
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F68_reg(r1: i32, r2: i32) { instr_660F68(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F68_mem(addr: i32, r: i32) {
    instr_660F68(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F69(source: u64, r: i32) {
    // punpckhwd mm, mm/m64
    let destination: [u16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [u16; 4] = std::mem::transmute(source);
    let result = [destination[2], source[2], destination[3], source[3]];
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F69_reg(r1: i32, r2: i32) { instr_0F69(read_mmx64s(r1), r2); }
pub unsafe fn instr_0F69_mem(addr: i32, r: i32) {
    instr_0F69(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F69(source: reg128, r: i32) {
    // punpckhwd xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..4 {
        result.u16[2 * i + 0] = destination.u16[i + 4];
        result.u16[2 * i + 1] = source.u16[i + 4];
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F69_reg(r1: i32, r2: i32) { instr_660F69(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F69_mem(addr: i32, r: i32) {
    instr_660F69(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F6A(source: u64, r: i32) {
    // punpckhdq mm, mm/m64
    let destination = read_mmx64s(r);
    write_mmx_reg64(r, (destination >> 32) | (source >> 32 << 32));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F6A_reg(r1: i32, r2: i32) { instr_0F6A(read_mmx64s(r1), r2); }
pub unsafe fn instr_0F6A_mem(addr: i32, r: i32) {
    instr_0F6A(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F6A(source: reg128, r: i32) {
    // punpckhdq xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    write_xmm128(
        r,
        destination.u32[2] as i32,
        source.u32[2] as i32,
        destination.u32[3] as i32,
        source.u32[3] as i32,
    );
}
pub unsafe fn instr_660F6A_reg(r1: i32, r2: i32) { instr_660F6A(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F6A_mem(addr: i32, r: i32) {
    instr_660F6A(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F6B(source: u64, r: i32) {
    // packssdw mm, mm/m64
    let destination: [u32; 2] = std::mem::transmute(read_mmx64s(r));
    let source: [u32; 2] = std::mem::transmute(source);
    let mut result = [0; 4];
    for i in 0..2 {
        result[i + 0] = saturate_sd_to_sw(destination[i]);
        result[i + 2] = saturate_sd_to_sw(source[i]);
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F6B_reg(r1: i32, r2: i32) { instr_0F6B(read_mmx64s(r1), r2); }
pub unsafe fn instr_0F6B_mem(addr: i32, r: i32) {
    instr_0F6B(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F6B(source: reg128, r: i32) {
    // packssdw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..4 {
        result.u16[i + 0] = saturate_sd_to_sw(destination.u32[i]);
        result.u16[i + 4] = saturate_sd_to_sw(source.u32[i]);
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F6B_reg(r1: i32, r2: i32) { instr_660F6B(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F6B_mem(addr: i32, r: i32) {
    instr_660F6B(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F6C_mem(_addr: i32, _r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_0F6C_reg(_r1: i32, _r2: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_660F6C(source: reg128, r: i32) {
    // punpcklqdq xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    write_xmm128(
        r,
        destination.u32[0] as i32,
        destination.u32[1] as i32,
        source.u32[0] as i32,
        source.u32[1] as i32,
    );
}
pub unsafe fn instr_660F6C_reg(r1: i32, r2: i32) { instr_660F6C(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F6C_mem(addr: i32, r: i32) {
    instr_660F6C(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F6D_mem(_addr: i32, _r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_0F6D_reg(_r1: i32, _r2: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_660F6D(source: reg128, r: i32) {
    // punpckhqdq xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    write_xmm128(
        r,
        destination.u32[2] as i32,
        destination.u32[3] as i32,
        source.u32[2] as i32,
        source.u32[3] as i32,
    );
}
pub unsafe fn instr_660F6D_reg(r1: i32, r2: i32) { instr_660F6D(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F6D_mem(addr: i32, r: i32) {
    instr_660F6D(return_on_pagefault!(safe_read128s(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_0F6E(source: i32, r: i32) {
    // movd mm, r/m32
    write_mmx_reg64(r, source as u32 as u64);
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F6E_reg(r1: i32, r2: i32) { instr_0F6E(read_reg32(r1), r2); }
pub unsafe fn instr_0F6E_mem(addr: i32, r: i32) {
    instr_0F6E(return_on_pagefault!(safe_read32s(addr)), r);
}
pub unsafe fn instr_660F6E(source: i32, r: i32) {
    // movd mm, r/m32
    write_xmm128(r, source, 0, 0, 0);
}
pub unsafe fn instr_660F6E_reg(r1: i32, r2: i32) { instr_660F6E(read_reg32(r1), r2); }
pub unsafe fn instr_660F6E_mem(addr: i32, r: i32) {
    instr_660F6E(return_on_pagefault!(safe_read32s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F6F(source: u64, r: i32) {
    // movq mm, mm/m64
    write_mmx_reg64(r, source);
    transition_fpu_to_mmx();
}
#[no_mangle]
pub unsafe fn instr_0F6F_reg(r1: i32, r2: i32) { instr_0F6F(read_mmx64s(r1), r2); }
pub unsafe fn instr_0F6F_mem(addr: i32, r: i32) {
    instr_0F6F(return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr_660F6F(source: reg128, r: i32) {
    // movdqa xmm, xmm/mem128
    // XXX: Aligned access or #gp
    mov_rm_r128(source, r);
}
pub unsafe fn instr_660F6F_reg(r1: i32, r2: i32) { instr_660F6F(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F6F_mem(addr: i32, r: i32) {
    instr_660F6F(return_on_pagefault!(safe_read128s(addr)), r);
}
pub unsafe fn instr_F30F6F(source: reg128, r: i32) {
    // movdqu xmm, xmm/m128
    mov_rm_r128(source, r);
}
pub unsafe fn instr_F30F6F_reg(r1: i32, r2: i32) { instr_F30F6F(read_xmm128s(r1), r2); }
pub unsafe fn instr_F30F6F_mem(addr: i32, r: i32) {
    instr_F30F6F(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F70(source: u64, r: i32, imm8: i32) {
    // pshufw mm1, mm2/m64, imm8
    let source: [u16; 4] = std::mem::transmute(source);
    let mut result = [0; 4];
    for i in 0..4 {
        result[i] = source[(imm8 >> (2 * i) & 3) as usize]
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F70_reg(r1: i32, r2: i32, imm: i32) { instr_0F70(read_mmx64s(r1), r2, imm); }
pub unsafe fn instr_0F70_mem(addr: i32, r: i32, imm: i32) {
    instr_0F70(return_on_pagefault!(safe_read64s(addr)), r, imm);
}
pub unsafe fn instr_660F70(source: reg128, r: i32, imm8: i32) {
    // pshufd xmm, xmm/mem128, imm8
    // XXX: Aligned access or #gp
    write_xmm128(
        r,
        source.u32[(imm8 & 3) as usize] as i32,
        source.u32[(imm8 >> 2 & 3) as usize] as i32,
        source.u32[(imm8 >> 4 & 3) as usize] as i32,
        source.u32[(imm8 >> 6 & 3) as usize] as i32,
    );
}
pub unsafe fn instr_660F70_reg(r1: i32, r2: i32, imm: i32) {
    instr_660F70(read_xmm128s(r1), r2, imm);
}
pub unsafe fn instr_660F70_mem(addr: i32, r: i32, imm: i32) {
    instr_660F70(return_on_pagefault!(safe_read128s(addr)), r, imm);
}

#[no_mangle]
pub unsafe fn instr_F20F70(source: reg128, r: i32, imm8: i32) {
    // pshuflw xmm, xmm/m128, imm8
    // XXX: Aligned access or #gp
    write_xmm128(
        r,
        source.u16[(imm8 & 3) as usize] as i32
            | (source.u16[(imm8 >> 2 & 3) as usize] as i32) << 16,
        source.u16[(imm8 >> 4 & 3) as usize] as i32
            | (source.u16[(imm8 >> 6 & 3) as usize] as i32) << 16,
        source.u32[2] as i32,
        source.u32[3] as i32,
    );
}
pub unsafe fn instr_F20F70_reg(r1: i32, r2: i32, imm: i32) {
    instr_F20F70(read_xmm128s(r1), r2, imm);
}
pub unsafe fn instr_F20F70_mem(addr: i32, r: i32, imm: i32) {
    instr_F20F70(return_on_pagefault!(safe_read128s(addr)), r, imm);
}
#[no_mangle]
pub unsafe fn instr_F30F70(source: reg128, r: i32, imm8: i32) {
    // pshufhw xmm, xmm/m128, imm8
    // XXX: Aligned access or #gp
    write_xmm128(
        r,
        source.u32[0] as i32,
        source.u32[1] as i32,
        source.u16[(imm8 & 3 | 4) as usize] as i32
            | (source.u16[(imm8 >> 2 & 3 | 4) as usize] as i32) << 16,
        source.u16[(imm8 >> 4 & 3 | 4) as usize] as i32
            | (source.u16[(imm8 >> 6 & 3 | 4) as usize] as i32) << 16,
    );
}
pub unsafe fn instr_F30F70_reg(r1: i32, r2: i32, imm: i32) {
    instr_F30F70(read_xmm128s(r1), r2, imm);
}
pub unsafe fn instr_F30F70_mem(addr: i32, r: i32, imm: i32) {
    instr_F30F70(return_on_pagefault!(safe_read128s(addr)), r, imm);
}
pub unsafe fn instr_0F71_2_mem(_addr: i32, _r: i32) { trigger_ud(); }
pub unsafe fn instr_0F71_4_mem(_addr: i32, _r: i32) { trigger_ud(); }
pub unsafe fn instr_0F71_6_mem(_addr: i32, _r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_0F71_2_reg(r: i32, imm8: i32) {
    // psrlw mm, imm8
    psrlw_r64(r, imm8 as u64);
}
#[no_mangle]
pub unsafe fn instr_0F71_4_reg(r: i32, imm8: i32) {
    // psraw mm, imm8
    psraw_r64(r, imm8 as u64);
}
#[no_mangle]
pub unsafe fn instr_0F71_6_reg(r: i32, imm8: i32) {
    // psllw mm, imm8
    psllw_r64(r, imm8 as u64);
}
pub unsafe fn instr_660F71_2_mem(_addr: i32, _r: i32) { trigger_ud(); }
pub unsafe fn instr_660F71_4_mem(_addr: i32, _r: i32) { trigger_ud(); }
pub unsafe fn instr_660F71_6_mem(_addr: i32, _r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_660F71_2_reg(r: i32, imm8: i32) {
    // psrlw xmm, imm8
    psrlw_r128(r, imm8 as u64);
}
#[no_mangle]
pub unsafe fn instr_660F71_4_reg(r: i32, imm8: i32) {
    // psraw xmm, imm8
    psraw_r128(r, imm8 as u64);
}
#[no_mangle]
pub unsafe fn instr_660F71_6_reg(r: i32, imm8: i32) {
    // psllw xmm, imm8
    psllw_r128(r, imm8 as u64);
}
pub unsafe fn instr_0F72_2_mem(_addr: i32, _r: i32) { trigger_ud(); }
pub unsafe fn instr_0F72_4_mem(_addr: i32, _r: i32) { trigger_ud(); }
pub unsafe fn instr_0F72_6_mem(_addr: i32, _r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_0F72_2_reg(r: i32, imm8: i32) {
    // psrld mm, imm8
    psrld_r64(r, imm8 as u64);
}
#[no_mangle]
pub unsafe fn instr_0F72_4_reg(r: i32, imm8: i32) {
    // psrad mm, imm8
    psrad_r64(r, imm8 as u64);
}
#[no_mangle]
pub unsafe fn instr_0F72_6_reg(r: i32, imm8: i32) {
    // pslld mm, imm8
    pslld_r64(r, imm8 as u64);
}
pub unsafe fn instr_660F72_2_mem(_addr: i32, _r: i32) { trigger_ud(); }
pub unsafe fn instr_660F72_4_mem(_addr: i32, _r: i32) { trigger_ud(); }
pub unsafe fn instr_660F72_6_mem(_addr: i32, _r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_660F72_2_reg(r: i32, imm8: i32) {
    // psrld xmm, imm8
    psrld_r128(r, imm8 as u64);
}
#[no_mangle]
pub unsafe fn instr_660F72_4_reg(r: i32, imm8: i32) {
    // psrad xmm, imm8
    psrad_r128(r, imm8 as u64);
}
#[no_mangle]
pub unsafe fn instr_660F72_6_reg(r: i32, imm8: i32) {
    // pslld xmm, imm8
    pslld_r128(r, imm8 as u64);
}
pub unsafe fn instr_0F73_2_mem(_addr: i32, _r: i32) { trigger_ud(); }
pub unsafe fn instr_0F73_6_mem(_addr: i32, _r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_0F73_2_reg(r: i32, imm8: i32) {
    // psrlq mm, imm8
    psrlq_r64(r, imm8 as u64);
}
#[no_mangle]
pub unsafe fn instr_0F73_6_reg(r: i32, imm8: i32) {
    // psllq mm, imm8
    psllq_r64(r, imm8 as u64);
}
pub unsafe fn instr_660F73_2_mem(_addr: i32, _r: i32) { trigger_ud(); }
pub unsafe fn instr_660F73_3_mem(_addr: i32, _r: i32) { trigger_ud(); }
pub unsafe fn instr_660F73_6_mem(_addr: i32, _r: i32) { trigger_ud(); }
pub unsafe fn instr_660F73_7_mem(_addr: i32, _r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_660F73_2_reg(r: i32, imm8: i32) {
    // psrlq xmm, imm8
    psrlq_r128(r, imm8 as u64);
}
#[no_mangle]
pub unsafe fn instr_660F73_3_reg(r: i32, imm8: i32) {
    // psrldq xmm, imm8
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    if imm8 == 0 {
        return;
    }
    let shift = (if imm8 > 15 { 128 } else { imm8 << 3 }) as u32;
    if shift <= 63 {
        result.u64[0] = destination.u64[0] >> shift | destination.u64[1] << (64 - shift);
        result.u64[1] = destination.u64[1] >> shift
    }
    else if shift <= 127 {
        result.u64[0] = destination.u64[1] >> (shift - 64);
        result.u64[1] = 0
    }
    write_xmm_reg128(r, result);
}
#[no_mangle]
pub unsafe fn instr_660F73_6_reg(r: i32, imm8: i32) {
    // psllq xmm, imm8
    psllq_r128(r, imm8 as u64);
}
#[no_mangle]
pub unsafe fn instr_660F73_7_reg(r: i32, imm8: i32) {
    // pslldq xmm, imm8
    if imm8 == 0 {
        return;
    }
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    let shift = (if imm8 > 15 { 128 } else { imm8 << 3 }) as u32;
    if shift <= 63 {
        result.u64[0] = destination.u64[0] << shift;
        result.u64[1] = destination.u64[1] << shift | destination.u64[0] >> (64 - shift)
    }
    else if shift <= 127 {
        result.u64[0] = 0;
        result.u64[1] = destination.u64[0] << (shift - 64)
    }
    write_xmm_reg128(r, result);
}

#[no_mangle]
pub unsafe fn instr_0F74(source: u64, r: i32) {
    // pcmpeqb mm, mm/m64
    let destination: [u8; 8] = u64::to_le_bytes(read_mmx64s(r));
    let source: [u8; 8] = u64::to_le_bytes(source);
    let mut result: [u8; 8] = [0; 8];
    for i in 0..8 {
        result[i] = if destination[i] == source[i] { 255 } else { 0 };
    }
    write_mmx_reg64(r, u64::from_le_bytes(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F74_reg(r1: i32, r2: i32) { instr_0F74(read_mmx64s(r1), r2); }
pub unsafe fn instr_0F74_mem(addr: i32, r: i32) {
    instr_0F74(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F74(source: reg128, r: i32) {
    // pcmpeqb xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..16 {
        result.u8[i] = if source.u8[i] == destination.u8[i] { 255 } else { 0 }
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F74_reg(r1: i32, r2: i32) { instr_660F74(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F74_mem(addr: i32, r: i32) {
    instr_660F74(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F75(source: u64, r: i32) {
    // pcmpeqw mm, mm/m64
    let destination: [i16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [i16; 4] = std::mem::transmute(source);
    let mut result: [u16; 4] = [0; 4];
    for i in 0..4 {
        result[i] = if destination[i] == source[i] { 0xFFFF } else { 0 };
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F75_reg(r1: i32, r2: i32) { instr_0F75(read_mmx64s(r1), r2); }
pub unsafe fn instr_0F75_mem(addr: i32, r: i32) {
    instr_0F75(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F75(source: reg128, r: i32) {
    // pcmpeqw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.u16[i] =
            (if source.u16[i] as i32 == destination.u16[i] as i32 { 0xFFFF } else { 0 }) as u16;
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F75_reg(r1: i32, r2: i32) { instr_660F75(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F75_mem(addr: i32, r: i32) {
    instr_660F75(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F76(source: u64, r: i32) {
    // pcmpeqd mm, mm/m64
    let destination: [i32; 2] = std::mem::transmute(read_mmx64s(r));
    let source: [i32; 2] = std::mem::transmute(source);
    let mut result = [0; 2];
    for i in 0..2 {
        result[i] = if destination[i] == source[i] { -1 } else { 0 }
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F76_reg(r1: i32, r2: i32) { instr_0F76(read_mmx64s(r1), r2); }
pub unsafe fn instr_0F76_mem(addr: i32, r: i32) {
    instr_0F76(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660F76(source: reg128, r: i32) {
    // pcmpeqd xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..4 {
        result.i32[i] = if source.u32[i] == destination.u32[i] { -1 } else { 0 }
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660F76_reg(r1: i32, r2: i32) { instr_660F76(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F76_mem(addr: i32, r: i32) {
    instr_660F76(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0F77() {
    // emms
    fpu_set_tag_word(0xFFFF);
}

#[no_mangle]
pub unsafe fn instr_0F78() { unimplemented_sse(); }
#[no_mangle]
pub unsafe fn instr_0F79() { unimplemented_sse(); }
#[no_mangle]
pub unsafe fn instr_0F7A() { unimplemented_sse(); }
#[no_mangle]
pub unsafe fn instr_0F7B() { unimplemented_sse(); }

#[no_mangle]
pub unsafe fn instr_0F7C() { unimplemented_sse(); }
#[no_mangle]
pub unsafe fn instr_0F7D() { unimplemented_sse(); }

#[no_mangle]
pub unsafe fn instr_660F7C(source: reg128, r: i32) {
    // haddpd xmm1, xmm2/m128
    let destination = read_xmm128s(r);
    write_xmm_reg128(
        r,
        reg128 {
            f64: [
                destination.f64[0] + destination.f64[1],
                source.f64[0] + source.f64[1],
            ],
        },
    );
}
pub unsafe fn instr_660F7C_reg(r1: i32, r2: i32) { instr_660F7C(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F7C_mem(addr: i32, r: i32) {
    instr_660F7C(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_F20F7C(source: reg128, r: i32) {
    // haddps xmm, xmm/mem128
    let destination = read_xmm128s(r);
    write_xmm_reg128(
        r,
        reg128 {
            f32: [
                destination.f32[0] + destination.f32[1],
                destination.f32[2] + destination.f32[3],
                source.f32[0] + source.f32[1],
                source.f32[2] + source.f32[3],
            ],
        },
    );
}
pub unsafe fn instr_F20F7C_reg(r1: i32, r2: i32) { instr_F20F7C(read_xmm128s(r1), r2); }
pub unsafe fn instr_F20F7C_mem(addr: i32, r: i32) {
    instr_F20F7C(return_on_pagefault!(safe_read128s(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_660F7D(source: reg128, r: i32) {
    // hsubpd xmm1, xmm2/m128
    let destination = read_xmm128s(r);
    write_xmm_reg128(
        r,
        reg128 {
            f64: [
                destination.f64[0] - destination.f64[1],
                source.f64[0] - source.f64[1],
            ],
        },
    );
}
pub unsafe fn instr_660F7D_reg(r1: i32, r2: i32) { instr_660F7D(read_xmm128s(r1), r2); }
pub unsafe fn instr_660F7D_mem(addr: i32, r: i32) {
    instr_660F7D(return_on_pagefault!(safe_read128s(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_F20F7D(source: reg128, r: i32) {
    // hsubps xmm1, xmm2/m128
    let destination = read_xmm128s(r);
    write_xmm_reg128(
        r,
        reg128 {
            f32: [
                destination.f32[0] - destination.f32[1],
                destination.f32[2] - destination.f32[3],
                source.f32[0] - source.f32[1],
                source.f32[2] - source.f32[3],
            ],
        },
    );
}
pub unsafe fn instr_F20F7D_reg(r1: i32, r2: i32) { instr_F20F7D(read_xmm128s(r1), r2); }
pub unsafe fn instr_F20F7D_mem(addr: i32, r: i32) {
    instr_F20F7D(return_on_pagefault!(safe_read128s(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_0F7E(r: i32) -> i32 {
    // movd r/m32, mm
    return read_mmx64s(r) as i32;
}
pub unsafe fn instr_0F7E_reg(r1: i32, r2: i32) {
    write_reg32(r1, instr_0F7E(r2));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0F7E_mem(addr: i32, r: i32) {
    return_on_pagefault!(safe_write32(addr, instr_0F7E(r)));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_660F7E(r: i32) -> i32 {
    // movd r/m32, xmm
    let data = read_xmm64s(r);
    return data as i32;
}
pub unsafe fn instr_660F7E_reg(r1: i32, r2: i32) { write_reg32(r1, instr_660F7E(r2)); }
pub unsafe fn instr_660F7E_mem(addr: i32, r: i32) {
    return_on_pagefault!(safe_write32(addr, instr_660F7E(r)));
}
pub unsafe fn instr_F30F7E_mem(addr: i32, r: i32) {
    // movq xmm, xmm/mem64
    let data = return_on_pagefault!(safe_read64s(addr));
    write_xmm128_2(r, data, 0);
}
pub unsafe fn instr_F30F7E_reg(r1: i32, r2: i32) {
    // movq xmm, xmm/mem64
    write_xmm128_2(r2, read_xmm64s(r1), 0);
}

#[no_mangle]
pub unsafe fn instr_0F7F(r: i32) -> u64 {
    // movq mm/m64, mm
    read_mmx64s(r)
}
pub unsafe fn instr_0F7F_mem(addr: i32, r: i32) {
    // movq mm/m64, mm
    mov_r_m64(addr, r);
}
#[no_mangle]
pub unsafe fn instr_0F7F_reg(r1: i32, r2: i32) {
    // movq mm/m64, mm
    write_mmx_reg64(r1, read_mmx64s(r2));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_660F7F_mem(addr: i32, r: i32) {
    // movdqa xmm/m128, xmm
    // XXX: Aligned write or #gp
    mov_r_m128(addr, r);
}
pub unsafe fn instr_660F7F_reg(r1: i32, r2: i32) {
    // movdqa xmm/m128, xmm
    // XXX: Aligned access or #gp
    mov_r_r128(r1, r2);
}
pub unsafe fn instr_F30F7F_mem(addr: i32, r: i32) {
    // movdqu xmm/m128, xmm
    mov_r_m128(addr, r);
}
pub unsafe fn instr_F30F7F_reg(r1: i32, r2: i32) {
    // movdqu xmm/m128, xmm
    mov_r_r128(r1, r2);
}

pub unsafe fn instr16_0F80(imm: i32) { jmpcc16(test_o(), imm); }
pub unsafe fn instr32_0F80(imm: i32) { jmpcc32(test_o(), imm); }
pub unsafe fn instr16_0F81(imm: i32) { jmpcc16(!test_o(), imm); }
pub unsafe fn instr32_0F81(imm: i32) { jmpcc32(!test_o(), imm); }
pub unsafe fn instr16_0F82(imm: i32) { jmpcc16(test_b(), imm); }
pub unsafe fn instr32_0F82(imm: i32) { jmpcc32(test_b(), imm); }
pub unsafe fn instr16_0F83(imm: i32) { jmpcc16(!test_b(), imm); }
pub unsafe fn instr32_0F83(imm: i32) { jmpcc32(!test_b(), imm); }
pub unsafe fn instr16_0F84(imm: i32) { jmpcc16(test_z(), imm); }
pub unsafe fn instr32_0F84(imm: i32) { jmpcc32(test_z(), imm); }
pub unsafe fn instr16_0F85(imm: i32) { jmpcc16(!test_z(), imm); }
pub unsafe fn instr32_0F85(imm: i32) { jmpcc32(!test_z(), imm); }
pub unsafe fn instr16_0F86(imm: i32) { jmpcc16(test_be(), imm); }
pub unsafe fn instr32_0F86(imm: i32) { jmpcc32(test_be(), imm); }
pub unsafe fn instr16_0F87(imm: i32) { jmpcc16(!test_be(), imm); }
pub unsafe fn instr32_0F87(imm: i32) { jmpcc32(!test_be(), imm); }
pub unsafe fn instr16_0F88(imm: i32) { jmpcc16(test_s(), imm); }
pub unsafe fn instr32_0F88(imm: i32) { jmpcc32(test_s(), imm); }
pub unsafe fn instr16_0F89(imm: i32) { jmpcc16(!test_s(), imm); }
pub unsafe fn instr32_0F89(imm: i32) { jmpcc32(!test_s(), imm); }
pub unsafe fn instr16_0F8A(imm: i32) { jmpcc16(test_p(), imm); }
pub unsafe fn instr32_0F8A(imm: i32) { jmpcc32(test_p(), imm); }
pub unsafe fn instr16_0F8B(imm: i32) { jmpcc16(!test_p(), imm); }
pub unsafe fn instr32_0F8B(imm: i32) { jmpcc32(!test_p(), imm); }
pub unsafe fn instr16_0F8C(imm: i32) { jmpcc16(test_l(), imm); }
pub unsafe fn instr32_0F8C(imm: i32) { jmpcc32(test_l(), imm); }
pub unsafe fn instr16_0F8D(imm: i32) { jmpcc16(!test_l(), imm); }
pub unsafe fn instr32_0F8D(imm: i32) { jmpcc32(!test_l(), imm); }
pub unsafe fn instr16_0F8E(imm: i32) { jmpcc16(test_le(), imm); }
pub unsafe fn instr32_0F8E(imm: i32) { jmpcc32(test_le(), imm); }
pub unsafe fn instr16_0F8F(imm: i32) { jmpcc16(!test_le(), imm); }
pub unsafe fn instr32_0F8F(imm: i32) { jmpcc32(!test_le(), imm); }

pub unsafe fn instr_0F90_reg(r: i32, _: i32) { setcc_reg(test_o(), r); }
pub unsafe fn instr_0F91_reg(r: i32, _: i32) { setcc_reg(!test_o(), r); }
pub unsafe fn instr_0F92_reg(r: i32, _: i32) { setcc_reg(test_b(), r); }
pub unsafe fn instr_0F93_reg(r: i32, _: i32) { setcc_reg(!test_b(), r); }
pub unsafe fn instr_0F94_reg(r: i32, _: i32) { setcc_reg(test_z(), r); }
pub unsafe fn instr_0F95_reg(r: i32, _: i32) { setcc_reg(!test_z(), r); }
pub unsafe fn instr_0F96_reg(r: i32, _: i32) { setcc_reg(test_be(), r); }
pub unsafe fn instr_0F97_reg(r: i32, _: i32) { setcc_reg(!test_be(), r); }
pub unsafe fn instr_0F98_reg(r: i32, _: i32) { setcc_reg(test_s(), r); }
pub unsafe fn instr_0F99_reg(r: i32, _: i32) { setcc_reg(!test_s(), r); }
pub unsafe fn instr_0F9A_reg(r: i32, _: i32) { setcc_reg(test_p(), r); }
pub unsafe fn instr_0F9B_reg(r: i32, _: i32) { setcc_reg(!test_p(), r); }
pub unsafe fn instr_0F9C_reg(r: i32, _: i32) { setcc_reg(test_l(), r); }
pub unsafe fn instr_0F9D_reg(r: i32, _: i32) { setcc_reg(!test_l(), r); }
pub unsafe fn instr_0F9E_reg(r: i32, _: i32) { setcc_reg(test_le(), r); }
pub unsafe fn instr_0F9F_reg(r: i32, _: i32) { setcc_reg(!test_le(), r); }
pub unsafe fn instr_0F90_mem(addr: i32, _: i32) { setcc_mem(test_o(), addr); }
pub unsafe fn instr_0F91_mem(addr: i32, _: i32) { setcc_mem(!test_o(), addr); }
pub unsafe fn instr_0F92_mem(addr: i32, _: i32) { setcc_mem(test_b(), addr); }
pub unsafe fn instr_0F93_mem(addr: i32, _: i32) { setcc_mem(!test_b(), addr); }
pub unsafe fn instr_0F94_mem(addr: i32, _: i32) { setcc_mem(test_z(), addr); }
pub unsafe fn instr_0F95_mem(addr: i32, _: i32) { setcc_mem(!test_z(), addr); }
pub unsafe fn instr_0F96_mem(addr: i32, _: i32) { setcc_mem(test_be(), addr); }
pub unsafe fn instr_0F97_mem(addr: i32, _: i32) { setcc_mem(!test_be(), addr); }
pub unsafe fn instr_0F98_mem(addr: i32, _: i32) { setcc_mem(test_s(), addr); }
pub unsafe fn instr_0F99_mem(addr: i32, _: i32) { setcc_mem(!test_s(), addr); }
pub unsafe fn instr_0F9A_mem(addr: i32, _: i32) { setcc_mem(test_p(), addr); }
pub unsafe fn instr_0F9B_mem(addr: i32, _: i32) { setcc_mem(!test_p(), addr); }
pub unsafe fn instr_0F9C_mem(addr: i32, _: i32) { setcc_mem(test_l(), addr); }
pub unsafe fn instr_0F9D_mem(addr: i32, _: i32) { setcc_mem(!test_l(), addr); }
pub unsafe fn instr_0F9E_mem(addr: i32, _: i32) { setcc_mem(test_le(), addr); }
pub unsafe fn instr_0F9F_mem(addr: i32, _: i32) { setcc_mem(!test_le(), addr); }

pub unsafe fn instr16_0FA0() {
    return_on_pagefault!(push16(*sreg.offset(FS as isize) as i32));
}
pub unsafe fn instr32_0FA0() { return_on_pagefault!(push32_sreg(FS)) }
#[no_mangle]
pub unsafe fn instr16_0FA1() {
    if !switch_seg(FS, return_on_pagefault!(safe_read16(get_stack_pointer(0)))) {
        return;
    }
    else {
        adjust_stack_reg(2);
        return;
    };
}
#[no_mangle]
pub unsafe fn instr32_0FA1() {
    if !switch_seg(
        FS,
        return_on_pagefault!(safe_read32s(get_stack_pointer(0))) & 0xFFFF,
    ) {
        return;
    }
    else {
        adjust_stack_reg(4);
        return;
    };
}
#[no_mangle]
pub unsafe fn instr_0FA2() {
    // cpuid
    // TODO: Fill in with less bogus values

    // http://lxr.linux.no/linux+%2a/arch/x86/include/asm/cpufeature.h
    // http://www.sandpile.org/x86/cpuid.htm
    // https://gitlab.com/x86-cpuid.org/x86-cpuid-db
    let mut eax = 0;
    let mut ecx = 0;
    let mut edx = 0;
    let mut ebx = 0;

    let level = read_reg32(EAX) as u32;

    match level {
        0 => {
            // maximum supported level (default 0x16, overwritten to 2 as a workaround for Windows NT)
            eax = cpuid_level as i32;

            ebx = 0x756E6547 | 0; // Genu
            edx = 0x49656E69 | 0; // ineI
            ecx = 0x6C65746E | 0; // ntel
        },

        1 => {
            eax = 9 | 14 << 4 | 6 << 8 | 8 << 16; // i7-8500Y (family 6, model 142)
            ebx = 1 << 16 | 8 << 8; // cpu count, clflush size
            // pclmul, sse3, ssse3, fma, sse4.1, sse4.2, movbe, popcnt, aes,
            // xsave, osxsave, avx, rdrand. Bit assignments:
            // https://gitlab.com/x86-cpuid.org/x86-cpuid-db
            ecx = 1 << 0 | 1 << 1 | 1 << 9 | 1 << 12 | 1 << 19 | 1 << 20 | 1 << 22 | 1 << 23
                | 1 << 25 | 1 << 26 | 1 << 27 | 1 << 28 | 1 << 30;
            let vme = 0 << 1;
            if config::VMWARE_HYPERVISOR_PORT {
                ecx |= 1 << 31
            }; // hypervisor
            edx = 1 |      // fpu
                    vme | 1 << 3 | 1 << 4 | 1 << 5 | 1 << 6 |  // vme, pse, tsc, msr, pae
                    1 << 8 | 1 << 11 | 1 << 13 | 1 << 15 | // cx8, sep, pge, cmov
                    1 << 23 | 1 << 24 | 1 << 25 | 1 << 26; // mmx, fxsr, sse1, sse2

            if *acpi_enabled
            //&& this.apic_enabled[0])
            {
                edx |= 1 << 9; // apic
            }
        },

        2 => {
            // Taken from http://siyobik.info.gf/main/reference/instruction/CPUID
            eax = 0x665B5001;
            ebx = 0;
            ecx = 0;
            edx = 0x007A7000;
        },

        4 => {
            // from my local machine
            match read_reg32(ECX) {
                0 => {
                    eax = 0x00000121;
                    ebx = 0x01c0003f;
                    ecx = 0x0000003f;
                    edx = 0x00000001;
                },
                1 => {
                    eax = 0x00000122;
                    ebx = 0x01c0003f;
                    ecx = 0x0000003f;
                    edx = 0x00000001;
                },
                2 => {
                    eax = 0x00000143;
                    ebx = 0x05c0003f;
                    ecx = 0x00000fff;
                    edx = 0x00000001;
                },
                _ => {},
            }
        },

        5 => {
            // from my local machine
            eax = 0x40;
            ebx = 0x40;
            ecx = 3;
            edx = 0x00142120;
        },

        7 => {
            if read_reg32(ECX) == 0 {
                eax = 0; // maximum supported sub-level
                // bmi1, avx2, bmi2, erms, rdseed, adx
                // https://gitlab.com/x86-cpuid.org/x86-cpuid-db
                ebx = 1 << 3 | 1 << 5 | 1 << 8 | 1 << 9 | 1 << 18 | 1 << 19;
                ecx = 0;
                edx = 0;
            }
        },

        // XSAVE features: x87 (160 @ 0), SSE (256 @ 160) and AVX (256 @ 576).
        0xD => {
            match read_reg32(ECX) {
                0 => {
                    let xcr0 = crate::cpu::interp::interp64::get_xcr0();
                    // EAX/EDX: supported XCR0 bits. EBX: size required by the
                    // enabled XCR0 features (512 legacy + 64 header, plus 256
                    // for AVX). ECX: size required by all supported features.
                    eax = 7;
                    ebx = 576 + if xcr0 & 4 != 0 { 256 } else { 0 };
                    ecx = 832;
                    edx = 0;
                },
                // Sub-leaf n describes the state component for XCR0 bit n
                // (Linux reads sub-leaf XFEATURE_YMM == 2 for the AVX state).
                // Sub-leaf 1: XSAVE sub-features (none) and the XSAVES size.
                1 => {
                    eax = 0; // no XSAVEOPT/XSAVEC/XGETBV/XSAVES
                    ebx = 832;
                    ecx = 0;
                    edx = 0;
                },
                2 => {
                    eax = 256; // YMM (AVX) size
                    ebx = 576; // YMM offset
                    ecx = 0;
                    edx = 0;
                },
                _ => {
                    eax = 0;
                    ebx = 0;
                    ecx = 0;
                    edx = 0;
                },
            }
        },

        0x80000000 => {
            eax = 0x80000004u32 as i32; // brand string in 0x80000002-4
        },

        0x80000001 => {
            eax = 9 | 14 << 4 | 6 << 8 | 8 << 16;
            edx = 1 << 29 | 1 << 11; // LM, SYSCALL (NX and RDTSCP are not implemented)
        },

        0x80000002 | 0x80000003 | 0x80000004 => {
            let brand = b"Intel(R) Core(TM) i7-8500Y CPU @ 1.50GHz        ";
            let off = (level - 0x80000002) as usize * 16;
            eax = i32::from_le_bytes(brand[off..off + 4].try_into().unwrap());
            ebx = i32::from_le_bytes(brand[off + 4..off + 8].try_into().unwrap());
            ecx = i32::from_le_bytes(brand[off + 8..off + 12].try_into().unwrap());
            edx = i32::from_le_bytes(brand[off + 12..off + 16].try_into().unwrap());
        },

        0x80000008 => {
            // physical address bits 32 (wasm32), virtual address bits 48
            eax = 32 | 48 << 8;
        },

        0x40000000 => {
            // hypervisor
            if config::VMWARE_HYPERVISOR_PORT {
                // h("Ware".split("").reduce((a, c, i) => a | c.charCodeAt(0) << i * 8, 0))
                ebx = 0x61774D56 | 0; // VMwa
                ecx = 0x4D566572 | 0; // reVM
                edx = 0x65726177 | 0; // ware
            }
        },

        0x15 => {
            eax = 1; // denominator
            ebx = 1; // numerator
            ecx = (TSC_RATE * 1000.0) as u32 as i32; // core crystal clock frequency in Hz
            dbg_assert!(ecx > 0);
            //  (TSC frequency = core crystal clock frequency * EBX/EAX)
        },

        0x16 => {
            eax = (TSC_RATE / 1000.0).floor() as u32 as i32; // core base frequency in MHz
            ebx = 4200; // core maximum frequency in MHz
            ecx = 100; // bus (reference) frequency in MHz

            // 16-bit values
            dbg_assert!(eax < 0x10000);
            dbg_assert!(ebx < 0x10000);
            dbg_assert!(ecx < 0x10000);
        },

        x => {
            dbg_log!("cpuid: unimplemented eax: {:x}", x);
        },
    }

    if level == 4 || level == 7 {
        dbg_log!(
            "cpuid: eax={:08x} ecx={:02x}",
            read_reg32(EAX),
            read_reg32(ECX),
        );
    }
    else if level != 0 && level != 2 && level != 0x80000000 {
        dbg_log!("cpuid: eax={:08x}", read_reg32(EAX));
    }

    write_reg32(EAX, eax);
    write_reg32(ECX, ecx);
    write_reg32(EDX, edx);
    write_reg32(EBX, ebx);
}
pub unsafe fn instr16_0FA3_reg(r1: i32, r2: i32) { bt_reg(read_reg16(r1), read_reg16(r2) & 15); }
pub unsafe fn instr16_0FA3_mem(addr: i32, r: i32) { bt_mem(addr, read_reg16(r) << 16 >> 16); }
pub unsafe fn instr32_0FA3_reg(r1: i32, r2: i32) { bt_reg(read_reg32(r1), read_reg32(r2) & 31); }
pub unsafe fn instr32_0FA3_mem(addr: i32, r: i32) { bt_mem(addr, read_reg32(r)); }
pub unsafe fn instr16_0FA4_mem(addr: i32, r: i32, imm: i32) {
    safe_read_write16(addr, &|x| shld16(x, read_reg16(r), imm & 31))
}
pub unsafe fn instr16_0FA4_reg(r1: i32, r: i32, imm: i32) {
    write_reg16(r1, shld16(read_reg16(r1), read_reg16(r), imm & 31));
}
pub unsafe fn instr32_0FA4_mem(addr: i32, r: i32, imm: i32) {
    safe_read_write32(addr, &|x| shld32(x, read_reg32(r), imm & 31))
}
pub unsafe fn instr32_0FA4_reg(r1: i32, r: i32, imm: i32) {
    write_reg32(r1, shld32(read_reg32(r1), read_reg32(r), imm & 31));
}
pub unsafe fn instr16_0FA5_mem(addr: i32, r: i32) {
    safe_read_write16(addr, &|x| shld16(x, read_reg16(r), read_reg8(CL) & 31))
}
pub unsafe fn instr16_0FA5_reg(r1: i32, r: i32) {
    write_reg16(
        r1,
        shld16(read_reg16(r1), read_reg16(r), read_reg8(CL) & 31),
    );
}
pub unsafe fn instr32_0FA5_mem(addr: i32, r: i32) {
    safe_read_write32(addr, &|x| shld32(x, read_reg32(r), read_reg8(CL) & 31))
}
pub unsafe fn instr32_0FA5_reg(r1: i32, r: i32) {
    write_reg32(
        r1,
        shld32(read_reg32(r1), read_reg32(r), read_reg8(CL) & 31),
    );
}
#[no_mangle]
pub unsafe fn instr_0FA6() {
    // obsolete cmpxchg (os/2)
    trigger_ud();
}
#[no_mangle]
pub unsafe fn instr_0FA7() { undefined_instruction(); }
pub unsafe fn instr16_0FA8() {
    return_on_pagefault!(push16(*sreg.offset(GS as isize) as i32));
}
pub unsafe fn instr32_0FA8() { return_on_pagefault!(push32_sreg(GS)) }
#[no_mangle]
pub unsafe fn instr16_0FA9() {
    if !switch_seg(GS, return_on_pagefault!(safe_read16(get_stack_pointer(0)))) {
        return;
    }
    else {
        adjust_stack_reg(2);
        return;
    };
}
#[no_mangle]
pub unsafe fn instr32_0FA9() {
    if !switch_seg(
        GS,
        return_on_pagefault!(safe_read32s(get_stack_pointer(0))) & 0xFFFF,
    ) {
        return;
    }
    else {
        adjust_stack_reg(4);
        return;
    };
}
#[no_mangle]
pub unsafe fn instr_0FAA() {
    // rsm
    undefined_instruction();
}
#[no_mangle]
pub unsafe fn instr16_0FAB_reg(r1: i32, r2: i32) {
    write_reg16(r1, bts_reg(read_reg16(r1), read_reg16(r2) & 15));
}
#[no_mangle]
pub unsafe fn instr16_0FAB_mem(addr: i32, r: i32) { bts_mem(addr, read_reg16(r) << 16 >> 16); }
#[no_mangle]
pub unsafe fn instr32_0FAB_reg(r1: i32, r2: i32) {
    write_reg32(r1, bts_reg(read_reg32(r1), read_reg32(r2) & 31));
}
#[no_mangle]
pub unsafe fn instr32_0FAB_mem(addr: i32, r: i32) { bts_mem(addr, read_reg32(r)); }
pub unsafe fn instr16_0FAC_mem(addr: i32, r: i32, imm: i32) {
    safe_read_write16(addr, &|x| shrd16(x, read_reg16(r), imm & 31))
}
pub unsafe fn instr16_0FAC_reg(r1: i32, r: i32, imm: i32) {
    write_reg16(r1, shrd16(read_reg16(r1), read_reg16(r), imm & 31));
}
pub unsafe fn instr32_0FAC_mem(addr: i32, r: i32, imm: i32) {
    safe_read_write32(addr, &|x| shrd32(x, read_reg32(r), imm & 31))
}
pub unsafe fn instr32_0FAC_reg(r1: i32, r: i32, imm: i32) {
    write_reg32(r1, shrd32(read_reg32(r1), read_reg32(r), imm & 31));
}
pub unsafe fn instr16_0FAD_mem(addr: i32, r: i32) {
    safe_read_write16(addr, &|x| shrd16(x, read_reg16(r), read_reg8(CL) & 31))
}
pub unsafe fn instr16_0FAD_reg(r1: i32, r: i32) {
    write_reg16(
        r1,
        shrd16(read_reg16(r1), read_reg16(r), read_reg8(CL) & 31),
    );
}
pub unsafe fn instr32_0FAD_mem(addr: i32, r: i32) {
    safe_read_write32(addr, &|x| shrd32(x, read_reg32(r), read_reg8(CL) & 31))
}
pub unsafe fn instr32_0FAD_reg(r1: i32, r: i32) {
    write_reg32(
        r1,
        shrd32(read_reg32(r1), read_reg32(r), read_reg8(CL) & 31),
    );
}
#[no_mangle]
pub unsafe fn instr_0FAE_0_reg(_r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_0FAE_0_mem(addr: i32) { fxsave(addr); }
#[no_mangle]
pub unsafe fn instr_0FAE_1_reg(_r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_0FAE_1_mem(addr: i32) { fxrstor(addr); }
#[no_mangle]
pub unsafe fn instr_0FAE_2_reg(_r: i32) { unimplemented_sse(); }
#[no_mangle]
pub unsafe fn instr_0FAE_2_mem(addr: i32) {
    // ldmxcsr
    let new_mxcsr = return_on_pagefault!(safe_read32s(addr));
    if 0 != new_mxcsr & !MXCSR_MASK {
        dbg_log!("Invalid mxcsr bits: {:x}", new_mxcsr & !MXCSR_MASK);
        trigger_gp(0);
        return;
    }
    else {
        set_mxcsr(new_mxcsr);
        return;
    };
}
#[no_mangle]
pub unsafe fn instr_0FAE_3_reg(_r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_0FAE_3_mem(addr: i32) {
    // stmxcsr
    return_on_pagefault!(safe_write32(addr, *mxcsr));
}
#[no_mangle]
pub unsafe fn instr_0FAE_4_reg(_r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_0FAE_4_mem(addr: i32) {
    // XSAVE: the legacy FXSAVE area plus the XSTATE_BV header, and the YMM
    // halves when AVX is enabled in XCR0. The 32-bit kernel uses this for FPU
    // context switches, so a stub #UDs early in boot.
    // cf. Intel SDM Vol. 1, XSAVE.
    fxsave(addr);
    let mask = read_reg32(EAX) as u32 as u64 | (read_reg32(EDX) as u32 as u64) << 32;
    let requested = mask & crate::cpu::interp::interp64::get_xcr0();
    return_on_pagefault!(safe_write64(addr + 512, requested));
    return_on_pagefault!(safe_write64(addr + 520, 0));
    if requested & 4 != 0 {
        for i in 0..8 {
            let hi = *crate::cpu::global_pointers::ymm_high_ptr(i);
            return_on_pagefault!(safe_write64(addr + 576 + i * 16, hi.u64[0]));
            return_on_pagefault!(safe_write64(addr + 576 + i * 16 + 8, hi.u64[1]));
        }
    }
}
pub unsafe fn instr_0FAE_5_reg(_r: i32) {
    // lfence
}
#[no_mangle]
pub unsafe fn instr_0FAE_5_mem(addr: i32) {
    // XRSTOR: restore the legacy area; the header selects the extra state.
    let xstate_bv = return_on_pagefault!(safe_read64s(addr + 512));
    fxrstor(addr);
    if xstate_bv & 4 != 0 {
        for i in 0..8 {
            let lo = return_on_pagefault!(safe_read64s(addr + 576 + i * 16));
            let hi = return_on_pagefault!(safe_read64s(addr + 576 + i * 16 + 8));
            *crate::cpu::global_pointers::ymm_high_ptr(i) = reg128 { u64: [lo, hi] };
        }
    }
}
#[no_mangle]
pub unsafe fn instr_0FAE_6_reg(_r: i32) {
    // mfence
}
#[no_mangle]
pub unsafe fn instr_0FAE_6_mem(addr: i32) {
    // xsaveopt: same state as XSAVE, no dirty-tracking optimisation.
    instr_0FAE_4_mem(addr);
}
#[no_mangle]
pub unsafe fn instr_0FAE_7_reg(_r: i32) {
    // sfence
}
#[no_mangle]
pub unsafe fn instr_0FAE_7_mem(_addr: i32) {
    // clflush
    undefined_instruction();
}
pub unsafe fn instr16_0FAF_mem(addr: i32, r: i32) {
    write_reg16(
        r,
        imul_reg16(read_reg16(r), return_on_pagefault!(safe_read16(addr))),
    );
}
pub unsafe fn instr16_0FAF_reg(r1: i32, r: i32) {
    write_reg16(r, imul_reg16(read_reg16(r), read_reg16(r1)));
}
pub unsafe fn instr32_0FAF_mem(addr: i32, r: i32) {
    write_reg32(
        r,
        imul_reg32(read_reg32(r), return_on_pagefault!(safe_read32s(addr))),
    );
}
pub unsafe fn instr32_0FAF_reg(r1: i32, r: i32) {
    write_reg32(r, imul_reg32(read_reg32(r), read_reg32(r1)));
}

#[no_mangle]
pub unsafe fn instr_0FB0_reg(r1: i32, r2: i32) { write_reg8(r1, cmpxchg8(read_reg8(r1), r2)); }
#[no_mangle]
pub unsafe fn instr_0FB0_mem(addr: i32, r: i32) { safe_read_write8(addr, &|x| cmpxchg8(x, r)) }
pub unsafe fn instr16_0FB1_reg(r1: i32, r2: i32) { write_reg16(r1, cmpxchg16(read_reg16(r1), r2)); }
pub unsafe fn instr16_0FB1_mem(addr: i32, r: i32) { safe_read_write16(addr, &|x| cmpxchg16(x, r)) }
pub unsafe fn instr32_0FB1_reg(r1: i32, r2: i32) { write_reg32(r1, cmpxchg32(read_reg32(r1), r2)); }
pub unsafe fn instr32_0FB1_mem(addr: i32, r: i32) { safe_read_write32(addr, &|x| cmpxchg32(x, r)) }

#[no_mangle]
pub unsafe fn instr16_0FB2_reg(_unused: i32, _unused2: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr16_0FB2_mem(addr: i32, r: i32) { lss16(addr, r, SS); }
#[no_mangle]
pub unsafe fn instr32_0FB2_reg(_unused: i32, _unused2: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr32_0FB2_mem(addr: i32, r: i32) { lss32(addr, r, SS); }
#[no_mangle]
pub unsafe fn instr16_0FB3_reg(r1: i32, r2: i32) {
    write_reg16(r1, btr_reg(read_reg16(r1), read_reg16(r2) & 15));
}
#[no_mangle]
pub unsafe fn instr16_0FB3_mem(addr: i32, r: i32) { btr_mem(addr, read_reg16(r) << 16 >> 16); }
#[no_mangle]
pub unsafe fn instr32_0FB3_reg(r1: i32, r2: i32) {
    write_reg32(r1, btr_reg(read_reg32(r1), read_reg32(r2) & 31));
}
#[no_mangle]
pub unsafe fn instr32_0FB3_mem(addr: i32, r: i32) { btr_mem(addr, read_reg32(r)); }
#[no_mangle]
pub unsafe fn instr16_0FB4_reg(_unused: i32, _unused2: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr16_0FB4_mem(addr: i32, r: i32) { lss16(addr, r, FS); }
#[no_mangle]
pub unsafe fn instr32_0FB4_reg(_unused: i32, _unused2: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr32_0FB4_mem(addr: i32, r: i32) { lss32(addr, r, FS); }
#[no_mangle]
pub unsafe fn instr16_0FB5_reg(_unused: i32, _unused2: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr16_0FB5_mem(addr: i32, r: i32) { lss16(addr, r, GS); }
#[no_mangle]
pub unsafe fn instr32_0FB5_reg(_unused: i32, _unused2: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr32_0FB5_mem(addr: i32, r: i32) { lss32(addr, r, GS); }
pub unsafe fn instr16_0FB6_mem(addr: i32, r: i32) {
    write_reg16(r, return_on_pagefault!(safe_read8(addr)));
}
pub unsafe fn instr16_0FB6_reg(r1: i32, r: i32) { write_reg16(r, read_reg8(r1)); }
pub unsafe fn instr32_0FB6_mem(addr: i32, r: i32) {
    write_reg32(r, return_on_pagefault!(safe_read8(addr)));
}
pub unsafe fn instr32_0FB6_reg(r1: i32, r: i32) { write_reg32(r, read_reg8(r1)); }
pub unsafe fn instr16_0FB7_mem(addr: i32, r: i32) {
    write_reg16(r, return_on_pagefault!(safe_read16(addr)));
}
pub unsafe fn instr16_0FB7_reg(r1: i32, r: i32) { write_reg16(r, read_reg16(r1)); }
pub unsafe fn instr32_0FB7_mem(addr: i32, r: i32) {
    write_reg32(r, return_on_pagefault!(safe_read16(addr)));
}
pub unsafe fn instr32_0FB7_reg(r1: i32, r: i32) { write_reg32(r, read_reg16(r1)); }
#[no_mangle]
pub unsafe fn instr16_0FB8_reg(_r1: i32, _r2: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr16_0FB8_mem(_addr: i32, _r: i32) { trigger_ud(); }
pub unsafe fn instr16_F30FB8_mem(addr: i32, r: i32) {
    write_reg16(r, popcnt(return_on_pagefault!(safe_read16(addr))));
}
pub unsafe fn instr16_F30FB8_reg(r1: i32, r: i32) { write_reg16(r, popcnt(read_reg16(r1))); }
#[no_mangle]
pub unsafe fn instr32_0FB8_reg(_r1: i32, _r2: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr32_0FB8_mem(_addr: i32, _r: i32) { trigger_ud(); }
pub unsafe fn instr32_F30FB8_mem(addr: i32, r: i32) {
    write_reg32(r, popcnt(return_on_pagefault!(safe_read32s(addr))));
}
pub unsafe fn instr32_F30FB8_reg(r1: i32, r: i32) { write_reg32(r, popcnt(read_reg32(r1))); }
#[no_mangle]
pub unsafe fn instr_0FB9() {
    // UD2
    trigger_ud();
}
pub unsafe fn instr16_0FBA_4_reg(r: i32, imm: i32) { bt_reg(read_reg16(r), imm & 15); }
pub unsafe fn instr16_0FBA_4_mem(addr: i32, imm: i32) { bt_mem(addr, imm & 15); }
#[no_mangle]
pub unsafe fn instr16_0FBA_5_reg(r: i32, imm: i32) {
    write_reg16(r, bts_reg(read_reg16(r), imm & 15));
}
#[no_mangle]
pub unsafe fn instr16_0FBA_5_mem(addr: i32, imm: i32) { bts_mem(addr, imm & 15); }
#[no_mangle]
pub unsafe fn instr16_0FBA_6_reg(r: i32, imm: i32) {
    write_reg16(r, btr_reg(read_reg16(r), imm & 15));
}
#[no_mangle]
pub unsafe fn instr16_0FBA_6_mem(addr: i32, imm: i32) { btr_mem(addr, imm & 15); }
#[no_mangle]
pub unsafe fn instr16_0FBA_7_reg(r: i32, imm: i32) {
    write_reg16(r, btc_reg(read_reg16(r), imm & 15));
}
#[no_mangle]
pub unsafe fn instr16_0FBA_7_mem(addr: i32, imm: i32) { btc_mem(addr, imm & 15); }
pub unsafe fn instr32_0FBA_4_reg(r: i32, imm: i32) { bt_reg(read_reg32(r), imm & 31); }
pub unsafe fn instr32_0FBA_4_mem(addr: i32, imm: i32) { bt_mem(addr, imm & 31); }
#[no_mangle]
pub unsafe fn instr32_0FBA_5_reg(r: i32, imm: i32) {
    write_reg32(r, bts_reg(read_reg32(r), imm & 31));
}
#[no_mangle]
pub unsafe fn instr32_0FBA_5_mem(addr: i32, imm: i32) { bts_mem(addr, imm & 31); }
#[no_mangle]
pub unsafe fn instr32_0FBA_6_reg(r: i32, imm: i32) {
    write_reg32(r, btr_reg(read_reg32(r), imm & 31));
}
#[no_mangle]
pub unsafe fn instr32_0FBA_6_mem(addr: i32, imm: i32) { btr_mem(addr, imm & 31); }
#[no_mangle]
pub unsafe fn instr32_0FBA_7_reg(r: i32, imm: i32) {
    write_reg32(r, btc_reg(read_reg32(r), imm & 31));
}
#[no_mangle]
pub unsafe fn instr32_0FBA_7_mem(addr: i32, imm: i32) { btc_mem(addr, imm & 31); }
#[no_mangle]
pub unsafe fn instr16_0FBB_reg(r1: i32, r2: i32) {
    write_reg16(r1, btc_reg(read_reg16(r1), read_reg16(r2) & 15));
}
#[no_mangle]
pub unsafe fn instr16_0FBB_mem(addr: i32, r: i32) { btc_mem(addr, read_reg16(r) << 16 >> 16); }
#[no_mangle]
pub unsafe fn instr32_0FBB_reg(r1: i32, r2: i32) {
    write_reg32(r1, btc_reg(read_reg32(r1), read_reg32(r2) & 31));
}
#[no_mangle]
pub unsafe fn instr32_0FBB_mem(addr: i32, r: i32) { btc_mem(addr, read_reg32(r)); }
pub unsafe fn instr16_0FBC_mem(addr: i32, r: i32) {
    write_reg16(
        r,
        bsf16(read_reg16(r), return_on_pagefault!(safe_read16(addr))),
    );
}
pub unsafe fn instr16_0FBC_reg(r1: i32, r: i32) {
    write_reg16(r, bsf16(read_reg16(r), read_reg16(r1)));
}
pub unsafe fn instr32_0FBC_mem(addr: i32, r: i32) {
    write_reg32(
        r,
        bsf32(read_reg32(r), return_on_pagefault!(safe_read32s(addr))),
    );
}
pub unsafe fn instr32_0FBC_reg(r1: i32, r: i32) {
    write_reg32(r, bsf32(read_reg32(r), read_reg32(r1)));
}
pub unsafe fn instr16_0FBD_mem(addr: i32, r: i32) {
    write_reg16(
        r,
        bsr16(read_reg16(r), return_on_pagefault!(safe_read16(addr))),
    );
}
pub unsafe fn instr16_0FBD_reg(r1: i32, r: i32) {
    write_reg16(r, bsr16(read_reg16(r), read_reg16(r1)));
}
pub unsafe fn instr32_0FBD_mem(addr: i32, r: i32) {
    write_reg32(
        r,
        bsr32(read_reg32(r), return_on_pagefault!(safe_read32s(addr))),
    );
}
pub unsafe fn instr32_0FBD_reg(r1: i32, r: i32) {
    write_reg32(r, bsr32(read_reg32(r), read_reg32(r1)));
}
pub unsafe fn instr16_0FBE_mem(addr: i32, r: i32) {
    write_reg16(r, return_on_pagefault!(safe_read8(addr)) << 24 >> 24);
}
pub unsafe fn instr16_0FBE_reg(r1: i32, r: i32) { write_reg16(r, read_reg8(r1) << 24 >> 24); }
pub unsafe fn instr32_0FBE_mem(addr: i32, r: i32) {
    write_reg32(r, return_on_pagefault!(safe_read8(addr)) << 24 >> 24);
}
pub unsafe fn instr32_0FBE_reg(r1: i32, r: i32) { write_reg32(r, read_reg8(r1) << 24 >> 24); }
pub unsafe fn instr16_0FBF_mem(addr: i32, r: i32) {
    write_reg16(r, return_on_pagefault!(safe_read16(addr)) << 16 >> 16);
}
pub unsafe fn instr16_0FBF_reg(r1: i32, r: i32) { write_reg16(r, read_reg16(r1) << 16 >> 16); }
pub unsafe fn instr32_0FBF_mem(addr: i32, r: i32) {
    write_reg32(r, return_on_pagefault!(safe_read16(addr)) << 16 >> 16);
}
pub unsafe fn instr32_0FBF_reg(r1: i32, r: i32) { write_reg32(r, read_reg16(r1) << 16 >> 16); }
#[no_mangle]
pub unsafe fn instr_0FC0_mem(addr: i32, r: i32) { safe_read_write8(addr, &|x| xadd8(x, r)) }
#[no_mangle]
pub unsafe fn instr_0FC0_reg(r1: i32, r: i32) { write_reg8(r1, xadd8(read_reg8(r1), r)); }
pub unsafe fn instr16_0FC1_mem(addr: i32, r: i32) { safe_read_write16(addr, &|x| xadd16(x, r)) }
pub unsafe fn instr16_0FC1_reg(r1: i32, r: i32) { write_reg16(r1, xadd16(read_reg16(r1), r)); }
pub unsafe fn instr32_0FC1_mem(addr: i32, r: i32) { safe_read_write32(addr, &|x| xadd32(x, r)) }
pub unsafe fn instr32_0FC1_reg(r1: i32, r: i32) { write_reg32(r1, xadd32(read_reg32(r1), r)); }

#[no_mangle]
pub unsafe fn instr_0FC2(source: reg128, r: i32, imm8: i32) {
    // cmpps xmm, xmm/m128
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..4 {
        result.i32[i] = if sse_comparison(imm8, destination.f32[i] as f64, source.f32[i] as f64) {
            -1
        }
        else {
            0
        };
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_0FC2_reg(r1: i32, r2: i32, imm: i32) { instr_0FC2(read_xmm128s(r1), r2, imm); }
pub unsafe fn instr_0FC2_mem(addr: i32, r: i32, imm: i32) {
    instr_0FC2(return_on_pagefault!(safe_read128s(addr)), r, imm);
}
#[no_mangle]
pub unsafe fn instr_660FC2(source: reg128, r: i32, imm8: i32) {
    // cmppd xmm, xmm/m128
    let destination = read_xmm128s(r);
    let result = reg128 {
        i64: [
            (if sse_comparison(imm8, destination.f64[0], source.f64[0]) { -1 } else { 0 }) as i64,
            (if sse_comparison(imm8, destination.f64[1], source.f64[1]) { -1 } else { 0 }) as i64,
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FC2_reg(r1: i32, r2: i32, imm: i32) {
    instr_660FC2(read_xmm128s(r1), r2, imm);
}
pub unsafe fn instr_660FC2_mem(addr: i32, r: i32, imm: i32) {
    instr_660FC2(return_on_pagefault!(safe_read128s(addr)), r, imm);
}
#[no_mangle]
pub unsafe fn instr_F20FC2(source: u64, r: i32, imm8: i32) {
    // cmpsd xmm, xmm/m64
    let destination = read_xmm64s(r);
    write_xmm64(
        r,
        if sse_comparison(imm8, f64::from_bits(destination), f64::from_bits(source)) {
            (-1i32) as u64
        }
        else {
            0
        },
    );
}
pub unsafe fn instr_F20FC2_reg(r1: i32, r2: i32, imm: i32) {
    instr_F20FC2(read_xmm64s(r1), r2, imm);
}
pub unsafe fn instr_F20FC2_mem(addr: i32, r: i32, imm: i32) {
    instr_F20FC2(return_on_pagefault!(safe_read64s(addr)), r, imm);
}
#[no_mangle]
pub unsafe fn instr_F30FC2(source: i32, r: i32, imm8: i32) {
    // cmpss xmm, xmm/m32
    let destination = read_xmm_f32(r);
    let source: f32 = f32::from_bits(i32::cast_unsigned(source));
    let result = if sse_comparison(imm8, destination as f64, source as f64) { -1 } else { 0 };
    write_xmm32(r, result);
}
pub unsafe fn instr_F30FC2_reg(r1: i32, r2: i32, imm: i32) {
    instr_F30FC2(read_xmm64s(r1) as i32, r2, imm);
}
pub unsafe fn instr_F30FC2_mem(addr: i32, r: i32, imm: i32) {
    instr_F30FC2(return_on_pagefault!(safe_read32s(addr)), r, imm);
}

pub unsafe fn instr_0FC3_reg(_r1: i32, _r2: i32) { trigger_ud(); }
pub unsafe fn instr_0FC3_mem(addr: i32, r: i32) {
    // movnti
    return_on_pagefault!(safe_write32(addr, read_reg32(r)));
}

#[no_mangle]
pub unsafe fn instr_0FC4(source: i32, r: i32, imm8: i32) {
    // pinsrw mm, r32/m16, imm8
    let mut destination: [u16; 4] = std::mem::transmute(read_mmx64s(r));
    destination[(imm8 & 3) as usize] = source as u16;
    write_mmx_reg64(r, std::mem::transmute(destination));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FC4_reg(r1: i32, r2: i32, imm: i32) { instr_0FC4(read_reg32(r1), r2, imm); }
pub unsafe fn instr_0FC4_mem(addr: i32, r: i32, imm: i32) {
    instr_0FC4(return_on_pagefault!(safe_read16(addr)), r, imm);
}
pub unsafe fn instr_660FC4(source: i32, r: i32, imm8: i32) {
    // pinsrw xmm, r32/m16, imm8
    let mut destination = read_xmm128s(r);
    let index = (imm8 & 7) as u32;
    destination.u16[index as usize] = (source & 0xFFFF) as u16;
    write_xmm_reg128(r, destination);
}
pub unsafe fn instr_660FC4_reg(r1: i32, r2: i32, imm: i32) {
    instr_660FC4(read_reg32(r1), r2, imm);
}
pub unsafe fn instr_660FC4_mem(addr: i32, r: i32, imm: i32) {
    instr_660FC4(return_on_pagefault!(safe_read16(addr)), r, imm);
}
pub unsafe fn instr_0FC5_mem(_addr: i32, _r: i32, _imm8: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_0FC5_reg(r1: i32, r2: i32, imm8: i32) {
    // pextrw r32, mm, imm8
    let data: [u16; 4] = std::mem::transmute(read_mmx64s(r1));
    write_reg32(r2, data[(imm8 & 3) as usize] as i32);
    transition_fpu_to_mmx();
}
pub unsafe fn instr_660FC5_mem(_addr: i32, _r: i32, _imm8: i32) { trigger_ud(); }
pub unsafe fn instr_660FC5_reg(r1: i32, r2: i32, imm8: i32) {
    // pextrw r32, xmm, imm8
    let data = read_xmm128s(r1);
    let index = (imm8 & 7) as u32;
    let result = data.u16[index as usize] as u32;
    write_reg32(r2, result as i32);
}

#[no_mangle]
pub unsafe fn instr_0FC6(source: reg128, r: i32, imm8: i32) {
    // shufps xmm, xmm/mem128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    write_xmm128(
        r,
        destination.u32[(imm8 & 3) as usize] as i32,
        destination.u32[(imm8 >> 2 & 3) as usize] as i32,
        source.u32[(imm8 >> 4 & 3) as usize] as i32,
        source.u32[(imm8 >> 6 & 3) as usize] as i32,
    );
}
pub unsafe fn instr_0FC6_reg(r1: i32, r2: i32, imm: i32) { instr_0FC6(read_xmm128s(r1), r2, imm); }
pub unsafe fn instr_0FC6_mem(addr: i32, r: i32, imm: i32) {
    instr_0FC6(return_on_pagefault!(safe_read128s(addr)), r, imm);
}

#[no_mangle]
pub unsafe fn instr_660FC6(source: reg128, r: i32, imm8: i32) {
    // shufpd xmm, xmm/mem128
    let destination = read_xmm128s(r);
    let result = reg128 {
        i64: [
            destination.i64[imm8 as usize & 1],
            source.i64[imm8 as usize >> 1 & 1],
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FC6_reg(r1: i32, r2: i32, imm: i32) {
    instr_660FC6(read_xmm128s(r1), r2, imm);
}
pub unsafe fn instr_660FC6_mem(addr: i32, r: i32, imm: i32) {
    instr_660FC6(return_on_pagefault!(safe_read128s(addr)), r, imm);
}

pub unsafe fn instr16_0FC7_1_reg(_r: i32) { trigger_ud(); }
pub unsafe fn instr32_0FC7_1_reg(_r: i32) { trigger_ud(); }
pub unsafe fn instr16_0FC7_1_mem(addr: i32) {
    // cmpxchg8b
    return_on_pagefault!(writable_or_pagefault(addr, 8));
    let m64 = safe_read64s(addr).unwrap();
    let m64_low = m64 as i32;
    let m64_high = (m64 >> 32) as i32;
    if read_reg32(EAX) == m64_low && read_reg32(EDX) == m64_high {
        *flags |= FLAG_ZERO;
        safe_write64(
            addr,
            read_reg32(EBX) as u32 as u64 | (read_reg32(ECX) as u32 as u64) << 32,
        )
        .unwrap();
    }
    else {
        *flags &= !FLAG_ZERO;
        write_reg32(EAX, m64_low);
        write_reg32(EDX, m64_high);
    }
    *flags_changed &= !FLAG_ZERO;
}
pub unsafe fn instr32_0FC7_1_mem(addr: i32) { instr16_0FC7_1_mem(addr) }

#[no_mangle]
pub unsafe fn instr16_0FC7_6_reg(r: i32) {
    // rdrand
    let rand = js::get_rand_int();
    write_reg16(r, rand);
    *flags &= !FLAGS_ALL;
    *flags |= 1;
    *flags_changed = 0;
}
#[no_mangle]
pub unsafe fn instr32_0FC7_6_reg(r: i32) {
    // rdrand
    let rand = js::get_rand_int();
    write_reg32(r, rand);
    *flags &= !FLAGS_ALL;
    *flags |= 1;
    *flags_changed = 0;
}

#[no_mangle]
pub unsafe fn instr16_0FC7_6_mem(_addr: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr32_0FC7_6_mem(_addr: i32) { trigger_ud(); }

// RDSEED (0F C7 /7): same register/flag contract as RDRAND (CF = success); the
// entropy source is not modelled. Linux probes RDSEED via CPUID and uses it to
// seed the random pool, so a missing implementation #UDs early in boot.
#[no_mangle]
pub unsafe fn instr16_0FC7_7_reg(r: i32) {
    let rand = js::get_rand_int();
    write_reg16(r, rand);
    *flags &= !FLAGS_ALL;
    *flags |= 1;
    *flags_changed = 0;
}
#[no_mangle]
pub unsafe fn instr32_0FC7_7_reg(r: i32) {
    let rand = js::get_rand_int();
    write_reg32(r, rand);
    *flags &= !FLAGS_ALL;
    *flags |= 1;
    *flags_changed = 0;
}
#[no_mangle]
pub unsafe fn instr16_0FC7_7_mem(_addr: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr32_0FC7_7_mem(_addr: i32) { trigger_ud(); }

#[no_mangle]
pub unsafe fn instr_0FC8() { bswap(EAX); }
#[no_mangle]
pub unsafe fn instr_0FC9() { bswap(ECX); }
#[no_mangle]
pub unsafe fn instr_0FCA() { bswap(EDX); }
#[no_mangle]
pub unsafe fn instr_0FCB() { bswap(EBX); }
#[no_mangle]
pub unsafe fn instr_0FCC() { bswap(ESP); }
#[no_mangle]
pub unsafe fn instr_0FCD() { bswap(EBP); }
#[no_mangle]
pub unsafe fn instr_0FCE() { bswap(ESI); }
#[no_mangle]
pub unsafe fn instr_0FCF() { bswap(EDI); }
#[no_mangle]
pub unsafe fn instr_0FD0() { unimplemented_sse(); }
#[no_mangle]
pub unsafe fn instr_0FD1(source: u64, r: i32) {
    // psrlw mm, mm/m64
    psrlw_r64(r, source);
}
pub unsafe fn instr_0FD1_reg(r1: i32, r2: i32) { instr_0FD1(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FD1_mem(addr: i32, r: i32) {
    instr_0FD1(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FD1(source: reg128, r: i32) {
    // psrlw xmm, xmm/m128
    // XXX: Aligned access or #gp
    psrlw_r128(r, source.u64[0]);
}
pub unsafe fn instr_660FD1_reg(r1: i32, r2: i32) { instr_660FD1(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FD1_mem(addr: i32, r: i32) {
    instr_660FD1(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FD2(source: u64, r: i32) {
    // psrld mm, mm/m64
    psrld_r64(r, source);
}
pub unsafe fn instr_0FD2_reg(r1: i32, r2: i32) { instr_0FD2(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FD2_mem(addr: i32, r: i32) {
    instr_0FD2(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FD2(source: reg128, r: i32) {
    // psrld xmm, xmm/m128
    // XXX: Aligned access or #gp
    psrld_r128(r, source.u64[0]);
}
pub unsafe fn instr_660FD2_reg(r1: i32, r2: i32) { instr_660FD2(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FD2_mem(addr: i32, r: i32) {
    instr_660FD2(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FD3(source: u64, r: i32) {
    // psrlq mm, mm/m64
    psrlq_r64(r, source);
}
pub unsafe fn instr_0FD3_reg(r1: i32, r2: i32) { instr_0FD3(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FD3_mem(addr: i32, r: i32) {
    instr_0FD3(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FD3(source: reg128, r: i32) {
    // psrlq xmm, mm/m64
    psrlq_r128(r, source.u64[0]);
}
pub unsafe fn instr_660FD3_reg(r1: i32, r2: i32) { instr_660FD3(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FD3_mem(addr: i32, r: i32) {
    instr_660FD3(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FD4(source: u64, r: i32) {
    // paddq mm, mm/m64
    let destination = read_mmx64s(r);
    write_mmx_reg64(r, source + destination);
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FD4_reg(r1: i32, r2: i32) { instr_0FD4(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FD4_mem(addr: i32, r: i32) {
    instr_0FD4(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FD4(source: reg128, r: i32) {
    // paddq xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    result.u64[0] = destination.u64[0] + source.u64[0];
    result.u64[1] = destination.u64[1] + source.u64[1];
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FD4_reg(r1: i32, r2: i32) { instr_660FD4(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FD4_mem(addr: i32, r: i32) {
    instr_660FD4(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FD5(source: u64, r: i32) {
    // pmullw mm, mm/m64
    let destination: [i16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [i16; 4] = std::mem::transmute(source);
    let mut result = [0; 4];
    for i in 0..4 {
        result[i] = destination[i] * source[i];
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FD5_reg(r1: i32, r2: i32) { instr_0FD5(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FD5_mem(addr: i32, r: i32) {
    instr_0FD5(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FD5(source: reg128, r: i32) {
    // pmullw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.u16[i] = destination.u16[i] * source.u16[i]
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FD5_reg(r1: i32, r2: i32) { instr_660FD5(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FD5_mem(addr: i32, r: i32) {
    instr_660FD5(return_on_pagefault!(safe_read128s(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_0FD6_mem(_addr: i32, _r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_0FD6_reg(_r1: i32, _r2: i32) { trigger_ud(); }
pub unsafe fn instr_660FD6_mem(addr: i32, r: i32) {
    // movq xmm/m64, xmm
    movl_r128_m64(addr, r);
}
pub unsafe fn instr_660FD6_reg(r1: i32, r2: i32) {
    // movq xmm/m64, xmm
    write_xmm128_2(r1, read_xmm64s(r2), 0);
}

#[no_mangle]
pub unsafe fn instr_F20FD6_mem(_addr: i32, _r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_F20FD6_reg(r1: i32, r2: i32) {
    // movdq2q mm, xmm
    write_mmx_reg64(r2, read_xmm128s(r1).u64[0]);
    transition_fpu_to_mmx();
}
#[no_mangle]
pub unsafe fn instr_F30FD6_mem(_addr: i32, _r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_F30FD6_reg(r1: i32, r2: i32) {
    // movq2dq xmm, mm
    let source = read_mmx64s(r1);
    write_xmm_reg128(r2, reg128 { u64: [source, 0] });
    transition_fpu_to_mmx();
}

pub unsafe fn instr_0FD7_mem(_addr: i32, _r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_0FD7(r1: i32) -> i32 {
    // pmovmskb r, mm
    let x: [u8; 8] = u64::to_le_bytes(read_mmx64s(r1));
    let mut result = 0;
    for i in 0..8 {
        result |= x[i] as i32 >> 7 << i
    }
    transition_fpu_to_mmx();
    result
}
pub unsafe fn instr_0FD7_reg(r1: i32, r2: i32) { write_reg32(r2, instr_0FD7(r1)); }
pub unsafe fn instr_660FD7_mem(_addr: i32, _r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_660FD7(r1: i32) -> i32 {
    // pmovmskb reg, xmm
    let x = read_xmm128s(r1);
    let mut result = 0;
    for i in 0..16 {
        result |= x.u8[i] as i32 >> 7 << i
    }
    result
}
pub unsafe fn instr_660FD7_reg(r1: i32, r2: i32) { write_reg32(r2, instr_660FD7(r1)) }
#[no_mangle]
pub unsafe fn instr_0FD8(source: u64, r: i32) {
    // psubusb mm, mm/m64
    let destination: [u8; 8] = u64::to_le_bytes(read_mmx64s(r));
    let source: [u8; 8] = u64::to_le_bytes(source);
    let mut result = [0; 8];
    for i in 0..8 {
        result[i] = saturate_sd_to_ub(destination[i] as i32 - source[i] as i32) as u8;
    }
    write_mmx_reg64(r, u64::from_le_bytes(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FD8_reg(r1: i32, r2: i32) { instr_0FD8(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FD8_mem(addr: i32, r: i32) {
    instr_0FD8(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FD8(source: reg128, r: i32) {
    // psubusb xmm, xmm/m128
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..16 {
        result.u8[i] = saturate_sd_to_ub(destination.u8[i] as i32 - source.u8[i] as i32) as u8;
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FD8_reg(r1: i32, r2: i32) { instr_660FD8(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FD8_mem(addr: i32, r: i32) {
    instr_660FD8(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FD9(source: u64, r: i32) {
    // psubusw mm, mm/m64
    let destination: [u16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [u16; 4] = std::mem::transmute(source);
    let mut result = [0; 4];
    for i in 0..4 {
        result[i] = saturate_uw(destination[i] as u32 - source[i] as u32)
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FD9_reg(r1: i32, r2: i32) { instr_0FD9(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FD9_mem(addr: i32, r: i32) {
    instr_0FD9(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FD9(source: reg128, r: i32) {
    // psubusw xmm, xmm/m128
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.u16[i] = saturate_uw(destination.u16[i] as u32 - source.u16[i] as u32)
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FD9_reg(r1: i32, r2: i32) { instr_660FD9(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FD9_mem(addr: i32, r: i32) {
    instr_660FD9(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FDA(source: u64, r: i32) {
    // pminub mm, mm/m64
    let destination: [u8; 8] = u64::to_le_bytes(read_mmx64s(r));
    let source: [u8; 8] = u64::to_le_bytes(source);
    let mut result = [0; 8];
    for i in 0..8 {
        result[i] = u8::min(source[i], destination[i])
    }
    write_mmx_reg64(r, u64::from_le_bytes(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FDA_reg(r1: i32, r2: i32) { instr_0FDA(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FDA_mem(addr: i32, r: i32) {
    instr_0FDA(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FDA(source: reg128, r: i32) {
    // pminub xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { u8: [0; 16] };
    for i in 0..16 {
        result.u8[i] = u8::min(source.u8[i], destination.u8[i]);
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FDA_reg(r1: i32, r2: i32) { instr_660FDA(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FDA_mem(addr: i32, r: i32) {
    instr_660FDA(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FDB(source: u64, r: i32) {
    // pand mm, mm/m64
    let destination = read_mmx64s(r);
    write_mmx_reg64(r, source & destination);
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FDB_reg(r1: i32, r2: i32) { instr_0FDB(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FDB_mem(addr: i32, r: i32) {
    instr_0FDB(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FDB(source: reg128, r: i32) {
    // pand xmm, xmm/m128
    // XXX: Aligned access or #gp
    pand_r128(source, r);
}
pub unsafe fn instr_660FDB_reg(r1: i32, r2: i32) { instr_660FDB(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FDB_mem(addr: i32, r: i32) {
    instr_660FDB(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FDC(source: u64, r: i32) {
    // paddusb mm, mm/m64
    let destination: [u8; 8] = u64::to_le_bytes(read_mmx64s(r));
    let source: [u8; 8] = u64::to_le_bytes(source);
    let mut result = [0; 8];
    for i in 0..8 {
        result[i] = saturate_ud_to_ub(destination[i] as u32 + source[i] as u32);
    }
    write_mmx_reg64(r, u64::from_le_bytes(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FDC_reg(r1: i32, r2: i32) { instr_0FDC(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FDC_mem(addr: i32, r: i32) {
    instr_0FDC(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FDC(source: reg128, r: i32) {
    // paddusb xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..16 {
        result.u8[i] = saturate_ud_to_ub(source.u8[i] as u32 + destination.u8[i] as u32);
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FDC_reg(r1: i32, r2: i32) { instr_660FDC(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FDC_mem(addr: i32, r: i32) {
    instr_660FDC(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FDD(source: u64, r: i32) {
    // paddusw mm, mm/m64
    let destination: [u16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [u16; 4] = std::mem::transmute(source);
    let mut result = [0; 4];
    for i in 0..4 {
        result[i] = saturate_uw(destination[i] as u32 + source[i] as u32)
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FDD_reg(r1: i32, r2: i32) { instr_0FDD(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FDD_mem(addr: i32, r: i32) {
    instr_0FDD(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FDD(source: reg128, r: i32) {
    // paddusw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.u16[i] = saturate_uw(source.u16[i] as u32 + destination.u16[i] as u32)
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FDD_reg(r1: i32, r2: i32) { instr_660FDD(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FDD_mem(addr: i32, r: i32) {
    instr_660FDD(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FDE(source: u64, r: i32) {
    // pmaxub mm, mm/m64
    let destination: [u8; 8] = u64::to_le_bytes(read_mmx64s(r));
    let source: [u8; 8] = u64::to_le_bytes(source);
    let mut result = [0; 8];
    for i in 0..8 {
        result[i] = u8::max(source[i], destination[i])
    }
    write_mmx_reg64(r, u64::from_le_bytes(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FDE_reg(r1: i32, r2: i32) { instr_0FDE(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FDE_mem(addr: i32, r: i32) {
    instr_0FDE(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FDE(source: reg128, r: i32) {
    // pmaxub xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..16 {
        result.u8[i] = u8::max(source.u8[i], destination.u8[i]);
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FDE_reg(r1: i32, r2: i32) { instr_660FDE(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FDE_mem(addr: i32, r: i32) {
    instr_660FDE(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FDF(source: u64, r: i32) {
    // pandn mm, mm/m64
    let destination = read_mmx64s(r);
    write_mmx_reg64(r, source & !destination);
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FDF_reg(r1: i32, r2: i32) { instr_0FDF(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FDF_mem(addr: i32, r: i32) {
    instr_0FDF(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FDF(source: reg128, r: i32) {
    // pandn xmm, xmm/m128
    // XXX: Aligned access or #gp
    pandn_r128(source, r);
}
pub unsafe fn instr_660FDF_reg(r1: i32, r2: i32) { instr_660FDF(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FDF_mem(addr: i32, r: i32) {
    instr_660FDF(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FE0(source: u64, r: i32) {
    // pavgb mm, mm/m64
    let destination: [u8; 8] = u64::to_le_bytes(read_mmx64s(r));
    let source: [u8; 8] = u64::to_le_bytes(source);
    let mut result = [0; 8];
    for i in 0..8 {
        result[i] = (destination[i] as i32 + source[i] as i32 + 1 >> 1) as u8;
    }
    write_mmx_reg64(r, u64::from_le_bytes(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FE0_reg(r1: i32, r2: i32) { instr_0FE0(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FE0_mem(addr: i32, r: i32) {
    instr_0FE0(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FE0(source: reg128, r: i32) {
    // pavgb xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..16 {
        result.u8[i] = (destination.u8[i] as i32 + source.u8[i] as i32 + 1 >> 1) as u8;
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FE0_reg(r1: i32, r2: i32) { instr_660FE0(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FE0_mem(addr: i32, r: i32) {
    instr_660FE0(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FE1(source: u64, r: i32) {
    // psraw mm, mm/m64
    psraw_r64(r, source);
}
pub unsafe fn instr_0FE1_reg(r1: i32, r2: i32) { instr_0FE1(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FE1_mem(addr: i32, r: i32) {
    instr_0FE1(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FE1(source: reg128, r: i32) {
    // psraw xmm, xmm/m128
    // XXX: Aligned access or #gp
    psraw_r128(r, source.u64[0]);
}
pub unsafe fn instr_660FE1_reg(r1: i32, r2: i32) { instr_660FE1(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FE1_mem(addr: i32, r: i32) {
    instr_660FE1(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FE2(source: u64, r: i32) {
    // psrad mm, mm/m64
    psrad_r64(r, source);
}
pub unsafe fn instr_0FE2_reg(r1: i32, r2: i32) { instr_0FE2(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FE2_mem(addr: i32, r: i32) {
    instr_0FE2(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FE2(source: reg128, r: i32) {
    // psrad xmm, xmm/m128
    // XXX: Aligned access or #gp
    psrad_r128(r, source.u64[0]);
}
pub unsafe fn instr_660FE2_reg(r1: i32, r2: i32) { instr_660FE2(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FE2_mem(addr: i32, r: i32) {
    instr_660FE2(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FE3(source: u64, r: i32) {
    // pavgw mm, mm/m64
    let destination: [u16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [u16; 4] = std::mem::transmute(source);
    let mut result = [0; 4];
    for i in 0..4 {
        result[i] = (destination[i] as i32 + source[i] as i32 + 1 >> 1) as u16
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FE3_reg(r1: i32, r2: i32) { instr_0FE3(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FE3_mem(addr: i32, r: i32) {
    instr_0FE3(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FE3(source: reg128, r: i32) {
    // pavgw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let mut destination = read_xmm128s(r);
    for i in 0..8 {
        destination.u16[i] = (destination.u16[i] as i32 + source.u16[i] as i32 + 1 >> 1) as u16;
    }
    write_xmm_reg128(r, destination);
}
pub unsafe fn instr_660FE3_reg(r1: i32, r2: i32) { instr_660FE3(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FE3_mem(addr: i32, r: i32) {
    instr_660FE3(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FE4(source: u64, r: i32) {
    // pmulhuw mm, mm/m64
    let destination: [u16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [u16; 4] = std::mem::transmute(source);
    let mut result = [0; 4];
    for i in 0..4 {
        result[i] = ((destination[i] as i32 * source[i] as i32) >> 16) as u16
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FE4_reg(r1: i32, r2: i32) { instr_0FE4(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FE4_mem(addr: i32, r: i32) {
    instr_0FE4(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FE4(source: reg128, r: i32) {
    // pmulhuw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.u16[i] = (source.u16[i] as i32 * destination.u16[i] as i32 >> 16) as u16;
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FE4_reg(r1: i32, r2: i32) { instr_660FE4(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FE4_mem(addr: i32, r: i32) {
    instr_660FE4(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FE5(source: u64, r: i32) {
    // pmulhw mm, mm/m64
    let destination: [i16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [i16; 4] = std::mem::transmute(source);
    let mut result = [0; 4];
    for i in 0..4 {
        result[i] = ((destination[i] as i32 * source[i] as i32) >> 16) as i16
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FE5_reg(r1: i32, r2: i32) { instr_0FE5(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FE5_mem(addr: i32, r: i32) {
    instr_0FE5(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FE5(source: reg128, r: i32) {
    // pmulhw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.u16[i] = (destination.i16[i] as i32 * source.i16[i] as i32 >> 16) as u16
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FE5_reg(r1: i32, r2: i32) { instr_660FE5(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FE5_mem(addr: i32, r: i32) {
    instr_660FE5(return_on_pagefault!(safe_read128s(addr)), r);
}

#[no_mangle]
pub unsafe fn instr_0FE6_mem(_addr: i32, _r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn instr_0FE6_reg(_r1: i32, _r2: i32) { trigger_ud(); }

#[no_mangle]
pub unsafe fn instr_660FE6(source: reg128, r: i32) {
    // cvttpd2dq xmm1, xmm2/m128
    let result = reg128 {
        i32: [
            sse_convert_with_truncation_f64_to_i32(source.f64[0]),
            sse_convert_with_truncation_f64_to_i32(source.f64[1]),
            0,
            0,
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FE6_mem(addr: i32, r: i32) {
    instr_660FE6(return_on_pagefault!(safe_read128s(addr)), r);
}
pub unsafe fn instr_660FE6_reg(r1: i32, r2: i32) { instr_660FE6(read_xmm128s(r1), r2); }

#[no_mangle]
pub unsafe fn instr_F20FE6(source: reg128, r: i32) {
    // cvtpd2dq xmm1, xmm2/m128
    let result = reg128 {
        i32: [
            // XXX: Precision exception
            sse_convert_f64_to_i32(source.f64[0]),
            sse_convert_f64_to_i32(source.f64[1]),
            0,
            0,
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_F20FE6_mem(addr: i32, r: i32) {
    instr_F20FE6(return_on_pagefault!(safe_read128s(addr)), r);
}
pub unsafe fn instr_F20FE6_reg(r1: i32, r2: i32) { instr_F20FE6(read_xmm128s(r1), r2); }

#[no_mangle]
pub unsafe fn instr_F30FE6(source: u64, r: i32) {
    // cvtdq2pd xmm1, xmm2/m64
    let result = reg128 {
        f64: [
            // Note: Conversion never fails (i32 fits into f64)
            source as i32 as f64,
            (source >> 32) as i32 as f64,
        ],
    };
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_F30FE6_mem(addr: i32, r: i32) {
    instr_F30FE6(return_on_pagefault!(safe_read64s(addr)), r);
}
pub unsafe fn instr_F30FE6_reg(r1: i32, r2: i32) { instr_F30FE6(read_xmm64s(r1), r2); }

#[no_mangle]
pub unsafe fn instr_0FE7_mem(addr: i32, r: i32) {
    // movntq m64, mm
    mov_r_m64(addr, r);
}
#[no_mangle]
pub unsafe fn instr_0FE7_reg(_r1: i32, _r2: i32) { trigger_ud(); }
pub unsafe fn instr_660FE7_reg(_r1: i32, _r2: i32) { trigger_ud(); }
pub unsafe fn instr_660FE7_mem(addr: i32, r: i32) {
    // movntdq m128, xmm
    mov_r_m128(addr, r);
}
#[no_mangle]
pub unsafe fn instr_0FE8(source: u64, r: i32) {
    // psubsb mm, mm/m64
    let destination: [i8; 8] = std::mem::transmute(read_mmx64s(r));
    let source: [i8; 8] = std::mem::transmute(source);
    let mut result = [0; 8];
    for i in 0..8 {
        result[i] = saturate_sd_to_sb(destination[i] as u32 - source[i] as u32);
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FE8_reg(r1: i32, r2: i32) { instr_0FE8(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FE8_mem(addr: i32, r: i32) {
    instr_0FE8(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FE8(source: reg128, r: i32) {
    // psubsb xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..16 {
        result.i8[i] = saturate_sd_to_sb(destination.i8[i] as u32 - source.i8[i] as u32);
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FE8_reg(r1: i32, r2: i32) { instr_660FE8(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FE8_mem(addr: i32, r: i32) {
    instr_660FE8(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FE9(source: u64, r: i32) {
    // psubsw mm, mm/m64
    let destination: [i16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [i16; 4] = std::mem::transmute(source);
    let mut result = [0; 4];
    for i in 0..4 {
        result[i] = saturate_sd_to_sw(destination[i] as u32 - source[i] as u32)
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FE9_reg(r1: i32, r2: i32) { instr_0FE9(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FE9_mem(addr: i32, r: i32) {
    instr_0FE9(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FE9(source: reg128, r: i32) {
    // psubsw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.u16[i] = saturate_sd_to_sw(destination.i16[i] as u32 - source.i16[i] as u32)
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FE9_reg(r1: i32, r2: i32) { instr_660FE9(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FE9_mem(addr: i32, r: i32) {
    instr_660FE9(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FEA(source: u64, r: i32) {
    // pminsw mm, mm/m64
    let destination: [i16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [i16; 4] = std::mem::transmute(source);
    let mut result = [0; 4];
    for i in 0..4 {
        result[i] = i16::min(destination[i], source[i])
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FEA_reg(r1: i32, r2: i32) { instr_0FEA(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FEA_mem(addr: i32, r: i32) {
    instr_0FEA(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FEA(source: reg128, r: i32) {
    // pminsw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.i16[i] = i16::min(destination.i16[i], source.i16[i])
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FEA_reg(r1: i32, r2: i32) { instr_660FEA(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FEA_mem(addr: i32, r: i32) {
    instr_660FEA(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FEB(source: u64, r: i32) {
    // por mm, mm/m64
    let destination = read_mmx64s(r);
    write_mmx_reg64(r, source | destination);
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FEB_reg(r1: i32, r2: i32) { instr_0FEB(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FEB_mem(addr: i32, r: i32) {
    instr_0FEB(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FEB(source: reg128, r: i32) {
    // por xmm, xmm/m128
    // XXX: Aligned access or #gp
    por_r128(source, r);
}
pub unsafe fn instr_660FEB_reg(r1: i32, r2: i32) { instr_660FEB(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FEB_mem(addr: i32, r: i32) {
    instr_660FEB(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FEC(source: u64, r: i32) {
    // paddsb mm, mm/m64
    let destination: [i8; 8] = std::mem::transmute(read_mmx64s(r));
    let source: [i8; 8] = std::mem::transmute(source);
    let mut result = [0; 8];
    for i in 0..8 {
        result[i] = saturate_sd_to_sb(destination[i] as u32 + source[i] as u32);
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FEC_reg(r1: i32, r2: i32) { instr_0FEC(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FEC_mem(addr: i32, r: i32) {
    instr_0FEC(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FEC(source: reg128, r: i32) {
    // paddsb xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..16 {
        result.i8[i] = saturate_sd_to_sb(destination.i8[i] as u32 + source.i8[i] as u32);
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FEC_reg(r1: i32, r2: i32) { instr_660FEC(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FEC_mem(addr: i32, r: i32) {
    instr_660FEC(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FED(source: u64, r: i32) {
    // paddsw mm, mm/m64
    let destination: [i16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [i16; 4] = std::mem::transmute(source);
    let mut result = [0; 4];
    for i in 0..4 {
        result[i] = saturate_sd_to_sw(destination[i] as u32 + source[i] as u32)
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FED_reg(r1: i32, r2: i32) { instr_0FED(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FED_mem(addr: i32, r: i32) {
    instr_0FED(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FED(source: reg128, r: i32) {
    // paddsw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.u16[i] = saturate_sd_to_sw(destination.i16[i] as u32 + source.i16[i] as u32)
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FED_reg(r1: i32, r2: i32) { instr_660FED(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FED_mem(addr: i32, r: i32) {
    instr_660FED(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FEE(source: u64, r: i32) {
    // pmaxsw mm, mm/m64
    let destination: [i16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [i16; 4] = std::mem::transmute(source);
    let mut result = [0; 4];
    for i in 0..4 {
        result[i] = i16::max(destination[i], source[i])
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FEE_reg(r1: i32, r2: i32) { instr_0FEE(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FEE_mem(addr: i32, r: i32) {
    instr_0FEE(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FEE(source: reg128, r: i32) {
    // pmaxsw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.i16[i] = i16::max(destination.i16[i], source.i16[i])
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FEE_reg(r1: i32, r2: i32) { instr_660FEE(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FEE_mem(addr: i32, r: i32) {
    instr_660FEE(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FEF(source: u64, r: i32) {
    // pxor mm, mm/m64
    let destination = read_mmx64s(r);
    write_mmx_reg64(r, source ^ destination);
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FEF_reg(r1: i32, r2: i32) { instr_0FEF(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FEF_mem(addr: i32, r: i32) {
    instr_0FEF(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FEF(source: reg128, r: i32) {
    // pxor xmm, xmm/m128
    // XXX: Aligned access or #gp
    pxor_r128(source, r);
}
pub unsafe fn instr_660FEF_reg(r1: i32, r2: i32) { instr_660FEF(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FEF_mem(addr: i32, r: i32) {
    instr_660FEF(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FF0() { unimplemented_sse(); }
#[no_mangle]
pub unsafe fn instr_0FF1(source: u64, r: i32) {
    // psllw mm, mm/m64
    psllw_r64(r, source);
}
pub unsafe fn instr_0FF1_reg(r1: i32, r2: i32) { instr_0FF1(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FF1_mem(addr: i32, r: i32) {
    instr_0FF1(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FF1(source: reg128, r: i32) {
    // psllw xmm, xmm/m128
    // XXX: Aligned access or #gp
    psllw_r128(r, source.u64[0]);
}
pub unsafe fn instr_660FF1_reg(r1: i32, r2: i32) { instr_660FF1(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FF1_mem(addr: i32, r: i32) {
    instr_660FF1(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FF2(source: u64, r: i32) {
    // pslld mm, mm/m64
    pslld_r64(r, source);
}
pub unsafe fn instr_0FF2_reg(r1: i32, r2: i32) { instr_0FF2(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FF2_mem(addr: i32, r: i32) {
    instr_0FF2(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FF2(source: reg128, r: i32) {
    // pslld xmm, xmm/m128
    // XXX: Aligned access or #gp
    pslld_r128(r, source.u64[0]);
}
pub unsafe fn instr_660FF2_reg(r1: i32, r2: i32) { instr_660FF2(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FF2_mem(addr: i32, r: i32) {
    instr_660FF2(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FF3(source: u64, r: i32) {
    // psllq mm, mm/m64
    psllq_r64(r, source);
}
pub unsafe fn instr_0FF3_reg(r1: i32, r2: i32) { instr_0FF3(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FF3_mem(addr: i32, r: i32) {
    instr_0FF3(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FF3(source: reg128, r: i32) {
    // psllq xmm, xmm/m128
    // XXX: Aligned access or #gp
    psllq_r128(r, source.u64[0]);
}
pub unsafe fn instr_660FF3_reg(r1: i32, r2: i32) { instr_660FF3(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FF3_mem(addr: i32, r: i32) {
    instr_660FF3(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FF4(source: u64, r: i32) {
    // pmuludq mm, mm/m64
    let destination = read_mmx64s(r);
    write_mmx_reg64(r, (source as u32 as u64) * (destination as u32 as u64));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FF4_reg(r1: i32, r2: i32) { instr_0FF4(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FF4_mem(addr: i32, r: i32) {
    instr_0FF4(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FF4(source: reg128, r: i32) {
    // pmuludq xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    result.u64[0] = source.u32[0] as u64 * destination.u32[0] as u64;
    result.u64[1] = source.u32[2] as u64 * destination.u32[2] as u64;
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FF4_reg(r1: i32, r2: i32) { instr_660FF4(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FF4_mem(addr: i32, r: i32) {
    instr_660FF4(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FF5(source: u64, r: i32) {
    // pmaddwd mm, mm/m64
    let destination: [i16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [i16; 4] = std::mem::transmute(source);
    let mul0 = destination[0] as i32 * source[0] as i32;
    let mul1 = destination[1] as i32 * source[1] as i32;
    let mul2 = destination[2] as i32 * source[2] as i32;
    let mul3 = destination[3] as i32 * source[3] as i32;
    let low = mul0 + mul1;
    let high = mul2 + mul3;
    write_mmx_reg64(r, low as u32 as u64 | (high as u64) << 32);
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FF5_reg(r1: i32, r2: i32) { instr_0FF5(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FF5_mem(addr: i32, r: i32) {
    instr_0FF5(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FF5(source: reg128, r: i32) {
    // pmaddwd xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..4 {
        result.i32[i] = destination.i16[2 * i] as i32 * source.i16[2 * i] as i32
            + destination.i16[2 * i + 1] as i32 * source.i16[2 * i + 1] as i32
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FF5_reg(r1: i32, r2: i32) { instr_660FF5(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FF5_mem(addr: i32, r: i32) {
    instr_660FF5(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FF6(source: u64, r: i32) {
    // psadbw mm, mm/m64
    let destination: [u8; 8] = u64::to_le_bytes(read_mmx64s(r));
    let source: [u8; 8] = u64::to_le_bytes(source);
    let mut sum = 0;
    for i in 0..8 {
        sum += (destination[i] as i32 - source[i] as i32).abs() as u64;
    }
    write_mmx_reg64(r, sum);
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FF6_reg(r1: i32, r2: i32) { instr_0FF6(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FF6_mem(addr: i32, r: i32) {
    instr_0FF6(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FF6(source: reg128, r: i32) {
    // psadbw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut sum0 = 0;
    let mut sum1 = 0;
    for i in 0..8 {
        sum0 += (destination.u8[i + 0] as i32 - source.u8[i + 0] as i32).abs() as u32;
        sum1 += (destination.u8[i + 8] as i32 - source.u8[i + 8] as i32).abs() as u32;
    }
    write_xmm128(r, sum0 as i32, 0, sum1 as i32, 0);
}
pub unsafe fn instr_660FF6_reg(r1: i32, r2: i32) { instr_660FF6(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FF6_mem(addr: i32, r: i32) {
    instr_660FF6(return_on_pagefault!(safe_read128s(addr)), r);
}

pub unsafe fn instr_0FF7_mem(_addr: i32, _r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn maskmovq(r1: i32, r2: i32, addr: i32) {
    // maskmovq mm, mm
    // the caller must have called writable_or_pagefault
    let source: [u8; 8] = u64::to_le_bytes(read_mmx64s(r2));
    let mask: [u8; 8] = u64::to_le_bytes(read_mmx64s(r1));
    for i in 0..8 {
        if 0 != mask[i] & 0x80 {
            safe_write8(addr + i as i32, source[i] as i32).unwrap();
        }
    }
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FF7_reg(r1: i32, r2: i32) {
    let addr = return_on_pagefault!(get_seg_prefix_ds(get_reg_asize(EDI)));
    return_on_pagefault!(writable_or_pagefault(addr, 8));
    maskmovq(r1, r2, addr)
}

pub unsafe fn instr_660FF7_mem(_addr: i32, _r: i32) { trigger_ud(); }
#[no_mangle]
pub unsafe fn maskmovdqu(r1: i32, r2: i32, addr: i32) {
    // maskmovdqu xmm, xmm
    // the caller must have called writable_or_pagefault
    let source = read_xmm128s(r2);
    let mask = read_xmm128s(r1);
    for i in 0..16 {
        if 0 != mask.u8[i] & 0x80 {
            safe_write8(addr + i as i32, source.u8[i] as i32).unwrap();
        }
    }
}
pub unsafe fn instr_660FF7_reg(r1: i32, r2: i32) {
    let addr = return_on_pagefault!(get_seg_prefix_ds(get_reg_asize(EDI)));
    return_on_pagefault!(writable_or_pagefault(addr, 16));
    maskmovdqu(r1, r2, addr)
}
#[no_mangle]
pub unsafe fn instr_0FF8(source: u64, r: i32) {
    // psubb mm, mm/m64
    let destination: [u8; 8] = u64::to_le_bytes(read_mmx64s(r));
    let source: [u8; 8] = u64::to_le_bytes(source);
    let mut result = [0; 8];
    for i in 0..8 {
        result[i] = destination[i] - source[i];
    }
    write_mmx_reg64(r, u64::from_le_bytes(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FF8_reg(r1: i32, r2: i32) { instr_0FF8(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FF8_mem(addr: i32, r: i32) {
    instr_0FF8(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FF8(source: reg128, r: i32) {
    // psubb xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..16 {
        result.u8[i] = destination.u8[i] - source.u8[i];
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FF8_reg(r1: i32, r2: i32) { instr_660FF8(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FF8_mem(addr: i32, r: i32) {
    instr_660FF8(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FF9(source: u64, r: i32) {
    // psubw mm, mm/m64
    let destination: [i16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [i16; 4] = std::mem::transmute(source);
    let mut result = [0; 4];
    for i in 0..4 {
        result[i] = destination[i] - source[i]
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FF9_reg(r1: i32, r2: i32) { instr_0FF9(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FF9_mem(addr: i32, r: i32) {
    instr_0FF9(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FF9(source: reg128, r: i32) {
    // psubw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.i16[i] = destination.i16[i] - source.i16[i]
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FF9_reg(r1: i32, r2: i32) { instr_660FF9(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FF9_mem(addr: i32, r: i32) {
    instr_660FF9(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FFA(source: u64, r: i32) {
    // psubd mm, mm/m64
    let destination: [i32; 2] = std::mem::transmute(read_mmx64s(r));
    let source: [i32; 2] = std::mem::transmute(source);
    let mut result = [0; 2];
    for i in 0..2 {
        result[i] = destination[i] - source[i]
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FFA_reg(r1: i32, r2: i32) { instr_0FFA(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FFA_mem(addr: i32, r: i32) {
    instr_0FFA(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FFA(source: reg128, r: i32) {
    // psubd xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    write_xmm128(
        r,
        destination.i32[0] - source.i32[0],
        destination.i32[1] - source.i32[1],
        destination.i32[2] - source.i32[2],
        destination.i32[3] - source.i32[3],
    );
}
pub unsafe fn instr_660FFA_reg(r1: i32, r2: i32) { instr_660FFA(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FFA_mem(addr: i32, r: i32) {
    instr_660FFA(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FFB(source: u64, r: i32) {
    // psubq mm, mm/m64
    write_mmx_reg64(r, read_mmx64s(r) - source);
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FFB_reg(r1: i32, r2: i32) { instr_0FFB(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FFB_mem(addr: i32, r: i32) {
    instr_0FFB(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FFB(source: reg128, r: i32) {
    // psubq xmm, xmm/m128
    // XXX: Aligned access or #gp
    let mut destination = read_xmm128s(r);
    destination.u64[0] = destination.u64[0] - source.u64[0];
    destination.u64[1] = destination.u64[1] - source.u64[1];
    write_xmm_reg128(r, destination);
}
pub unsafe fn instr_660FFB_reg(r1: i32, r2: i32) { instr_660FFB(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FFB_mem(addr: i32, r: i32) {
    instr_660FFB(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FFC(source: u64, r: i32) {
    // paddb mm, mm/m64
    let destination: [u8; 8] = u64::to_le_bytes(read_mmx64s(r));
    let source: [u8; 8] = u64::to_le_bytes(source);
    let mut result = [0; 8];
    for i in 0..8 {
        result[i] = destination[i] + source[i];
    }
    write_mmx_reg64(r, u64::from_le_bytes(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FFC_reg(r1: i32, r2: i32) { instr_0FFC(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FFC_mem(addr: i32, r: i32) {
    instr_0FFC(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FFC(source: reg128, r: i32) {
    // paddb xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..16 {
        result.u8[i] = destination.u8[i] + source.u8[i];
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FFC_reg(r1: i32, r2: i32) { instr_660FFC(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FFC_mem(addr: i32, r: i32) {
    instr_660FFC(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FFD(source: u64, r: i32) {
    // paddw mm, mm/m64
    let destination: [u16; 4] = std::mem::transmute(read_mmx64s(r));
    let source: [u16; 4] = std::mem::transmute(source);
    let mut result = [0; 4];
    for i in 0..4 {
        result[i] = destination[i] + source[i]
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FFD_reg(r1: i32, r2: i32) { instr_0FFD(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FFD_mem(addr: i32, r: i32) {
    instr_0FFD(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FFD(source: reg128, r: i32) {
    // paddw xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let mut result = reg128 { i8: [0; 16] };
    for i in 0..8 {
        result.u16[i] = (destination.u16[i] as i32 + source.u16[i] as i32 & 0xFFFF) as u16;
    }
    write_xmm_reg128(r, result);
}
pub unsafe fn instr_660FFD_reg(r1: i32, r2: i32) { instr_660FFD(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FFD_mem(addr: i32, r: i32) {
    instr_660FFD(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FFE(source: u64, r: i32) {
    // paddd mm, mm/m64
    let destination: [i32; 2] = std::mem::transmute(read_mmx64s(r));
    let source: [i32; 2] = std::mem::transmute(source);
    let mut result = [0; 2];
    for i in 0..2 {
        result[i] = destination[i] + source[i]
    }
    write_mmx_reg64(r, std::mem::transmute(result));
    transition_fpu_to_mmx();
}
pub unsafe fn instr_0FFE_reg(r1: i32, r2: i32) { instr_0FFE(read_mmx64s(r1), r2); }
pub unsafe fn instr_0FFE_mem(addr: i32, r: i32) {
    instr_0FFE(return_on_pagefault!(safe_read64s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_660FFE(source: reg128, r: i32) {
    // paddd xmm, xmm/m128
    // XXX: Aligned access or #gp
    let destination = read_xmm128s(r);
    let dword0 = destination.i32[0] + source.i32[0];
    let dword1 = destination.i32[1] + source.i32[1];
    let dword2 = destination.i32[2] + source.i32[2];
    let dword3 = destination.i32[3] + source.i32[3];
    write_xmm128(r, dword0, dword1, dword2, dword3);
}
pub unsafe fn instr_660FFE_reg(r1: i32, r2: i32) { instr_660FFE(read_xmm128s(r1), r2); }
pub unsafe fn instr_660FFE_mem(addr: i32, r: i32) {
    instr_660FFE(return_on_pagefault!(safe_read128s(addr)), r);
}
#[no_mangle]
pub unsafe fn instr_0FFF() {
    // Windows 98
    dbg_log!("#ud: 0F FF");
    trigger_ud();
}

// --- AVX (VEX-encoded) for 32-bit mode ------------------------------------
//
// C4/C5 are LES/LDS in 16/32-bit mode; a second byte with bits[7:6] == 11 is
// an invalid LES/LDS encoding, so it is decoded as a VEX prefix instead. The
// lane semantics come from the shared simd_instr layer.

unsafe fn vex32_rm(modrm_byte: i32, l: bool) -> crate::paging::OrPageFault<(reg128, reg128)> {
    if modrm_byte < 0xC0 {
        let addr = modrm_resolve(modrm_byte)?.addr() as i32;
        let lo = safe_read128s(addr)?;
        let hi = if l { safe_read128s(addr + 16)? } else { reg128 { u64: [0, 0] } };
        Ok((lo, hi))
    }
    else {
        let r = (modrm_byte & 7) as i32;
        let lo = read_xmm128s(r);
        let hi = if l { *crate::cpu::global_pointers::ymm_high_ptr(r) } else { reg128 { u64: [0, 0] } };
        Ok((lo, hi))
    }
}

unsafe fn vex32_set(r: i32, lo: reg128, hi: reg128, l: bool) {
    write_xmm_reg128(r, lo);
    *crate::cpu::global_pointers::ymm_high_ptr(r) = if l { hi } else { reg128 { u64: [0, 0] } };
}

unsafe fn vex32_store(modrm_byte: i32, lo: reg128, hi: reg128, l: bool) {
    if modrm_byte < 0xC0 {
        let addr = return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32;
        return_on_pagefault!(safe_write128(addr, lo));
        if l {
            return_on_pagefault!(safe_write128(addr + 16, hi));
        }
    }
    else {
        vex32_set((modrm_byte & 7) as i32, lo, hi, l);
    }
}

unsafe fn vex32_v(vvvv: u8, l: bool) -> (reg128, reg128) {
    (
        read_xmm128s(vvvv as i32),
        if l { *crate::cpu::global_pointers::ymm_high_ptr(vvvv as i32) } else { reg128 { u64: [0, 0] } },
    )
}

unsafe fn vex32_arith(opcode: u8, pp: u8, l: bool, vvvv: u8, modrm_byte: i32) {
    let (s2lo, s2hi) = return_on_pagefault!(vex32_rm(modrm_byte, l));
    let (s1lo, s1hi) = vex32_v(vvvv, l);
    let dst = (modrm_byte >> 3 & 7) as i32;
    if pp == 2 {
        let mut r = s1lo;
        r.f32[0] = crate::cpu::interp::simd_instr::arith_f32(opcode, s1lo.f32[0], s2lo.f32[0]);
        vex32_set(dst, r, reg128 { u64: [0, 0] }, false);
    }
    else if pp == 3 {
        let mut r = s1lo;
        r.f64[0] = crate::cpu::interp::simd_instr::arith_f64(opcode, s1lo.f64[0], s2lo.f64[0]);
        vex32_set(dst, r, reg128 { u64: [0, 0] }, false);
    }
    else if pp == 1 {
        let mut rlo = s1lo;
        let mut rhi = s1hi;
        for i in 0..2 {
            rlo.f64[i] = crate::cpu::interp::simd_instr::arith_f64(opcode, s1lo.f64[i], s2lo.f64[i]);
            if l {
                rhi.f64[i] = crate::cpu::interp::simd_instr::arith_f64(opcode, s1hi.f64[i], s2hi.f64[i]);
            }
        }
        vex32_set(dst, rlo, rhi, l);
    }
    else {
        let mut rlo = s1lo;
        let mut rhi = s1hi;
        for i in 0..4 {
            rlo.f32[i] = crate::cpu::interp::simd_instr::arith_f32(opcode, s1lo.f32[i], s2lo.f32[i]);
            if l {
                rhi.f32[i] = crate::cpu::interp::simd_instr::arith_f32(opcode, s1hi.f32[i], s2hi.f32[i]);
            }
        }
        vex32_set(dst, rlo, rhi, l);
    }
}

unsafe fn vex32_logic(opcode: u8, l: bool, vvvv: u8, modrm_byte: i32) {
    let (s2lo, s2hi) = return_on_pagefault!(vex32_rm(modrm_byte, l));
    let (s1lo, s1hi) = vex32_v(vvvv, l);
    let dst = (modrm_byte >> 3 & 7) as i32;
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
    let rhi = if l { apply(s1hi, s2hi) } else { reg128 { u64: [0, 0] } };
    vex32_set(dst, rlo, rhi, l);
}

unsafe fn vex32_int(opcode: u8, l: bool, vvvv: u8, modrm_byte: i32) {
    let (s2lo, s2hi) = return_on_pagefault!(vex32_rm(modrm_byte, l));
    let (s1lo, s1hi) = vex32_v(vvvv, l);
    let dst = (modrm_byte >> 3 & 7) as i32;
    let rlo = match crate::cpu::interp::simd_instr::int_apply(opcode, s1lo, s2lo) {
        Some(r) => r,
        None =>
        {
            trigger_ud();
            return;
        },
    };
    let rhi = if l {
        crate::cpu::interp::simd_instr::int_apply(opcode, s1hi, s2hi).unwrap_or(reg128 { u64: [0, 0] })
    }
    else {
        reg128 { u64: [0, 0] }
    };
    vex32_set(dst, rlo, rhi, l);
}

unsafe fn vex32_mov(opcode: u8, pp: u8, l: bool, modrm_byte: i32) {
    let dst = (modrm_byte >> 3 & 7) as i32;
    if opcode == 0x10 || opcode == 0x11 {
        // VMOVSS/SD (scalar) and VMOVUPS/UPD
        if pp == 2 || pp == 3 {
            let bits = if pp == 2 { 32 } else { 64 };
            if opcode == 0x10 {
                if modrm_byte < 0xC0 {
                    let addr = return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32;
                    let value = return_on_pagefault!(safe_read64s(addr));
                    let mut r = reg128 { u64: [0, 0] };
                    if bits == 32 {
                        r.u32[0] = value as u32;
                    }
                    else {
                        r.u64[0] = value;
                    }
                    vex32_set(dst, r, reg128 { u64: [0, 0] }, false);
                }
                else {
                    let src = read_xmm128s((modrm_byte & 7) as i32);
                    let mut r = reg128 { u64: [0, 0] };
                    if bits == 32 {
                        r.u32[0] = src.u32[0];
                    }
                    else {
                        r.u64[0] = src.u64[0];
                    }
                    vex32_set(dst, r, reg128 { u64: [0, 0] }, false);
                }
            }
            else {
                let src = read_xmm128s(dst);
                if modrm_byte < 0xC0 {
                    let addr = return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32;
                    return_on_pagefault!(safe_write64(addr, if bits == 32 { src.u32[0] as u64 } else { src.u64[0] }));
                }
                else {
                    let mut r = read_xmm128s((modrm_byte & 7) as i32);
                    if bits == 32 {
                        r.u32[0] = src.u32[0];
                    }
                    else {
                        r.u64[0] = src.u64[0];
                    }
                    vex32_set((modrm_byte & 7) as i32, r, reg128 { u64: [0, 0] }, false);
                }
            }
            return;
        }
    }
    if opcode == 0x10 || opcode == 0x28 || opcode == 0x6F {
        let (lo, hi) = return_on_pagefault!(vex32_rm(modrm_byte, l));
        vex32_set(dst, lo, hi, l);
    }
    else {
        let lo = read_xmm128s(dst);
        let hi = if l { *crate::cpu::global_pointers::ymm_high_ptr(dst) } else { reg128 { u64: [0, 0] } };
        vex32_store(modrm_byte, lo, hi, l);
    }
}

unsafe fn vex32_0f(opcode: u8, pp: u8, l: bool, w: bool, vvvv: u8) {
    if opcode == 0x77 && pp == 0 {
        for r in 0..8i32 {
            *crate::cpu::global_pointers::ymm_high_ptr(r) = reg128 { u64: [0, 0] };
            if l {
                write_xmm_reg128(r, reg128 { u64: [0, 0] });
            }
        }
        return;
    }
    let modrm_byte = return_on_pagefault!(read_imm8());
    let dst = (modrm_byte >> 3 & 7) as i32;
    match opcode {
        0x10 | 0x11 | 0x28 | 0x29 | 0x6F | 0x7F if pp == 0 || pp == 1 || pp == 2 || pp == 3 => {
            vex32_mov(opcode, pp, l, modrm_byte)
        },
        0x51 | 0x58 | 0x59 | 0x5C | 0x5D | 0x5E | 0x5F => vex32_arith(opcode, pp, l, vvvv, modrm_byte),
        0x54 | 0x55 | 0x56 | 0x57 => vex32_logic(opcode, l, vvvv, modrm_byte),
        0x14 | 0x15 => {
            // VUNPCKLPS/PD (14) and VUNPCKHPS/PD (15)
            let (s2lo, s2hi) = return_on_pagefault!(vex32_rm(modrm_byte, l));
            let (s1lo, s1hi) = vex32_v(vvvv, l);
            let f64 = pp == 1;
            let high = opcode == 0x15;
            let unpck = |a: reg128, b: reg128| -> reg128 {
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
            let rlo = unpck(s1lo, s2lo);
            let rhi = if l { unpck(s1hi, s2hi) } else { reg128 { u64: [0, 0] } };
            vex32_set(dst, rlo, rhi, l);
        },
        0x70 => {
            let (slo, shi) = return_on_pagefault!(vex32_rm(modrm_byte, l));
            let control = return_on_pagefault!(read_imm8()) as u8;
            let variant = if pp == 2 { 1 } else if pp == 3 { 2 } else { 0 };
            let rlo = crate::cpu::interp::simd_instr::pshuf_apply(variant, control, slo);
            let rhi = if l { crate::cpu::interp::simd_instr::pshuf_apply(variant, control, shi) } else { reg128 { u64: [0, 0] } };
            vex32_set(dst, rlo, rhi, l);
        },
        0x71 | 0x72 | 0x73 if pp == 1 => {
            let (slo, shi) = return_on_pagefault!(vex32_rm(modrm_byte, l));
            let imm = return_on_pagefault!(read_imm8()) as u64;
            let group = (modrm_byte >> 3 & 7) as u8;
            let rlo = match crate::cpu::interp::simd_instr::shift_imm_apply(opcode, group, imm, slo) {
                Some(r) => r,
                None =>
                {
                    trigger_ud();
                    return;
                },
            };
            let rhi = if l {
                crate::cpu::interp::simd_instr::shift_imm_apply(opcode, group, imm, shi).unwrap_or(reg128 { u64: [0, 0] })
            }
            else {
                reg128 { u64: [0, 0] }
            };
            vex32_store(modrm_byte, rlo, rhi, l);
        },
        // VUCOMISS/SD (2E) and VCOMISS/SD (2F)
        0x2E | 0x2F => {
            let (src, _) = return_on_pagefault!(vex32_rm(modrm_byte, false));
            let a = read_xmm128s(dst);
            let (unordered, less, equal) = if pp == 1 {
                (a.f64[0].is_nan() || src.f64[0].is_nan(), a.f64[0] < src.f64[0], a.f64[0] == src.f64[0])
            }
            else {
                (a.f32[0].is_nan() || src.f32[0].is_nan(), a.f32[0] < src.f32[0], a.f32[0] == src.f32[0])
            };
            *flags &= !((FLAG_CARRY | FLAG_PARITY | FLAG_ZERO | FLAG_OVERFLOW | FLAG_SIGN | FLAG_ADJUST) as i32);
            if unordered {
                *flags |= (FLAG_CARRY | FLAG_PARITY | FLAG_ZERO) as i32;
            }
            else if less {
                *flags |= FLAG_CARRY as i32;
            }
            else if equal {
                *flags |= FLAG_ZERO as i32;
            }
            *flags_changed = 0;
        },
        // VMOVMSKPS/PD (50)
        0x50 => {
            let src = if modrm_byte < 0xC0 {
                return_on_pagefault!(safe_read128s(return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32))
            }
            else {
                read_xmm128s(modrm_byte & 7)
            };
            let mask = if pp == 1 {
                ((src.f64[0].to_bits() >> 63) as i32) | (((src.f64[1].to_bits() >> 63) as i32) << 1)
            }
            else {
                let mut m = 0;
                for i in 0..4 {
                    m |= ((src.f32[i].to_bits() >> 31) as i32) << i;
                }
                m
            };
            write_reg32(dst, mask);
        },
        // VCVTPS2PD/PD2PS/SS2SD/SD2SS (5A) and CVTDQ2PS/PS2DQ/TTPS2DQ (5B)
        0x5A | 0x5B => {
            let (src, srchi) = return_on_pagefault!(vex32_rm(modrm_byte, l));
            let (s1lo, s1hi) = vex32_v(vvvv, l);
            let mut rlo = s1lo;
            let mut rhi = s1hi;
            if opcode == 0x5A {
                if pp == 0 {
                    for i in 0..2 {
                        rlo.f64[i] = src.f32[i] as f64;
                        if l {
                            rhi.f64[i] = srchi.f32[i] as f64;
                        }
                    }
                }
                else if pp == 1 {
                    for i in 0..2 {
                        rlo.f32[i] = src.f64[i] as f32;
                    }
                    rlo.u64[1] = 0;
                    rhi = reg128 { u64: [0, 0] };
                    if l {
                        for i in 0..2 {
                            rhi.f32[i] = srchi.f64[i] as f32;
                        }
                    }
                }
                else if pp == 2 {
                    rlo.f64[0] = src.f32[0] as f64;
                    vex32_set(dst, rlo, reg128 { u64: [0, 0] }, false);
                    return;
                }
                else {
                    rlo.f32[0] = src.f64[0] as f32;
                    vex32_set(dst, rlo, reg128 { u64: [0, 0] }, false);
                    return;
                }
            }
            else if pp == 0 {
                for i in 0..4 {
                    rlo.f32[i] = src.i32[i] as f32;
                    if l {
                        rhi.f32[i] = srchi.i32[i] as f32;
                    }
                }
            }
            else if pp == 1 {
                for i in 0..4 {
                    rlo.i32[i] = src.f32[i].round() as i32;
                    if l {
                        rhi.i32[i] = srchi.f32[i].round() as i32;
                    }
                }
            }
            else {
                for i in 0..4 {
                    rlo.i32[i] = if src.f32[i].is_nan() { i32::MIN } else { src.f32[i].trunc() as i32 };
                    if l {
                        rhi.i32[i] = if srchi.f32[i].is_nan() { i32::MIN } else { srchi.f32[i].trunc() as i32 };
                    }
                }
            }
            vex32_set(dst, rlo, rhi, l);
        },
        // VCVTSI2SS/SD (2A), VCVTTSS2SI/SD (2C), VCVTSS2SI/SD (2D)
        0x2A | 0x2C | 0x2D => {
            let f64 = pp == 3;
            if opcode == 0x2A {
                let source = if modrm_byte < 0xC0 {
                    let addr = return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32;
                    if w { return_on_pagefault!(safe_read64s(addr)) } else { return_on_pagefault!(safe_read32s(addr)) as u32 as u64 }
                }
                else if w {
                    read_reg64(modrm_byte & 7)
                }
                else {
                    read_reg32(modrm_byte & 7) as u32 as u64
                };
                let mut r = read_xmm128s(dst);
                if f64 {
                    r.f64[0] = if w { source as i64 as f64 } else { source as u32 as i32 as f64 };
                }
                else {
                    r.f32[0] = if w { source as i64 as f32 } else { source as u32 as i32 as f32 };
                }
                vex32_set(dst, r, reg128 { u64: [0, 0] }, false);
                return;
            }
            let (src, _) = return_on_pagefault!(vex32_rm(modrm_byte, false));
            let truncate = opcode == 0x2C;
            let value = if f64 {
                if truncate { src.f64[0].trunc() } else { src.f64[0].round() }
            }
            else if truncate {
                src.f32[0].trunc() as f64
            }
            else {
                src.f32[0].round() as f64
            };
            let mask = if w { u64::MAX } else { 0xFFFF_FFFF };
            let bits = if value.is_nan() || value <= i64::MIN as f64 {
                0x8000_0000_0000_0000
            }
            else if value >= i64::MAX as f64 {
                0x7FFF_FFFF_FFFF_FFFF
            }
            else {
                value as i64 as u64
            };
            write_reg64(dst, bits & mask);
        },
        0x74 | 0x75 | 0x76 if pp == 1 => vex32_int(opcode, l, vvvv, modrm_byte),
        _ => {
            if pp == 1 {
                vex32_int(opcode, l, vvvv, modrm_byte);
            }
            else {
                trigger_ud();
            }
        },
    }
}

unsafe fn vex32_fma(opcode: u8, l: bool, w: bool, vvvv: u8) {
    let modrm_byte = return_on_pagefault!(read_imm8());
    let (s2lo, s2hi) = return_on_pagefault!(vex32_rm(modrm_byte, l));
    let (s1lo, s1hi) = vex32_v(vvvv, l);
    let dst = (modrm_byte >> 3 & 7) as i32;
    let dlo = read_xmm128s(dst);
    let dhi = if l { *crate::cpu::global_pointers::ymm_high_ptr(dst) } else { reg128 { u64: [0, 0] } };
    let group = if opcode < 0xA0 { 0 } else if opcode < 0xB0 { 1 } else { 2 };
    let sub = (opcode & 0xF) - 6;
    let scalar = sub % 2 == 1;
    let f64 = w;
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
        vex32_set(dst, rlo, reg128 { u64: [0, 0] }, false);
        return;
    }
    for i in 0..lanes {
        if f64 {
            rlo.f64[i] = calc(dlo, s1lo, s2lo, i);
            if l {
                rhi.f64[i] = calc(dhi, s1hi, s2hi, i);
            }
        }
        else {
            rlo.f32[i] = calc(dlo, s1lo, s2lo, i) as f32;
            if l {
                rhi.f32[i] = calc(dhi, s1hi, s2hi, i) as f32;
            }
        }
    }
    vex32_set(dst, rlo, rhi, l);
}

unsafe fn vex32_ymm_lane(r: i32, i: usize, bytes: u64) -> u64 {
    let bits = bytes * 8;
    let bit = i as u32 * bits as u32;
    let word = if bit < 128 {
        read_xmm128s(r).u64[(bit / 64) as usize]
    }
    else {
        (*crate::cpu::global_pointers::ymm_high_ptr(r)).u64[((bit - 128) / 64) as usize]
    };
    let mask = if bytes == 8 { u64::MAX } else { (1u64 << bits) - 1 };
    word >> (bit % 64) & mask
}

unsafe fn vex32_ymm_set_lane(r: i32, i: usize, bytes: u64, value: u64) {
    let bits = bytes * 8;
    let bit = i as u32 * bits as u32;
    let mask = if bytes == 8 { u64::MAX } else { (1u64 << bits) - 1 };
    let shift = bit % 64;
    if bit < 128 {
        let mut lo = read_xmm128s(r);
        let w = &mut lo.u64[(bit / 64) as usize];
        *w = (*w & !(mask << shift)) | ((value & mask) << shift);
        write_xmm_reg128(r, lo);
    }
    else {
        let mut hi = *crate::cpu::global_pointers::ymm_high_ptr(r);
        let w = &mut hi.u64[((bit - 128) / 64) as usize];
        *w = (*w & !(mask << shift)) | ((value & mask) << shift);
        *crate::cpu::global_pointers::ymm_high_ptr(r) = hi;
    }
}

unsafe fn vex32_gather(opcode: u8, l: bool, w: bool, vvvv: u8) {
    let modrm_byte = return_on_pagefault!(read_imm8());
    if modrm_byte >= 0xC0 || modrm_byte & 7 != 4 {
        trigger_ud();
        return;
    }
    let sib = return_on_pagefault!(read_imm8()) as u8;
    let scale = 1u64 << (sib >> 6);
    let index_reg = (sib >> 3 & 7) as i32;
    let base_low = sib & 7;
    let mut base = if base_low == 5 && modrm_byte >> 6 == 0 {
        return_on_pagefault!(read_imm32s()) as u32 as u64
    }
    else {
        read_reg32(base_low as i32) as u32 as u64
    };
    match modrm_byte >> 6 {
        0 => {},
        1 => base = base.wrapping_add(return_on_pagefault!(read_imm8s()) as i32 as u32 as u64),
        _ => base = base.wrapping_add(return_on_pagefault!(read_imm32s()) as u32 as u64),
    }
    if *prefixes & crate::cpu::decode::prefix::PREFIX_MASK_SEGMENT != 0 {
        let seg = (*prefixes & crate::cpu::decode::prefix::PREFIX_MASK_SEGMENT) - 1;
        base = base.wrapping_add(*segment_offsets.offset(seg as isize) as u32 as u64);
    }
    let index64 = opcode == 0x91 || opcode == 0x93;
    let data64 = w;
    let idx_size = if index64 { 8u64 } else { 4u64 };
    let data_size = if data64 { 8u64 } else { 4u64 };
    let vec_bytes = if l { 32usize } else { 16usize };
    let count = core::cmp::min(vec_bytes / data_size as usize, vec_bytes / idx_size as usize);
    let mask_reg = vvvv as i32;
    let dst = (modrm_byte >> 3 & 7) as i32;
    let sign = 1u64 << (data_size * 8 - 1);
    for i in 0..count {
        let mask_elem = vex32_ymm_lane(mask_reg, i, data_size);
        if mask_elem & sign == 0 {
            continue;
        }
        let index_val = vex32_ymm_lane(index_reg, i, idx_size);
        let offset = if idx_size == 8 { index_val as i64 as u64 } else { index_val as u32 as i32 as i64 as u64 };
        let addr = base.wrapping_add(offset.wrapping_mul(scale));
        let value = if data64 {
            return_on_pagefault!(safe_read64s(addr as i32)) as u64
        }
        else {
            return_on_pagefault!(safe_read32s(addr as i32)) as u32 as u64
        };
        vex32_ymm_set_lane(dst, i, data_size, value);
        vex32_ymm_set_lane(mask_reg, i, data_size, 0);
    }
    if !l {
        *crate::cpu::global_pointers::ymm_high_ptr(dst) = reg128 { u64: [0, 0] };
    }
}

unsafe fn vex32_0f38(opcode: u8, pp: u8, l: bool, w: bool, vvvv: u8) {
    if pp != 1 {
        trigger_ud();
        return;
    }
    if (0x96..=0x9F).contains(&opcode) || (0xA6..=0xAF).contains(&opcode) || (0xB6..=0xBF).contains(&opcode) {
        vex32_fma(opcode, l, w, vvvv);
        return;
    }
    if (0x90..=0x93).contains(&opcode) {
        vex32_gather(opcode, l, w, vvvv);
        return;
    }
    let modrm_byte = return_on_pagefault!(read_imm8());
    let dst = (modrm_byte >> 3 & 7) as i32;
    let (s2lo, s2hi) = return_on_pagefault!(vex32_rm(modrm_byte, l));
    let (s1lo, s1hi) = vex32_v(vvvv, l);
    match opcode {
        0xDB | 0xDC | 0xDD | 0xDE | 0xDF => {
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
            let rhi = if l { one(s1hi, s2hi) } else { reg128 { u64: [0, 0] } };
            vex32_set(dst, rlo, rhi, l);
        },
        // VPERMILPS/PD, variable form
        0x0C | 0x0D => {
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
            let rhi = if l { perm(s1hi, s2hi, f64) } else { reg128 { u64: [0, 0] } };
            vex32_set(dst, rlo, rhi, l);
        },
        // VTESTPS/PD
        0x0E | 0x0F => {
            let (and, andn) = crate::cpu::interp::simd_instr::ptest(s1lo, s2lo);
            *flags &= !((FLAG_CARRY | FLAG_ZERO) as i32);
            if and == 0 {
                *flags |= FLAG_ZERO as i32;
            }
            if andn == 0 {
                *flags |= FLAG_CARRY as i32;
            }
            *flags_changed = 0;
        },
        // VPERMPS/VPERMD (256-bit dword permute)
        0x16 | 0x36 => {
            if !l {
                trigger_ud();
                return;
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
            vex32_set(dst, rlo, rhi, true);
        },
        // VPSRLVD/VPSRAVD/VPSLLVD
        0x45 | 0x46 | 0x47 => {
            let kind = match opcode {
                0x45 => 0,
                0x46 => 1,
                _ => 2,
            };
            let rlo = crate::cpu::interp::simd_instr::variable_shift32(s1lo, s2lo, kind);
            let rhi = if l {
                crate::cpu::interp::simd_instr::variable_shift32(s1hi, s2hi, kind)
            }
            else {
                reg128 { u64: [0, 0] }
            };
            vex32_set(dst, rlo, rhi, l);
        },
        // VMASKMOVPS/PD load (2C/2D) and store (2E/2F)
        0x2C | 0x2D | 0x2E | 0x2F => {
            if modrm_byte >= 0xC0 {
                trigger_ud();
                return;
            }
            let addr = return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32;
            let store = opcode == 0x2E || opcode == 0x2F;
            let f64 = opcode == 0x2D || opcode == 0x2F;
            let lanes = if l { 8 } else { 4 };
            let mut mask = [0u32; 8];
            for i in 0..4 {
                mask[i] = s1lo.u32[i];
                mask[i + 4] = s1hi.u32[i];
            }
            if store {
                let src = read_xmm128s(dst);
                let mut i = 0;
                while i < lanes {
                    let active = if f64 { mask[i | 1] & 0x8000_0000 != 0 } else { mask[i] & 0x8000_0000 != 0 };
                    if active {
                        if f64 {
                            return_on_pagefault!(safe_write64(addr + (i as i32) * 4, src.u64[i / 2]));
                        }
                        else {
                            return_on_pagefault!(safe_write32(addr + (i as i32) * 4, src.u32[i] as i32));
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
                            let v = return_on_pagefault!(safe_read64s(addr + (i as i32) * 4));
                            if i < 4 {
                                rlo.u64[i / 2] = v;
                            }
                            else {
                                rhi.u64[(i - 4) / 2] = v;
                            }
                        }
                        else {
                            let v = return_on_pagefault!(safe_read32s(addr + (i as i32) * 4)) as u32;
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
                vex32_set(dst, rlo, rhi, l);
            }
        },
        // VPBROADCASTD/Q/B/W and VBROADCASTI128
        0x58 | 0x59 | 0x78 | 0x79 | 0x5A => {
            let mut rlo = reg128 { u64: [0, 0] };
            let mut rhi = reg128 { u64: [0, 0] };
            match opcode {
                0x58 => {
                    let value = s2lo.u32[0];
                    for i in 0..4 {
                        rlo.u32[i] = value;
                        rhi.u32[i] = value;
                    }
                },
                0x59 => {
                    let value = s2lo.u64[0];
                    rlo.u64[0] = value;
                    rlo.u64[1] = value;
                    rhi.u64[0] = value;
                    rhi.u64[1] = value;
                },
                0x78 => {
                    let value = s2lo.u8[0];
                    for i in 0..16 {
                        rlo.u8[i] = value;
                        rhi.u8[i] = value;
                    }
                },
                0x79 => {
                    let value = s2lo.u16[0];
                    for i in 0..8 {
                        rlo.u16[i] = value;
                        rhi.u16[i] = value;
                    }
                },
                _ => {
                    rlo = s2lo;
                    rhi = s2lo;
                },
            }
            vex32_set(dst, rlo, rhi, l);
        },
        // VPMASKMOVD/Q load (8C/8D) and store (8E/8F)
        0x8C | 0x8D | 0x8E | 0x8F => {
            if modrm_byte >= 0xC0 {
                trigger_ud();
                return;
            }
            let addr = return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32;
            let store = opcode == 0x8E || opcode == 0x8F;
            let qword = opcode == 0x8D || opcode == 0x8F;
            let lanes = if l { 8 } else { 4 };
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
                let src = read_xmm128s(dst);
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
                        return_on_pagefault!(safe_write32(addr + (i as i32) * 4, value[i] as i32));
                    }
                }
            }
            else {
                let mut out = [0u32; 8];
                for i in 0..lanes {
                    if active(i) {
                        out[i] = return_on_pagefault!(safe_read32s(addr + (i as i32) * 4)) as u32;
                    }
                }
                let mut rlo = reg128 { u64: [0, 0] };
                let mut rhi = reg128 { u64: [0, 0] };
                for i in 0..4 {
                    rlo.u32[i] = out[i];
                    rhi.u32[i] = out[i + 4];
                }
                vex32_set(dst, rlo, rhi, l);
            }
        },
        _ => {
            if let Some(rlo) = crate::cpu::interp::simd_instr::sse4_38_apply(opcode, s1lo, s2lo) {
                let rhi = if l {
                    crate::cpu::interp::simd_instr::sse4_38_apply(opcode, s1hi, s2hi).unwrap_or(reg128 { u64: [0, 0] })
                }
                else {
                    reg128 { u64: [0, 0] }
                };
                vex32_set(dst, rlo, rhi, l);
            }
            else if let Some(rlo) = crate::cpu::interp::simd_instr::ssse3_apply(opcode, s1lo, s2lo) {
                let rhi = if l {
                    crate::cpu::interp::simd_instr::ssse3_apply(opcode, s1hi, s2hi).unwrap_or(reg128 { u64: [0, 0] })
                }
                else {
                    reg128 { u64: [0, 0] }
                };
                vex32_set(dst, rlo, rhi, l);
            }
            else {
                trigger_ud();
            }
        },
    }
}

unsafe fn vex32_0f3a(opcode: u8, pp: u8, l: bool, _w: bool, vvvv: u8) {
    if pp != 1 {
        trigger_ud();
        return;
    }
    let modrm_byte = return_on_pagefault!(read_imm8());
    let dst = (modrm_byte >> 3 & 7) as i32;
    let (s2lo, s2hi) = return_on_pagefault!(vex32_rm(modrm_byte, l));
    let (s1lo, s1hi) = vex32_v(vvvv, l);
    let imm = return_on_pagefault!(read_imm8()) as u8;
    match opcode {
        0x0F => {
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
            let rhi = if l { align(s1hi, s2hi) } else { reg128 { u64: [0, 0] } };
            vex32_set(dst, rlo, rhi, l);
        },
        0x0C | 0x0D | 0x0E => {
            let (lanes, bits) = match opcode {
                0x0C => (4, 32),
                0x0D => (2, 64),
                _ => (8, 16),
            };
            let rlo = crate::cpu::interp::simd_instr::blend_imm_apply(s1lo, s2lo, imm, lanes, bits);
            let rhi = if l {
                crate::cpu::interp::simd_instr::blend_imm_apply(s1hi, s2hi, imm, lanes, bits)
            }
            else {
                reg128 { u64: [0, 0] }
            };
            vex32_set(dst, rlo, rhi, l);
        },
        0x08 | 0x09 | 0x0A | 0x0B => {
            if opcode == 0x08 || opcode == 0x0A {
                let lanes = if opcode == 0x08 { 4 } else { 1 };
                let mut rlo = s1lo;
                let mut rhi = s1hi;
                for i in 0..lanes {
                    rlo.f32[i] = crate::cpu::interp::simd_instr::round_apply(s2lo.f32[i] as f64, imm) as f32;
                }
                if opcode == 0x08 && l {
                    for i in 0..4 {
                        rhi.f32[i] = crate::cpu::interp::simd_instr::round_apply(s2hi.f32[i] as f64, imm) as f32;
                    }
                }
                vex32_set(dst, rlo, rhi, l);
            }
            else {
                let lanes = if opcode == 0x09 { 2 } else { 1 };
                let mut rlo = s1lo;
                let mut rhi = s1hi;
                for i in 0..lanes {
                    rlo.f64[i] = crate::cpu::interp::simd_instr::round_apply(s2lo.f64[i], imm);
                }
                if opcode == 0x09 && l {
                    for i in 0..2 {
                        rhi.f64[i] = crate::cpu::interp::simd_instr::round_apply(s2hi.f64[i], imm);
                    }
                }
                vex32_set(dst, rlo, rhi, l);
            }
        },
        0x44 => {
            let rlo = crate::cpu::interp::simd_instr::pclmulqdq(s1lo, s2lo, imm);
            let rhi = if l { crate::cpu::interp::simd_instr::pclmulqdq(s1hi, s2hi, imm) } else { reg128 { u64: [0, 0] } };
            vex32_set(dst, rlo, rhi, l);
        },
        0xDF => {
            let rlo = crate::cpu::interp::simd_instr::aeskeygenassist(s2lo, imm);
            let rhi = if l { crate::cpu::interp::simd_instr::aeskeygenassist(s2hi, imm) } else { reg128 { u64: [0, 0] } };
            vex32_set(dst, rlo, rhi, l);
        },
        // VPBLENDD
        0x02 => {
            let rlo = crate::cpu::interp::simd_instr::blend_d(s1lo, s2lo, imm & 0x0F);
            let rhi = if l {
                crate::cpu::interp::simd_instr::blend_d(s1hi, s2hi, imm >> 4)
            }
            else {
                reg128 { u64: [0, 0] }
            };
            vex32_set(dst, rlo, rhi, l);
        },
        // VPERMILPS/PD with an immediate
        0x04 | 0x05 => {
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
            let rhi = if l { perm(s2hi, f64) } else { reg128 { u64: [0, 0] } };
            vex32_set(dst, rlo, rhi, l);
        },
        // VPERM2F128 (06) / VPERM2I128 (46)
        0x06 | 0x46 => {
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
            vex32_set(dst, rlo, rhi, l);
        },
        // VPEXTRB/W/D/Q and VEXTRACTPS (source is vvvv, destination is rm)
        0x14 | 0x15 | 0x16 | 0x17 => {
            let (value, size) = match opcode {
                0x14 => (s1lo.u8[imm as usize] as i32, 8),
                0x15 => (s1lo.u16[imm as usize & 7] as i32, 16),
                _ => (s1lo.u32[imm as usize & 3] as i32, 32),
            };
            if modrm_byte < 0xC0 {
                let addr = return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32;
                match size {
                    8 => return_on_pagefault!(safe_write8(addr, value)),
                    16 => return_on_pagefault!(safe_write16(addr, value)),
                    _ => return_on_pagefault!(safe_write32(addr, value)),
                }
            }
            else {
                let r = (modrm_byte & 7) as i32;
                match size {
                    8 => write_reg8(r, value),
                    16 => write_reg16(r, value),
                    _ => write_reg32(r, value),
                }
            }
        },
        // VINSERTF128 (18) / VEXTRACTF128 (19) / VINSERTI128 (38) / VEXTRACTI128 (39)
        0x18 | 0x38 => {
            let mut rlo = s1lo;
            let mut rhi = s1hi;
            if imm & 1 != 0 {
                rhi = s2lo;
            }
            else {
                rlo = s2lo;
            }
            vex32_set(dst, rlo, rhi, true);
        },
        0x19 | 0x39 => {
            let value = if imm & 1 != 0 { s1hi } else { s1lo };
            if modrm_byte < 0xC0 {
                let addr = return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32;
                return_on_pagefault!(safe_write128(addr, value));
            }
            else {
                vex32_set((modrm_byte & 7) as i32, value, reg128 { u64: [0, 0] }, false);
            }
        },
        // VPINSRB/D and VINSERTPS
        0x20 | 0x21 | 0x22 => {
            if opcode == 0x21 {
                vex32_set(dst, crate::cpu::interp::simd_instr::insertps(s1lo, s2lo, imm), s1hi, l);
            }
            else {
                let value = if modrm_byte < 0xC0 {
                    let addr = return_on_pagefault!(modrm_resolve(modrm_byte)).addr() as i32;
                    if opcode == 0x20 {
                        return_on_pagefault!(safe_read8(addr)) as u32
                    }
                    else {
                        return_on_pagefault!(safe_read32s(addr)) as u32
                    }
                }
                else if opcode == 0x20 {
                    read_reg8(modrm_byte & 7) as u8 as u32
                }
                else {
                    read_reg32(modrm_byte & 7) as u32
                };
                let mut rlo = s1lo;
                if opcode == 0x20 {
                    rlo.u8[imm as usize] = value as u8;
                }
                else {
                    rlo.u32[imm as usize & 3] = value;
                }
                vex32_set(dst, rlo, s1hi, l);
            }
        },
        // VDPPS/VDPPD/VMPSADBW
        0x40 | 0x41 | 0x42 => {
            let (rlo, rhi) = match opcode {
                0x40 => (
                    crate::cpu::interp::simd_instr::dpps(s1lo, s2lo, imm),
                    if l { crate::cpu::interp::simd_instr::dpps(s1hi, s2hi, imm) } else { reg128 { u64: [0, 0] } },
                ),
                0x41 => (
                    crate::cpu::interp::simd_instr::dppd(s1lo, s2lo, imm),
                    if l { crate::cpu::interp::simd_instr::dppd(s1hi, s2hi, imm) } else { reg128 { u64: [0, 0] } },
                ),
                _ => (
                    crate::cpu::interp::simd_instr::mpsadbw(s1lo, s2lo, imm),
                    if l { crate::cpu::interp::simd_instr::mpsadbw(s1hi, s2hi, imm) } else { reg128 { u64: [0, 0] } },
                ),
            };
            vex32_set(dst, rlo, rhi, l);
        },
        // VBLENDVPS/VBLENDVPD/VPBLENDVB (mask register in imm[7:4])
        0x4A | 0x4B | 0x4C => {
            let mask = read_xmm128s((imm >> 4) as i32);
            let rlo = crate::cpu::interp::simd_instr::blendv_apply(s1lo, s2lo, mask);
            let rhi = if l {
                let mhi = *crate::cpu::global_pointers::ymm_high_ptr((imm >> 4) as i32);
                crate::cpu::interp::simd_instr::blendv_apply(s1hi, s2hi, mhi)
            }
            else {
                reg128 { u64: [0, 0] }
            };
            vex32_set(dst, rlo, rhi, l);
        },
        _ => trigger_ud(),
    }
}

pub unsafe fn instr_vex32(first: i32, byte2: i32) {
    let (map, pp, l, w, vvvv);
    if first == 0xC5 {
        let b = byte2 as u8;
        map = 1;
        w = false;
        vvvv = !(b >> 3) & 0xF;
        l = b & 4 != 0;
        pp = b & 3;
    }
    else {
        let b1 = byte2 as u8;
        let b2 = return_on_pagefault!(read_imm8()) as u8;
        map = b1 & 0x1F;
        w = b2 & 0x80 != 0;
        vvvv = !(b2 >> 3) & 0xF;
        l = b2 & 4 != 0;
        pp = b2 & 3;
    }
    let opcode = return_on_pagefault!(read_imm8()) as u8;
    match map {
        1 => vex32_0f(opcode, pp, l, w, vvvv),
        2 => vex32_0f38(opcode, pp, l, w, vvvv),
        3 => vex32_0f3a(opcode, pp, l, w, vvvv),
        _ => trigger_ud(),
    }
}
