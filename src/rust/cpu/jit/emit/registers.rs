use super::*;

pub fn gen_get_reg8(ctx: &mut JitContext, r: u32) {
    match r {
        regs::AL | regs::CL | regs::DL | regs::BL => {
            ctx.builder.get_local(&ctx.register_locals[r as usize]);
            ctx.builder.const_i32(0xFF);
            ctx.builder.and_i32();
        },
        regs::AH | regs::CH | regs::DH | regs::BH => {
            ctx.builder
                .get_local(&ctx.register_locals[(r - 4) as usize]);
            ctx.builder.const_i32(8);
            ctx.builder.shr_u_i32();
            ctx.builder.const_i32(0xFF);
            ctx.builder.and_i32();
        },
        _ => assert!(false),
    }
}

/// Return a new local referencing one of the 8 bit registers or a direct reference to one of the
/// register locals. Higher bits might be garbage (suitable for gen_cmp8 etc.). Must be freed with
/// gen_free_reg8_or_alias.
pub fn gen_get_reg8_or_alias_to_reg32(ctx: &mut JitContext, r: u32) -> WasmLocal {
    match r {
        regs::AL | regs::CL | regs::DL | regs::BL => ctx.register_locals[r as usize].unsafe_clone(),
        regs::AH | regs::CH | regs::DH | regs::BH => {
            ctx.builder
                .get_local(&ctx.register_locals[(r - 4) as usize]);
            ctx.builder.const_i32(8);
            ctx.builder.shr_u_i32();
            ctx.builder.set_new_local()
        },
        _ => panic!(),
    }
}

pub fn gen_free_reg8_or_alias(ctx: &mut JitContext, r: u32, local: WasmLocal) {
    match r {
        regs::AL | regs::CL | regs::DL | regs::BL => {},
        regs::AH | regs::CH | regs::DH | regs::BH => ctx.builder.free_local(local),
        _ => panic!(),
    }
}

pub fn gen_get_reg16(ctx: &mut JitContext, r: u32) {
    ctx.builder.get_local(&ctx.register_locals[r as usize]);
    ctx.builder.const_i32(0xFFFF);
    ctx.builder.and_i32();
}

pub fn gen_get_reg32(ctx: &mut JitContext, r: u32) {
    ctx.builder.get_local(&ctx.register_locals[r as usize]);
}

pub fn gen_set_reg8(ctx: &mut JitContext, r: u32) {
    match r {
        regs::AL | regs::CL | regs::DL | regs::BL => {
            // reg32[r] = stack_value & 0xFF | reg32[r] & ~0xFF
            ctx.builder.const_i32(0xFF);
            ctx.builder.and_i32();

            ctx.builder.get_local(&ctx.register_locals[r as usize]);
            ctx.builder.const_i32(!0xFF);
            ctx.builder.and_i32();

            ctx.builder.or_i32();
            ctx.builder.set_local(&ctx.register_locals[r as usize]);
        },
        regs::AH | regs::CH | regs::DH | regs::BH => {
            // reg32[r] = stack_value << 8 & 0xFF00 | reg32[r] & ~0xFF00
            ctx.builder.const_i32(8);
            ctx.builder.shl_i32();
            ctx.builder.const_i32(0xFF00);
            ctx.builder.and_i32();

            ctx.builder
                .get_local(&ctx.register_locals[(r - 4) as usize]);
            ctx.builder.const_i32(!0xFF00);
            ctx.builder.and_i32();

            ctx.builder.or_i32();
            ctx.builder
                .set_local(&ctx.register_locals[(r - 4) as usize]);
        },
        _ => assert!(false),
    }
}

pub fn gen_set_reg8_unmasked(ctx: &mut JitContext, r: u32) {
    if cfg!(debug_assertions) {
        let val = ctx.builder.set_new_local();
        ctx.builder.get_local(&val);
        ctx.builder.const_i32(!0xFF);
        ctx.builder.and_i32();
        ctx.builder.if_void();
        ctx.builder.unreachable();
        ctx.builder.block_end();
        ctx.builder.get_local(&val);
        ctx.builder.free_local(val);
    }

    match r {
        regs::AL | regs::CL | regs::DL | regs::BL => {
            // reg32[r] = stack_value | reg32[r] & ~0xFF
            ctx.builder.get_local(&ctx.register_locals[r as usize]);
            ctx.builder.const_i32(!0xFF);
            ctx.builder.and_i32();

            ctx.builder.or_i32();
            ctx.builder.set_local(&ctx.register_locals[r as usize]);
        },
        regs::AH | regs::CH | regs::DH | regs::BH => {
            // reg32[r] = stack_value << 8 | reg32[r] & ~0xFF00
            ctx.builder.const_i32(8);
            ctx.builder.shl_i32();
            ctx.builder.const_i32(0xFF00);
            ctx.builder.and_i32();

            ctx.builder
                .get_local(&ctx.register_locals[(r - 4) as usize]);
            ctx.builder.const_i32(!0xFF00);
            ctx.builder.and_i32();

            ctx.builder.or_i32();
            ctx.builder
                .set_local(&ctx.register_locals[(r - 4) as usize]);
        },
        _ => assert!(false),
    }
}

pub fn gen_set_reg16(ctx: &mut JitContext, r: u32) {
    gen_set_reg16_local(ctx.builder, &ctx.register_locals[r as usize]);
}

pub fn gen_set_reg16_unmasked(ctx: &mut JitContext, r: u32) {
    if cfg!(debug_assertions) {
        let val = ctx.builder.set_new_local();
        ctx.builder.get_local(&val);
        ctx.builder.const_i32(!0xFFFF);
        ctx.builder.and_i32();
        ctx.builder.if_void();
        ctx.builder.unreachable();
        ctx.builder.block_end();
        ctx.builder.get_local(&val);
        ctx.builder.free_local(val);
    }

    ctx.builder.get_local(&ctx.reg(r));
    ctx.builder.const_i32(!0xFFFF);
    ctx.builder.and_i32();
    ctx.builder.or_i32();
    ctx.builder.set_local(&ctx.reg(r));
}

pub fn gen_set_reg16_local(builder: &mut WasmBuilder, local: &WasmLocal) {
    // reg32[r] = v & 0xFFFF | reg32[r] & ~0xFFFF
    builder.const_i32(0xFFFF);
    builder.and_i32();
    builder.get_local(local);
    builder.const_i32(!0xFFFF);
    builder.and_i32();
    builder.or_i32();
    builder.set_local(local);
}

pub fn gen_set_reg32(ctx: &mut JitContext, r: u32) {
    ctx.builder.set_local(&ctx.register_locals[r as usize]);
}

pub fn decr_exc_asize(ctx: &mut JitContext) {
    gen_get_reg32(ctx, regs::ECX);
    ctx.builder.const_i32(1);
    ctx.builder.sub_i32();
    if ctx.cpu.asize_32() {
        gen_set_reg32(ctx, regs::ECX);
    }
    else {
        gen_set_reg16(ctx, regs::CX);
    }
}

pub fn gen_read_reg_xmm128_into_scratch(ctx: &mut JitContext, r: u32) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    let dest = global_pointers::get_reg_xmm_offset(r);
    ctx.builder.const_i32(dest as i32);
    ctx.builder.load_aligned_i64(0);
    ctx.builder.store_aligned_i64(0);

    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32 + 8);
    let dest = global_pointers::get_reg_xmm_offset(r) + 8;
    ctx.builder.const_i32(dest as i32);
    ctx.builder.load_aligned_i64(0);
    ctx.builder.store_aligned_i64(0);
}

pub fn gen_get_sreg(ctx: &mut JitContext, r: u32) {
    ctx.builder
        .load_fixed_u16(global_pointers::get_sreg_offset(r))
}

pub fn gen_get_ss_offset(ctx: &mut JitContext) {
    ctx.builder
        .load_fixed_i32(global_pointers::get_seg_offset(regs::SS));
}

