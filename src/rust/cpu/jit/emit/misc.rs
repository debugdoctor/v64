use super::*;

pub fn gen_fpu_get_sti(ctx: &mut JitContext, i: u32) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    ctx.builder.const_i32(i as i32);
    ctx.builder.call_fn2("fpu_get_sti_jit");
    ctx.builder
        .load_fixed_i64(global_pointers::sse_scratch_register as u32);
    ctx.builder
        .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
}

pub fn gen_fpu_load_m32(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    gen_modrm_resolve_safe_read32(ctx, modrm_byte);
    ctx.builder.call_fn2("f32_to_f80_jit");
    ctx.builder
        .load_fixed_i64(global_pointers::sse_scratch_register as u32);
    ctx.builder
        .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
}

pub fn gen_fpu_load_m64(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    gen_modrm_resolve_safe_read64(ctx, modrm_byte);
    ctx.builder.call_fn2_i32_i64("f64_to_f80_jit");
    ctx.builder
        .load_fixed_i64(global_pointers::sse_scratch_register as u32);
    ctx.builder
        .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
}

pub fn gen_fpu_load_i16(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    gen_modrm_resolve_safe_read16(ctx, modrm_byte);
    sign_extend_i16(ctx.builder);
    ctx.builder.call_fn2("i32_to_f80_jit");
    ctx.builder
        .load_fixed_i64(global_pointers::sse_scratch_register as u32);
    ctx.builder
        .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
}
pub fn gen_fpu_load_i32(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    gen_modrm_resolve_safe_read32(ctx, modrm_byte);
    ctx.builder.call_fn2("i32_to_f80_jit");
    ctx.builder
        .load_fixed_i64(global_pointers::sse_scratch_register as u32);
    ctx.builder
        .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
}
pub fn gen_fpu_load_i64(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    ctx.builder
        .const_i32(global_pointers::sse_scratch_register as i32);
    gen_modrm_resolve_safe_read64(ctx, modrm_byte);
    ctx.builder.call_fn2_i32_i64("i64_to_f80_jit");
    ctx.builder
        .load_fixed_i64(global_pointers::sse_scratch_register as u32);
    ctx.builder
        .load_fixed_u16(global_pointers::sse_scratch_register as u32 + 8);
}

pub fn gen_trigger_de(ctx: &mut JitContext) {
    gen_fn1_const(
        ctx.builder,
        "trigger_de_jit",
        ctx.start_of_current_instruction & 0xFFF,
    );
    gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
    ctx.builder.br(ctx.exit_with_fault_label);
}

pub fn gen_trigger_ud(ctx: &mut JitContext) {
    gen_fn1_const(
        ctx.builder,
        "trigger_ud_jit",
        ctx.start_of_current_instruction & 0xFFF,
    );
    gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
    ctx.builder.br(ctx.exit_with_fault_label);
}

pub fn gen_trigger_gp(ctx: &mut JitContext, error_code: u32) {
    gen_fn2_const(
        ctx.builder,
        "trigger_gp_jit",
        error_code,
        ctx.start_of_current_instruction & 0xFFF,
    );
    gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
    ctx.builder.br(ctx.exit_with_fault_label);
}

pub fn gen_condition_fn_negated(ctx: &mut JitContext, condition: u8) {
    gen_condition_fn(ctx, condition ^ 1)
}

pub fn gen_condition_fn(ctx: &mut JitContext, condition: u8) {
    if condition & 0xF0 == 0x00 || condition & 0xF0 == 0x70 || condition & 0xF0 == 0x80 {
        match condition & 0xF {
            0x0 => {
                gen_getof(ctx);
            },
            0x1 => {
                gen_getof(ctx);
                ctx.builder.eqz_i32();
            },
            0x2 => {
                gen_getcf(ctx, ConditionNegate::False);
            },
            0x3 => {
                gen_getcf(ctx, ConditionNegate::True);
            },
            0x4 => {
                gen_getzf(ctx, ConditionNegate::False);
            },
            0x5 => {
                gen_getzf(ctx, ConditionNegate::True);
            },
            0x6 => {
                gen_test_be(ctx, ConditionNegate::False);
            },
            0x7 => {
                gen_test_be(ctx, ConditionNegate::True);
            },
            0x8 => {
                gen_getsf(ctx, ConditionNegate::False);
            },
            0x9 => {
                gen_getsf(ctx, ConditionNegate::True);
            },
            0xA => {
                gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
                gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED_PF);
                ctx.builder.call_fn0_ret("test_p");
            },
            0xB => {
                gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
                gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED_PF);
                ctx.builder.call_fn0_ret("test_np");
            },
            0xC => {
                gen_test_l(ctx, ConditionNegate::False);
            },
            0xD => {
                gen_test_l(ctx, ConditionNegate::True);
            },
            0xE => {
                gen_test_le(ctx, ConditionNegate::False);
            },
            0xF => {
                gen_test_le(ctx, ConditionNegate::True);
            },
            _ => {
                dbg_assert!(false);
            },
        }
    }
    else {
        // loop, loopnz, loopz, jcxz
        dbg_assert!(condition & !0x3 == 0xE0);
        if condition == 0xE0 {
            gen_test_loopnz(ctx, ctx.cpu.asize_32());
        }
        else if condition == 0xE1 {
            gen_test_loopz(ctx, ctx.cpu.asize_32());
        }
        else if condition == 0xE2 {
            gen_test_loop(ctx, ctx.cpu.asize_32());
        }
        else if condition == 0xE3 {
            gen_test_jcxz(ctx, ctx.cpu.asize_32());
        }
    }
}

pub fn gen_move_registers_from_locals_to_memory(ctx: &mut JitContext) {
    if cfg!(feature = "profiler") {
        let instruction = memory::read32s(ctx.start_of_current_instruction) as u32;
        opstats::gen_opstat_unguarded_register(ctx.builder, instruction);
    }

    for i in 0..8 {
        ctx.builder
            .const_i32(global_pointers::get_reg32_offset(i as u32) as i32);
        ctx.builder.get_local(&ctx.register_locals[i]);
        ctx.builder.store_aligned_i32(0);
    }
}
pub fn gen_move_registers_from_memory_to_locals(ctx: &mut JitContext) {
    if cfg!(feature = "profiler") {
        let instruction = memory::read32s(ctx.start_of_current_instruction) as u32;
        opstats::gen_opstat_unguarded_register(ctx.builder, instruction);
    }

    for i in 0..8 {
        ctx.builder
            .const_i32(global_pointers::get_reg32_offset(i as u32) as i32);
        ctx.builder.load_aligned_i32(0);
        ctx.builder.set_local(&ctx.register_locals[i]);
    }
}

pub fn gen_profiler_stat_increment(builder: &mut WasmBuilder, stat: profiler::stat) {
    if !cfg!(feature = "profiler") {
        return;
    }
    let addr = unsafe { &raw mut profiler::stat_array[stat as usize] } as u32;
    builder.increment_fixed_i64(addr, 1)
}

pub fn gen_debug_track_jit_exit(builder: &mut WasmBuilder, address: u32) {
    if cfg!(feature = "profiler") {
        gen_fn1_const(builder, "track_jit_exit", address);
    }
}
