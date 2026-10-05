use super::*;

pub fn gen_pop16_ss16(ctx: &mut JitContext) {
    // sp = segment_offsets[SS] + reg16[SP] (or just reg16[SP] if has_flat_segmentation)
    gen_get_reg16(ctx, regs::SP);

    if !ctx.cpu.has_flat_segmentation() {
        gen_get_ss_offset(ctx);
        ctx.builder.add_i32();
    }

    // result = safe_read16(sp)
    let address_local = ctx.builder.set_new_local();
    gen_safe_read16(ctx, &address_local);
    ctx.builder.free_local(address_local);

    // reg16[SP] += 2;
    gen_get_reg16(ctx, regs::SP);
    ctx.builder.const_i32(2);
    ctx.builder.add_i32();
    gen_set_reg16(ctx, regs::SP);

    // return value is already on stack
}

pub fn gen_pop16_ss32(ctx: &mut JitContext) {
    // esp = segment_offsets[SS] + reg32[ESP] (or just reg32[ESP] if has_flat_segmentation)
    gen_get_reg32(ctx, regs::ESP);

    if !ctx.cpu.has_flat_segmentation() {
        gen_get_ss_offset(ctx);
        ctx.builder.add_i32();
    }

    // result = safe_read16(esp)
    let address_local = ctx.builder.set_new_local();
    gen_safe_read16(ctx, &address_local);
    ctx.builder.free_local(address_local);

    // reg32[ESP] += 2;
    gen_get_reg32(ctx, regs::ESP);
    ctx.builder.const_i32(2);
    ctx.builder.add_i32();
    gen_set_reg32(ctx, regs::ESP);

    // return value is already on stack
}

pub fn gen_pop16(ctx: &mut JitContext) {
    if ctx.cpu.ssize_32() {
        gen_pop16_ss32(ctx);
    }
    else {
        gen_pop16_ss16(ctx);
    }
}

pub fn gen_pop32s_ss16(ctx: &mut JitContext) {
    // sp = reg16[SP]
    gen_get_reg16(ctx, regs::SP);

    // result = safe_read32s(segment_offsets[SS] + sp) (or just sp if has_flat_segmentation)
    if !ctx.cpu.has_flat_segmentation() {
        gen_get_ss_offset(ctx);
        ctx.builder.add_i32();
    }

    let address_local = ctx.builder.set_new_local();
    gen_safe_read32(ctx, &address_local);
    ctx.builder.free_local(address_local);

    // reg16[SP] = sp + 4;
    gen_get_reg16(ctx, regs::SP);
    ctx.builder.const_i32(4);
    ctx.builder.add_i32();
    gen_set_reg16(ctx, regs::SP);

    // return value is already on stack
}

pub fn gen_pop32s_ss32(ctx: &mut JitContext) {
    if !ctx.cpu.has_flat_segmentation() {
        gen_get_reg32(ctx, regs::ESP);
        gen_get_ss_offset(ctx);
        ctx.builder.add_i32();
        let address_local = ctx.builder.set_new_local();
        gen_safe_read32(ctx, &address_local);
        ctx.builder.free_local(address_local);
    }
    else {
        let reg = ctx.register_locals[regs::ESP as usize].unsafe_clone();
        gen_safe_read32(ctx, &reg);
    }

    gen_get_reg32(ctx, regs::ESP);
    ctx.builder.const_i32(4);
    ctx.builder.add_i32();
    gen_set_reg32(ctx, regs::ESP);

    // return value is already on stack
}

pub fn gen_pop32s(ctx: &mut JitContext) {
    if ctx.cpu.ssize_32() {
        gen_pop32s_ss32(ctx);
    }
    else {
        gen_pop32s_ss16(ctx);
    }
}

pub fn gen_adjust_stack_reg(ctx: &mut JitContext, offset: u32) {
    if ctx.cpu.ssize_32() {
        gen_get_reg32(ctx, regs::ESP);
        ctx.builder.const_i32(offset as i32);
        ctx.builder.add_i32();
        gen_set_reg32(ctx, regs::ESP);
    }
    else {
        gen_get_reg16(ctx, regs::SP);
        ctx.builder.const_i32(offset as i32);
        ctx.builder.add_i32();
        gen_set_reg16(ctx, regs::SP);
    }
}

pub fn gen_leave(ctx: &mut JitContext, os32: bool) {
    // [e]bp = safe_read{16,32}([e]bp)

    if ctx.cpu.ssize_32() {
        gen_get_reg32(ctx, regs::EBP);
    }
    else {
        gen_get_reg16(ctx, regs::BP);
    }

    let old_vbp = ctx.builder.tee_new_local();

    if !ctx.cpu.has_flat_segmentation() {
        gen_get_ss_offset(ctx);
        ctx.builder.add_i32();
    }
    if os32 {
        let address_local = ctx.builder.set_new_local();
        gen_safe_read32(ctx, &address_local);
        ctx.builder.free_local(address_local);
        gen_set_reg32(ctx, regs::EBP);
    }
    else {
        let address_local = ctx.builder.set_new_local();
        gen_safe_read16(ctx, &address_local);
        ctx.builder.free_local(address_local);
        gen_set_reg16(ctx, regs::BP);
    }

    // [e]sp = [e]bp + (os32 ? 4 : 2)

    if ctx.cpu.ssize_32() {
        ctx.builder.get_local(&old_vbp);
        ctx.builder.const_i32(if os32 { 4 } else { 2 });
        ctx.builder.add_i32();
        gen_set_reg32(ctx, regs::ESP);
    }
    else {
        ctx.builder.get_local(&old_vbp);
        ctx.builder.const_i32(if os32 { 4 } else { 2 });
        ctx.builder.add_i32();
        gen_set_reg16(ctx, regs::SP);
    }

    ctx.builder.free_local(old_vbp);
}

pub fn gen_task_switch_test(ctx: &mut JitContext) {
    // generate if(cr[0] & (CR0_EM | CR0_TS)) { task_switch_test_jit(); goto exit_with_fault; }
    let cr0_offset = global_pointers::get_creg_offset(0);

    dbg_assert!(regs::CR0_EM | regs::CR0_TS <= 0xFF);
    ctx.builder.load_fixed_u8(cr0_offset);
    ctx.builder.const_i32((regs::CR0_EM | regs::CR0_TS) as i32);
    ctx.builder.and_i32();

    ctx.builder.if_void();
    {
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        gen_fn1_const(
            ctx.builder,
            "task_switch_test_jit",
            ctx.start_of_current_instruction & 0xFFF,
        );
        ctx.builder.br(ctx.exit_with_fault_label);
    }
    ctx.builder.block_end();
}

pub fn gen_task_switch_test_mmx(ctx: &mut JitContext) {
    // generate if(cr[0] & (CR0_EM | CR0_TS)) { task_switch_test_mmx_jit(); goto exit_with_fault; }
    let cr0_offset = global_pointers::get_creg_offset(0);

    dbg_assert!(regs::CR0_EM | regs::CR0_TS <= 0xFF);
    ctx.builder.load_fixed_u8(cr0_offset);
    ctx.builder.const_i32((regs::CR0_EM | regs::CR0_TS) as i32);
    ctx.builder.and_i32();

    ctx.builder.if_void();
    {
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        gen_fn1_const(
            ctx.builder,
            "task_switch_test_mmx_jit",
            ctx.start_of_current_instruction & 0xFFF,
        );
        ctx.builder.br(ctx.exit_with_fault_label);
    }
    ctx.builder.block_end();
}

pub fn gen_push16(ctx: &mut JitContext, value_local: &WasmLocal) {
    if ctx.cpu.ssize_32() {
        gen_get_reg32(ctx, regs::ESP);
    }
    else {
        gen_get_reg16(ctx, regs::SP);
    };

    ctx.builder.const_i32(2);
    ctx.builder.sub_i32();

    let reg_updated_local = if !ctx.cpu.ssize_32() || !ctx.cpu.has_flat_segmentation() {
        let reg_updated_local = ctx.builder.tee_new_local();
        if !ctx.cpu.ssize_32() {
            ctx.builder.const_i32(0xFFFF);
            ctx.builder.and_i32();
        }

        if !ctx.cpu.has_flat_segmentation() {
            gen_get_ss_offset(ctx);
            ctx.builder.add_i32();
        }

        let sp_local = ctx.builder.set_new_local();
        gen_safe_write16(ctx, &sp_local, &value_local);
        ctx.builder.free_local(sp_local);

        ctx.builder.get_local(&reg_updated_local);
        reg_updated_local
    }
    else {
        // short path: The address written to is equal to ESP/SP minus two
        let reg_updated_local = ctx.builder.tee_new_local();
        gen_safe_write16(ctx, &reg_updated_local, &value_local);
        reg_updated_local
    };

    if ctx.cpu.ssize_32() {
        gen_set_reg32(ctx, regs::ESP);
    }
    else {
        gen_set_reg16(ctx, regs::SP);
    };
    ctx.builder.free_local(reg_updated_local);
}

pub fn gen_push32(ctx: &mut JitContext, value_local: &WasmLocal) {
    if ctx.cpu.ssize_32() {
        gen_get_reg32(ctx, regs::ESP);
    }
    else {
        gen_get_reg16(ctx, regs::SP);
    };

    ctx.builder.const_i32(4);
    ctx.builder.sub_i32();

    let new_sp_local = if !ctx.cpu.ssize_32() || !ctx.cpu.has_flat_segmentation() {
        let new_sp_local = ctx.builder.tee_new_local();
        if !ctx.cpu.ssize_32() {
            ctx.builder.const_i32(0xFFFF);
            ctx.builder.and_i32();
        }

        if !ctx.cpu.has_flat_segmentation() {
            gen_get_ss_offset(ctx);
            ctx.builder.add_i32();
        }

        let sp_local = ctx.builder.set_new_local();

        gen_safe_write32(ctx, &sp_local, &value_local);
        ctx.builder.free_local(sp_local);

        ctx.builder.get_local(&new_sp_local);
        new_sp_local
    }
    else {
        // short path: The address written to is equal to ESP/SP minus four
        let new_sp_local = ctx.builder.tee_new_local();
        gen_safe_write32(ctx, &new_sp_local, &value_local);
        new_sp_local
    };

    if ctx.cpu.ssize_32() {
        gen_set_reg32(ctx, regs::ESP);
    }
    else {
        gen_set_reg16(ctx, regs::SP);
    };
    ctx.builder.free_local(new_sp_local);
}

pub fn gen_push32_sreg(ctx: &mut JitContext, reg: u32) {
    gen_get_sreg(ctx, reg);
    let value_local = ctx.builder.set_new_local();

    if ctx.cpu.ssize_32() {
        gen_get_reg32(ctx, regs::ESP);
    }
    else {
        gen_get_reg16(ctx, regs::SP);
    };

    ctx.builder.const_i32(4);
    ctx.builder.sub_i32();

    let new_sp_local = if !ctx.cpu.ssize_32() || !ctx.cpu.has_flat_segmentation() {
        let new_sp_local = ctx.builder.tee_new_local();
        if !ctx.cpu.ssize_32() {
            ctx.builder.const_i32(0xFFFF);
            ctx.builder.and_i32();
        }

        if !ctx.cpu.has_flat_segmentation() {
            gen_get_ss_offset(ctx);
            ctx.builder.add_i32();
        }

        let sp_local = ctx.builder.set_new_local();

        gen_safe_write16(ctx, &sp_local, &value_local);
        ctx.builder.free_local(sp_local);

        ctx.builder.get_local(&new_sp_local);
        new_sp_local
    }
    else {
        // short path: The address written to is equal to ESP/SP minus four
        let new_sp_local = ctx.builder.tee_new_local();
        gen_safe_write16(ctx, &new_sp_local, &value_local);
        new_sp_local
    };

    if ctx.cpu.ssize_32() {
        gen_set_reg32(ctx, regs::ESP);
    }
    else {
        gen_set_reg16(ctx, regs::SP);
    };
    ctx.builder.free_local(new_sp_local);
    ctx.builder.free_local(value_local);
}

