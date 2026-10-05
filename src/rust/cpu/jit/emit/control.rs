use super::*;

pub fn gen_add_cs_offset(ctx: &mut JitContext) {
    if !ctx.cpu.has_flat_segmentation() {
        ctx.builder
            .load_fixed_i32(global_pointers::get_seg_offset(regs::CS));
        ctx.builder.add_i32();
    }
}

pub fn gen_get_eip(builder: &mut WasmBuilder) {
    builder.load_fixed_i32(global_pointers::instruction_pointer as u32);
}

pub fn gen_set_eip_to_after_current_instruction(ctx: &mut JitContext) {
    ctx.builder
        .const_i32(global_pointers::instruction_pointer as i32);
    gen_get_eip(ctx.builder);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.const_i32(ctx.cpu.eip as i32 & 0xFFF);
    ctx.builder.or_i32();
    ctx.builder.store_aligned_i32(0);
}

pub fn gen_set_previous_eip_offset_from_eip_with_low_bits(
    builder: &mut WasmBuilder,
    low_bits: i32,
) {
    // previous_ip = instruction_pointer & ~0xFFF | low_bits;
    dbg_assert!(low_bits & !0xFFF == 0);
    builder.const_i32(global_pointers::previous_ip as i32);
    gen_get_eip(builder);
    builder.const_i32(!0xFFF);
    builder.and_i32();
    builder.const_i32(low_bits);
    builder.or_i32();
    builder.store_aligned_i32(0);
}

pub fn gen_set_eip_low_bits(builder: &mut WasmBuilder, low_bits: i32) {
    // instruction_pointer = instruction_pointer & ~0xFFF | low_bits;
    dbg_assert!(low_bits & !0xFFF == 0);
    builder.const_i32(global_pointers::instruction_pointer as i32);
    gen_get_eip(builder);
    builder.const_i32(!0xFFF);
    builder.and_i32();
    builder.const_i32(low_bits);
    builder.or_i32();
    builder.store_aligned_i32(0);
}

pub fn gen_set_eip_low_bits_and_jump_rel32(builder: &mut WasmBuilder, low_bits: i32, n: i32) {
    // instruction_pointer = (instruction_pointer & ~0xFFF | low_bits) + n;
    dbg_assert!(low_bits & !0xFFF == 0);
    builder.const_i32(global_pointers::instruction_pointer as i32);
    gen_get_eip(builder);
    builder.const_i32(!0xFFF);
    builder.and_i32();
    builder.const_i32(low_bits);
    builder.or_i32();
    if n != 0 {
        builder.const_i32(n);
        builder.add_i32();
    }
    builder.store_aligned_i32(0);
}

pub fn gen_relative_jump(builder: &mut WasmBuilder, n: i32) {
    // add n to instruction_pointer
    if n != 0 {
        builder.const_i32(global_pointers::instruction_pointer as i32);
        gen_get_eip(builder);
        builder.const_i32(n);
        builder.add_i32();
        builder.store_aligned_i32(0);
    }
}

pub fn gen_page_switch_check(
    ctx: &mut JitContext,
    next_block_addr: u32,
    last_instruction_addr: u32,
) {
    // After switching a page while in jitted code, check if the page mapping still holds

    gen_get_eip(ctx.builder);
    let address_local = ctx.builder.set_new_local();
    gen_get_phys_eip_plus_mem(ctx, &address_local);
    ctx.builder.free_local(address_local);

    ctx.builder
        .const_i32(next_block_addr as i32 + unsafe { memory::mem8 } as i32);
    ctx.builder.ne_i32();

    if cfg!(debug_assertions) {
        ctx.builder.if_void();
        gen_profiler_stat_increment(ctx.builder, profiler::stat::FAILED_PAGE_CHANGE);
        gen_debug_track_jit_exit(ctx.builder, last_instruction_addr);
        ctx.builder.br(ctx.exit_label);
        ctx.builder.block_end();
    }
    else {
        ctx.builder.br_if(ctx.exit_label);
    }
}

pub fn gen_update_instruction_counter(ctx: &mut JitContext) {
    ctx.builder
        .const_i32(global_pointers::instruction_counter as i32);
    ctx.builder
        .load_fixed_i32(global_pointers::instruction_counter as u32);
    ctx.builder.get_local(&ctx.instruction_counter);
    ctx.builder.add_i32();
    ctx.builder.store_aligned_i32(0);
}

pub fn gen_jmp_rel16(builder: &mut WasmBuilder, rel16: u16) {
    let cs_offset_addr = global_pointers::get_seg_offset(regs::CS);
    builder.load_fixed_i32(cs_offset_addr);
    let local = builder.set_new_local();

    // generate:
    // *instruction_pointer = cs_offset + ((*instruction_pointer - cs_offset + rel16) & 0xFFFF);
    {
        builder.const_i32(global_pointers::instruction_pointer as i32);

        gen_get_eip(builder);
        builder.get_local(&local);
        builder.sub_i32();

        builder.const_i32(rel16 as i32);
        builder.add_i32();

        builder.const_i32(0xFFFF);
        builder.and_i32();

        builder.get_local(&local);
        builder.add_i32();

        builder.store_aligned_i32(0);
    }
    builder.free_local(local);
}

