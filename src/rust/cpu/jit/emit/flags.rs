use super::*;

pub fn gen_get_flags(builder: &mut WasmBuilder) {
    builder.load_fixed_i32(global_pointers::flags as u32);
}
fn gen_get_flags_changed(builder: &mut WasmBuilder) {
    builder.load_fixed_i32(global_pointers::flags_changed as u32);
}
fn gen_get_last_result(builder: &mut WasmBuilder, previous_instruction: &Instruction) {
    match previous_instruction {
        Instruction::Add {
            dest: InstructionOperandDest::WasmLocal(l),
            opsize: OPSIZE_32,
            ..
        }
        | Instruction::AdcSbb {
            dest: InstructionOperandDest::WasmLocal(l),
            opsize: OPSIZE_32,
            ..
        }
        | Instruction::Sub {
            dest: InstructionOperandDest::WasmLocal(l),
            opsize: OPSIZE_32,
            ..
        }
        | Instruction::Bitwise {
            dest: InstructionOperandDest::WasmLocal(l),
            opsize: OPSIZE_32,
        }
        | Instruction::NonZeroShift {
            dest: InstructionOperandDest::WasmLocal(l),
            opsize: OPSIZE_32,
        } => builder.get_local(&l),
        Instruction::Cmp {
            dest: InstructionOperandDest::WasmLocal(l),
            source,
            opsize: OPSIZE_32,
        } => {
            if source.is_zero() {
                builder.get_local(&l)
            }
            else {
                builder.load_fixed_i32(global_pointers::last_result as u32)
            }
        },
        _ => builder.load_fixed_i32(global_pointers::last_result as u32),
    }
}
fn gen_get_last_op_size(builder: &mut WasmBuilder) {
    builder.load_fixed_i32(global_pointers::last_op_size as u32);
}
fn gen_get_last_op1(builder: &mut WasmBuilder, previous_instruction: &Instruction) {
    match previous_instruction {
        Instruction::Cmp {
            dest: InstructionOperandDest::WasmLocal(l),
            source: _,
            opsize: OPSIZE_32,
        } => builder.get_local(&l),
        _ => builder.load_fixed_i32(global_pointers::last_op1 as u32),
    }
}

pub fn gen_get_real_eip(ctx: &mut JitContext) {
    gen_get_eip(ctx.builder);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.const_i32(ctx.cpu.eip as i32 & 0xFFF);
    ctx.builder.or_i32();
    if !ctx.cpu.has_flat_segmentation() {
        ctx.builder
            .load_fixed_i32(global_pointers::get_seg_offset(regs::CS));
        ctx.builder.sub_i32();
    }
}

pub fn gen_set_last_op1(builder: &mut WasmBuilder, source: &WasmLocal) {
    builder.const_i32(global_pointers::last_op1 as i32);
    builder.get_local(&source);
    builder.store_aligned_i32(0);
}

pub fn gen_set_last_result(builder: &mut WasmBuilder, source: &WasmLocal) {
    builder.const_i32(global_pointers::last_result as i32);
    builder.get_local(&source);
    builder.store_aligned_i32(0);
}

pub fn gen_clear_flags_changed_bits(builder: &mut WasmBuilder, bits_to_clear: i32) {
    builder.const_i32(global_pointers::flags_changed as i32);
    gen_get_flags_changed(builder);
    builder.const_i32(!bits_to_clear);
    builder.and_i32();
    builder.store_aligned_i32(0);
}

pub fn gen_set_last_op_size_and_flags_changed(
    builder: &mut WasmBuilder,
    last_op_size: i32,
    flags_changed: i32,
) {
    dbg_assert!(last_op_size == OPSIZE_8 || last_op_size == OPSIZE_16 || last_op_size == OPSIZE_32);
    dbg_assert!(global_pointers::last_op_size as i32 % 8 == 0);
    dbg_assert!(global_pointers::last_op_size as i32 + 4 == global_pointers::flags_changed as i32);
    builder.const_i32(global_pointers::last_op_size as i32);
    builder.const_i64(last_op_size as u32 as i64 | (flags_changed as u32 as i64) << 32);
    builder.store_aligned_i64(0);
}

pub fn gen_set_flags_bits(builder: &mut WasmBuilder, bits_to_set: i32) {
    builder.const_i32(global_pointers::flags as i32);
    gen_get_flags(builder);
    builder.const_i32(bits_to_set);
    builder.or_i32();
    builder.store_aligned_i32(0);
}

pub fn gen_clear_flags_bits(builder: &mut WasmBuilder, bits_to_clear: i32) {
    builder.const_i32(global_pointers::flags as i32);
    gen_get_flags(builder);
    builder.const_i32(!bits_to_clear);
    builder.and_i32();
    builder.store_aligned_i32(0);
}

#[derive(PartialEq)]
pub enum ConditionNegate {
    True,
    False,
}

pub fn gen_getzf(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp {
            dest: InstructionOperandDest::WasmLocal(dest),
            source: InstructionOperand::WasmLocal(source),
            opsize: OPSIZE_32,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            ctx.builder.get_local(dest);
            ctx.builder.get_local(source);
            if negate == ConditionNegate::False {
                ctx.builder.eq_i32();
            }
            else {
                ctx.builder.ne_i32();
            }
        },
        Instruction::Cmp {
            dest: InstructionOperandDest::WasmLocal(dest),
            source: InstructionOperand::Immediate(0),
            opsize: OPSIZE_32,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            ctx.builder.get_local(dest);
            if negate == ConditionNegate::False {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Cmp {
            dest: InstructionOperandDest::WasmLocal(dest),
            source: InstructionOperand::Immediate(i),
            opsize: OPSIZE_32,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            ctx.builder.get_local(dest);
            ctx.builder.const_i32(*i);
            if negate == ConditionNegate::False {
                ctx.builder.eq_i32();
            }
            else {
                ctx.builder.ne_i32();
            }
        },
        Instruction::Cmp { .. }
        | Instruction::Sub { .. }
        | Instruction::Add { .. }
        | Instruction::AdcSbb { .. }
        | Instruction::NonZeroShift { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            if negate == ConditionNegate::False {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Bitwise { opsize, .. } => {
            let &opsize = opsize;
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            // Note: Necessary because test{8,16} don't mask either last_result or any of their operands
            // TODO: Use local instead of last_result for 8-bit/16-bit
            if opsize == OPSIZE_32 {
                gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            }
            else if opsize == OPSIZE_16 {
                ctx.builder
                    .load_fixed_u16(global_pointers::last_result as u32);
            }
            else if opsize == OPSIZE_8 {
                ctx.builder
                    .load_fixed_u8(global_pointers::last_result as u32);
            }
            if negate == ConditionNegate::False {
                ctx.builder.eqz_i32();
            }
        },
        &Instruction::Other => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            gen_get_flags_changed(ctx.builder);
            ctx.builder.const_i32(FLAG_ZERO);
            ctx.builder.and_i32();
            ctx.builder.if_i32();

            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            let last_result = ctx.builder.tee_new_local();
            ctx.builder.const_i32(-1);
            ctx.builder.xor_i32();
            ctx.builder.get_local(&last_result);
            ctx.builder.free_local(last_result);
            ctx.builder.const_i32(1);
            ctx.builder.sub_i32();
            ctx.builder.and_i32();
            gen_get_last_op_size(ctx.builder);
            ctx.builder.shr_u_i32();
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();

            ctx.builder.else_();
            gen_get_flags(ctx.builder);
            ctx.builder.const_i32(FLAG_ZERO);
            ctx.builder.and_i32();
            ctx.builder.block_end();

            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_getcf(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp { source, opsize, .. }
        | Instruction::Sub {
            source,
            opsize,
            is_dec: false,
            ..
        } => {
            // Note: x < y and x < x - y can be used interchangeably (see getcf)
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            match (opsize, source) {
                (&OPSIZE_32, InstructionOperand::WasmLocal(l)) => ctx.builder.get_local(l),
                (_, &InstructionOperand::Immediate(i)) => ctx.builder.const_i32(i),
                _ => gen_get_last_result(ctx.builder, &ctx.previous_instruction),
            }
            if negate == ConditionNegate::True {
                ctx.builder.geu_i32();
            }
            else {
                ctx.builder.ltu_i32();
            }
        },
        Instruction::Add {
            source,
            opsize,
            is_inc: false,
            ..
        } => {
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            match (opsize, source) {
                (&OPSIZE_32, InstructionOperand::WasmLocal(l)) => ctx.builder.get_local(l),
                (_, &InstructionOperand::Immediate(i)) => ctx.builder.const_i32(i),
                _ => gen_get_last_op1(ctx.builder, &ctx.previous_instruction),
            }
            if negate == ConditionNegate::True {
                ctx.builder.geu_i32();
            }
            else {
                ctx.builder.ltu_i32();
            }
        },
        Instruction::Add { is_inc: true, .. } | Instruction::Sub { is_dec: true, .. } => {
            gen_get_flags(ctx.builder);
            ctx.builder.const_i32(FLAG_CARRY);
            ctx.builder.and_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Bitwise { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            ctx.builder
                .const_i32(if negate == ConditionNegate::True { 1 } else { 0 });
        },
        Instruction::NonZeroShift { .. } | Instruction::AdcSbb { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_flags(ctx.builder);
            ctx.builder.const_i32(FLAG_CARRY);
            ctx.builder.and_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
        &Instruction::Other => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);

            gen_get_flags_changed(ctx.builder);
            let flags_changed = ctx.builder.tee_new_local();
            ctx.builder.const_i32(FLAG_CARRY);
            ctx.builder.and_i32();
            ctx.builder.if_i32();

            ctx.builder.get_local(&flags_changed);
            ctx.builder.const_i32(31);
            ctx.builder.shr_s_i32();
            ctx.builder.free_local(flags_changed);
            let sub_mask = ctx.builder.set_new_local();

            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            ctx.builder.get_local(&sub_mask);
            ctx.builder.xor_i32();

            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            ctx.builder.get_local(&sub_mask);
            ctx.builder.xor_i32();

            ctx.builder.ltu_i32();

            ctx.builder.else_();
            gen_get_flags(ctx.builder);
            ctx.builder.const_i32(FLAG_CARRY);
            ctx.builder.and_i32();
            ctx.builder.block_end();

            ctx.builder.free_local(sub_mask);

            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_getsf(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp { opsize, .. }
        | Instruction::Sub { opsize, .. }
        | Instruction::Add { opsize, .. }
        | Instruction::AdcSbb { opsize, .. }
        | Instruction::Bitwise { opsize, .. }
        | Instruction::NonZeroShift { opsize, .. } => {
            let &opsize = opsize;
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            if opsize == OPSIZE_32 {
                ctx.builder.const_i32(0);
                if negate == ConditionNegate::True {
                    ctx.builder.ge_i32();
                }
                else {
                    ctx.builder.lt_i32();
                }
            }
            else {
                // TODO: use register (see get_last_result)
                ctx.builder
                    .const_i32(if opsize == OPSIZE_16 { 0x8000 } else { 0x80 });
                ctx.builder.and_i32();
                if negate == ConditionNegate::True {
                    ctx.builder.eqz_i32();
                }
            }
        },
        &Instruction::Other => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            gen_get_flags_changed(ctx.builder);
            ctx.builder.const_i32(FLAG_SIGN);
            ctx.builder.and_i32();
            ctx.builder.if_i32();
            {
                gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                gen_get_last_op_size(ctx.builder);
                ctx.builder.shr_u_i32();
                ctx.builder.const_i32(1);
                ctx.builder.and_i32();
            }
            ctx.builder.else_();
            {
                gen_get_flags(ctx.builder);
                ctx.builder.const_i32(FLAG_SIGN);
                ctx.builder.and_i32();
            }
            ctx.builder.block_end();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_getof(ctx: &mut JitContext) {
    match &ctx.previous_instruction {
        Instruction::Cmp { opsize, .. } | Instruction::Sub { opsize, .. } => {
            // TODO: a better formula might be possible
            let &opsize = opsize;
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            ctx.builder.xor_i32();

            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            ctx.builder.sub_i32();
            ctx.builder.xor_i32();
            ctx.builder.and_i32();

            ctx.builder.const_i32(if opsize == OPSIZE_32 {
                0x8000_0000u32 as i32
            }
            else if opsize == OPSIZE_16 {
                0x8000
            }
            else {
                0x80
            });
            ctx.builder.and_i32();
        },
        Instruction::Add { opsize, .. } => {
            // TODO: a better formula might be possible
            let &opsize = opsize;
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            ctx.builder.xor_i32();

            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            gen_get_last_result(ctx.builder, &ctx.previous_instruction);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            ctx.builder.sub_i32();
            ctx.builder.xor_i32();
            ctx.builder.and_i32();

            ctx.builder.const_i32(if opsize == OPSIZE_32 {
                0x8000_0000u32 as i32
            }
            else if opsize == OPSIZE_16 {
                0x8000
            }
            else {
                0x80
            });
            ctx.builder.and_i32();
        },
        Instruction::Bitwise { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            ctx.builder.const_i32(0);
        },
        Instruction::NonZeroShift { .. } | Instruction::AdcSbb { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_flags(ctx.builder);
            ctx.builder.const_i32(FLAG_OVERFLOW);
            ctx.builder.and_i32();
        },
        &Instruction::Other => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            gen_get_flags_changed(ctx.builder);
            let flags_changed = ctx.builder.tee_new_local();
            ctx.builder.const_i32(FLAG_OVERFLOW);
            ctx.builder.and_i32();
            ctx.builder.if_i32();
            {
                gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                let last_op1 = ctx.builder.tee_new_local();
                gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                let last_result = ctx.builder.tee_new_local();
                ctx.builder.xor_i32();

                ctx.builder.get_local(&last_result);
                ctx.builder.get_local(&last_op1);
                ctx.builder.sub_i32();
                gen_get_flags_changed(ctx.builder);
                ctx.builder.const_i32(31);
                ctx.builder.shr_u_i32();
                ctx.builder.sub_i32();

                ctx.builder.get_local(&last_result);
                ctx.builder.xor_i32();

                ctx.builder.and_i32();

                gen_get_last_op_size(ctx.builder);
                ctx.builder.shr_u_i32();
                ctx.builder.const_i32(1);
                ctx.builder.and_i32();

                ctx.builder.free_local(last_op1);
                ctx.builder.free_local(last_result);
            }
            ctx.builder.else_();
            {
                gen_get_flags(ctx.builder);
                ctx.builder.const_i32(FLAG_OVERFLOW);
                ctx.builder.and_i32();
            }
            ctx.builder.block_end();
            ctx.builder.free_local(flags_changed);
        },
    }
}

pub fn gen_test_be(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp {
            dest,
            source,
            opsize,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            match dest {
                InstructionOperandDest::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 0xFF } else { 0xFFFF });
                        ctx.builder.and_i32();
                    }
                },
                InstructionOperandDest::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                },
            }
            match source {
                InstructionOperand::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 0xFF } else { 0xFFFF });
                        ctx.builder.and_i32();
                    }
                },
                InstructionOperand::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 0xFF } else { 0xFFFF });
                        ctx.builder.and_i32();
                    }
                },
                &InstructionOperand::Immediate(i) => {
                    dbg_assert!(*opsize != OPSIZE_8 || i >= 0 && i < 0x100);
                    dbg_assert!(*opsize != OPSIZE_16 || i >= 0 && i < 0x10000);
                    ctx.builder.const_i32(i);
                },
            }

            if negate == ConditionNegate::True {
                ctx.builder.gtu_i32();
            }
            else {
                ctx.builder.leu_i32();
            }
        },
        Instruction::Sub {
            opsize,
            source,
            is_dec: false,
            ..
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);

            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            match (opsize, source) {
                (&OPSIZE_32, InstructionOperand::WasmLocal(l)) => ctx.builder.get_local(l),
                (_, &InstructionOperand::Immediate(i)) => ctx.builder.const_i32(i),
                _ => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 0xFF } else { 0xFFFF });
                        ctx.builder.and_i32();
                    }
                },
            }

            if negate == ConditionNegate::True {
                ctx.builder.gtu_i32();
            }
            else {
                ctx.builder.leu_i32();
            }
        },
        &Instruction::Bitwise { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_getzf(ctx, negate);
        },
        &Instruction::Add { .. } | &Instruction::Sub { is_dec: true, .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            // not the best code generation, but reasonable for this fairly uncommon case
            gen_getcf(ctx, ConditionNegate::False);
            gen_getzf(ctx, ConditionNegate::False);
            ctx.builder.or_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Other | Instruction::NonZeroShift { .. } | Instruction::AdcSbb { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            gen_getcf(ctx, ConditionNegate::False);
            gen_getzf(ctx, ConditionNegate::False);
            ctx.builder.or_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_test_l(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp {
            dest,
            source,
            opsize,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            match dest {
                InstructionOperandDest::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                InstructionOperandDest::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            match source {
                InstructionOperand::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                InstructionOperand::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                &InstructionOperand::Immediate(i) => {
                    ctx.builder.const_i32(i);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            if negate == ConditionNegate::True {
                ctx.builder.ge_i32();
            }
            else {
                ctx.builder.lt_i32();
            }
        },
        Instruction::Sub { opsize, source, .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                ctx.builder
                    .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                ctx.builder.shl_i32();
            }
            match (opsize, source) {
                (&OPSIZE_32, InstructionOperand::WasmLocal(l)) => ctx.builder.get_local(l),
                (_, &InstructionOperand::Immediate(i)) => ctx.builder.const_i32(
                    i << if *opsize == OPSIZE_32 {
                        0
                    }
                    else if *opsize == OPSIZE_16 {
                        16
                    }
                    else {
                        24
                    },
                ),
                _ => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            if negate == ConditionNegate::True {
                ctx.builder.ge_i32();
            }
            else {
                ctx.builder.lt_i32();
            }
        },
        &Instruction::Bitwise { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_getsf(ctx, negate);
        },
        &Instruction::Other
        | Instruction::Add { .. }
        | Instruction::NonZeroShift { .. }
        | Instruction::AdcSbb { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            if let Instruction::Add { .. } = ctx.previous_instruction {
                gen_profiler_stat_increment(
                    ctx.builder,
                    profiler::stat::CONDITION_UNOPTIMISED_UNHANDLED_L,
                );
            }
            gen_getsf(ctx, ConditionNegate::False);
            ctx.builder.eqz_i32();
            gen_getof(ctx);
            ctx.builder.eqz_i32();
            ctx.builder.xor_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_test_le(ctx: &mut JitContext, negate: ConditionNegate) {
    match &ctx.previous_instruction {
        Instruction::Cmp {
            dest,
            source,
            opsize,
        } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            match dest {
                InstructionOperandDest::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                InstructionOperandDest::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            match source {
                InstructionOperand::WasmLocal(l) => {
                    ctx.builder.get_local(l);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                InstructionOperand::Other => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
                &InstructionOperand::Immediate(i) => {
                    ctx.builder.const_i32(i);
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            if negate == ConditionNegate::True {
                ctx.builder.gt_i32();
            }
            else {
                ctx.builder.le_i32();
            }
        },
        Instruction::Sub { opsize, source, .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
            if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                ctx.builder
                    .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                ctx.builder.shl_i32();
            }
            match (opsize, source) {
                (&OPSIZE_32, InstructionOperand::WasmLocal(l)) => ctx.builder.get_local(l),
                (_, &InstructionOperand::Immediate(i)) => ctx.builder.const_i32(
                    i << if *opsize == OPSIZE_32 {
                        0
                    }
                    else if *opsize == OPSIZE_16 {
                        16
                    }
                    else {
                        24
                    },
                ),
                _ => {
                    gen_get_last_op1(ctx.builder, &ctx.previous_instruction);
                    gen_get_last_result(ctx.builder, &ctx.previous_instruction);
                    ctx.builder.sub_i32();
                    if *opsize == OPSIZE_8 || *opsize == OPSIZE_16 {
                        ctx.builder
                            .const_i32(if *opsize == OPSIZE_8 { 24 } else { 16 });
                        ctx.builder.shl_i32();
                    }
                },
            }
            if negate == ConditionNegate::True {
                ctx.builder.gt_i32();
            }
            else {
                ctx.builder.le_i32();
            }
        },
        &Instruction::Bitwise { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_OPTIMISED);
            // TODO: Could probably be improved (<= 0)
            gen_test_l(ctx, ConditionNegate::False);
            gen_getzf(ctx, ConditionNegate::False);
            ctx.builder.or_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
        Instruction::Other
        | Instruction::Add { .. }
        | Instruction::NonZeroShift { .. }
        | Instruction::AdcSbb { .. } => {
            gen_profiler_stat_increment(ctx.builder, profiler::stat::CONDITION_UNOPTIMISED);
            if let Instruction::Add { .. } = ctx.previous_instruction {
                gen_profiler_stat_increment(
                    ctx.builder,
                    profiler::stat::CONDITION_UNOPTIMISED_UNHANDLED_LE,
                );
            }
            gen_test_l(ctx, ConditionNegate::False);
            gen_getzf(ctx, ConditionNegate::False);
            ctx.builder.or_i32();
            if negate == ConditionNegate::True {
                ctx.builder.eqz_i32();
            }
        },
    }
}

pub fn gen_test_loopnz(ctx: &mut JitContext, is_asize_32: bool) {
    gen_test_loop(ctx, is_asize_32);
    ctx.builder.eqz_i32();
    gen_getzf(ctx, ConditionNegate::False);
    ctx.builder.or_i32();
    ctx.builder.eqz_i32();
}
pub fn gen_test_loopz(ctx: &mut JitContext, is_asize_32: bool) {
    gen_test_loop(ctx, is_asize_32);
    ctx.builder.eqz_i32();
    gen_getzf(ctx, ConditionNegate::False);
    ctx.builder.eqz_i32();
    ctx.builder.or_i32();
    ctx.builder.eqz_i32();
}
pub fn gen_test_loop(ctx: &mut JitContext, is_asize_32: bool) {
    if is_asize_32 {
        gen_get_reg32(ctx, regs::ECX);
    }
    else {
        gen_get_reg16(ctx, regs::CX);
    }
}
pub fn gen_test_jcxz(ctx: &mut JitContext, is_asize_32: bool) {
    if is_asize_32 {
        gen_get_reg32(ctx, regs::ECX);
    }
    else {
        gen_get_reg16(ctx, regs::CX);
    }
    ctx.builder.eqz_i32();
}

