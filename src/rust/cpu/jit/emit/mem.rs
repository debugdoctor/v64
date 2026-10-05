use super::*;

pub fn gen_readable_or_pagefault(ctx: &mut JitContext, address_local: &WasmLocal, size: i32) {
    ctx.builder.get_local(address_local);
    ctx.builder.const_i32(size);
    ctx.builder
        .const_i32(ctx.start_of_current_instruction as i32 & 0xFFF);
    ctx.builder.call_fn3_ret("readable_or_pagefault_jit");
    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        ctx.builder.br(ctx.exit_with_fault_label);
        ctx.builder.block_end();
    }
    else {
        ctx.builder.br_if(ctx.exit_with_fault_label);
    }
}

pub fn gen_writable_or_pagefault(ctx: &mut JitContext, address_local: &WasmLocal, size: i32) {
    ctx.builder.get_local(address_local);
    ctx.builder.const_i32(size);
    // packed lower bits of eip and wasm table index
    ctx.builder.const_i32(
        ctx.start_of_current_instruction as i32 & 0xFFF
            | (ctx.wasm_table_index.to_u16() as i32) << 16,
    );
    ctx.builder.call_fn3_ret("writable_or_pagefault_jit");
    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        ctx.builder.br(ctx.exit_with_fault_label);
        ctx.builder.block_end();
    }
    else {
        ctx.builder.br_if(ctx.exit_with_fault_label);
    }
}

/// sign-extend a byte value on the stack and leave it on the stack
pub fn sign_extend_i8(builder: &mut WasmBuilder) {
    builder.const_i32(24);
    builder.shl_i32();
    builder.const_i32(24);
    builder.shr_s_i32();
}

/// sign-extend a two byte value on the stack and leave it on the stack
pub fn sign_extend_i16(builder: &mut WasmBuilder) {
    builder.const_i32(16);
    builder.shl_i32();
    builder.const_i32(16);
    builder.shr_s_i32();
}

pub fn gen_fn0_const(builder: &mut WasmBuilder, name: &str) { builder.call_fn0(name) }
pub fn gen_fn1_const(builder: &mut WasmBuilder, name: &str, arg0: u32) {
    builder.const_i32(arg0 as i32);
    builder.call_fn1(name);
}
pub fn gen_fn2_const(builder: &mut WasmBuilder, name: &str, arg0: u32, arg1: u32) {
    builder.const_i32(arg0 as i32);
    builder.const_i32(arg1 as i32);
    builder.call_fn2(name);
}

// helper functions for gen/generate_jit.js
pub fn gen_modrm_fn0(builder: &mut WasmBuilder, name: &str) {
    // generates: fn( _ )
    builder.call_fn1(name);
}
pub fn gen_modrm_fn1(builder: &mut WasmBuilder, name: &str, arg0: u32) {
    // generates: fn( _, arg0 )
    builder.const_i32(arg0 as i32);
    builder.call_fn2(name);
}

pub fn gen_modrm_resolve(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    modrm::gen(ctx, modrm_byte, 0)
}
pub fn gen_modrm_resolve_with_local(
    ctx: &mut JitContext,
    modrm_byte: ModrmByte,
    gen: &dyn Fn(&mut JitContext, &WasmLocal),
) {
    if let Some(r) = modrm::get_as_reg_index_if_possible(ctx, &modrm_byte) {
        gen(ctx, &ctx.reg(r));
    }
    else {
        gen_modrm_resolve(ctx, modrm_byte);
        let address = ctx.builder.set_new_local();
        gen(ctx, &address);
        ctx.builder.free_local(address);
    }
}
pub fn gen_modrm_resolve_with_esp_offset(
    ctx: &mut JitContext,
    modrm_byte: ModrmByte,
    esp_offset: i32,
) {
    modrm::gen(ctx, modrm_byte, esp_offset)
}

pub fn gen_set_reg8_r(ctx: &mut JitContext, dest: u32, src: u32) {
    // generates: reg8[r_dest] = reg8[r_src]
    if src != dest {
        gen_get_reg8(ctx, src);
        gen_set_reg8_unmasked(ctx, dest);
    }
}
pub fn gen_set_reg16_r(ctx: &mut JitContext, dest: u32, src: u32) {
    // generates: reg16[r_dest] = reg16[r_src]
    if src != dest {
        gen_get_reg16(ctx, src);
        gen_set_reg16_unmasked(ctx, dest);
    }
}
pub fn gen_set_reg32_r(ctx: &mut JitContext, dest: u32, src: u32) {
    // generates: reg32[r_dest] = reg32[r_src]
    if src != dest {
        gen_get_reg32(ctx, src);
        gen_set_reg32(ctx, dest);
    }
}

pub fn gen_modrm_resolve_safe_read8(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_modrm_resolve_with_local(ctx, modrm_byte, &|ctx, addr| gen_safe_read8(ctx, addr));
}
pub fn gen_modrm_resolve_safe_read16(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_modrm_resolve_with_local(ctx, modrm_byte, &|ctx, addr| gen_safe_read16(ctx, addr));
}
pub fn gen_modrm_resolve_safe_read32(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_modrm_resolve_with_local(ctx, modrm_byte, &|ctx, addr| gen_safe_read32(ctx, addr));
}
pub fn gen_modrm_resolve_safe_read64(ctx: &mut JitContext, modrm_byte: ModrmByte) {
    gen_modrm_resolve_with_local(ctx, modrm_byte, &|ctx, addr| gen_safe_read64(ctx, addr));
}
pub fn gen_modrm_resolve_safe_read128(
    ctx: &mut JitContext,
    modrm_byte: ModrmByte,
    where_to_write: u32,
) {
    gen_modrm_resolve_with_local(ctx, modrm_byte, &|ctx, addr| {
        gen_safe_read128(ctx, addr, where_to_write)
    });
}

pub fn gen_safe_read8(ctx: &mut JitContext, address_local: &WasmLocal) {
    gen_safe_read(ctx, BitSize::BYTE, address_local, None);
}
pub fn gen_safe_read16(ctx: &mut JitContext, address_local: &WasmLocal) {
    gen_safe_read(ctx, BitSize::WORD, address_local, None);
}
pub fn gen_safe_read32(ctx: &mut JitContext, address_local: &WasmLocal) {
    gen_safe_read(ctx, BitSize::DWORD, address_local, None);
}
pub fn gen_safe_read64(ctx: &mut JitContext, address_local: &WasmLocal) {
    gen_safe_read(ctx, BitSize::QWORD, &address_local, None);
}
pub fn gen_safe_read128(ctx: &mut JitContext, address_local: &WasmLocal, where_to_write: u32) {
    gen_safe_read(ctx, BitSize::DQWORD, &address_local, Some(where_to_write));
}

// only used internally for gen_safe_write
enum GenSafeWriteValue<'a> {
    I32(&'a WasmLocal),
    I64(&'a WasmLocalI64),
    TwoI64s(&'a WasmLocalI64, &'a WasmLocalI64),
}

enum GenSafeReadWriteValue {
    I32(WasmLocal),
    I64(WasmLocalI64),
}

#[derive(Copy, Clone, Eq, PartialEq)]
pub enum BitSize {
    BYTE,
    WORD,
    DWORD,
    QWORD,
    DQWORD,
}
impl BitSize {
    pub fn bytes(&self) -> u32 {
        match self {
            BitSize::BYTE => 1,
            BitSize::WORD => 2,
            BitSize::DWORD => 4,
            BitSize::QWORD => 8,
            BitSize::DQWORD => 16,
        }
    }
}

pub fn gen_safe_write8(ctx: &mut JitContext, address_local: &WasmLocal, value_local: &WasmLocal) {
    gen_safe_write(
        ctx,
        BitSize::BYTE,
        address_local,
        GenSafeWriteValue::I32(value_local),
    )
}
pub fn gen_safe_write16(ctx: &mut JitContext, address_local: &WasmLocal, value_local: &WasmLocal) {
    gen_safe_write(
        ctx,
        BitSize::WORD,
        address_local,
        GenSafeWriteValue::I32(value_local),
    )
}
pub fn gen_safe_write32(ctx: &mut JitContext, address_local: &WasmLocal, value_local: &WasmLocal) {
    gen_safe_write(
        ctx,
        BitSize::DWORD,
        address_local,
        GenSafeWriteValue::I32(value_local),
    )
}
pub fn gen_safe_write64(
    ctx: &mut JitContext,
    address_local: &WasmLocal,
    value_local: &WasmLocalI64,
) {
    gen_safe_write(
        ctx,
        BitSize::QWORD,
        address_local,
        GenSafeWriteValue::I64(value_local),
    )
}

pub fn gen_safe_write128(
    ctx: &mut JitContext,
    address_local: &WasmLocal,
    value_local_low: &WasmLocalI64,
    value_local_high: &WasmLocalI64,
) {
    gen_safe_write(
        ctx,
        BitSize::DQWORD,
        address_local,
        GenSafeWriteValue::TwoI64s(value_local_low, value_local_high),
    )
}

fn gen_safe_read(
    ctx: &mut JitContext,
    bits: BitSize,
    address_local: &WasmLocal,
    where_to_write: Option<u32>,
) {
    // Execute a virtual memory read. All slow paths (memory-mapped IO, tlb miss, page fault and
    // read across page boundary are handled in safe_read_jit_slow

    //   entry <- tlb_data[addr >> 12 << 2]
    //   if entry & MASK == TLB_VALID && (addr & 0xFFF) <= 0x1000 - bytes: goto fast
    //   entry <- safe_read_jit_slow(addr, instruction_pointer)
    //   if page_fault: goto exit-with-pagefault
    //   fast: mem[(entry & ~0xFFF) ^ addr]

    let cont = ctx.builder.block_void();
    ctx.builder.get_local(&address_local);

    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    ctx.builder.const_i32(2);
    ctx.builder.shl_i32();

    ctx.builder
        .load_aligned_i32(unsafe { &tlb_data[0] as *const i32 as u32 });
    let entry_local = ctx.builder.tee_new_local();

    ctx.builder.const_i32(
        (0xFFF
            & !TLB_READONLY
            & !TLB_GLOBAL
            & !TLB_HAS_CODE
            & !(if ctx.cpu.cpl3() { 0 } else { TLB_NO_USER })) as i32,
    );
    ctx.builder.and_i32();

    ctx.builder.const_i32(TLB_VALID as i32);
    ctx.builder.eq_i32();

    if bits != BitSize::BYTE {
        ctx.builder.get_local(&address_local);
        ctx.builder.const_i32(0xFFF);
        ctx.builder.and_i32();
        ctx.builder.const_i32(0x1000 - bits.bytes() as i32);
        ctx.builder.le_i32();

        ctx.builder.and_i32();
    }

    ctx.builder.br_if(cont);

    if cfg!(feature = "profiler") {
        ctx.builder.get_local(&address_local);
        ctx.builder.get_local(&entry_local);
        ctx.builder.call_fn2("report_safe_read_jit_slow");
    }

    ctx.builder.get_local(&address_local);
    ctx.builder
        .const_i32(ctx.start_of_current_instruction as i32 & 0xFFF);
    match bits {
        BitSize::BYTE => {
            ctx.builder.call_fn2_ret("safe_read8_slow_jit");
        },
        BitSize::WORD => {
            ctx.builder.call_fn2_ret("safe_read16_slow_jit");
        },
        BitSize::DWORD => {
            ctx.builder.call_fn2_ret("safe_read32s_slow_jit");
        },
        BitSize::QWORD => {
            ctx.builder.call_fn2_ret("safe_read64s_slow_jit");
        },
        BitSize::DQWORD => {
            ctx.builder.call_fn2_ret("safe_read128s_slow_jit");
        },
    }
    ctx.builder.tee_local(&entry_local);
    ctx.builder.const_i32(1);
    ctx.builder.and_i32();

    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        ctx.builder.block_end();

        ctx.builder.get_local(&entry_local);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
    }

    ctx.builder.br_if(ctx.exit_with_fault_label);

    ctx.builder.block_end();

    gen_profiler_stat_increment(ctx.builder, profiler::stat::SAFE_READ_FAST); // XXX: Both fast and slow

    ctx.builder.get_local(&entry_local);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.get_local(&address_local);
    ctx.builder.xor_i32();

    // where_to_write is only used by dqword
    dbg_assert!((where_to_write != None) == (bits == BitSize::DQWORD));

    match bits {
        BitSize::BYTE => {
            ctx.builder.load_u8(0);
        },
        BitSize::WORD => {
            ctx.builder.load_unaligned_u16(0);
        },
        BitSize::DWORD => {
            ctx.builder.load_unaligned_i32(0);
        },
        BitSize::QWORD => {
            ctx.builder.load_unaligned_i64(0);
        },
        BitSize::DQWORD => {
            let where_to_write = where_to_write.unwrap();
            let virt_address_local = ctx.builder.set_new_local();
            ctx.builder.const_i32(0);
            ctx.builder.get_local(&virt_address_local);
            ctx.builder.load_unaligned_i64(0);
            ctx.builder.store_unaligned_i64(where_to_write);

            ctx.builder.const_i32(0);
            ctx.builder.get_local(&virt_address_local);
            ctx.builder.load_unaligned_i64(8);
            ctx.builder.store_unaligned_i64(where_to_write + 8);

            ctx.builder.free_local(virt_address_local);
        },
    }

    ctx.builder.free_local(entry_local);
}

pub fn gen_get_phys_eip_plus_mem(ctx: &mut JitContext, address_local: &WasmLocal) {
    // Similar to gen_safe_read, but return the physical eip + memory::mem rather than reading from memory
    // In functions that need to use this value we need to fix it by substracting memory::mem
    // this is done in order to remove one instruction from the fast path of memory accesses (no need to add
    // memory::mem anymore ).
    // We need to account for this in gen_page_switch_check and we compare with next_block_addr + memory::mem8
    // We cannot the same while processing an AbsoluteEip flow control change so there we need to fix the value
    // by subscracting memory::mem. Overall, since AbsoluteEip is encountered less often than memory accesses so
    // this ends up improving perf.
    // Does not (need to) handle mapped memory
    // XXX: Currently does not use ctx.start_of_current_instruction, but rather assumes that eip is
    //      already correct (pointing at the current instruction)

    let cont = ctx.builder.block_void();
    ctx.builder.get_local(&address_local);

    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    ctx.builder.const_i32(2);
    ctx.builder.shl_i32();

    ctx.builder
        .load_aligned_i32(unsafe { &tlb_data[0] as *const i32 as u32 });
    let entry_local = ctx.builder.tee_new_local();

    ctx.builder.const_i32(
        (0xFFF
            & !TLB_READONLY
            & !TLB_GLOBAL
            & !TLB_HAS_CODE
            & !(if ctx.cpu.cpl3() { 0 } else { TLB_NO_USER })) as i32,
    );
    ctx.builder.and_i32();

    ctx.builder.const_i32(TLB_VALID as i32);
    ctx.builder.eq_i32();

    ctx.builder.br_if(cont);

    if cfg!(feature = "profiler") {
        ctx.builder.get_local(&address_local);
        ctx.builder.get_local(&entry_local);
        ctx.builder.call_fn2("report_safe_read_jit_slow");
    }

    ctx.builder.get_local(&address_local);
    ctx.builder.call_fn1_ret("get_phys_eip_slow_jit");

    ctx.builder.tee_local(&entry_local);
    ctx.builder.const_i32(1);
    ctx.builder.and_i32();

    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction); // XXX
        ctx.builder.block_end();

        ctx.builder.get_local(&entry_local);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
    }

    ctx.builder.br_if(ctx.exit_with_fault_label);

    ctx.builder.block_end();

    gen_profiler_stat_increment(ctx.builder, profiler::stat::SAFE_READ_FAST); // XXX: Both fast and slow

    ctx.builder.get_local(&entry_local);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.get_local(&address_local);
    ctx.builder.xor_i32();

    ctx.builder.free_local(entry_local);
}

fn gen_safe_write(
    ctx: &mut JitContext,
    bits: BitSize,
    address_local: &WasmLocal,
    value_local: GenSafeWriteValue,
) {
    // Execute a virtual memory write. All slow paths (memory-mapped IO, tlb miss, page fault,
    // write across page boundary and page containing jitted code are handled in safe_write_jit_slow

    //   entry <- tlb_data[addr >> 12 << 2]
    //   if entry & MASK == TLB_VALID && (addr & 0xFFF) <= 0x1000 - bytes: goto fast
    //   entry <- safe_write_jit_slow(addr, value, instruction_pointer)
    //   if page_fault: goto exit-with-pagefault
    //   fast: mem[(entry & ~0xFFF) ^ addr] <- value

    let cont = ctx.builder.block_void();
    ctx.builder.get_local(&address_local);

    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    ctx.builder.const_i32(2);
    ctx.builder.shl_i32();

    ctx.builder
        .load_aligned_i32(unsafe { &tlb_data[0] as *const i32 as u32 });
    let entry_local = ctx.builder.tee_new_local();

    ctx.builder
        .const_i32((0xFFF & !TLB_GLOBAL & !(if ctx.cpu.cpl3() { 0 } else { TLB_NO_USER })) as i32);
    ctx.builder.and_i32();

    ctx.builder.const_i32(TLB_VALID as i32);
    ctx.builder.eq_i32();

    if bits != BitSize::BYTE {
        ctx.builder.get_local(&address_local);
        ctx.builder.const_i32(0xFFF);
        ctx.builder.and_i32();
        ctx.builder.const_i32(0x1000 - bits.bytes() as i32);
        ctx.builder.le_i32();

        ctx.builder.and_i32();
    }

    ctx.builder.br_if(cont);

    if cfg!(feature = "profiler") {
        ctx.builder.get_local(&address_local);
        ctx.builder.get_local(&entry_local);
        ctx.builder.call_fn2("report_safe_write_jit_slow");
    }

    ctx.builder.get_local(&address_local);
    match value_local {
        GenSafeWriteValue::I32(local) => ctx.builder.get_local(local),
        GenSafeWriteValue::I64(local) => ctx.builder.get_local_i64(local),
        GenSafeWriteValue::TwoI64s(local1, local2) => {
            ctx.builder.get_local_i64(local1);
            ctx.builder.get_local_i64(local2)
        },
    }

    // packed lower bits of eip and wasm table index
    ctx.builder.const_i32(
        ctx.start_of_current_instruction as i32 & 0xFFF
            | (ctx.wasm_table_index.to_u16() as i32) << 16,
    );

    match bits {
        BitSize::BYTE => {
            ctx.builder.call_fn3_ret("safe_write8_slow_jit");
        },
        BitSize::WORD => {
            ctx.builder.call_fn3_ret("safe_write16_slow_jit");
        },
        BitSize::DWORD => {
            ctx.builder.call_fn3_ret("safe_write32_slow_jit");
        },
        BitSize::QWORD => {
            ctx.builder
                .call_fn3_i32_i64_i32_ret("safe_write64_slow_jit");
        },
        BitSize::DQWORD => {
            ctx.builder
                .call_fn4_i32_i64_i64_i32_ret("safe_write128_slow_jit");
        },
    }
    ctx.builder.tee_local(&entry_local);
    ctx.builder.const_i32(1);
    ctx.builder.and_i32();

    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        ctx.builder.block_end();

        ctx.builder.get_local(&entry_local);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
    }

    ctx.builder.br_if(ctx.exit_with_fault_label);

    ctx.builder.block_end();

    gen_profiler_stat_increment(ctx.builder, profiler::stat::SAFE_WRITE_FAST); // XXX: Both fast and slow

    ctx.builder.get_local(&entry_local);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.get_local(&address_local);
    ctx.builder.xor_i32();

    match value_local {
        GenSafeWriteValue::I32(local) => ctx.builder.get_local(local),
        GenSafeWriteValue::I64(local) => ctx.builder.get_local_i64(local),
        GenSafeWriteValue::TwoI64s(local1, local2) => {
            assert!(bits == BitSize::DQWORD);

            let virt_address_local = ctx.builder.tee_new_local();
            ctx.builder.get_local_i64(local1);
            ctx.builder.store_unaligned_i64(0);

            ctx.builder.get_local(&virt_address_local);
            ctx.builder.get_local_i64(local2);
            ctx.builder.store_unaligned_i64(8);
            ctx.builder.free_local(virt_address_local);
        },
    }
    match bits {
        BitSize::BYTE => {
            ctx.builder.store_u8(0);
        },
        BitSize::WORD => {
            ctx.builder.store_unaligned_u16(0);
        },
        BitSize::DWORD => {
            ctx.builder.store_unaligned_i32(0);
        },
        BitSize::QWORD => {
            ctx.builder.store_unaligned_i64(0);
        },
        BitSize::DQWORD => {}, // handled above
    }

    ctx.builder.free_local(entry_local);
}

pub fn gen_safe_read_write(
    ctx: &mut JitContext,
    bits: BitSize,
    address_local: &WasmLocal,
    f: &dyn Fn(&mut JitContext),
) {
    // Execute a virtual memory read+write. All slow paths (memory-mapped IO, tlb miss, page fault,
    // write across page boundary and page containing jitted code are handled in
    // safe_read_write_jit_slow

    //   entry <- tlb_data[addr >> 12 << 2]
    //   can_use_fast_path <- entry & MASK == TLB_VALID && (addr & 0xFFF) <= 0x1000 - bytes
    //   if can_use_fast_path: goto fast
    //   entry <- safe_read_write_jit_slow(addr, instruction_pointer)
    //   if page_fault: goto exit-with-pagefault
    //   fast: value <- f(mem[(entry & ~0xFFF) ^ addr])
    //   if !can_use_fast_path { safe_write_jit_slow(addr, value, instruction_pointer) }
    //   mem[(entry & ~0xFFF) ^ addr] <- value

    let cont = ctx.builder.block_void();
    ctx.builder.get_local(address_local);

    ctx.builder.const_i32(12);
    ctx.builder.shr_u_i32();
    ctx.builder.const_i32(2);
    ctx.builder.shl_i32();

    ctx.builder
        .load_aligned_i32(unsafe { &tlb_data[0] as *const i32 as u32 });
    let entry_local = ctx.builder.tee_new_local();

    ctx.builder
        .const_i32((0xFFF & !TLB_GLOBAL & !(if ctx.cpu.cpl3() { 0 } else { TLB_NO_USER })) as i32);
    ctx.builder.and_i32();

    ctx.builder.const_i32(TLB_VALID as i32);
    ctx.builder.eq_i32();

    if bits != BitSize::BYTE {
        ctx.builder.get_local(&address_local);
        ctx.builder.const_i32(0xFFF);
        ctx.builder.and_i32();
        ctx.builder.const_i32(0x1000 - bits.bytes() as i32);
        ctx.builder.le_i32();
        ctx.builder.and_i32();
    }

    let can_use_fast_path_local = ctx.builder.tee_new_local();

    ctx.builder.br_if(cont);

    if cfg!(feature = "profiler") {
        ctx.builder.get_local(&address_local);
        ctx.builder.get_local(&entry_local);
        ctx.builder.call_fn2("report_safe_read_write_jit_slow");
    }

    ctx.builder.get_local(&address_local);

    // packed lower bits of eip and wasm table index
    ctx.builder.const_i32(
        ctx.start_of_current_instruction as i32 & 0xFFF
            | (ctx.wasm_table_index.to_u16() as i32) << 16,
    );

    match bits {
        BitSize::BYTE => {
            ctx.builder.call_fn2_ret("safe_read_write8_slow_jit");
        },
        BitSize::WORD => {
            ctx.builder.call_fn2_ret("safe_read_write16_slow_jit");
        },
        BitSize::DWORD => {
            ctx.builder.call_fn2_ret("safe_read_write32s_slow_jit");
        },
        BitSize::QWORD => {
            ctx.builder.call_fn2_ret("safe_read_write64_slow_jit");
        },
        BitSize::DQWORD => {
            dbg_assert!(false);
        },
    }
    ctx.builder.tee_local(&entry_local);
    ctx.builder.const_i32(1);
    ctx.builder.and_i32();

    if cfg!(feature = "profiler") {
        ctx.builder.if_void();
        gen_debug_track_jit_exit(ctx.builder, ctx.start_of_current_instruction);
        ctx.builder.block_end();

        ctx.builder.get_local(&entry_local);
        ctx.builder.const_i32(1);
        ctx.builder.and_i32();
    }

    ctx.builder.br_if(ctx.exit_with_fault_label);

    ctx.builder.block_end();

    gen_profiler_stat_increment(ctx.builder, profiler::stat::SAFE_READ_WRITE_FAST); // XXX: Also slow

    ctx.builder.get_local(&entry_local);
    ctx.builder.const_i32(!0xFFF);
    ctx.builder.and_i32();
    ctx.builder.get_local(&address_local);
    ctx.builder.xor_i32();

    ctx.builder.free_local(entry_local);
    let phys_addr_local = ctx.builder.tee_new_local();

    match bits {
        BitSize::BYTE => {
            ctx.builder.load_u8(0);
        },
        BitSize::WORD => {
            ctx.builder.load_unaligned_u16(0);
        },
        BitSize::DWORD => {
            ctx.builder.load_unaligned_i32(0);
        },
        BitSize::QWORD => {
            ctx.builder.load_unaligned_i64(0);
        },
        BitSize::DQWORD => assert!(false), // not used
    }

    // value is now on stack

    f(ctx);

    // TODO: Could get rid of this local by returning one from f
    let value_local = if bits == BitSize::QWORD {
        GenSafeReadWriteValue::I64(ctx.builder.set_new_local_i64())
    }
    else {
        GenSafeReadWriteValue::I32(ctx.builder.set_new_local())
    };

    ctx.builder.get_local(&can_use_fast_path_local);

    ctx.builder.eqz_i32();
    ctx.builder.if_void();
    {
        ctx.builder.get_local(&address_local);

        match &value_local {
            GenSafeReadWriteValue::I32(l) => ctx.builder.get_local(l),
            GenSafeReadWriteValue::I64(l) => ctx.builder.get_local_i64(l),
        }

        // packed lower bits of eip and wasm table index
        ctx.builder.const_i32(
            ctx.start_of_current_instruction as i32 & 0xFFF
                | (ctx.wasm_table_index.to_u16() as i32) << 16,
        );

        match bits {
            BitSize::BYTE => {
                ctx.builder.call_fn3_ret("safe_write8_slow_jit");
            },
            BitSize::WORD => {
                ctx.builder.call_fn3_ret("safe_write16_slow_jit");
            },
            BitSize::DWORD => {
                ctx.builder.call_fn3_ret("safe_write32_slow_jit");
            },
            BitSize::QWORD => {
                ctx.builder
                    .call_fn3_i32_i64_i32_ret("safe_write64_slow_jit");
            },
            BitSize::DQWORD => {
                dbg_assert!(false);
            },
        }

        if cfg!(debug_assertions) {
            ctx.builder.const_i32(1);
            ctx.builder.and_i32();

            ctx.builder.if_void();
            {
                // handled above
                ctx.builder.const_i32(match bits {
                    BitSize::BYTE => 8,
                    BitSize::WORD => 16,
                    BitSize::DWORD => 32,
                    BitSize::QWORD => 64,
                    _ => {
                        dbg_assert!(false);
                        0
                    },
                });
                ctx.builder.get_local(&address_local);
                ctx.builder.call_fn2("bug_gen_safe_read_write_page_fault");
            }
            ctx.builder.block_end();
        }
        else {
            ctx.builder.drop_();
        }
    }
    ctx.builder.block_end();

    ctx.builder.get_local(&phys_addr_local);
    match &value_local {
        GenSafeReadWriteValue::I32(l) => ctx.builder.get_local(l),
        GenSafeReadWriteValue::I64(l) => ctx.builder.get_local_i64(l),
    }

    match bits {
        BitSize::BYTE => {
            ctx.builder.store_u8(0);
        },
        BitSize::WORD => {
            ctx.builder.store_unaligned_u16(0);
        },
        BitSize::DWORD => {
            ctx.builder.store_unaligned_i32(0);
        },
        BitSize::QWORD => {
            ctx.builder.store_unaligned_i64(0);
        },
        BitSize::DQWORD => {
            dbg_assert!(false);
        },
    }

    match value_local {
        GenSafeReadWriteValue::I32(l) => ctx.builder.free_local(l),
        GenSafeReadWriteValue::I64(l) => ctx.builder.free_local_i64(l),
    }
    ctx.builder.free_local(can_use_fast_path_local);
    ctx.builder.free_local(phys_addr_local);
}

#[cfg(debug_assertions)]
#[no_mangle]
pub fn bug_gen_safe_read_write_page_fault(bits: i32, addr: u32) {
    dbg_log!("bug: gen_safe_read_write_page_fault {} {:x}", bits, addr);
    dbg_assert!(false);
}

