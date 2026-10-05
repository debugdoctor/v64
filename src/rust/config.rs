pub const LOG_PAGE_FAULTS: bool = false;

pub const VMWARE_HYPERVISOR_PORT: bool = true;

// BEGIN GENERATED config keys (config/cpu_config.json) -- do not edit
// @generated from config/cpu_config.json by tools/gen_cpu_config.js -- do not edit
// JIT master switch (jit.rs + jit64.rs)
pub const JIT_DISABLE: u32 = 0;

// 32-bit JIT (jit.rs)
pub const MAX_PAGES: u32 = 1;
pub const LOOP_SAFETY: u32 = 2;
pub const EXTRA_BASIC_BLOCKS: u32 = 3;

// Long-mode JIT (jit64.rs)
pub const JIT64_INLINE_MEMORY: u32 = 17;
pub const JIT64_INLINE_WRITE: u32 = 18;
pub const JIT64_SSE: u32 = 19;
pub const JIT64_PARTIAL_BLOCKS: u32 = 20;
pub const JIT64_ENTRY_CACHE: u32 = 21;
pub const JIT64_HOT_CACHE: u32 = 22;
pub const JIT64_DIRECT_CODE_READ: u32 = 23;
pub const JIT64_MAX_BLOCKS: u32 = 24;
pub const JIT64_CHAIN_BUDGET: u32 = 25;
pub const JIT64_BLOCK_PROLOGUE: u32 = 26;
pub const JIT64_SUPERBLOCKS: u32 = 27;
pub const JIT64_BLOCK_LIMIT: u32 = 28;
pub const JIT64_SMC_BAIL: u32 = 29;
pub const JIT64_SELFCHECK: u32 = 30;
pub const JIT64_SELFCHECK_MIN: u32 = 31;
pub const JIT64_SELFCHECK_REPEAT: u32 = 32;
pub const JIT64_EXIT_STATS: u32 = 33;

// Long-mode interpreter (interp64.rs)
pub const INTERP64_BATCH: u32 = 48;
pub const INTERP64_MEM_SINGLE: u32 = 50;
pub const INTERP64_DECODE_CACHE: u32 = 51;
pub const INTERP64_OPCODE_STATS: u32 = 52;

// 32-bit interpreter (interpreter.rs)
pub const INTERP32_BATCH: u32 = 49;
// END GENERATED config keys

// One switchboard for the JITs and the long-mode interpreter: every tunable
// goes through `set_cpu_config`/`get_cpu_config`. Each subsystem answers for
// its own key range.
//
// The 64-bit BAIL mask and address ranges carry u64s, so they stay as
// dedicated exports in jit64.rs.

// A single bitmask for every JIT: a set bit turns that JIT off. It is the one
// source of truth for JIT enablement, shared by the 32-bit JIT and jit64, so a
// single write can disable several JITs at once.
pub const JIT_DISABLE_32: u32 = 1 << 0; // 32-bit JIT (jit.rs)
pub const JIT_DISABLE_64: u32 = 1 << 1; // long-mode JIT (jit64.rs)

pub static mut JIT_DISABLE_MASK: u32 = 0;

#[inline]
pub unsafe fn jit_disabled(bit: u32) -> bool {
    JIT_DISABLE_MASK & bit != 0
}

#[no_mangle]
pub unsafe fn set_cpu_config(index: u32, value: u32) {
    if index == crate::config::JIT_DISABLE {
        let was = JIT_DISABLE_MASK;
        JIT_DISABLE_MASK = value & (JIT_DISABLE_32 | JIT_DISABLE_64);
        // The decode caches validate against per-page write counters, which the
        // JITs' inline stores bypass. Drop them when a JIT turns back on so no
        // stale block can survive a JIT-enabled period.
        if was & JIT_DISABLE_64 != 0 && JIT_DISABLE_MASK & JIT_DISABLE_64 == 0 {
            crate::cpu::interp::interp64::clear_decode_cache();
        }
        if was & JIT_DISABLE_32 != 0 && JIT_DISABLE_MASK & JIT_DISABLE_32 == 0 {
            crate::cpu::interp::interp32_core::clear_decode_cache();
        }
        return;
    }

    if index == crate::config::INTERP32_BATCH {
        crate::cpu::core::interp32_set_batch(value);
        return;
    }

    if index == crate::config::INTERP64_DECODE_CACHE {
        crate::cpu::interp::interp64::interp64_set_decode_cache(value);
        return;
    }

    if crate::cpu::jit::set_config(index, value)
        || crate::cpu::jit::jit64::set_config(index, value)
        || crate::cpu::interp::interp64::set_config(index, value)
    {
        return;
    }

    dbg_assert!(false, "set_cpu_config: unknown key {}", index);
}

#[no_mangle]
pub unsafe fn get_cpu_config(index: u32) -> u32 {
    if index == crate::config::JIT_DISABLE {
        return JIT_DISABLE_MASK;
    }

    if index == crate::config::INTERP32_BATCH {
        return crate::cpu::core::INTERP32_BATCH as u32;
    }

    crate::cpu::jit::get_config(index)
        .or_else(|| crate::cpu::jit::jit64::get_config(index))
        .or_else(|| crate::cpu::interp::interp64::get_config(index))
        .unwrap_or_else(|| {
            dbg_assert!(false, "get_cpu_config: unknown key {}", index);
            0
        })
}
