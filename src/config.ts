// CPU (JIT/interpreter) option keys for the wasm switchboard. The table below
// is generated from config/cpu_config.json.

// BEGIN GENERATED config keys (config/cpu_config.json) -- do not edit
// @generated from config/cpu_config.json by tools/gen_cpu_config.js -- do not edit
export const CPU_CONFIG = {
    // JIT master switch (jit.rs + jit64.rs)
    "JIT_DISABLE": 0,

    // 32-bit JIT (jit.rs)
    "MAX_PAGES": 1,
    "LOOP_SAFETY": 2,
    "EXTRA_BASIC_BLOCKS": 3,

    // Long-mode JIT (jit64.rs)
    "JIT64_INLINE_MEMORY": 17,
    "JIT64_INLINE_WRITE": 18,
    "JIT64_SSE": 19,
    "JIT64_PARTIAL_BLOCKS": 20,
    "JIT64_ENTRY_CACHE": 21,
    "JIT64_HOT_CACHE": 22,
    "JIT64_DIRECT_CODE_READ": 23,
    "JIT64_MAX_BLOCKS": 24,
    "JIT64_CHAIN_BUDGET": 25,
    "JIT64_BLOCK_PROLOGUE": 26,
    "JIT64_SUPERBLOCKS": 27,
    "JIT64_BLOCK_LIMIT": 28,
    "JIT64_SMC_BAIL": 29,
    "JIT64_SELFCHECK": 30,
    "JIT64_SELFCHECK_MIN": 31,
    "JIT64_SELFCHECK_REPEAT": 32,
    "JIT64_EXIT_STATS": 33,

    // Long-mode interpreter (interp64.rs)
    "INTERP64_BATCH": 48,
    "INTERP64_MEM_SINGLE": 50,
    "INTERP64_DECODE_CACHE": 51,
    "INTERP64_OPCODE_STATS": 52,

    // 32-bit interpreter (interpreter.rs)
    "INTERP32_BATCH": 49,

};
// END GENERATED config keys

// Bits in the `JIT_DISABLE` mask: a set bit turns that JIT off.
export const JIT_DISABLE_32 = 1 << 0; // 32-bit JIT (jit.rs)
export const JIT_DISABLE_64 = 1 << 1; // long-mode JIT (jit64.rs)

function key(name)
{
    const index = CPU_CONFIG[name];
    if(index === undefined)
    {
        throw new Error("unknown cpu config key: " + name);
    }
    return index;
}

// Look the wasm export up through a runtime value: Closure renames dot accesses
// and string literals on the exports object.
function wasm_export(exports, name)
{
    return exports[name];
}

export function set_cpu_config(exports, name, value)
{
    wasm_export(exports, "set_cpu_config")(key(name), value);
}

export function get_cpu_config(exports, name)
{
    return wasm_export(exports, "get_cpu_config")(key(name));
}
