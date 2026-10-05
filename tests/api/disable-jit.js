#!/usr/bin/env node

// `JIT_DISABLE` is a single bitmask of JITs to turn off, so one value can
// disable several at once: bit 0 = the 32-bit JIT (jit.rs), bit 1 = jit64.
// `disable_jit: true` turns off every JIT; a number selects individual bits.
import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";
import { get_cpu_config, set_cpu_config, JIT_DISABLE_32, JIT_DISABLE_64 } from "../../src/config.js";

const make = disable_jit =>
{
    const emulator = new v64({
        autostart: false,
        memory_size: 1024 * 1024,
        log_level: 0,
        disable_jit,
        wasm_path: process.env.WASM_PATH || undefined,
    });
    return new Promise(resolve =>
    {
        emulator.add_listener("emulator-loaded", () =>
        {
            resolve({ emulator, ex: emulator.v86.cpu.wm.exports });
        });
    });
};

const check = (ex, want, name) =>
{
    assert.equal(get_cpu_config(ex, "JIT_DISABLE"), want, name + ": JIT_DISABLE mask");
};

// Default: both JITs on.
{
    const { emulator, ex } = await make(undefined);
    check(ex, 0, "default");
    emulator.destroy();
}

// `true` disables every JIT (both bits).
{
    const { emulator, ex } = await make(true);
    check(ex, JIT_DISABLE_32 | JIT_DISABLE_64, "disable_jit true");
    emulator.destroy();
}

// A number picks individual JITs.
{
    const { emulator, ex } = await make(JIT_DISABLE_32);
    check(ex, JIT_DISABLE_32, "disable_jit 32-bit only");
    emulator.destroy();
}
{
    const { emulator, ex } = await make(JIT_DISABLE_64);
    check(ex, JIT_DISABLE_64, "disable_jit jit64 only");
    emulator.destroy();
}

// Writing the mask at runtime round-trips through get.
{
    const { emulator, ex } = await make(undefined);
    set_cpu_config(ex, "JIT_DISABLE", JIT_DISABLE_32 | JIT_DISABLE_64);
    check(ex, JIT_DISABLE_32 | JIT_DISABLE_64, "runtime all");
    set_cpu_config(ex, "JIT_DISABLE", JIT_DISABLE_32);
    check(ex, JIT_DISABLE_32, "runtime 32-bit only");
    emulator.destroy();
}

console.log("api disable_jit: test passed");
