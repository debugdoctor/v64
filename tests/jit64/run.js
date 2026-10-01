#!/usr/bin/env node

// Test for the 64-bit JIT prototype (src/rust/jit64.rs), targeting wasm32.
//
// Decodes/compiles a small x86-64 program (cmp/inc flags + conditional jump) to a
// wasm32 module, runs it and checks the register file, flags and next RIP.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/jit64/run.js`

import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";

const emulator = new v64({
    autostart: false,
    memory_size: 2 * 1024 * 1024,
    disable_jit: 1,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;

    // CPU state offsets (global_pointers.rs)
    const REG_LOW = 64; // r0-r7 low 32 bits
    const REG_HIGH = 128; // r0-r7 high 32 bits
    const FLAGS = 120;
    const IN_HLT = 616;
    const RIP = 232;

    const ptr = ex.jit64_proto();
    const len = ex.jit64_proto_len();
    const bytes = new Uint8Array(ex.memory.buffer, ptr, len).slice();

    const memory = new WebAssembly.Memory({ initial: 64 });
    const view = new DataView(memory.buffer);

    const module = new WebAssembly.Module(bytes);
    const instance = new WebAssembly.Instance(module, { e: {
        m: memory,
        jit64_sync_flags: ex.jit64_sync_flags,
        jit64_clear_exception_flag: ex.jit64_clear_exception_flag,
        jit64_hlt: () => view.setUint8(IN_HLT, 1),
    } });
    // This test calls the block directly, so do the entry setup here.
    ex.jit64_sync_flags();
    ex.jit64_clear_exception_flag();
    instance.exports.f(0);

    // mov rax,0; cmp rax,1 (CF=1); inc rax preserves CF; jc +2 -> target 21
    const rax_low = view.getUint32(REG_LOW + 0, true);
    const rax_high = view.getUint32(REG_HIGH + 0, true);
    const flags = view.getInt32(FLAGS, true);
    const ip = view.getBigUint64(RIP, true);
    const in_hlt = new Uint8Array(memory.buffer)[IN_HLT];

    assert.equal(rax_low, 1, "rax low dword");
    assert.equal(rax_high, 0, "rax high dword");
    assert.equal(flags & (1 << 11), 0, "OF clear");
    assert.equal(flags & (1 << 4), 0, "AF clear");
    assert.equal(flags & (1 << 2), 0, "PF clear");
    assert.equal(flags & (1 << 7), 0, "SF clear");
    assert.equal(flags & (1 << 6), 0, "ZF clear");
    assert.ok(flags & 1, "CF preserved");
    assert.equal(ip, 21n, "jc taken -> rip");
    assert.equal(in_hlt, 0, "hlt not reached");

    console.log("jit64: test passed");
    process.exit(0);
});
