#!/usr/bin/env node

// Entry-cache stress: more blocks than the JIT cap, forcing compile/evict/
// recompile churn while the direct-mapped cache is live. JIT and interpreter
// must agree on the final register state.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/jit64/entry-cache.js`

import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";
import { set_cpu_config, JIT_DISABLE_64 } from "../../src/config.js";

const BLOCKS = +process.env.JIT64_ENTRY_BLOCKS || 4000;
const PASSES = +process.env.JIT64_ENTRY_PASSES || 700;
const CODE = 0x100000;
const STRIDE = 16;

const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const PT = 0x13000;

const emulator = new v64({
    autostart: false,
    memory_size: 64 * 1024 * 1024,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    if(process.env.DIRECT_CODE_READ === "0") set_cpu_config(ex, "JIT64_DIRECT_CODE_READ", 0);
    // Force eviction with a small cap so the block cache actually trims.
    set_cpu_config(ex, "JIT64_MAX_BLOCKS", 1000);
    const write64 = (address, value) => {
        ex.write32(address, Number(value & 0xFFFFFFFFn));
        ex.write32(address + 4, Number(value >> 32n));
    };

    // Identity-map the first 1 GiB.
    write64(PML4, BigInt(PDPT) | 3n);
    write64(PDPT, BigInt(PD) | 3n);
    write64(PD, BigInt(PT) | 3n);
    for(let i = 0; i < 512; i++) write64(PT + i * 8, BigInt(i) * 0x1000n | 3n);
    for(let i = 1; i < 512; i++) write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);

    // Each block: add rax, imm; jmp next. The last decrements r11d and loops
    // until the pass counter is exhausted, then falls through to hlt.
    const code = new Uint8Array(BLOCKS * STRIDE);
    for(let i = 0; i < BLOCKS; i++)
    {
        const at = i * STRIDE;
        const imm = (i % 7) + 1;
        code.set([0x48, 0x05, imm, 0, 0, 0], at); // add rax, imm32
        if(i < BLOCKS - 1)
        {
            const rel = STRIDE - 11; // next block start - end of this jmp
            code.set([0xE9, rel, 0, 0, 0], at + 6); // jmp next
        }
        else
        {
            code.set([0x41, 0xFF, 0xCB], at + 6); // dec r11d
            const after = at + 15;
            const rel = -(after); // back to block 0
            code.set([0x0F, 0x85,
                rel & 0xFF, rel >> 8 & 0xFF, rel >> 16 & 0xFF, rel >> 24 & 0xFF], at + 9);
            code[at + 15] = 0xF4; // hlt on fallthrough
        }
    }
    cpu.mem8.set(code, CODE);

    const expected = PASSES * (() => {
        let sum = 0;
        for(let i = 0; i < BLOCKS; i++) sum += (i % 7) + 1;
        return sum;
    })();

    const run = jitEnabled => {
        set_cpu_config(ex, "JIT_DISABLE", jitEnabled ? 0 : JIT_DISABLE_64);
        ex.jit64_clear_cache();
        const view = new DataView(cpu.wasm_memory.buffer);
        // REG_LOW = 64 (low 32 bits), REG_HIGH = 128, REG_EXT = 160 (r8-r15).
        view.setUint32(64, 0, true); // RAX low
        view.setUint32(128, 0, true); // RAX high
        view.setUint32(80, 0x9000, true); // RSP low
        view.setUint32(144, 0, true); // RSP high
        cpu.cpl[0] = 0;
        cpu.flags[0] = 2;
        cpu.in_hlt[0] = 0;
        cpu.instruction_pointer[0] = CODE;
        ex.enter_long_mode(PML4);
        view.setBigUint64(184, BigInt(PASSES), true); // r11 = passes (REG_EXT + 24)
        view.setBigUint64(1064, 0x12345678n, true); // CR2 sentinel

        const started = process.hrtime.bigint();
        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 100000000) ex.main_loop();
        const ms = Number(process.hrtime.bigint() - started) / 1e6;
        assert.equal(cpu.in_hlt[0], 1, "program halted");
        // The wasm memory may have grown during the run; re-read its buffer.
        const out = new DataView(cpu.wasm_memory.buffer);
        assert.equal(out.getBigUint64(1064, true), 0x12345678n, "no guest fault");
        const rax = BigInt(out.getUint32(64, true)) | BigInt(out.getUint32(128, true)) << 32n;
        return {
            rax,
            r11: out.getBigUint64(184, true),
            rip: out.getBigUint64(232, true),
            ms,
        };
    };

    const interpreted = run(0);
    const compiled = run(1);
    assert.equal(interpreted.rax, BigInt(expected), "interpreter result");
    assert.equal(compiled.rax, BigInt(expected), "jit result");
    assert.equal(compiled.r11, interpreted.r11, "pass counter matches");
    assert.equal(compiled.rip, interpreted.rip, "final rip matches");
    assert.ok(ex.jit64_stat(5) <= 1000, "block count stays at or below the cap");

    console.log(
        "jit64 entry cache: " + BLOCKS + " blocks x " + PASSES + " passes, " +
        "rax=" + compiled.rax + ", compiled blocks=" + ex.jit64_stat(3) +
        ", interp=" + interpreted.ms.toFixed(0) + "ms jit=" + compiled.ms.toFixed(0) + "ms",
    );
    process.exit(0);
});
