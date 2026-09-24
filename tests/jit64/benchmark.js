#!/usr/bin/env node

// Long-mode JIT performance benchmark: runs the same hot loop once through the
// interpreter and once through the JIT, and reports the throughput ratio.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/jit64/benchmark.js`

import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";

const BASE = 0x1000;
const ITERATIONS = +process.env.JIT64_BENCH_ITERATIONS || 500000;

const emulator = new v64({
    autostart: false,
    memory_size: 2 * 1024 * 1024,
    disable_jit: 0,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const buffer = ex.memory.buffer;
    const u32 = new Uint32Array(buffer);

    const write64 = (address, value) => {
        ex.write32(address, Number(value & 0xFFFF_FFFFn));
        ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
    };

    // Identity-map the first 1 GiB with 2 MiB pages
    const PML4 = 0x10000;
    const PDPT = 0x11000;
    const PD = 0x12000;
    const PT = 0x13000;
    write64(PML4, BigInt(PDPT) | 0x3n);
    write64(PDPT, BigInt(PD) | 0x3n);
    write64(PD, BigInt(PT) | 0x3n);
    for(let i = 0; i < 512; i++)
    {
        write64(PT + i * 8, BigInt(i) * 0x1000n | 0x3n);
    }
    for(let i = 1; i < 512; i++)
    {
        write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);
    }

    // mov rax, 1; mov rbx, 3; mov rcx, N;
    // loop: add rax, rbx; xor rbx, rax; shl rbx, 1; sub rcx, 1; jnz loop; hlt
    const program = [
        0x48, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov rax, 1
        0x48, 0xC7, 0xC3, 0x03, 0x00, 0x00, 0x00, // mov rbx, 3
        0x48, 0xC7, 0xC1, ITERATIONS & 0xFF, ITERATIONS >> 8 & 0xFF, ITERATIONS >> 16 & 0xFF, ITERATIONS >> 24 & 0xFF, // mov rcx, N
        0x48, 0x01, 0xD8, // add rax, rbx
        0x48, 0x31, 0xC3, // xor rbx, rax
        0x48, 0xD1, 0xE3, // shl rbx, 1
        0x48, 0x83, 0xE9, 0x01, // sub rcx, 1
        0x75, 0xF1, // jnz loop
        0xF4, // hlt
    ];
    for(let i = 0; i < program.length; i++)
    {
        ex.write8(BASE + i, program[i]);
    }

    const run = jitEnabled => {
        ex.jit64_set_enabled(jitEnabled ? 1 : 0);
        ex.jit64_clear_cache();
        cpu.instruction_pointer[0] = BASE;
        u32[16 + 4] = 0x80000; // rsp
        cpu.flags[0] &= ~(1 << 9); // cli
        cpu.in_hlt[0] = 0;
        ex.enter_long_mode(PML4);

        const before = u32[166];
        const start = process.hrtime.bigint();
        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 100000000)
        {
            ex.main_loop();
        }
        const elapsed = Number(process.hrtime.bigint() - start) / 1e6;
        assert.equal(cpu.in_hlt[0], 1, "the benchmark program halted");
        const instructions = (u32[166] - before) >>> 0;
        return {
            ms: elapsed,
            instructions,
            mips: instructions / elapsed / 1000,
        };
    };

    // Warm up the module once so instantiation cost is not measured.
    run(0);

    const interpreted = run(0);
    const compiled = run(1);

    const ratio = interpreted.ms / compiled.ms;
    console.log(
        "jit64 benchmark: " + ITERATIONS + " iterations, " +
        compiled.instructions + " loop instructions",
    );
    console.log(
        "  interpreter: " + interpreted.ms.toFixed(1) + " ms (" +
        interpreted.mips.toFixed(1) + " M instr/s)",
    );
    console.log(
        "  jit64:       " + compiled.ms.toFixed(1) + " ms (" +
        compiled.mips.toFixed(1) + " M instr/s)",
    );
    console.log("  speedup:     " + ratio.toFixed(2) + "x");

    assert.ok(compiled.instructions > 0, "the JIT ran the loop");
    assert.ok(ratio > 1, "jit64 is faster than the interpreter");
    process.exit(0);
});
