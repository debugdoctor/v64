#!/usr/bin/env node

// Long-mode CPU microbenchmarks: each workload runs once interpreted and once
// through the 64-bit JIT, for before/after comparison of CPU changes.
//
// Requires `make build/v64-debug.wasm`. Options:
//   JIT64_BENCH_ITERATIONS=500000 MICROBENCH_ONLY=load_stride_page \
//   MICROBENCH_OUT=build/microbench.json node tests/benchmark/cpu64.js
//
// The measured JIT run does not clear the block cache or rewrite the code page,
// so one-time compile cost is excluded.
// HOTSPOTS=1 reports fallback counts and enables sampling during measurement.

import assert from "node:assert/strict";
import { writeFileSync as write_file_sync } from "node:fs";
import { v64 } from "../../src/main.js";
import { set_cpu_config, JIT_DISABLE_64 } from "../../src/config.js";

const ITERATIONS = +process.env.JIT64_BENCH_ITERATIONS || 500000;

// Code at 0x1000. PD[0] maps the first 2 MiB with 4 KiB pages; PD[1..] use 2 MiB
// huge pages. Scratch regions stay in the 4 KiB window so each page is a PTE.
const CODE = 0x1000;
const SCRATCH = 0x100000; // 1 MiB
const SCRATCH_MASK = 0x1FFFFF; // stay inside the first 2 MiB while striding

const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const PT = 0x13000;

const imm32 = value => [value & 0xFF, value >> 8 & 0xFF, value >> 16 & 0xFF, value >> 24 & 0xFF];
const imm64 = value => {
    const out = [];
    for(let i = 0; i < 8; i++) out.push(Number(value >> BigInt(8 * i) & 0xFFn));
    return out;
};

// Body runs `iterations` times; `outer` runs once before the loop.
const wrap = (outer, body, iterations) => {
    const bytes = [];
    const put = (...b) => bytes.push(...b);
    put(0x41, 0xBB, ...imm32(iterations)); // mov r11d, iterations
    if(outer.length) put(...outer);
    const head = bytes.length;
    put(0x41, 0xFF, 0xCB); // dec r11d
    const jz_at = bytes.length;
    put(0x0F, 0x84, 0, 0, 0, 0); // jz end
    put(...body);
    const jmp_at = bytes.length;
    put(0xE9, 0, 0, 0, 0); // jmp head
    const end = bytes.length;
    put(0xF4); // hlt
    for(let i = 0; i < 4; i++) bytes[jz_at + 2 + i] = end - (jz_at + 6) >> (8 * i) & 0xFF;
    for(let i = 0; i < 4; i++) bytes[jmp_at + 1 + i] = head - (jmp_at + 5) >> (8 * i) & 0xFF;
    return bytes;
};

// head / dec / jz end / call fn / jmp head / fn: ret / end: hlt
const call_ret_program = iterations => [
    0x41, 0xBB, ...imm32(iterations), // mov r11d, iterations
    0x41, 0xFF, 0xCB,                 // head: dec r11d
    0x0F, 0x84, 11, 0, 0, 0,          // jz end
    0xE8, 5, 0, 0, 0,                 // call fn
    0xE9, 0xED, 0xFF, 0xFF, 0xFF,     // jmp head
    0xC3,                             // fn: ret
    0xF4,                             // end: hlt
];

const mov_rax_rbx = [0x48, 0x8B, 0x03];
const mov_rbx_rax = [0x48, 0x89, 0x03];
const mov_rbx_imm = value => [0x48, 0xBB, ...imm64(BigInt(value))];
const mov_rdi_imm = value => [0x48, 0xBF, ...imm64(BigInt(value))];
const mov_rsi_imm = value => [0x48, 0xBE, ...imm64(BigInt(value))];
const mov_rcx_imm = value => [0x48, 0xC7, 0xC1, ...imm32(value)];
const add_rbx_imm8 = value => [0x48, 0x83, 0xC3, value];
const add_rbx_imm32 = value => [0x48, 0x81, 0xC3, ...imm32(value)];
const and_rbx_imm32 = value => [0x48, 0x81, 0xE3, ...imm32(value)];
const add_rax_imm8 = value => [0x48, 0x83, 0xC0, value];

const REP_ITERATIONS = +process.env.JIT64_BENCH_REP_ITERATIONS || 4000;
const REP_BYTES = 64;

const WORKLOADS = [
    {
        name: "alu",
        note: "integer ALU + loop branch",
        iterations: ITERATIONS,
        outer: [],
        body: [
            0x48, 0x01, 0xD8, // add rax, rbx
            0x48, 0x31, 0xC3, // xor rbx, rax
            0x48, 0xD1, 0xE3, // shl rbx, 1
        ],
    },
    {
        name: "load_same_page",
        note: "loads inside one 4 KiB page",
        iterations: ITERATIONS,
        outer: mov_rbx_imm(SCRATCH),
        body: [
            ...mov_rax_rbx,          // mov rax, [rbx]
            ...add_rbx_imm8(8),      // add rbx, 8
            ...and_rbx_imm32(0xFF8), // and rbx, 0xFF8
        ],
    },
    {
        name: "store_same_page",
        note: "stores inside one 4 KiB page",
        iterations: ITERATIONS,
        outer: mov_rbx_imm(SCRATCH),
        body: [
            ...mov_rbx_rax,          // mov [rbx], rax
            ...add_rbx_imm8(8),      // add rbx, 8
            ...and_rbx_imm32(0xFF8), // and rbx, 0xFF8
        ],
    },
    {
        name: "load_store_mixed",
        note: "read-modify-write, one page",
        iterations: ITERATIONS,
        outer: mov_rbx_imm(SCRATCH),
        body: [
            ...mov_rax_rbx,          // mov rax, [rbx]
            ...add_rax_imm8(1),      // add rax, 1
            ...mov_rbx_rax,          // mov [rbx], rax
            ...add_rbx_imm8(8),      // add rbx, 8
            ...and_rbx_imm32(0xFF8), // and rbx, 0xFF8
        ],
    },
    {
        name: "load_stride_page",
        note: "one load per 4 KiB page (TLB pressure)",
        iterations: ITERATIONS,
        outer: mov_rbx_imm(SCRATCH),
        body: [
            ...mov_rax_rbx,                    // mov rax, [rbx]
            ...add_rbx_imm32(0x1000),          // add rbx, 0x1000
            ...and_rbx_imm32(SCRATCH_MASK),    // keep inside the mapped window
        ],
    },
    {
        name: "call_ret",
        note: "near call/ret pair",
        iterations: ITERATIONS,
        program: iterations => call_ret_program(iterations),
    },
    {
        name: "partial_prefix",
        note: "supported prefix followed by an undecodable pause",
        iterations: ITERATIONS,
        outer: [],
        body: [
            0x48, 0x01, 0xD8, // add rax, rbx
            0x48, 0x31, 0xC3, // xor rbx, rax
            0x48, 0xD1, 0xE3, // shl rbx, 1
            0xF3, 0x90,       // pause: the JIT decoder stops here
        ],
    },
    {
        name: "sse_movdqa",
        note: "movdqa 16-byte load + store (SSE2 codegen)",
        iterations: ITERATIONS,
        outer: mov_rbx_imm(SCRATCH),
        body: [
            0x66, 0x0F, 0x6F, 0x03,       // movdqa xmm0, [rbx]
            0x66, 0x0F, 0x7F, 0x43, 0x10, // movdqa [rbx+16], xmm0
            ...and_rbx_imm32(SCRATCH_MASK), // keep rbx inside the mapped window
        ],
    },
    {
        name: "sse_paddd",
        note: "packed integer SSE2 (paddd/pand/por)",
        iterations: ITERATIONS,
        outer: [],
        body: [
            0x66, 0x0F, 0xFE, 0xC1, // paddd xmm0, xmm1
            0x66, 0x0F, 0xDB, 0xC2, // pand xmm0, xmm2
            0x66, 0x0F, 0xEB, 0xC3, // por xmm0, xmm3
        ],
    },
    {
        name: "bmi_shifts",
        note: "SHLX/SHRX/SARX register shifts",
        iterations: ITERATIONS,
        body: [
            0xC4, 0xE2, 0xF1, 0xF7, 0xC8,
            0xC4, 0xE2, 0xF3, 0xF7, 0xC8,
            0xC4, 0xE2, 0xF2, 0xF7, 0xC8,
        ],
    },
    {
        name: "avx_addq",
        note: "VPADDQ/VPSUBQ register arithmetic",
        iterations: ITERATIONS,
        body: [0xC5, 0xFD, 0xD4, 0xC1, 0xC5, 0xFD, 0xFB, 0xC1],
    },
    {
        name: "bmi_rorx",
        note: "RORX register rotates used by SHA-256",
        iterations: ITERATIONS,
        body: [0xC4, 0xE3, 0xFB, 0xF0, 0xD8, 0x19], // nasm: rorx rbx,rax,25
    },
    {
        name: "sign_extend",
        note: "CDQE and CQO",
        iterations: ITERATIONS,
        body: [0x48, 0x98, 0x48, 0x99],
    },
    {
        name: "avx_vpxor",
        note: "VEX.256 register XOR",
        iterations: ITERATIONS,
        body: [0xC5, 0xFD, 0xEF, 0xC1], // nasm: vpxor ymm0,ymm0,ymm1
    },
    {
        name: "rep_stosb",
        note: "rep stosb, 64 bytes per iteration (known slow path)",
        iterations: REP_ITERATIONS,
        outer: [0x31, 0xC0], // xor eax, eax (al = 0)
        body: [
            ...mov_rdi_imm(SCRATCH),
            ...mov_rcx_imm(REP_BYTES),
            0xF3, 0xAA, // rep stosb
        ],
    },
    {
        name: "rep_movsb",
        note: "rep movsb, 64 bytes per iteration (known slow path)",
        iterations: REP_ITERATIONS,
        body: [
            ...mov_rsi_imm(SCRATCH),
            ...mov_rdi_imm(SCRATCH + 0x80000),
            ...mov_rcx_imm(REP_BYTES),
            0xF3, 0xA4, // rep movsb
        ],
    },
];

for(const workload of WORKLOADS)
{
    workload.program = workload.program
        ? workload.program(workload.iterations)
        : wrap(workload.outer || [], workload.body, workload.iterations);
}

const emulator = new v64({
    autostart: false,
    memory_size: 4 * 1024 * 1024,
    disable_jit: 0,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const buffer = ex.memory.buffer;
    const u32 = new Uint32Array(buffer);
    if(process.env.INLINE_MEMORY === "0") set_cpu_config(ex, "JIT64_INLINE_MEMORY", 0);
    if(process.env.INLINE_WRITE === "0") set_cpu_config(ex, "JIT64_INLINE_WRITE", 0);
    if(process.env.INTERP_MEM_SINGLE === "0") set_cpu_config(ex, "INTERP64_MEM_SINGLE", 0);
    if(process.env.INTERP_BATCH === "0") set_cpu_config(ex, "INTERP64_BATCH", 0);
    if(process.env.PARTIAL_BLOCKS === "0") set_cpu_config(ex, "JIT64_PARTIAL_BLOCKS", 0);
    if(process.env.ENTRY_CACHE === "0") set_cpu_config(ex, "JIT64_ENTRY_CACHE", 0);
    if(process.env.HOT_CACHE === "0") set_cpu_config(ex, "JIT64_HOT_CACHE", 0);
    if(process.env.SSE === "0") set_cpu_config(ex, "JIT64_SSE", 0);
    if(process.env.SSE === "1") set_cpu_config(ex, "JIT64_SSE", 1);
    if(process.env.CHAIN_BUDGET !== undefined) set_cpu_config(ex, "JIT64_CHAIN_BUDGET", parseInt(process.env.CHAIN_BUDGET, 10));
    if(process.env.BLOCK_PROLOGUE !== undefined) set_cpu_config(ex, "JIT64_BLOCK_PROLOGUE", parseInt(process.env.BLOCK_PROLOGUE, 10));
    if(process.env.SUPERBLOCKS !== undefined) set_cpu_config(ex, "JIT64_SUPERBLOCKS", parseInt(process.env.SUPERBLOCKS, 10));

    const write64 = (address, value) => {
        ex.write32(address, Number(value & 0xFFFF_FFFFn));
        ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
    };

    // Identity-map the first 1 GiB: a 4 KiB PT for the first 2 MiB, 2 MiB pages
    // above it.
    write64(PML4, BigInt(PDPT) | 0x3n);
    write64(PDPT, BigInt(PD) | 0x3n);
    write64(PD, BigInt(PT) | 0x3n);
    for(let i = 0; i < 512; i++) write64(PT + i * 8, BigInt(i) * 0x1000n | 0x3n);
    for(let i = 1; i < 512; i++) write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);

    const prepare = program => {
        for(let i = 0; i < program.length; i++) ex.write8(CODE + i, program[i]);
    };

    const run = (jitEnabled, clearCache) => {
        set_cpu_config(ex, "JIT_DISABLE", jitEnabled ? 0 : JIT_DISABLE_64);
        if(clearCache) ex.jit64_clear_cache();
        cpu.instruction_pointer[0] = CODE;
        u32[16 + 4] = 0x80000; // rsp
        cpu.flags[0] &= ~(1 << 9); // cli: no timer interrupts
        cpu.in_hlt[0] = 0;
        ex.enter_long_mode(PML4);

        const before = u32[166]; // instruction counter
        const start = process.hrtime.bigint();
        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 100000000)
        {
            ex.main_loop();
        }
        const elapsed = Number(process.hrtime.bigint() - start) / 1e6;
        assert.equal(cpu.in_hlt[0], 1, "workload halted");
        const instructions = (u32[166] - before) >>> 0;
        return {
            ms: elapsed,
            instructions,
            mips: instructions / elapsed / 1000,
        };
    };

    const results = {};
    console.log("jit64 microbench");
    console.log(
        "  " + "workload".padEnd(18) + "N".padStart(9) +
        "interp M/s".padStart(12) + "jit M/s".padStart(11) +
        "speedup".padStart(10) + "  guest instr",
    );
    const only = process.env.MICROBENCH_ONLY;
    for(const workload of WORKLOADS)
    {
        if(only && workload.name !== only) continue;

        // Interpreted baseline.
        prepare(workload.program);
        run(0, true); // warm wasm dispatch
        const interpreted = run(0, true);

        // Compile once, then measure without clearing the cache or rewriting code.
        prepare(workload.program);
        run(1, true);
        if(process.env.HOTSPOTS === "1") set_cpu_config(ex, "JIT64_PROFILE", 1);
        const compiled = run(1, false);
        if(process.env.HOTSPOTS === "1")
        {
            const count = ex.jit64_profile_snapshot();
            console.log("  fallback " + workload.name + ": interpreter=" + ex.jit64_profile_events(0) +
                " jit-helper=" + ex.jit64_profile_events(1) + " sampled_locations=" + count);
            set_cpu_config(ex, "JIT64_PROFILE", 0);
            assert.equal(ex.jit64_profile_events(0), 0n, "disabling profiling resets counters");
            assert.equal(ex.jit64_profile_snapshot(), 0, "disabling profiling clears samples");
        }

        const speedup = interpreted.ms / compiled.ms;
        results[workload.name] = {
            note: workload.note,
            iterations: workload.iterations,
            interpreted_mips: +interpreted.mips.toFixed(2),
            jit_mips: +compiled.mips.toFixed(2),
            speedup: +speedup.toFixed(3),
            instructions: compiled.instructions,
        };
        console.log(
            "  " + workload.name.padEnd(18) +
            String(workload.iterations).padStart(9) +
            interpreted.mips.toFixed(1).padStart(12) +
            compiled.mips.toFixed(1).padStart(11) +
            speedup.toFixed(2).padStart(9) + "x  " +
            compiled.instructions,
        );
        assert.ok(compiled.instructions > 0, workload.name + ": ran instructions");
    }

    const out = process.env.MICROBENCH_OUT;
    if(out)
    {
        write_file_sync(out, JSON.stringify({ iterations: ITERATIONS, workloads: results }, null, 2) + "\n");
        console.log("wrote " + out);
    }
    process.exit(0);
});
