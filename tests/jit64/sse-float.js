// Float-to-integer conversion conformance for the 64-bit interpreter and JIT.
//
// Out-of-range float-to-integer conversion diverged from qemu: busybox awk
// printing 2^62 gave -2147483648 under qemu and 0 here. qemu's answer is the x86
// "integer indefinite" value. SDM Vol. 2, CVTTSD2SI / CVTTSD2SQ: zero converts
// to zero, an in-range value truncates toward zero, and NaN, either infinity and
// anything outside the destination range all give integer indefinite --
// 0x80000000 at 32 bits, 0x8000000000000000 at 64. A result that looks like the
// low bits of the value means the conversion wrapped, which is what these cases
// catch.
//
// Operands are staged through memory and the GPRs (mov rax,[r15]; movq xmm0,rax)
// because XMM and GPR are separate register files: setting rax does not set xmm0.
// The harness is shared with tests/jit64/bmi1.js and is kept identical to it.
//
// Encodings from clang:
//   mov rax,[r15]      49 8b 07         mov eax,[r15]       41 8b 07
//   movq xmm0,rax      66 48 0f 6e c0   movd xmm0,eax       66 0f 6e c0
//   cvttsd2si ebx,xmm0 f2 0f 2c d8      cvttsd2si rbx,xmm0  f2 48 0f 2c d8
//   cvtsd2si  ebx,xmm0 f2 0f 2d d8      cvttss2si ebx,xmm0  f3 0f 2c d8
//   cvtss2si  ebx,xmm0 f3 0f 2d d8      cvttss2si rbx,xmm0  f3 48 0f 2c d8
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/jit64/sse-float.js`

import { v64 } from "../../src/main.js";
import { set_cpu_config, JIT_DISABLE_64 } from "../../src/config.js";

const BASE = 0x1000;
// Staging area for memory-operand cases; r15 points here.
const SCRATCH = 0x100000;
const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const PT = 0x13000;

// Register names used in the case tables, mapped to the emulator's indices.
const REG = { rax: 0, rcx: 1, rdx: 2, rbx: 3, rsp: 4, rbp: 5, rsi: 6, rdi: 7 };

const M = 0xFFFFFFFFFFFFFFFFn;
const u64 = v => BigInt.asUintN(64, v);

// Note on the single-precision cases: 1e18 is not representable as an f32, so
// the staged pattern decodes to 999999984306749440. That is the value the SDM
// conversion has to produce, and it happens to have zero low 32 bits -- worth
// knowing before reading a truncated-looking result as a bug.

const CASES = [
    {
        name: "cvttsd2si r32  0",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2C, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00] },
        set: {},
        check: { reg: 3, want: 0n },
    },
    {
        name: "cvttsd2si r32  1.9",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2C, 0xD8], mem: [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0xFE, 0x3F] },
        set: {},
        check: { reg: 3, want: 1n },
    },
    {
        name: "cvttsd2si r32  -1.9",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2C, 0xD8], mem: [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0xFE, 0xBF] },
        set: {},
        check: { reg: 3, want: 0xFFFFFFFFn },
    },
    {
        name: "cvttsd2si r32  INT32_MAX",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2C, 0xD8], mem: [0x00, 0x00, 0xC0, 0xFF, 0xFF, 0xFF, 0xDF, 0x41] },
        set: {},
        check: { reg: 3, want: 2147483647n },
    },
    {
        name: "cvttsd2si r32  INT32_MIN",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2C, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE0, 0xC1] },
        set: {},
        check: { reg: 3, want: 0x80000000n },
    },
    {
        name: "cvttsd2si r32  2^31 out of range",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2C, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE0, 0x41] },
        set: {},
        check: { reg: 3, want: 0x80000000n },
    },
    {
        name: "cvttsd2si r32  5000050000 out of range",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2C, 0xD8], mem: [0x00, 0x00, 0x00, 0x55, 0x6B, 0xA0, 0xF2, 0x41] },
        set: {},
        check: { reg: 3, want: 0x80000000n },
    },
    {
        name: "cvttsd2si r32  -2147483649 out of range",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2C, 0xD8], mem: [0x00, 0x00, 0x20, 0x00, 0x00, 0x00, 0xE0, 0xC1] },
        set: {},
        check: { reg: 3, want: 0x80000000n },
    },
    {
        name: "cvttsd2si r32  2^62 out of range",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2C, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xD0, 0x43] },
        set: {},
        check: { reg: 3, want: 0x80000000n },
    },
    {
        name: "cvttsd2si r32  NaN",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2C, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF8, 0x7F] },
        set: {},
        check: { reg: 3, want: 0x80000000n },
    },
    {
        name: "cvttsd2si r32  +Inf",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2C, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x7F] },
        set: {},
        check: { reg: 3, want: 0x80000000n },
    },
    {
        name: "cvttsd2si r32  -Inf",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2C, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0xFF] },
        set: {},
        check: { reg: 3, want: 0x80000000n },
    },
    {
        name: "cvttsd2si r64  0",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x48, 0x0F, 0x2C, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00] },
        set: {},
        check: { reg: 3, want: 0n },
    },
    {
        name: "cvttsd2si r64  1.9",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x48, 0x0F, 0x2C, 0xD8], mem: [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0xFE, 0x3F] },
        set: {},
        check: { reg: 3, want: 1n },
    },
    {
        name: "cvttsd2si r64  -1.9",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x48, 0x0F, 0x2C, 0xD8], mem: [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0xFE, 0xBF] },
        set: {},
        check: { reg: 3, want: 0xFFFFFFFFFFFFFFFFn },
    },
    {
        name: "cvttsd2si r64  5000050000",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x48, 0x0F, 0x2C, 0xD8], mem: [0x00, 0x00, 0x00, 0x55, 0x6B, 0xA0, 0xF2, 0x41] },
        set: {},
        check: { reg: 3, want: 5000050000n },
    },
    {
        name: "cvttsd2si r64  2^62",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x48, 0x0F, 0x2C, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xD0, 0x43] },
        set: {},
        check: { reg: 3, want: 4611686018427387904n },
    },
    {
        name: "cvttsd2si r64  2^63 out of range",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x48, 0x0F, 0x2C, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE0, 0x43] },
        set: {},
        check: { reg: 3, want: 0x8000000000000000n },
    },
    {
        name: "cvttsd2si r64  NaN",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x48, 0x0F, 0x2C, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF8, 0x7F] },
        set: {},
        check: { reg: 3, want: 0x8000000000000000n },
    },
    {
        name: "cvttsd2si r64  +Inf",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x48, 0x0F, 0x2C, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x7F] },
        set: {},
        check: { reg: 3, want: 0x8000000000000000n },
    },
    {
        name: "cvttss2si r32  1.9",
        code: { bytes: [0x41, 0x8B, 0x07, 0x66, 0x0F, 0x6E, 0xC0, 0xF3, 0x0F, 0x2C, 0xD8], mem: [0x33, 0x33, 0xF3, 0x3F] },
        set: {},
        check: { reg: 3, want: 1n },
    },
    {
        name: "cvttss2si r32  -1.9",
        code: { bytes: [0x41, 0x8B, 0x07, 0x66, 0x0F, 0x6E, 0xC0, 0xF3, 0x0F, 0x2C, 0xD8], mem: [0x33, 0x33, 0xF3, 0xBF] },
        set: {},
        check: { reg: 3, want: 0xFFFFFFFFn },
    },
    {
        name: "cvttss2si r32  3e9 out of range",
        code: { bytes: [0x41, 0x8B, 0x07, 0x66, 0x0F, 0x6E, 0xC0, 0xF3, 0x0F, 0x2C, 0xD8], mem: [0x5E, 0xD0, 0x32, 0x4F] },
        set: {},
        check: { reg: 3, want: 0x80000000n },
    },
    {
        name: "cvttss2si r64  1.9",
        code: { bytes: [0x41, 0x8B, 0x07, 0x66, 0x0F, 0x6E, 0xC0, 0xF3, 0x48, 0x0F, 0x2C, 0xD8], mem: [0x33, 0x33, 0xF3, 0x3F] },
        set: {},
        check: { reg: 3, want: 1n },
    },
    {
        name: "cvttss2si r64  1e18",
        code: { bytes: [0x41, 0x8B, 0x07, 0x66, 0x0F, 0x6E, 0xC0, 0xF3, 0x48, 0x0F, 0x2C, 0xD8], mem: [0x6B, 0x0B, 0x5E, 0x5D] },
        set: {},
        check: { reg: 3, want: 999999984306749440n },
    },
    {
        name: "cvttss2si r64  1e30 out of range",
        code: { bytes: [0x41, 0x8B, 0x07, 0x66, 0x0F, 0x6E, 0xC0, 0xF3, 0x48, 0x0F, 0x2C, 0xD8], mem: [0xCA, 0xF2, 0x49, 0x71] },
        set: {},
        check: { reg: 3, want: 0x8000000000000000n },
    },
    {
        name: "cvtsd2si  r32  1.5 -> 2",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2D, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF8, 0x3F] },
        set: {},
        check: { reg: 3, want: 2n },
    },
    {
        name: "cvtsd2si  r32  2.5 -> 2",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2D, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x40] },
        set: {},
        check: { reg: 3, want: 2n },
    },
    {
        name: "cvtsd2si  r32  -1.5 -> -2",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2D, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xF8, 0xBF] },
        set: {},
        check: { reg: 3, want: 0xFFFFFFFEn },
    },
    {
        name: "cvtsd2si  r32  0.5 -> 0",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2D, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE0, 0x3F] },
        set: {},
        check: { reg: 3, want: 0n },
    },
    {
        name: "cvtsd2si  r32  1.4 -> 1",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2D, 0xD8], mem: [0x66, 0x66, 0x66, 0x66, 0x66, 0x66, 0xF6, 0x3F] },
        set: {},
        check: { reg: 3, want: 1n },
    },
    {
        name: "cvtsd2si  r32  2^31 out of range",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0, 0xF2, 0x0F, 0x2D, 0xD8], mem: [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xE0, 0x41] },
        set: {},
        check: { reg: 3, want: 0x80000000n },
    },
    {
        name: "cvtss2si  r32  1.5 -> 2",
        code: { bytes: [0x41, 0x8B, 0x07, 0x66, 0x0F, 0x6E, 0xC0, 0xF3, 0x0F, 0x2D, 0xD8], mem: [0x00, 0x00, 0xC0, 0x3F] },
        set: {},
        check: { reg: 3, want: 2n },
    },
    {
        name: "cvtss2si  r32  2.5 -> 2",
        code: { bytes: [0x41, 0x8B, 0x07, 0x66, 0x0F, 0x6E, 0xC0, 0xF3, 0x0F, 0x2D, 0xD8], mem: [0x00, 0x00, 0x20, 0x40] },
        set: {},
        check: { reg: 3, want: 2n },
    },
    {
        name: "cvtss2si  r32  0.5 -> 0",
        code: { bytes: [0x41, 0x8B, 0x07, 0x66, 0x0F, 0x6E, 0xC0, 0xF3, 0x0F, 0x2D, 0xD8], mem: [0x00, 0x00, 0x00, 0x3F] },
        set: {},
        check: { reg: 3, want: 0n },
    },
    {
        // Self-check on the staging path: load a known bit pattern into rax,
        // move it into xmm0 and straight back. If this round trip is not
        // bit-exact then every conversion result below is measuring the wrong
        // input, which is exactly the trap a throwaway probe fell into.
        name: "movq xmm0<->rax round trip (staging self-check)",
        code: { bytes: [0x49, 0x8B, 0x07, 0x66, 0x48, 0x0F, 0x6E, 0xC0,
                        0x66, 0x48, 0x0F, 0x7E, 0xC3],
                mem: [0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0xF3, 0x3F] },
        set: {},
        check: { reg: 3, want: 0x3FF3333333333333n },
    },
];

const emulator = new v64({
    autostart: false,
    memory_size: 8 * 1024 * 1024,
    disable_jit: 0,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", async () =>
{
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const u32 = new Uint32Array(ex.memory.buffer);
    const ext = new BigUint64Array(ex.memory.buffer, 160, 8);

    const write64 = (a, v) =>
    {
        ex.write32(a, Number(v & 0xFFFF_FFFFn));
        ex.write32(a + 4, Number(v >> 32n & 0xFFFF_FFFFn));
    };
    write64(PML4, BigInt(PDPT) | 0x3n);
    write64(PDPT, BigInt(PD) | 0x3n);
    write64(PD, BigInt(PT) | 0x3n);
    for(let i = 0; i < 512; i++) write64(PT + i * 8, BigInt(i) * 0x1000n | 0x3n);

    const set64 = (i, v) =>
    {
        if(i < 8)
        {
            u32[16 + i] = Number(u64(v) & 0xFFFF_FFFFn);
            u32[32 + i] = Number((u64(v) >> 32n) & 0xFFFF_FFFFn);
        }
        else { ext[i - 8] = u64(v); }
    };
    const get64 = i =>
        (i < 8) ? ((BigInt(u32[32 + i]) << 32n) | BigInt(u32[16 + i])) : ext[i - 8];

    // ---- harness self-check -------------------------------------------
    // Write 16 mutually distinct values, read them back untouched. If any
    // mismatch, every later result is meaningless, so abort instead.
    {
        const distinct = [];
        for(let i = 0; i < 16; i++)
        {
            // Mask to 64 bits: the multiply overflows past i = 0, and an
            // unmasked expectation would make every register but r0 "fail".
            distinct.push(BigInt.asUintN(64, 0x9E3779B97F4A7C15n * BigInt(i + 1)));
        }
        const want4 = 0x80000n;
        for(let i = 0; i < 16; i++) set64(i, distinct[i]);
        set64(4, want4);
        ex.write8(BASE, 0xF4);
        cpu.instruction_pointer[0] = BASE;
        cpu.flags[0] = 0x2;
        cpu.in_hlt[0] = 0;
        ex.enter_long_mode(PML4);
        let g = 0;
        while(!cpu.in_hlt[0] && g++ < 20000) ex.main_loop();
        const bad = [];
        for(let i = 0; i < 16; i++)
        {
            const want = i === 4 ? want4 : distinct[i];
            if(get64(i) !== want)
            {
                bad.push("r" + i + "=0x" + get64(i).toString(16) +
                    " want 0x" + want.toString(16));
            }
        }
        if(bad.length)
        {
            console.log("harness read-back FAILED: " + bad.join(", "));
            console.log("results below would be meaningless; aborting");
            process.exit(2);
        }
        console.log("harness self-check: register read-back ok (16/16 distinct)\n");
    }

    const run = (code, set, jitEnabled, flags) =>
    {
        set_cpu_config(ex, "JIT_DISABLE", jitEnabled ? 0 : JIT_DISABLE_64);
        ex.jit64_clear_cache();
        const bytes = Array.isArray(code) ? code : code.bytes;
        const program = bytes.concat([0xF4]);
        for(let i = 0; i < program.length; i++) ex.write8(BASE + i, program[i]);
        for(let i = 0; i < 16; i++) set64(i, 0n);
        set64(4, 0x80000n);
        // r15 addresses the staging area for the memory-operand cases.
        set64(15, BigInt(SCRATCH));
        if(!Array.isArray(code) && code.mem)
        {
            for(let i = 0; i < code.mem.length; i++)
            {
                ex.write8(SCRATCH + i, code.mem[i] & 0xFF);
            }
        }
        for(const [r, v] of Object.entries(set || {})) set64(REG[r], v);
        // Start from a known flags value so OF cannot leak between cases: ADCX
        // uses CF, ADOX uses OF, and confusing the two is an easy way to
        // invent a failure that is not there.
        cpu.flags[0] = 0x2;
        if(flags)
        {
            if(flags.cf) cpu.flags[0] |= 1;
            if(flags.of) cpu.flags[0] |= (1 << 11);
        }
        cpu.instruction_pointer[0] = BASE;
        cpu.in_hlt[0] = 0;
        ex.enter_long_mode(PML4);
        let g = 0;
        while(!cpu.in_hlt[0] && g++ < 20000) ex.main_loop();
        return { halted: cpu.in_hlt[0] === 1, rip: cpu.instruction_pointer[0] >>> 0 };
    };

    const engines = [];
    const which = process.env.SSE_FLOAT_ENGINE || "both";
    if(which === "interp" || which === "both") engines.push(["interp", false]);
    if(which === "jit" || which === "both") engines.push(["jit", true]);

    let failures = 0;
    for(const c of CASES)
    {
        const checks = Array.isArray(c.check) ? c.check : [c.check];
        const bad = [];
        for(const [label, jitEnabled] of engines)
        {
            const r = run(c.code, c.set, jitEnabled, { cf: c.cf, of: c.of });
            if(!r.halted)
            {
                bad.push(label + ": did not halt (rip=0x" + r.rip.toString(16) + ")");
                continue;
            }
            for(const chk of checks)
            {
                if(chk.flag)
                {
                    const bit = chk.flag === "cf" ? 1 : (1 << 11);
                    const got = (cpu.flags[0] & bit) !== 0;
                    if(got !== chk.want)
                    {
                        bad.push(label + ": " + chk.flag.toUpperCase() + "=" + got +
                            " want " + chk.want);
                    }
                    continue;
                }
                const got = get64(chk.reg);
                if(got !== chk.want)
                {
                    bad.push(label + ": r" + chk.reg + "=0x" + got.toString(16) +
                        " want 0x" + chk.want.toString(16));
                }
            }
            // BMI1/BMI2/ADX write at most two registers, so any other source
            // that changed means the wrong register was written.
            for(const [name, idx] of Object.entries(c.set))
            {
                const reg = REG[name];
                if(checks.some(k => k.reg === reg)) continue;
                if(get64(reg) !== u64(c.set[name]))
                {
                    bad.push(label + ": source " + name + " clobbered (0x" +
                        get64(reg).toString(16) + ")");
                }
            }
        }
        if(bad.length) failures++;
        console.log((bad.length ? "FAIL " : "ok   ") + c.name);
        for(const b of bad) console.log("      " + b);
    }

    console.log("\n" + (failures ? failures + " failing case(s)"
        : "all float-to-integer conversions match the SDM"));
    process.exit(failures ? 1 : 0);
});
