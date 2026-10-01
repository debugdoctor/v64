#!/usr/bin/env node

// Differential fuzzing of jit64 against the interpreter.
//
// Random straight-line 64-bit programs are generated from a pool of encodings
// both engines decode, wrapped in a hot loop so the JIT compiles them. Each
// program is run once interpreted and once compiled; registers, the defined
// flags and a scratch page must match. Registers and memory are the
// architecture's outputs, so a mismatch is a bug in one of the two engines.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/jit64/differential.js`

import { v64 } from "../../src/main.js";

const BASE = 0x1000;
const SCRATCH = 0x100000;
const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const PT = 0x13000;
const ITERATIONS = 600;
const CASES = +process.env.JIT64_DIFF_CASES || 40;
// JIT64_DIFF_ONLY=n: use only the last n pool entries (the newest additions)
const ONLY = +process.env.JIT64_DIFF_ONLY || 0;

// Defined for every instruction in the pool.
const FLAG_MASK = (1 << 0) | (1 << 2) | (1 << 6) | (1 << 7); // CF, PF, ZF, SF

const POOL = [
    [0x48, 0x01, 0xD8], // add rax, rbx
    [0x48, 0x29, 0xD1], // sub rcx, rdx
    [0x48, 0x21, 0xFE], // and rsi, rdi
    [0x4D, 0x09, 0xC8], // or r8, r9
    [0x4D, 0x31, 0xDA], // xor r10, r11
    [0x48, 0x11, 0xC8], // adc rax, rcx
    [0x48, 0x19, 0xD3], // sbb rbx, rdx
    [0x48, 0x39, 0xD8], // cmp rax, rbx
    [0x48, 0x85, 0xC0], // test rax, rax
    [0x48, 0xFF, 0xC0], // inc rax
    [0x48, 0xFF, 0xCB], // dec rbx
    [0x48, 0xF7, 0xD9], // neg rcx
    [0x48, 0xF7, 0xD2], // not rdx
    [0x48, 0xC1, 0xE0, 0x03], // shl rax, 3
    [0x48, 0xC1, 0xE9, 0x05], // shr rcx, 5
    [0x48, 0xC1, 0xFA, 0x02], // sar rdx, 2
    [0x49, 0xD3, 0xE0], // shl r8, cl
    [0x49, 0xD3, 0xE9], // shr r9, cl
    [0x49, 0xD3, 0xFA], // sar r10, cl
    [0x0F, 0xB6, 0xC3], // movzx eax, bl
    [0x48, 0x0F, 0xBE, 0xC3], // movsx rax, bl
    [0x48, 0x63, 0xC3], // movsxd rax, ebx
    [0x48, 0x8D, 0x44, 0x8B, 0x08], // lea rax, [rbx + rcx*4 + 8]
    [0x4F, 0x8D, 0x04, 0xD1], // lea r8, [r9 + r10*8]
    [0x48, 0x0F, 0xC8], // bswap rax
    [0x49, 0x0F, 0xC9], // bswap r9
    [0x48, 0x0F, 0xAF, 0xC3], // imul rax, rbx
    [0x48, 0x6B, 0xCA, 0x07], // imul rcx, rdx, 7
    [0x48, 0x0F, 0x44, 0xC3], // cmove rax, rbx
    [0x48, 0x0F, 0x45, 0xCA], // cmovne rcx, rdx
    [0x0F, 0x94, 0xC0], // sete al
    [0x0F, 0x95, 0xC3], // setne bl
    [0x00, 0xD8], // add al, bl
    [0x66, 0x29, 0xD8], // sub ax, bx
    [0x01, 0xD8], // add eax, ebx
    [0x45, 0x31, 0xC8], // xor r8d, r9d
    [0x4D, 0x89, 0xDA], // mov r10, r11
    [0x49, 0x90], // xchg r8, rax (0x90 + REX.B)
    [0x49, 0x91], // xchg r9, rax
    [0x48, 0x91], // xchg rcx, rax
    [0x48, 0x92], // xchg rdx, rax
    [0x49, 0x89, 0x47, 0x08], // mov [r15+8], rax
    [0x49, 0x8B, 0x5F, 0x08], // mov rbx, [r15+8]
    [0x49, 0x01, 0x47, 0x10], // add qword [r15+16], rax
    [0x41, 0x89, 0x4F, 0x18], // mov [r15+24], ecx
    [0x41, 0x8B, 0x57, 0x18], // mov edx, [r15+24]
    [0x41, 0x89, 0xD6], // mov r14d, edx
    [0x41, 0xD1, 0xEE], // shr r14d, 1
    [0x41, 0x83, 0xFC, 0x02], // cmp r12d, 2
    [0x48, 0x0F, 0xA3, 0xD8], // bt rax, rbx
    [0x48, 0x0F, 0xBA, 0xE0, 0x03], // bt rax, 3
    [0x48, 0x0F, 0xBA, 0xE8, 0x07], // bts rax, 7
    [0x48, 0x0F, 0xBA, 0xF0, 0x01], // btr rax, 1
    [0x48, 0x0F, 0xBA, 0xF8, 0x02], // btc rax, 2
    [0x0F, 0xBA, 0xE1, 0x0D], // bt ecx, 13
    [0xF0, 0x48, 0x0F, 0xB1, 0xD8], // lock cmpxchg rax, rbx
    [0x48, 0x0F, 0xB1, 0xCB], // cmpxchg rbx, rcx
    [0x0F, 0xB1, 0xC3], // cmpxchg ebx, eax
    [0x0F, 0xB0, 0xC3], // cmpxchg bl, al
    [0xC0, 0xE3, 0x03], // shl bl, 3
    [0xC0, 0xEB, 0x07], // shr bl, 7
    [0xD0, 0xC8], // ror al, 1
    [0xD2, 0xD0], // rcl al, cl
    [0x48, 0x0F, 0xBC, 0xC3], // bsf rax, rbx
    [0x48, 0x0F, 0xBD, 0xC3], // bsr rax, rbx
    [0xF3, 0x0F, 0xBC, 0xC3], // rep bsf (TZCNT on a BMI1 CPU)
    [0x0F, 0xBC, 0xC8], // bsf ecx, eax
    [0x48, 0x98], // cdqe
    [0x66, 0x98], // cbw
    [0x48, 0x8C, 0xD8], // mov rax, ds
    [0x41, 0x0F, 0xB6, 0x47, 0x08], // movzx eax, byte [r15+8]
    [0xF3, 0x41, 0x0F, 0x6F, 0x07], // movdqu xmm0, [r15]
    [0xF3, 0x41, 0x0F, 0x7F, 0x07], // movdqu [r15], xmm0
    [0x66, 0x0F, 0x6F, 0xC8], // movdqa xmm1, xmm0
    [0x66, 0x0F, 0xEF, 0xC1], // pxor xmm0, xmm1
    [0x66, 0x41, 0x0F, 0xEF, 0x07], // pxor xmm0, [r15]
];
if(ONLY)
{
    POOL.splice(0, Math.max(0, POOL.length - ONLY));
}
const SKIP = +process.env.JIT64_DIFF_SKIP || 0;
if(SKIP)
{
    POOL.splice(POOL.length - SKIP, SKIP);
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
    const ext = new BigUint64Array(buffer, 160, 8);

    const write64 = (address, value) => {
        ex.write32(address, Number(value & 0xFFFF_FFFFn));
        ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
    };

    // Identity-map the first 1 GiB: 4 KiB pages for the first 2 MiB.
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

    // mulberry32
    const rng = seed => () => {
        seed |= 0;
        seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };

    const set_reg64 = (i, value) => {
        if(i < 8)
        {
            u32[16 + i] = Number(value & 0xFFFF_FFFFn);
            u32[32 + i] = Number(value >> 32n & 0xFFFF_FFFFn);
        }
        else
        {
            ext[i - 8] = value;
        }
    };
    const reg64 = i =>
        (i < 8)
            ? (BigInt(u32[32 + i]) << 32n) | BigInt(u32[16 + i])
            : ext[i - 8];

    const build_program = (random, iterations = ITERATIONS, bodyLimit = 0) => {
        const body = [];
        const count = bodyLimit || 12 + Math.floor(random() * 13);
        for(let i = 0; i < count; i++)
        {
            body.push(...POOL[Math.floor(random() * POOL.length)]);
        }
        const prefix = [0x41, 0xBC, iterations & 0xFF, iterations >> 8 & 0xFF, iterations >> 16 & 0xFF, iterations >> 24 & 0xFF];
        const jnz_at = prefix.length + body.length + 3; // + dec r12d
        const rel = prefix.length - (jnz_at + 6);
        const suffix = [
            0x41, 0xFF, 0xCC, // dec r12d
            0x0F, 0x85, rel & 0xFF, rel >> 8 & 0xFF, rel >> 16 & 0xFF, rel >> 24 & 0xFF,
            0xF4, // hlt
        ];
        return prefix.concat(body, suffix);
    };

    const wrap_body = (body, iterations = ITERATIONS) => {
        const prefix = [0x41, 0xBC, iterations & 0xFF, iterations >> 8 & 0xFF, iterations >> 16 & 0xFF, iterations >> 24 & 0xFF];
        const jnz_at = prefix.length + body.length + 3;
        const rel = prefix.length - (jnz_at + 6);
        const suffix = [
            0x41, 0xFF, 0xCC, // dec r12d
            0x0F, 0x85, rel & 0xFF, rel >> 8 & 0xFF, rel >> 16 & 0xFF, rel >> 24 & 0xFF,
            0xF4, // hlt
        ];
        return prefix.concat(body, suffix);
    };

    // Fixed sequences the random pool only reaches by chance.
    const FIXED = [
        {
            name: "cmpxchg + cmove + add [r15+16],rax + bt + imul",
            body: [
                0x48, 0x0F, 0x44, 0xC3, // cmove rax, rbx
                0x48, 0x0F, 0xB1, 0xCB, // cmpxchg rbx, rcx
                0x49, 0x01, 0x47, 0x10, // add [r15+16], rax
                0x48, 0x0F, 0xA3, 0xD8, // bt rax, rbx
                0x48, 0x0F, 0xAF, 0xC3, // imul rax, rbx
            ],
        },
        {
            name: "lock cmpxchg rax,rbx + add [r15+8],rax",
            body: [
                0xF0, 0x48, 0x0F, 0xB1, 0xD8, // lock cmpxchg rax, rbx
                0x49, 0x01, 0x47, 0x08, // add [r15+8], rax
            ],
        },
        {
            name: "SSE2: movdqu/movdqa/pxor round-trip",
            body: [
                0xF3, 0x41, 0x0F, 0x6F, 0x07, // movdqu xmm0, [r15]
                0xF3, 0x41, 0x0F, 0x6F, 0x4F, 0x10, // movdqu xmm1, [r15+16]
                0x66, 0x0F, 0xEF, 0xC1, // pxor xmm0, xmm1
                0xF3, 0x41, 0x0F, 0x7F, 0x07, // movdqu [r15], xmm0
                0x66, 0x0F, 0x6F, 0xD0, // movdqa xmm2, xmm0
                0x66, 0x41, 0x0F, 0xEF, 0x57, 0x10, // pxor xmm2, [r15+16]
                0x66, 0x41, 0x0F, 0x7F, 0x57, 0x10, // movdqa [r15+16], xmm2
            ],
        },
        {
            name: "SSE2: xmm8-15 (REX.R/B) round-trip",
            body: [
                0xF3, 0x44, 0x0F, 0x6F, 0x07, // movdqu xmm8, [r15]
                0xF3, 0x45, 0x0F, 0x6F, 0x4F, 0x10, // movdqu xmm9, [r15+16]
                0x66, 0x45, 0x0F, 0xEF, 0xC8, // pxor xmm9, xmm8
                0x66, 0x45, 0x0F, 0x6F, 0xD1, // movdqa xmm10, xmm9
                0xF3, 0x44, 0x0F, 0x7F, 0x07, // movdqu [r15], xmm8
                0xF3, 0x45, 0x0F, 0x7F, 0x57, 0x10, // movdqu [r15+16], xmm10
            ],
        },
        {
            name: "cmpxchg ebx,eax + setcc + cmovne",
            body: [
                0x0F, 0xB1, 0xC3, // cmpxchg ebx, eax
                0x0F, 0x94, 0xC0, // sete al
                0x48, 0x0F, 0x45, 0xCA, // cmovne rcx, rdx
            ],
        },
    ];

    const seed_regs = random => {
        const regs = new Array(16).fill(0n);
        for(const i of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14])
        {
            regs[i] = BigInt(Math.floor(random() * 2 ** 32)) | BigInt(Math.floor(random() * 2 ** 32)) << 32n;
        }
        return regs;
    };

    const run = (program, regs, memory, jitEnabled) => {
        ex.jit64_set_enabled(jitEnabled ? 1 : 0);
        ex.jit64_set_sse(1); // exercises the SSE2 codegen
        ex.jit64_clear_cache();
        for(let i = 0; i < program.length; i++) ex.write8(BASE + i, program[i]);
        for(let i = 0; i < 16; i++) set_reg64(i, regs[i]);
        set_reg64(12, 0n);
        set_reg64(15, BigInt(SCRATCH));
        set_reg64(4, 0x80000n); // rsp
        for(let i = 0; i < memory.length; i++) ex.write8(SCRATCH + i, memory[i]);

        cpu.instruction_pointer[0] = BASE;
        u32[16 + 4] = 0x80000;
        u32[32 + 4] = 0;
        cpu.flags[0] = 0x2;
        cpu.in_hlt[0] = 0;
        ex.enter_long_mode(PML4);

        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 20000)
        {
            ex.main_loop();
        }
        return {
            halted: cpu.in_hlt[0] === 1,
            regs: Array.from({ length: 16 }, (_, i) => reg64(i)),
            flags: cpu.flags[0] >>> 0,
            memory: Array.from({ length: memory.length }, (_, i) => ex.read8(SCRATCH + i)),
        };
    };

    // Run exactly n instructions with the given engine, for bisecting a
    // mismatch down to the instruction that first diverges.
    const step = (program, regs, memory, jitEnabled, n) => {
        ex.jit64_set_enabled(jitEnabled ? 1 : 0);
        ex.jit64_set_sse(1); // exercises the SSE2 codegen
        ex.jit64_clear_cache();
        for(let i = 0; i < program.length; i++) ex.write8(BASE + i, program[i]);
        for(let i = 0; i < 16; i++) set_reg64(i, regs[i]);
        set_reg64(12, 0n);
        set_reg64(15, BigInt(SCRATCH));
        for(let i = 0; i < memory.length; i++) ex.write8(SCRATCH + i, memory[i]);
        cpu.instruction_pointer[0] = BASE;
        cpu.flags[0] = 0x2;
        cpu.in_hlt[0] = 0;
        ex.enter_long_mode(PML4);
        const before = cpu.instruction_pointer[0] >>> 0;
        ex.run_exact_instructions(n);
        return {
            rip: cpu.instruction_pointer[0] >>> 0,
            before,
            regs: Array.from({ length: 16 }, (_, i) => reg64(i)),
            flags: cpu.flags[0] >>> 0,
            memory: Array.from({ length: memory.length }, (_, i) => ex.read8(SCRATCH + i)),
        };
    };

    // Find the smallest loop count for which the two engines disagree. The
    // count is a property of the program itself, so this is engine
    // independent, unlike counting instructions (the JIT counts per block).
    const localize = (seed, same_final) =>
    {
        for(let k = 1; k <= ITERATIONS; k++)
        {
            const random = rng(seed);
            const program = build_program(random, k);
            const regs = seed_regs(random);
            const memory = Array.from({ length: 32 }, () => Math.floor(random() * 256));
            const interpreted = run(program, regs, memory, false);
            const compiled = run(program, regs, memory, true);
            const problems = [];
            if(!interpreted.halted) problems.push("interp did not halt");
            if(!compiled.halted) problems.push("jit did not halt");
            if(problems.length === 0)
            {
                for(let i = 0; i < 16; i++)
                {
                    // r12 is the loop counter, exclude it from the comparison
                    if(i !== 12 && interpreted.regs[i] !== compiled.regs[i])
                    {
                        problems.push("r" + i + ": interp 0x" + interpreted.regs[i].toString(16) +
                            " vs jit 0x" + compiled.regs[i].toString(16));
                    }
                }
                if(interpreted.flags !== compiled.flags)
                {
                    problems.push("flags: interp 0x" + interpreted.flags.toString(16) +
                        " vs jit 0x" + compiled.flags.toString(16));
                }
                for(let i = 0; i < memory.length; i++)
                {
                    if(interpreted.memory[i] !== compiled.memory[i])
                    {
                        problems.push("mem+" + i);
                    }
                }
            }
            if(problems.length)
            {
                console.log("  smallest diverging loop count = " + k + ": " + problems.join(", "));
                // shrink the body: the smallest prefix of the body that still
                // diverges ends with the offending instruction
                const bodyCount = 12 + Math.floor(rng(seed)() * 13);
                let last = bodyCount;
                for(let m = 1; m <= bodyCount; m++)
                {
                    const r3 = rng(seed);
                    const p3 = build_program(r3, ITERATIONS, m);
                    const regs3 = seed_regs(r3);
                    const mem3 = Array.from({ length: 32 }, () => Math.floor(r3() * 256));
                    const a3 = run(p3, regs3, mem3, false);
                    const b3 = run(p3, regs3, mem3, true);
                    const differ = !a3.halted || !b3.halted ||
                        a3.flags !== b3.flags ||
                        a3.regs.some((v, i) => i !== 12 && v !== b3.regs[i]) ||
                        a3.memory.some((v, i) => v !== b3.memory[i]);
                    if(differ)
                    {
                        last = m;
                        console.log("    minimal diverging body = first " + m + " instruction(s)" +
                            " of " + bodyCount);
                        console.log("      bytes: [" + p3.slice(6, 6 + 40).join(", ") + "]");
                        break;
                    }
                }
                return k;
            }
        }
        console.log("  (no diverging loop count up to " + ITERATIONS + ")");
        return -1;
    };

    let fixed_failures = 0;
    for(const test of FIXED)
    {
        const program = wrap_body(test.body);
        const random = rng(12345);
        const regs = seed_regs(random);
        const memory = Array.from({ length: 32 }, () => Math.floor(random() * 256));
        const interpreted = run(program, regs, memory, false);
        const compiled = run(program, regs, memory, true);
        const differ = !interpreted.halted || !compiled.halted ||
            interpreted.flags !== compiled.flags ||
            interpreted.regs.some((v, i) => v !== compiled.regs[i]) ||
            interpreted.memory.some((v, i) => v !== compiled.memory[i]);
        if(differ)
        {
            fixed_failures++;
            console.log("FAIL fixed: " + test.name);
            for(let i = 0; i < 16; i++)
            {
                if(interpreted.regs[i] !== compiled.regs[i])
                {
                    console.log("    r" + i + ": interp 0x" + interpreted.regs[i].toString(16) +
                        " vs jit 0x" + compiled.regs[i].toString(16));
                }
            }
            console.log("    flags: interp 0x" + interpreted.flags.toString(16) +
                " vs jit 0x" + compiled.flags.toString(16));
            // shrink: which prefix of the sequence already diverges?
            const lens = [4, 8, 12, 16, 20];
            for(const len of lens)
            {
                if(len > test.body.length)
                {
                    continue;
                }
                const p2 = wrap_body(test.body.slice(0, len));
                const a2 = run(p2, regs, memory, false);
                const b2 = run(p2, regs, memory, true);
                const d2 = !a2.halted || !b2.halted ||
                    a2.regs.some((v, i) => v !== b2.regs[i]) ||
                    a2.memory.some((v, i) => v !== b2.memory[i]);
                console.log("    first " + len + " byte(s): " + (d2 ? "DIVERGES" : "ok"));
                if(d2)
                {
                    for(let i = 0; i < 16; i++)
                    {
                        if(a2.regs[i] !== b2.regs[i])
                        {
                            console.log("      r" + i + ": interp 0x" + a2.regs[i].toString(16) +
                                " vs jit 0x" + b2.regs[i].toString(16));
                        }
                    }
                    break;
                }
            }
            for(let i = 0; i < memory.length; i++)
            {
                if(interpreted.memory[i] !== compiled.memory[i])
                {
                    console.log("    mem+" + i + ": interp " + interpreted.memory[i] + " vs jit " + compiled.memory[i]);
                }
            }
        }
    }

    let compiled_any = false;
    let mismatches = 0;
    for(let seed = 1; seed <= CASES; seed++)
    {
        const random = rng(seed);
        const program = build_program(random);
        const regs = seed_regs(random);
        const memory = Array.from({ length: 32 }, () => Math.floor(random() * 256));

        const interpreted = run(program, regs, memory, false);
        const compiled = run(program, regs, memory, true);
        if(ex.jit64_compiled_count() > 0) compiled_any = true;

        const problems = [];
        if(!interpreted.halted) problems.push("interpreter did not halt");
        if(!compiled.halted) problems.push("JIT did not halt");
        for(let i = 0; i < 16; i++)
        {
            if(interpreted.regs[i] !== compiled.regs[i])
            {
                problems.push("r" + i + ": interp 0x" + interpreted.regs[i].toString(16) +
                    " vs jit 0x" + compiled.regs[i].toString(16));
            }
        }
        if((interpreted.flags & FLAG_MASK) !== (compiled.flags & FLAG_MASK))
        {
            problems.push("flags: interp 0x" + interpreted.flags.toString(16) +
                " vs jit 0x" + compiled.flags.toString(16));
        }
        for(let i = 0; i < memory.length; i++)
        {
            if(interpreted.memory[i] !== compiled.memory[i])
            {
                problems.push("mem+" + i + ": interp " + interpreted.memory[i] + " vs jit " + compiled.memory[i]);
            }
        }

        if(problems.length)
        {
            mismatches++;
            console.log("FAIL seed " + seed + ": " + problems[0]);
            for(const problem of problems.slice(1)) console.log("     " + problem);
            console.log("     program: [" + program.join(", ") + "]");
            localize(seed);
        }
    }

    if(!compiled_any)
    {
        console.log("FAIL no program was JIT-compiled");
        process.exit(1);
    }
    if(mismatches)
    {
        console.log("jit64 differential: " + mismatches + " mismatching program(s)");
        process.exit(1);
    }

    console.log("jit64 differential: " + CASES + " programs matched");
    process.exit(0);
});
