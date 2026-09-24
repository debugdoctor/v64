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
    [0x49, 0x89, 0x47, 0x08], // mov [r15+8], rax
    [0x49, 0x8B, 0x5F, 0x08], // mov rbx, [r15+8]
    [0x49, 0x01, 0x47, 0x10], // add qword [r15+16], rax
    [0x41, 0x89, 0x4F, 0x18], // mov [r15+24], ecx
    [0x41, 0x8B, 0x57, 0x18], // mov edx, [r15+24]
];

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

    const setReg64 = (i, value) => {
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

    const buildProgram = random => {
        const body = [];
        const count = 12 + Math.floor(random() * 13);
        for(let i = 0; i < count; i++)
        {
            body.push(...POOL[Math.floor(random() * POOL.length)]);
        }
        const prefix = [0x41, 0xBC, ITERATIONS & 0xFF, ITERATIONS >> 8 & 0xFF, ITERATIONS >> 16 & 0xFF, ITERATIONS >> 24 & 0xFF];
        const jnzAt = prefix.length + body.length + 3; // + dec r12d
        const rel = prefix.length - (jnzAt + 6);
        const suffix = [
            0x41, 0xFF, 0xCC, // dec r12d
            0x0F, 0x85, rel & 0xFF, rel >> 8 & 0xFF, rel >> 16 & 0xFF, rel >> 24 & 0xFF,
            0xF4, // hlt
        ];
        return prefix.concat(body, suffix);
    };

    const seedRegs = random => {
        const regs = new Array(16).fill(0n);
        for(const i of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14])
        {
            regs[i] = BigInt(Math.floor(random() * 2 ** 32)) | BigInt(Math.floor(random() * 2 ** 32)) << 32n;
        }
        return regs;
    };

    const run = (program, regs, memory, jitEnabled) => {
        ex.jit64_set_enabled(jitEnabled ? 1 : 0);
        ex.jit64_clear_cache();
        for(let i = 0; i < program.length; i++) ex.write8(BASE + i, program[i]);
        for(let i = 0; i < 16; i++) setReg64(i, regs[i]);
        setReg64(12, 0n);
        setReg64(15, BigInt(SCRATCH));
        setReg64(4, 0x80000n); // rsp
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

    let compiledAny = false;
    let mismatches = 0;
    for(let seed = 1; seed <= CASES; seed++)
    {
        const random = rng(seed);
        const program = buildProgram(random);
        const regs = seedRegs(random);
        const memory = Array.from({ length: 32 }, () => Math.floor(random() * 256));

        const interpreted = run(program, regs, memory, false);
        const compiled = run(program, regs, memory, true);
        if(ex.jit64_compiled_count() > 0) compiledAny = true;

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
        }
    }

    if(!compiledAny)
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
