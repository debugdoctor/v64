#!/usr/bin/env node

// Control-flow differential fuzzing of jit64 against the interpreter.
//
// differential.js only runs straight-line programs, so a block-level bug in
// branch handling stays invisible there. This generator emits forward `jcc`/
// `jmp` (rel8 and rel32) and backward branches to a counter-guarded loop head,
// then compares the final rip, registers, flags and scratch page of the
// interpreted and compiled runs.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/jit64/controlflow.js`

import { v64 } from "../../src/main.js";
import { set_cpu_config, JIT_DISABLE_64 } from "../../src/config.js";

const BASE = 0x1000;
const SCRATCH = 0x100000;
const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const PT = 0x13000;
const ITERATIONS = 800;
const CASES = +process.env.JIT64_CF_CASES || 300;

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
    [0x41, 0x89, 0xD6], // mov r14d, edx
    [0x41, 0xD1, 0xEE], // shr r14d, 1
    [0x41, 0x83, 0xFC, 0x02], // cmp r12d, 2
    [0x0F, 0x94, 0xC0], // sete al
    [0x0F, 0x95, 0xC3], // setne bl
    [0x00, 0xD8], // add al, bl
    [0x01, 0xD8], // add eax, ebx
    [0x45, 0x31, 0xC8], // xor r8d, r9d
    [0x4D, 0x89, 0xDA], // mov r10, r11
    [0x49, 0x89, 0x47, 0x08], // mov [r15+8], rax
    [0x49, 0x01, 0x47, 0x10], // add qword [r15+16], rax
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

    // Returns { program, targets } where targets lists the byte offsets that
    // must produce identical control flow.
    // Layout (offsets are absolute):
    //   0:  mov r12d, ITERATIONS      (6 bytes)
    //   6:  dec r12d                  (3 bytes)  <- the loop head every backward
    //   9:  jz end                    (6 bytes)     branch targets, so it always
    //  15:  <body>                                decrements and terminates
    //       jmp 6    (5 bytes)
    //       hlt      (1 byte)
    const HEAD = 6;
    const BODY_START = 15;
    const build_program = random => {
        const prefix = [0x41, 0xBC, ITERATIONS & 0xFF, ITERATIONS >> 8 & 0xFF, ITERATIONS >> 16 & 0xFF, ITERATIONS >> 24 & 0xFF, 0x41, 0xFF, 0xCC];

        const count = 8 + Math.floor(random() * 9);
        const steps = [];
        for(let i = 0; i < count; i++)
        {
            if(random() < 0.3)
            {
                const rel32 = random() < 0.5;
                steps.push({ branch: true, cc: Math.floor(random() * 16), rel32, len: rel32 ? 6 : 2 });
            }
            else
            {
                const bytes = POOL[Math.floor(random() * POOL.length)];
                steps.push({ bytes, len: bytes.length });
            }
        }

        const off = [];
        let cursor = BODY_START;
        for(const step of steps)
        {
            off.push(cursor);
            cursor += step.len;
        }
        const tail = cursor;
        const end = tail + 5;

        const body = [];
        for(let i = 0; i < steps.length; i++)
        {
            const step = steps[i];
            if(!step.branch)
            {
                body.push(...step.bytes);
                continue;
            }
            const after = off[i] + step.len;
            const candidates = [HEAD, tail, end];
            for(let j = i + 1; j < steps.length; j++) candidates.push(off[j]);
            // Keep only targets that fit the chosen displacement.
            const max = step.rel32 ? 0x7FFF_FFFF : 127;
            const min = step.rel32 ? -0x8000_0000 : -128;
            const choices = candidates.filter(t => t - after >= min && t - after <= max);
            const target = choices[Math.floor(random() * choices.length)];
            const rel = target - after;
            if(step.rel32)
            {
                body.push(0x0F, 0x80 + step.cc,
                    rel & 0xFF, rel >> 8 & 0xFF, rel >> 16 & 0xFF, rel >> 24 & 0xFF);
            }
            else
            {
                body.push(0x70 + step.cc, rel & 0xFF);
            }
        }

        const jz_rel = end - 15;
        const head = [0x0F, 0x84, jz_rel & 0xFF, jz_rel >> 8 & 0xFF, jz_rel >> 16 & 0xFF, jz_rel >> 24 & 0xFF];
        const tail_jmp_rel = HEAD - (tail + 5);
        const suffix = [
            0xE9, tail_jmp_rel & 0xFF, tail_jmp_rel >> 8 & 0xFF, tail_jmp_rel >> 16 & 0xFF, tail_jmp_rel >> 24 & 0xFF,
            0xF4, // hlt
        ];
        return prefix.concat(head, body, suffix);
    };

    const seed_regs = random => {
        const regs = new Array(16).fill(0n);
        for(const i of [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14])
        {
            regs[i] = BigInt(Math.floor(random() * 2 ** 32)) | BigInt(Math.floor(random() * 2 ** 32)) << 32n;
        }
        return regs;
    };

    const run = (program, regs, memory, jitEnabled) => {
        set_cpu_config(ex, "JIT_DISABLE", jitEnabled ? 0 : JIT_DISABLE_64);
        ex.jit64_clear_cache();
        for(let i = 0; i < program.length; i++) ex.write8(BASE + i, program[i]);
        for(let i = 0; i < 16; i++) set_reg64(i, regs[i]);
        set_reg64(12, 0n);
        set_reg64(15, BigInt(SCRATCH));
        set_reg64(4, 0x80000n);
        for(let i = 0; i < memory.length; i++) ex.write8(SCRATCH + i, memory[i]);

        cpu.instruction_pointer[0] = BASE;
        u32[16 + 4] = 0x80000;
        u32[32 + 4] = 0;
        cpu.flags[0] = 0x2;
        cpu.in_hlt[0] = 0;
        ex.enter_long_mode(PML4);

        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 400)
        {
            ex.main_loop();
        }
        return {
            halted: cpu.in_hlt[0] === 1,
            rip: cpu.instruction_pointer[0] >>> 0,
            regs: Array.from({ length: 16 }, (_, i) => reg64(i)),
            flags: cpu.flags[0] >>> 0,
            memory: Array.from({ length: memory.length }, (_, i) => ex.read8(SCRATCH + i)),
        };
    };

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
        if(interpreted.rip !== compiled.rip)
        {
            problems.push("rip: interp 0x" + interpreted.rip.toString(16) + " vs jit 0x" + compiled.rip.toString(16));
        }
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
            if(mismatches <= 5)
            {
                console.log("FAIL seed " + seed + ": " + problems[0]);
                for(const problem of problems.slice(1, 3)) console.log("     " + problem);
                console.log("     program: [" + program.join(", ") + "]");
                console.log("     regs: [" + regs.map(r => "0x" + r.toString(16)).join(", ") + "]");
                console.log("     mem: [" + memory.join(", ") + "]");
            }
        }
    }

    if(!compiled_any)
    {
        console.log("FAIL no program was JIT-compiled");
        process.exit(1);
    }
    if(mismatches)
    {
        console.log("jit64 controlflow: " + mismatches + " mismatching program(s)");
        process.exit(1);
    }

    console.log("jit64 controlflow: " + CASES + " programs matched");
    process.exit(0);
});
