#!/usr/bin/env node

// Reproduces the fill loop the kernel decompressor hits around 0x35c0cc0.
//
//   mov r12d, 1
//   mov [r10], cx
//   cmp r12, r14
//   je  exit
// loop:
//   mov [r10+r12*2], cx
//   mov [r10+r12*2+2], cx
//   add r12, 2
//   cmp r12, r14
//   jne loop
// exit:
//
// r12 starts at 1 and advances by 2, so r14 must be odd for the loop to end
// exactly when `cmp r12, r14` sees equality. The JIT was observed re-entering
// the loop with r12 == r14 in the real decompressor, which never terminates.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/jit64/fillloop.js`

import { v64 } from "../../src/main.js";
import { set_cpu_config, JIT_DISABLE_64 } from "../../src/config.js";

const BASE = 0x1000;
const SCRATCH = 0x100000;
const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const PT = 0x13000;
const ITERATIONS = 800;

const FLAG_MASK = (1 << 0) | (1 << 2) | (1 << 6) | (1 << 7);

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
    for(let i = 0; i < 512; i++) write64(PT + i * 8, BigInt(i) * 0x1000n | 0x3n);
    for(let i = 1; i < 512; i++) write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);

    const set_reg64 = (i, value) => {
        if(i < 8) { u32[16 + i] = Number(value & 0xFFFF_FFFFn); u32[32 + i] = Number(value >> 32n & 0xFFFF_FFFFn); }
        else ext[i - 8] = value;
    };
    const reg64 = i => (i < 8)
        ? (BigInt(u32[32 + i]) << 32n) | BigInt(u32[16 + i])
        : ext[i - 8];

    const imm32 = v => [v & 0xFF, v >> 8 & 0xFF, v >> 16 & 0xFF, v >> 24 & 0xFF];
    const imm64 = v => { const o = []; for(let i = 0; i < 8; i++) o.push(Number(v >> BigInt(8 * i) & 0xFFn)); return o; };

    // Built with absolute offsets; jumps are patched from the layout below.
    const build_program = () => {
        const bytes = [];
        const put = (...b) => bytes.push(...b);
        put(0x41, 0xBB, ...imm32(ITERATIONS));      // 0x00 mov r11d, ITERATIONS
        put(0x49, 0xBA, ...imm64(BigInt(SCRATCH))); // 0x06 mov r10, SCRATCH
        put(0x49, 0xBE, ...imm64(0n));              // 0x10 mov r14, 0      (patched)
        put(0x66, 0xB9, 0x34, 0x12);                // 0x1A mov cx, 0x1234  (patched)
        // 0x1E outer:
        put(0x41, 0xBC, 0x01, 0x00, 0x00, 0x00);    // 0x1E mov r12d, 1
        put(0x66, 0x41, 0x89, 0x0A);                // 0x24 mov [r10], cx
        put(0x4D, 0x39, 0xF4);                      // 0x28 cmp r12, r14
        put(0x74, 0x14);                            // 0x2B je exit (0x41)
        // 0x2D loop:
        put(0x66, 0x43, 0x89, 0x0C, 0x62);          // 0x2D mov [r10+r12*2], cx
        put(0x66, 0x43, 0x89, 0x4C, 0x62, 0x02);    // 0x32 mov [r10+r12*2+2], cx
        put(0x49, 0x83, 0xC4, 0x02);                // 0x38 add r12, 2
        put(0x4D, 0x39, 0xF4);                      // 0x3C cmp r12, r14
        put(0x75, 0xEC);                            // 0x3F jne loop (0x2D)
        // 0x41 exit:
        put(0x41, 0xFF, 0xCB);                      // 0x41 dec r11d
        put(0x0F, 0x85, 0xD4, 0xFF, 0xFF, 0xFF);    // 0x44 jnz outer (0x1E)
        put(0xF4);                                  // 0x4A hlt
        return bytes;
    };

    const run = (program, r14, cx, jitEnabled) => {
        set_cpu_config(ex, "JIT_DISABLE", jitEnabled ? 0 : JIT_DISABLE_64);
        ex.jit64_clear_cache();
        const bytes = program.slice();
        // patch r14 (imm64 at 0x12) and cx (imm16 at 0x1C)
        for(let i = 0; i < 8; i++) bytes[0x12 + i] = Number(r14 >> BigInt(8 * i) & 0xFFn);
        bytes[0x1C] = cx & 0xFF;
        bytes[0x1D] = cx >> 8 & 0xFF;
        for(let i = 0; i < bytes.length; i++) ex.write8(BASE + i, bytes[i]);
        for(let i = 0; i < 16; i++) set_reg64(i, 0n);
        set_reg64(4, 0x80000n);
        cpu.instruction_pointer[0] = BASE;
        u32[16 + 4] = 0x80000;
        u32[32 + 4] = 0;
        cpu.flags[0] = 0x2;
        cpu.in_hlt[0] = 0;
        ex.enter_long_mode(PML4);
        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 2000) ex.main_loop();
        return {
            halted: cpu.in_hlt[0] === 1,
            regs: Array.from({ length: 16 }, (_, i) => reg64(i)),
            flags: cpu.flags[0] >>> 0,
            memory: Array.from({ length: 32 }, (_, i) => ex.read8(SCRATCH + i)),
        };
    };

    const program = build_program();
    let compiled_any = false;
    let mismatches = 0;
    const cases = [];
    for(const r14 of [1n, 3n, 9n, 0x7Fn, 0x101n, 0x103n])
    {
        for(const cx of [0x1234, 0xABCD, 0x0066]) cases.push({ r14, cx });
    }

    for(const { r14, cx } of cases)
    {
        const interpreted = run(program, r14, cx, false);
        const compiled = run(program, r14, cx, true);
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
            problems.push("flags: interp 0x" + interpreted.flags.toString(16) + " vs jit 0x" + compiled.flags.toString(16));
        }
        for(let i = 0; i < 32; i++)
        {
            if(interpreted.memory[i] !== compiled.memory[i])
            {
                problems.push("mem+" + i + ": interp " + interpreted.memory[i] + " vs jit " + compiled.memory[i]);
                break;
            }
        }

        if(problems.length)
        {
            mismatches++;
            console.log("FAIL r14=0x" + r14.toString(16) + " cx=0x" + cx.toString(16) + ": " + problems[0]);
            for(const p of problems.slice(1, 4)) console.log("     " + p);
        }
    }

    if(!compiled_any)
    {
        console.log("FAIL no program was JIT-compiled");
        process.exit(1);
    }
    if(mismatches)
    {
        console.log("jit64 fillloop: " + mismatches + " mismatching case(s)");
        process.exit(1);
    }
    console.log("jit64 fillloop: " + cases.length + " cases matched");
    process.exit(0);
});
