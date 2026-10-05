#!/usr/bin/env node
// Compares the 64-bit interpreter (interp64.rs, long mode) against the 32-bit
// interpreter (gen/interpreter.rs, protected mode) on identical guest work.
// JIT is disabled for both, so this measures the interpreter floor only.
//
//   node tests/benchmark/interp-bitwidth.mjs [iterations]
//
// The 32-bit side enters protected mode the way firmware does (CR0.PE plus a
// far jump through a flat GDT, see tests/e2e/modes.js); setting CR0.PE=0 would
// leave the CPU in real mode, where the 32-bit opcodes below decode as 16-bit
// and the instruction stream desyncs.

import { writeFileSync as write_file_sync } from "node:fs";
import { v64 } from "../../src/main.js";

const ITERATIONS = +process.argv[2] || 500000;
const CODE = 0x1000;
const SCRATCH = 0x100000;
const RESET = 0xFFFF0;
const PROT = 0x2000;
const GDT = 0x1400;

const imm32 = v => [v & 0xFF, v >> 8 & 0xFF, v >> 16 & 0xFF, v >> 24 & 0xFF];

// mov eax,1 / mov ebx,3 / mov ecx,N / loop: 3 alu ops / sub ecx,1 / jnz / hlt
const alu_program = n => [
    0xB8, ...imm32(1),
    0xBB, ...imm32(3),
    0xB9, ...imm32(n),
    0x01, 0xD8,       // add eax, ebx
    0x31, 0xC3,       // xor ebx, eax
    0xD1, 0xE3,       // shl ebx, 1
    0x83, 0xE9, 0x01, // sub ecx, 1
    0x75, 0xF5,       // jnz loop (back to offset 0x0F)
    0xF4,
];

// mov ebx,SCRATCH / loop: mov eax,[ebx]; add ebx,8; and ebx,0xFF8 / sub ecx,1; jnz
const load_program = n => [
    0xBB, ...imm32(SCRATCH),
    0xB9, ...imm32(n),
    0x8B, 0x03,
    0x83, 0xC3, 0x08,
    0x81, 0xE3, 0xF8, 0x0F, 0x00, 0x00,
    0x83, 0xE9, 0x01,
    0x75, 0xF0,       // jnz loop (back to offset 0x0A)
    0xF4,
];

const store_program = n => [
    0xBB, ...imm32(SCRATCH),
    0xB9, ...imm32(n),
    0x89, 0x03,
    0x83, 0xC3, 0x08,
    0x81, 0xE3, 0xF8, 0x0F, 0x00, 0x00,
    0x83, 0xE9, 0x01,
    0x75, 0xF0,       // jnz loop (back to offset 0x0A)
    0xF4,
];

const CASES = [
    { name: "alu", note: "3 int ops + loop branch", program: alu_program },
    { name: "load", note: "load in one 4 KiB page", program: load_program },
    { name: "store", note: "store in one 4 KiB page", program: store_program },
];

// Flat GDT: null, 32-bit code (0x08), 32-bit data (0x10).
const GDT_BYTES = [
    0, 0, 0, 0, 0, 0, 0, 0,
    0xFF, 0xFF, 0, 0, 0, 0x9A, 0xCF, 0,
    0xFF, 0xFF, 0, 0, 0, 0x92, 0xCF, 0,
];

// Real-mode trampoline: set CR0.PE and far-jump into the 32-bit code segment.
// The prefix on the CR0 write and the far jump is how firmware leaves real mode.
const REAL_STUB = [
    0xFA,                               // cli
    0x66, 0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, CR0.PE
    0x0F, 0x22, 0xC0,                   // mov cr0, eax
    0x66, 0xEA, ...imm32(PROT), 0x08, 0x00, // jmp far 0008:2000
];

// Load flat data segments and a stack, then fall through into the workload.
const PROT_PREFIX = [
    0x66, 0xB8, 0x10, 0x00,             // mov ax, 0x10
    0x8E, 0xD8,                         // mov ds, ax
    0x8E, 0xC0,                         // mov es, ax
    0x8E, 0xD0,                         // mov ss, ax
    0xBC, 0x00, 0x80, 0x00, 0x00,       // mov esp, 0x8000
];

const emulator = new v64({
    autostart: false,
    memory_size: (+process.env.MEM_MB || 4) * 1024 * 1024,
    disable_jit: true,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || "build/v64-debug.wasm",
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const u32 = () => new Uint32Array(ex.memory.buffer);

    const write = (address, bytes) => {
        for(let i = 0; i < bytes.length; i++) ex.write8(address + i, bytes[i]);
    };

    // Put the CPU back where firmware starts: real mode, CS base 0xF0000, IP at
    // the reset vector. Every run has to do this, because a previous 64-bit run
    // leaves long mode and flat segments behind.
    const reset_to_reset_vector = () => {
        const seg = new Int32Array(ex.memory.buffer, 736, 8);
        seg.fill(0);
        seg[1] = 0xF0000;              // CS base in real mode
        cpu.is_32[0] = 0;
        new Uint8Array(ex.memory.buffer)[225] = 0; // long_mode
        cpu.cr[0] = 0x60000010;        // reset value, PE/PG clear
        cpu.cpl[0] = 0;
        cpu.flags[0] = 2;
        cpu.in_hlt[0] = 0;
        cpu.instruction_pointer[0] = RESET;
    };

    // Identity-map the first 1 GiB (4 KiB PT for the first 2 MiB, 2 MiB above).
    const map_identity = () => {
        const PML4 = 0x30000, PDPT = 0x31000, PD = 0x32000, PT = 0x33000;
        const w64 = (a, v) => { ex.write32(a, Number(v & 0xFFFFFFFFn)); ex.write32(a + 4, Number(v >> 32n)); };
        w64(PML4, BigInt(PDPT) | 3n);
        w64(PDPT, BigInt(PD) | 3n);
        w64(PD, BigInt(PT) | 3n);
        for(let i = 0; i < 512; i++) w64(PT + i * 8, BigInt(i) * 0x1000n | 3n);
        for(let i = 1; i < 512; i++) w64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);
        return PML4;
    };

    const run = (program, is64) => {
        reset_to_reset_vector();
        u32()[16 + 4] = 0x8000; // esp

        if(is64) {
            // Same shape as tests/benchmark/cpu64.js: place the program at CODE,
            // point RIP at it, then force the long-mode transition.
            write(CODE, program);
            cpu.instruction_pointer[0] = CODE;
            ex.enter_long_mode(map_identity());
        }
        else {
            // Real reset vector walks into protected mode; the program sits right
            // after the flat-segment setup prefix.
            write(GDT, GDT_BYTES);
            cpu.gdtr_size[0] = GDT_BYTES.length - 1;
            cpu.gdtr_offset[0] = GDT;
            write(RESET, REAL_STUB);
            write(PROT, [...PROT_PREFIX, ...program]);
        }

        const compiled_before = ex.jit64_compiled_count();
        const before = u32()[166];
        const start = process.hrtime.bigint();
        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 500000000) ex.main_loop();
        const ms = Number(process.hrtime.bigint() - start) / 1e6;
        const compiled = ex.jit64_compiled_count() - compiled_before;
        if(compiled !== 0) throw new Error("jit64 compiled " + compiled + " block(s); not interpreter-only");
        return { ms, mips: ((u32()[166] - before) >>> 0) / ms / 1000, halted: cpu.in_hlt[0] === 1 };
    };

    run(alu_program(1000), false); // warm up module instantiation
    run(alu_program(1000), true);  // and the long-mode interpreter path

    console.log("interpreter floor, " + ITERATIONS + " iterations per workload");
    console.log("  " + "workload".padEnd(10) + "32-bit M/s".padStart(12) + "64-bit M/s".padStart(12) + "64/32".padStart(9));
    const rows = [];
    for(const c of CASES) {
        const program = c.program(ITERATIONS);
        const a = run(program, false);
        const b = run(program, true);
        if(!a.halted) throw new Error(c.name + ": 32-bit run did not halt");
        if(!b.halted) throw new Error(c.name + ": 64-bit run did not halt");
        rows.push({ name: c.name, m32: +a.mips.toFixed(1), m64: +b.mips.toFixed(1), ratio: +(b.mips / a.mips).toFixed(2) });
        console.log("  " + c.name.padEnd(10) + a.mips.toFixed(1).padStart(12) + b.mips.toFixed(1).padStart(12) + (b.mips / a.mips).toFixed(2).padStart(9));
    }
    if(process.env.OUT) {
        write_file_sync(process.env.OUT, JSON.stringify({ iterations: ITERATIONS, rows }, null, 2) + "\n");
    }
    process.exit(0);
});
