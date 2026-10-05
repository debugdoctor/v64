#!/usr/bin/env node

// Memory-operand differential fuzzing of jit64 against the interpreter.
//
// differential.js only exercises `[r15+disp]`. Real kernel code (the bzImage
// decompressor) uses SIB forms such as `[r8+rcx*2]`, negative displacements
// like `[r10-1]` and 16-bit operand widths. Each candidate below is run in a
// counter-guarded loop with the base registers pinned to the scratch page, and
// the interpreted and compiled final states must match.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/jit64/addressing.js`

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

// Encodings taken from the kernel decompressor's hot loops plus nearby forms.
const CANDIDATES = [
    ["mov [r10-1], cl", [0x41, 0x88, 0x4A, 0xFF]],
    ["movzx r13d, word [r8+rcx*2]", [0x45, 0x0F, 0xB7, 0x2C, 0x48]],
    ["mov [r10+rcx*2], r13w", [0x66, 0x45, 0x89, 0x2C, 0x4A]],
    ["mov [r10+rcx*2], ecx", [0x41, 0x89, 0x0C, 0x4A]],
    ["mov ecx, [r8+rcx*2]", [0x41, 0x8B, 0x0C, 0x48]],
    ["mov rcx, [r8+rcx*2]", [0x49, 0x8B, 0x0C, 0x48]],
    ["movzx ecx, word [r8+rcx*2]", [0x41, 0x0F, 0xB7, 0x0C, 0x48]],
    ["movzx ecx, byte [r8]", [0x41, 0x0F, 0xB6, 0x08]],
    ["movzx eax, byte [rbx+rcx]", [0x0F, 0xB6, 0x04, 0x0B]],
    ["movzx eax, word [rbx+rcx*2+4]", [0x0F, 0xB7, 0x44, 0x4B, 0x04]],
    ["mov [rbx+rcx*2+2], cx", [0x66, 0x89, 0x4C, 0x4B, 0x02]],
    ["mov al, [rbx+rcx+8]", [0x8A, 0x44, 0x0B, 0x08]],
    ["add [r10+rcx*2], ecx", [0x41, 0x01, 0x0C, 0x4A]],
    ["mov [r10+rcx*4-2], ecx", [0x41, 0x89, 0x4C, 0x8A, 0xFE]],
    ["mov ecx, [rsp+0x18]", [0x8B, 0x4C, 0x24, 0x18]],
    ["and ecx, eax", [0x21, 0xC1]],
    ["shl r12d, cl", [0x41, 0xD3, 0xE4]],
    ["shr rax, cl", [0x48, 0xD3, 0xE8]],
    ["not r12d", [0x41, 0xF7, 0xD4]],
    ["and r12d, eax", [0x41, 0x21, 0xC4]],
    ["add edx, r12d", [0x44, 0x01, 0xE2]],
    ["sub esi, r14d", [0x44, 0x29, 0xF6]],
    ["cmp esi, 0xe", [0x83, 0xFE, 0x0E]],
    ["mov r12d, [rsp+0x18]", [0x44, 0x8B, 0x64, 0x24, 0x18]],
    // SIB index field 100 with REX.X set encodes r12 (not "no index").
    ["lea r10, [r10+r12*2]", [0x4F, 0x8D, 0x14, 0x62]],
    ["mov [r10+r12*2], cx", [0x66, 0x43, 0x89, 0x0C, 0x62]],
    ["mov [r10+r12*2+2], cx", [0x66, 0x43, 0x89, 0x4C, 0x62, 0x02]],
    ["movzx eax, word [rbx+r12*2]", [0x43, 0x0F, 0xB7, 0x04, 0x63]],
    // FS/GS overrides: the segment base is added to the effective address.
    ["mov ecx, %gs:[rax]", [0x65, 0x8B, 0x08]],
    ["mov %gs:[r10+rcx*2+4], ecx", [0x65, 0x41, 0x89, 0x4C, 0x4A, 0x04]],
    ["mov ecx, %fs:[rax+8]", [0x64, 0x8B, 0x48, 0x08]],
    ["cpuid", [0x0F, 0xA2]],
    ["multi-helper block", [
        0x0f, 0xa2,             // cpuid
        0x48, 0xc1, 0xe0, 0x01, // shl rax, 1
        0x0f, 0xa2,             // cpuid
        0x48, 0xc1, 0xe2, 0x02, // shl rdx, 2
        0x31, 0xc0, 0x31, 0xd2, // xor eax,eax / xor edx,edx
    ]],
    ["per-cpu TSC shape", [
        0x66, 0x90,
        0x31, 0xc0, 0x31, 0xd2,       // xor eax,eax / xor edx,edx (rdtsc stand-in)
        0x48, 0xc1, 0xe2, 0x20,       // shl rdx, 32
        0x48, 0x09, 0xd0,             // or rax, rdx
        0x65, 0x8b, 0x3d, 0x00, 0x00, 0x00, 0x00, // mov edi, gs:[rip+0]
        0x89, 0xfa,                   // mov edx, edi
        0x83, 0xe2, 0x01,             // and edx, 1
        0x48, 0xc1, 0xe2, 0x04,       // shl rdx, 4
        0x4c, 0x89, 0xc1,             // mov rcx, r8
        0x65, 0x49, 0x8b, 0x30,       // mov rsi, gs:[r8]
        0x41, 0x8b, 0x10,             // mov edx, [r8]
    ]],
    ["__cpuid body (verbatim)", [
        0x4c, 0x89, 0xc7,             // mov rdi, r8
        0x4c, 0x89, 0xc2,             // mov rdx, r8
        0x53,                         // push rbx
        0x4d, 0x89, 0xc1,             // mov r9, r8
        0x8b, 0x07,                   // mov eax, [rdi]
        0x8b, 0x0a,                   // mov ecx, [rdx]
        0x48, 0x89, 0xd6,             // mov rsi, rdx
        0x0f, 0xa2,                   // cpuid
        0x89, 0x07,                   // mov [rdi], eax
        0x41, 0x89, 0x19,             // mov [r9], ebx
        0x5b,                         // pop rbx
        0x89, 0x0e,                   // mov [rsi], ecx
        0x41, 0x89, 0x10,             // mov [r8], edx
        0x31, 0xc0, 0x31, 0xd2, 0x31, 0xc9, 0x31, 0xf6,
        0x31, 0xff, 0x45, 0x31, 0xc0, 0x45, 0x31, 0xc9, 0x45, 0x31, 0xd2,
    ]],
    ["__cpuid shape", [
        0x53,                         // push rbx
        0x49, 0x89, 0xc1,             // mov r9, r8
        0xb8, 0x00, 0x02, 0x00, 0x00, // mov eax, 0x200
        0x0f, 0xa2,                   // cpuid
        0x41, 0x89, 0x00,             // mov [r8], eax
        0x41, 0x89, 0x19,             // mov [r9], ebx
        0x5b,                         // pop rbx
        0x31, 0xff,                   // xor edi, edi (mitigation style)
    ]],
    ["x86 indirect thunk (call +1 / mov [rsp],rax / ret)", [
        0x48, 0x8d, 0x05, 0x0c, 0x00, 0x00, 0x00, // lea rax, [rip+0xc]
        0xe8, 0x01, 0x00, 0x00, 0x00,             // call +1
        0xcc,                                     // int3 (skipped by the call)
        0x48, 0x89, 0x04, 0x24,                   // mov [rsp], rax
        0xc3,                                     // ret -> jumps to rax
    ]],
    ["out 0x21, al", [0xE6, 0x21]],
    // Conditional branches: `cmp eax,ecx` then a Jcc whose taken/not-taken path
    // writes a different register value. The registers are randomised per seed,
    // so every flag combination is exercised.
    ["cmp eax,ecx / je", [0x39, 0xc8, 0x74, 0x02, 0x31, 0xc0]],
    ["cmp eax,ecx / jne", [0x39, 0xc8, 0x75, 0x02, 0x31, 0xc0]],
    ["cmp eax,ecx / jb", [0x39, 0xc8, 0x72, 0x02, 0x31, 0xc0]],
    ["cmp eax,ecx / jae", [0x39, 0xc8, 0x73, 0x02, 0x31, 0xc0]],
    ["cmp eax,ecx / js", [0x39, 0xc8, 0x78, 0x02, 0x31, 0xc0]],
    ["cmp eax,ecx / jl", [0x39, 0xc8, 0x7c, 0x02, 0x31, 0xc0]],
    ["cmp eax,ecx / jle", [0x39, 0xc8, 0x7e, 0x02, 0x31, 0xc0]],
    ["cmp eax,ecx / jo", [0x39, 0xc8, 0x70, 0x02, 0x31, 0xc0]],
    ["test eax,ecx / jz", [0x85, 0xc8, 0x74, 0x02, 0x31, 0xc0]],
    ["cmp r8d,r9d / je", [0x45, 0x39, 0xc8, 0x74, 0x02, 0x31, 0xc0]],
    ["cmp [r8],ecx / je", [0x41, 0x39, 0x08, 0x74, 0x02, 0x31, 0xc0]],
    // The kernel's entry code: testb on a *memory* operand then jz/jnz.
    ["testb $3,[rsp+8] / jz", [0xf6, 0x44, 0x24, 0x08, 0x03, 0x74, 0x03, 0x31, 0xc0, 0x90]],
    ["testb $3,[rsp+8] / jnz", [0xf6, 0x44, 0x24, 0x08, 0x03, 0x75, 0x03, 0x31, 0xc0, 0x90]],
    ["testb $3,[r8] / jz", [0x41, 0xf6, 0x00, 0x03, 0x74, 0x03, 0x31, 0xc0, 0x90]],
    ["testb $3,[r8] / jnz", [0x41, 0xf6, 0x00, 0x03, 0x75, 0x03, 0x31, 0xc0, 0x90]],
    ["test dword [r8],$3 / jz", [0x41, 0xf7, 0x00, 0x03, 0x00, 0x00, 0x00, 0x74, 0x03, 0x31, 0xc0, 0x90]],
    ["testb $3,[r8] / jz (2)", [0x41, 0xf6, 0x00, 0x03, 0x0f, 0x84, 0x02, 0x00, 0x00, 0x00, 0x31, 0xc0]],
    // The kernel's entry trampoline switches stacks with GS-relative accesses
    // that read/write RSP itself.
    ["mov rsp, gs:[r8+0x100]", [0x65, 0x49, 0x8b, 0xa0, 0x00, 0x01, 0x00, 0x00]],
    ["mov gs:[r8+0x100], rsp", [0x65, 0x49, 0x89, 0xa0, 0x00, 0x01, 0x00, 0x00]],
    ["mov rsp, gs:[rip+0]", [0x65, 0x48, 0x8b, 0x25, 0x00, 0x00, 0x00, 0x00]],
    ["mov gs:[rip+0], rsp", [0x65, 0x48, 0x89, 0x25, 0x00, 0x00, 0x00, 0x00]],
    ["mov rsp, [r8+0x100]", [0x49, 0x8b, 0xa0, 0x00, 0x01, 0x00, 0x00]],
    ["xchg rsp, r8", [0x49, 0x87, 0xe0]],
    // inc/dec on memory (preempt_disable/enable in the delay_tsc hot loop).
    ["incl gs:[r8+0x100]", [0x65, 0x41, 0xff, 0x80, 0x00, 0x01, 0x00, 0x00]],
    ["decl gs:[r8+0x100]", [0x65, 0x41, 0xff, 0x88, 0x00, 0x01, 0x00, 0x00]],
    ["inc dword [r8+0x100]", [0x41, 0xff, 0x80, 0x00, 0x01, 0x00, 0x00]],
    ["dec dword [r8+0x100]", [0x41, 0xff, 0x88, 0x00, 0x01, 0x00, 0x00]],
    ["inc byte gs:[r8+0x100]", [0x65, 0x41, 0xfe, 0x80, 0x00, 0x01, 0x00, 0x00]],
    ["dec byte [r8+0x100]", [0x41, 0xfe, 0x88, 0x00, 0x01, 0x00, 0x00]],
    ["inc word [r8+0x100]", [0x66, 0x41, 0xff, 0x80, 0x00, 0x01, 0x00, 0x00]],
    ["inc qword [r8+0x100]", [0x49, 0xff, 0x80, 0x00, 0x01, 0x00, 0x00]],
    ["inc qword [r8]", [0x49, 0xff, 0x00]],
    ["dec qword [r8]", [0x49, 0xff, 0x08]],
    // Stack manipulation, as used by the kernel's PUSH_AND_CLEAR_REGS macro:
    // a leak here walks the kernel stack page by page.
    ["push rbx / pop r9", [0x53, 0x41, 0x59]],
    ["push imm8 / pop r12", [0x6a, 0x11, 0x41, 0x5c]],
    ["pushfq / pop r15", [0x9c, 0x41, 0x5f]],
    ["mov [rsp+8], rdi", [0x48, 0x89, 0x7c, 0x24, 0x08]],
    ["mov rsi, [rsp+8]", [0x48, 0x8b, 0x74, 0x24, 0x08]],
    ["sub rsp,8 / add rsp,8", [0x48, 0x83, 0xec, 0x08, 0x48, 0x83, 0xc4, 0x08]],
    ["call/pop balance", [0xff, 0xc9, 0x53, 0x5b, 0xff, 0xc9]],
    ["in al, 0x21", [0xE4, 0x21]],
    // The DX forms (the kernel's port I/O helpers use `in al, dx`).
    ["in al, dx", [0xEC]],
    ["in ax, dx", [0x66, 0xED]],
    ["in eax, dx", [0xED]],
    ["out dx, al", [0xEE]],
    ["out dx, eax", [0xEF]],
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
    for(let i = 0; i < 512; i++) write64(PT + i * 8, BigInt(i) * 0x1000n | 0x3n);
    for(let i = 1; i < 512; i++) write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);

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
        else ext[i - 8] = value;
    };
    const reg64 = i => (i < 8)
        ? (BigInt(u32[32 + i]) << 32n) | BigInt(u32[16 + i])
        : ext[i - 8];

    const imm64 = value => {
        const out = [];
        for(let i = 0; i < 8; i++) out.push(Number(value >> BigInt(8 * i) & 0xFFn));
        return out;
    };
    const imm32 = value => {
        const out = [];
        for(let i = 0; i < 4; i++) out.push(value >> (8 * i) & 0xFF);
        return out;
    };
    const write32 = (bytes, at, value) => {
        for(let i = 0; i < 4; i++) bytes[at + i] = value >> (8 * i) & 0xFF;
    };

    const build_program = (instr, seed) => {
        const random = rng(seed);
        const bytes = [];
        const put = (...b) => bytes.push(...b);
        put(0x41, 0xBB, ...imm32(ITERATIONS));    // mov r11d, ITERATIONS
        put(0x49, 0xB8, ...imm64(BigInt(SCRATCH)));        // mov r8, SCRATCH
        put(0x49, 0xBA, ...imm64(BigInt(SCRATCH)));        // mov r10, SCRATCH
        put(0x41, 0xBC, 0x03, 0x00, 0x00, 0x00); // mov r12d, 3
        put(0x48, 0xBB, ...imm64(BigInt(SCRATCH)));        // mov rbx, SCRATCH
        put(0x31, 0xC9);                          // xor ecx, ecx
        put(0x31, 0xC0);                          // xor eax, eax
        put(0x45, 0x31, 0xED);                    // xor r13d, r13d
        const head = bytes.length;
        put(0x41, 0xFF, 0xCB);                    // dec r11d
        const jz_at = bytes.length;
        put(0x0F, 0x84, 0, 0, 0, 0);              // jz end
        put(...instr);
        put(0x48, 0x83, 0xC1, 0x01);              // add rcx, 1
        put(0x48, 0x83, 0xE1, 0x1F);              // and rcx, 0x1F
        const jmp_at = bytes.length;
        put(0xE9, 0, 0, 0, 0);                    // jmp head
        const end = bytes.length;
        put(0xF4);                                // hlt
        write32(bytes, jz_at + 2, end - (jz_at + 6));
        write32(bytes, jmp_at + 1, head - (jmp_at + 5));
        void random;
        return bytes;
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

        new BigUint64Array(cpu.wasm_memory.buffer, 1104, 1)[0] = 0x40n; // fs_base
        new BigUint64Array(cpu.wasm_memory.buffer, 1112, 1)[0] = 0x80n; // gs_base
        cpu.instruction_pointer[0] = BASE;
        u32[16 + 4] = 0x80000;
        u32[32 + 4] = 0;
        cpu.flags[0] = 0x2;
        cpu.in_hlt[0] = 0;
        ex.enter_long_mode(PML4);

        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 400) ex.main_loop();
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
    for(let c = 0; c < CANDIDATES.length; c++)
    {
        const [name, instr] = CANDIDATES[c];
        for(let seed = 1; seed <= 20; seed++)
        {
            const random = rng(seed * 101 + c);
            const program = build_program(instr, seed);
            const regs = new Array(16).fill(0n);
            for(const i of [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 13, 14])
            {
                regs[i] = BigInt(Math.floor(random() * 2 ** 32)) | BigInt(Math.floor(random() * 2 ** 32)) << 32n;
            }
            const memory = Array.from({ length: 256 }, () => Math.floor(random() * 256));

            const interpreted = run(program, regs, memory, false);
            const compiled = run(program, regs, memory, true);
            if(ex.jit64_compiled_count() > 0) compiled_any = true;

            const problems = [];
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
                problems.push("flags: interp 0x" + interpreted.flags.toString(16) + " vs jit 0x" + compiled.flags.toString(16));
            }
            for(let i = 0; i < memory.length; i++)
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
                if(mismatches <= 8)
                {
                    console.log("FAIL [" + name + "] seed " + seed + ": " + problems[0]);
                    for(const p of problems.slice(1, 3)) console.log("     " + p);
                    console.log("     instr: [" + instr.join(", ") + "]");
                }
            }
        }
    }

    // Systematic SIB sweep: every base/index/REX.B/REX.X/scale combination.
    // Hand-picked pools missed index field 100 with REX.X (r12) before, so
    // enumerate the cross product instead. The destination is always r13 and
    // every other register holds a distinct value, so a wrong base or index
    // shows up in the result.
    const sib_immediate = value => {
        const out = [];
        for(let i = 0; i < 8; i++) out.push(Number(value >> BigInt(8 * i) & 0xFFn));
        return out;
    };
    const build_sib_program = instr => {
        const bytes = [];
        const put = (...b) => bytes.push(...b);
        put(0x41, 0xBB, ...imm32(ITERATIONS));                // mov r11d, ITERATIONS
        const values = [[0, 0x11000n], [1, 0x12000n], [2, 0x13000n], [3, 0x14000n], [5, 0x16000n],
                        [6, 0x17000n], [7, 0x18000n], [8, 0x19000n], [9, 0x1a000n], [10, 0x1b000n],
                        [12, 0x1c000n], [13, 0x1d000n], [14, 0x1e000n], [15, 0x1f000n]];
        for(const [register, value] of values)
        {
            if(register < 8) put(0x48, 0xB8 + register);
            else put(0x49, 0xB8 + (register - 8));
            put(...sib_immediate(value));
        }
        const head = bytes.length;
        put(0x41, 0xFF, 0xCB);                                // dec r11d
        const jz_at = bytes.length;
        put(0x0F, 0x84, 0, 0, 0, 0);                          // jz end
        put(...instr);
        const jmp_at = bytes.length;
        put(0xE9, 0, 0, 0, 0);                                // jmp head
        const end = bytes.length;
        put(0xF4);                                            // hlt
        for(let i = 0; i < 4; i++) bytes[jz_at + 2 + i] = end - (jz_at + 6) >> (8 * i) & 0xFF;
        for(let i = 0; i < 4; i++) bytes[jmp_at + 1 + i] = head - (jmp_at + 5) >> (8 * i) & 0xFF;
        return bytes;
    };
    const sib_lea = (base_low, rex_b, index_low, rex_x, scale) => [
        0x40 | 0x08 | 0x04 | rex_x << 1 | rex_b,  // REX.W|REX.R(dst r13)|REX.X|REX.B
        0x8D,                                      // lea
        0x80 | 5 << 3 | 4,                         // mod=10, reg=r13, rm=SIB
        scale << 6 | index_low << 3 | base_low,    // SIB
        0x10, 0x00, 0x00, 0x00,                    // disp32
    ];

    let sib_cases = 0;
    for(let base_low = 0; base_low < 8; base_low++)
    {
        for(let rex_b = 0; rex_b < 2; rex_b++)
        {
            for(let index_low = 0; index_low < 8; index_low++)
            {
                for(let rex_x = 0; rex_x < 2; rex_x++)
                {
                    for(let scale = 0; scale < 4; scale++)
                    {
                        const program = build_sib_program(sib_lea(base_low, rex_b, index_low, rex_x, scale));
                        const regs = new Array(16).fill(0n);
                        const interpreted = run(program, regs, [], false);
                        const compiled = run(program, regs, [], true);
                        if(ex.jit64_compiled_count() > 0) compiled_any = true;
                        sib_cases++;

                        const problems = [];
                        for(let i = 0; i < 16; i++)
                        {
                            if(interpreted.regs[i] !== compiled.regs[i])
                            {
                                problems.push("r" + i + ": interp 0x" + interpreted.regs[i].toString(16) +
                                    " vs jit 0x" + compiled.regs[i].toString(16));
                            }
                        }
                        if(problems.length)
                        {
                            mismatches++;
                            if(mismatches <= 8)
                            {
                                console.log("FAIL SIB base=" + base_low + " rex_b=" + rex_b +
                                    " index=" + index_low + " rex_x=" + rex_x + " scale=" + scale +
                                    ": " + problems[0]);
                                console.log("     [0x" + sib_lea(base_low, rex_b, index_low, rex_x, scale)
                                    .map(x => x.toString(16)).join(", 0x") + "]");
                            }
                        }
                    }
                }
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
        console.log("jit64 addressing: " + mismatches + " mismatching case(s)");
        process.exit(1);
    }
    console.log("jit64 addressing: " + (CANDIDATES.length * 20 + sib_cases) + " cases matched");
    process.exit(0);
});
