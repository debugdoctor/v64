#!/usr/bin/env node

// Long-mode JIT integration test: runs a hot loop through the main loop, lets
// the JIT compile the block and checks the result.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/jit64/longmode.js`

import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";

const BASE = 0x1000;

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
    let buffer = ex.memory.buffer;
    let u32 = new Uint32Array(buffer);

    const write64 = (address, value) => {
        ex.write32(address, Number(value & 0xFFFF_FFFFn));
        ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
    };
    // Read guest physical memory (through the CPU), not the raw wasm buffer.
    const read_guest = (address, bytes) => {
        let value = 0n;
        for(let i = bytes - 1; i >= 0; i--)
        {
            value = value << 8n | BigInt(ex.read8(address + i));
        }
        return value;
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
    write64(PT + 0x70 * 8, 0x90000n | 0x3n); // virtual 0x70000 -> physical 0x90000
    write64(PT + 0x71 * 8, 0xC0000n | 0x3n); // virtual 0x71000 -> physical 0xC0000
    for(let i = 1; i < 512; i++)
    {
        write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);
    }

    // mov rax,10000; mov rcx,0; mov r12,body; loop: call r12; near jne loop; hlt.
    // body exercises lea, stack, arithmetic, logic and memory, then returns.
    const program = [
        0x48, 0xB8, 0x10, 0x27, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rax, 10000
        0x48, 0xB9, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rcx, 0
        0x49, 0xBC, 0x28, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov r12, body
        0x41, 0xFF, 0xD4, // call r12
        0x0F, 0x85, 0xF7, 0xFF, 0xFF, 0xFF, // near jne loop
        0xF4, // hlt
        // body:
        0xF3, 0x0F, 0x1E, 0xFA, // endbr64
        0x90, // nop
        0x0F, 0x1F, 0x00, // multi-byte nop
        0x55, // push rbp
        0x48, 0x89, 0xE5, // mov rbp, rsp
        0x48, 0x8D, 0x15, 0xC5, 0x00, 0x00, 0x00, // lea rdx, [rip + 0xc5] = 0x1100
        0x49, 0xBD, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, // mov r13, -1
        0x41, 0x83, 0xC5, 0x01, // add r13d, 1 (zero-extends to r13)
        0x52, // push rdx
        0x41, 0x59, // pop r9
        0x48, 0x83, 0xE8, 0x01, // sub rax, 1
        0x48, 0x83, 0xC1, 0x01, // add rcx, 1
        0x81, 0xF9, 0x10, 0x27, 0x00, 0x00, // cmp ecx, 10000
        0x49, 0x83, 0xC8, 0x03, // or r8, 3
        0x49, 0x83, 0xE0, 0x01, // and r8, 1
        0x4D, 0x31, 0xC0, // xor r8, r8
        0x48, 0x89, 0x54, 0x24, 0x10, // mov [rsp+16], rdx
        0x48, 0x83, 0x44, 0x24, 0x10, 0x01, // add qword [rsp+16], 1
        0x48, 0x83, 0x6C, 0x24, 0x10, 0x01, // sub qword [rsp+16], 1
        0x4C, 0x31, 0x44, 0x24, 0x10, // xor qword [rsp+16], r8
        0x48, 0x39, 0x54, 0x24, 0x10, // cmp qword [rsp+16], rdx
        0x48, 0x85, 0x54, 0x24, 0x10, // test qword [rsp+16], rdx
        0x4C, 0x8B, 0x54, 0x24, 0x10, // mov r10, [rsp+16]
        0x4C, 0x03, 0x5C, 0x24, 0x10, // add r11, [rsp+16]
        0x4C, 0x2B, 0x5C, 0x24, 0x10, // sub r11, [rsp+16]
        0x44, 0x0F, 0xB6, 0x7C, 0x24, 0x10, // movzx r15d, byte [rsp+16]
        0x48, 0x0F, 0xBF, 0x5C, 0x24, 0x10, // movsx rbx, word [rsp+16]
        0xC7, 0x44, 0x24, 0x18, 0xFF, 0xFF, 0xFF, 0xFF, // mov dword [rsp+24], -1
        0x83, 0x44, 0x24, 0x18, 0x01, // add dword [rsp+24], 1
        0x44, 0x8B, 0x74, 0x24, 0x18, // mov r14d, [rsp+24]
        0xC7, 0xC6, 0x01, 0x00, 0x00, 0x00, // mov esi, 1
        0xC7, 0xC7, 0x02, 0x00, 0x00, 0x00, // mov edi, 2
        0x87, 0xFE, // xchg esi, edi
        0xF7, 0xD6, // not esi
        0xF7, 0xDE, // neg esi
        0xFF, 0xC7, // inc edi
        0xFF, 0xCF, // dec edi
        0x48, 0xF7, 0x54, 0x24, 0x10, // not qword [rsp+16]
        0x48, 0xF7, 0x54, 0x24, 0x10, // not qword [rsp+16]
        0x48, 0xF7, 0x5C, 0x24, 0x10, // neg qword [rsp+16]
        0x48, 0xF7, 0x5C, 0x24, 0x10, // neg qword [rsp+16]
        0x48, 0xFF, 0x44, 0x24, 0x10, // inc qword [rsp+16]
        0x48, 0xFF, 0x4C, 0x24, 0x10, // dec qword [rsp+16]
        0x48, 0x87, 0x7C, 0x24, 0x10, // xchg [rsp+16], rdi
        0x48, 0x87, 0x7C, 0x24, 0x10, // xchg [rsp+16], rdi
        0xD1, 0xE6, // shl esi, 1
        0xD1, 0xEE, // shr esi, 1
        0x41, 0xC7, 0xC5, 0xFE, 0xFF, 0xFF, 0xFF, // mov r13d, -2
        0x41, 0xD1, 0xFD, // sar r13d, 1
        0x41, 0xC1, 0xE5, 0x21, // shl r13d, 33 (masked to 1)
        0x41, 0xD1, 0xFD, // sar r13d, 1 (back to -1)
        0x4D, 0x63, 0xED, // movsxd r13, r13d
        0x0F, 0xAF, 0xF7, // imul esi, edi
        0x6B, 0xFF, 0x01, // imul edi, edi, 1
        0x48, 0x6B, 0x5C, 0x24, 0x10, 0x01, // imul rbx, [rsp+16], 1
        0x83, 0xFF, 0x01, // cmp edi, 1
        0x44, 0x0F, 0x44, 0xF7, // cmove r14d, edi
        0x41, 0x0F, 0x94, 0xC7, // sete r15b
        0x41, 0x0F, 0xCE, // bswap r14d
        0x41, 0x0F, 0xCE, // bswap r14d (restore)
        0x41, 0xC7, 0xC0, 0x01, 0x00, 0x00, 0x00, // mov r8d, 1
        0x41, 0xD3, 0xE0, // shl r8d, cl
        0x41, 0xD3, 0xE8, // shr r8d, cl
        0x83, 0xFF, 0x02, // cmp edi, 2 (CF=1)
        0x41, 0x83, 0xD0, 0x00, // adc r8d, 0
        0x83, 0xFF, 0x02, // cmp edi, 2 (CF=1)
        0x41, 0x83, 0xD8, 0x00, // sbb r8d, 0
        0x83, 0xFF, 0x02, // cmp edi, 2 (CF=1)
        0x48, 0x83, 0x54, 0x24, 0x10, 0x00, // adc qword [rsp+16], 0
        0x83, 0xFF, 0x02, // cmp edi, 2 (CF=1)
        0x48, 0x83, 0x5C, 0x24, 0x10, 0x00, // sbb qword [rsp+16], 0
        0x48, 0x89, 0x14, 0x25, 0x00, 0x00, 0x07, 0x00, // mov [0x70000], rdx
        0x4C, 0x8B, 0x14, 0x25, 0x00, 0x00, 0x07, 0x00, // mov r10, [0x70000]
        0x49, 0xBB, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // mov r11, 0x1122334455667788
        0x4C, 0x89, 0x1C, 0x25, 0xFC, 0x0F, 0x07, 0x00, // mov [0x70ffc], r11
        0x4D, 0x31, 0xDB, // xor r11, r11
        0x4C, 0x8B, 0x1C, 0x25, 0xFC, 0x0F, 0x07, 0x00, // mov r11, [0x70ffc]
        0x85, 0xC0, // test eax, eax
        0xC9, // leave
        0xC3, // ret
    ];
    for(let i = 0; i < program.length; i++)
    {
        ex.write8(BASE + i, program[i]);
    }
    for(let i = 0; i < 4; i++) ex.write8(0x8000C + i, 0xAA);

    cpu.instruction_pointer[0] = BASE;
    u32[16 + 4] = 0x80000; // rsp
    cpu.flags[0] &= ~(1 << 9); // cli
    cpu.in_hlt[0] = 0;

    ex.enter_long_mode(PML4);
    let guard = 0;
    while(!cpu.in_hlt[0] && guard++ < 100000)
    {
        ex.main_loop();
    }

    // The guest can grow the wasm heap while it runs, detaching these views.
    buffer = ex.memory.buffer;
    u32 = new Uint32Array(buffer);

    const rax = (BigInt(u32[32]) << 32n) | BigInt(u32[16]);
    const rcx = (BigInt(u32[32 + 1]) << 32n) | BigInt(u32[16 + 1]);
    const rdx = (BigInt(u32[32 + 2]) << 32n) | BigInt(u32[16 + 2]);
    const r8 = new DataView(buffer).getBigUint64(160, true);
    const r9 = new DataView(buffer).getBigUint64(168, true);
    const r10 = new DataView(buffer).getBigUint64(176, true);
    const r11 = new DataView(buffer).getBigUint64(184, true);
    const r13 = new DataView(buffer).getBigUint64(200, true);
    const r14 = new DataView(buffer).getBigUint64(208, true);
    const r15 = new DataView(buffer).getBigUint64(216, true);
    const rbx = (BigInt(u32[32 + 3]) << 32n) | BigInt(u32[16 + 3]);
    const rsi = (BigInt(u32[32 + 6]) << 32n) | BigInt(u32[16 + 6]);
    const rdi = (BigInt(u32[32 + 7]) << 32n) | BigInt(u32[16 + 7]);
    const rsp = (BigInt(u32[32 + 4]) << 32n) | BigInt(u32[16 + 4]);
    const compiled = ex.jit64_compiled_count();

    assert.equal(rax, 0n, "rax");
    assert.equal(rcx, 10000n, "rcx");
    assert.equal(rdx, 0x1100n, "rip-relative lea result");
    assert.equal(r8, 1n, "CL-count shift result");
    assert.equal(r9, 0x1100n, "push/pop round trip");
    assert.equal(r10, 0x1100n, "memory round trip");
    assert.equal(r11, 0x1122334455667788n, "cross-page load result");
    assert.equal(r13, 0xFFFF_FFFF_FFFF_FFFFn, "movsxd sign-extends");
    assert.equal(r14, 1n, "cmov 32-bit result");
    assert.equal(r15, 1n, "setcc result");
    assert.equal(rbx, 0x1100n, "movsx memory result");
    assert.equal(rsi, 3n, "xchg/not/neg 32-bit result");
    assert.equal(rdi, 1n, "xchg 32-bit result");
    assert.equal(rsp, 0x80000n, "stack balanced");
    const stored = BigInt(ex.read8(0x80000)) | BigInt(ex.read8(0x80001)) << 8n;
    assert.equal(stored, 0x1100n, "guest memory store");
    const translated = BigInt(ex.read8(0x90000)) | BigInt(ex.read8(0x90001)) << 8n;
    assert.equal(translated, 0x1100n, "non-identity translated store");
    assert.equal(ex.read8(0x70000), 0, "unmapped physical alias untouched");
    const cross_page_low = Array.from({ length: 4 }, (_, i) => ex.read8(0x90FFC + i));
    const cross_page_high = Array.from({ length: 4 }, (_, i) => ex.read8(0xC0000 + i));
    assert.deepEqual(cross_page_low, [0x88, 0x77, 0x66, 0x55], "cross-page low bytes");
    assert.deepEqual(cross_page_high, [0x44, 0x33, 0x22, 0x11], "cross-page high bytes");
    assert.deepEqual(
        Array.from({ length: 4 }, (_, i) => ex.read8(0x8000C + i)),
        [0xAA, 0xAA, 0xAA, 0xAA],
        "dword store preserves adjacent bytes",
    );
    assert.equal(cpu.in_hlt[0], 1, "halted");
    assert.ok(compiled > 0, "at least one block was JIT-compiled (got " + compiled + ")");

    // dec/jnz is one block. 500 interpreted iterations, then 100 compiled
    // iterations of both instructions: 1 + 500*2 + 100*2 + 1 = 1202.
    const loop_base = 0x2000;
    const loop = [
        0xB9, 0x58, 0x02, 0x00, 0x00, // mov ecx, 600
        0xFF, 0xC9,                   // dec ecx
        0x75, 0xFC,                   // jnz dec
        0xF4,                         // hlt
    ];
    for(let i = 0; i < loop.length; i++)
    {
        ex.write8(loop_base + i, loop[i]);
    }
    cpu.in_hlt[0] = 0;
    new DataView(buffer).setBigUint64(232, BigInt(loop_base), true);
    const before = u32[166];
    let loop_guard = 0;
    while(!cpu.in_hlt[0] && loop_guard++ < 100000)
    {
        ex.main_loop();
    }
    assert.equal(u32[166] - before, 1202, "compiled block counts each instruction");

    // Self-modifying code: a guest store to the physical code page must drop the
    // blocks compiled from that page, without clearing the whole cache.
    const compiled_before_smc = ex.jit64_compiled_count();
    const store_base = 0x3000; // page 3, distinct from the loop at page 2
    const store = [
        0xC6, 0x04, 0x25, 0x00, 0x21, 0x00, 0x00, 0x01, // mov byte [0x2100], 1
        0xF4,                                           // hlt
    ];
    for(let i = 0; i < store.length; i++)
    {
        ex.write8(store_base + i, store[i]);
    }
    cpu.in_hlt[0] = 0;
    new DataView(buffer).setBigUint64(232, BigInt(store_base), true);
    let smc_guard = 0;
    while(!cpu.in_hlt[0] && smc_guard++ < 1000)
    {
        ex.main_loop();
    }
    assert.equal(ex.read8(0x2100), 1, "store landed in the code page");
    assert.ok(
        ex.jit64_compiled_count() < compiled_before_smc,
        "self-modifying code dropped the block(s) from that page",
    );

    // cmov/setcc with memory operands, compiled as a hot block.
    {
        const cmov_base = 0x4000;
        ex.write8(0x60000, 0x34);
        ex.write8(0x60001, 0x12);
        ex.write8(0x60002, 0x00);
        ex.write8(0x60003, 0x00);
        ex.write8(0x60008, 0xAA);
        const cmov_program = [
            0xB9, 0x58, 0x02, 0x00, 0x00,                   // mov ecx, 600
            0x39, 0xC9,                                     // cmp ecx, ecx (ZF=1)
            0xB8, 0x00, 0x00, 0x00, 0x00,                   // mov eax, 0
            0x0F, 0x44, 0x04, 0x25, 0x00, 0x00, 0x06, 0x00, // cmove eax, [0x60000]
            0x0F, 0x94, 0x04, 0x25, 0x08, 0x00, 0x06, 0x00, // sete byte [0x60008]
            0xFF, 0xC9,                                     // dec ecx
            0x75, 0xE5,                                     // jnz cmove (not the mov ecx,600)
            0xF4,                                           // hlt
        ];
        for(let i = 0; i < cmov_program.length; i++)
        {
            ex.write8(cmov_base + i, cmov_program[i]);
        }
        cpu.in_hlt[0] = 0;
        new DataView(buffer).setBigUint64(232, BigInt(cmov_base), true);
        let g = 0;
        while(!cpu.in_hlt[0] && g++ < 100000)
        {
            ex.main_loop();
        }
        assert.equal(u32[16] >>> 0, 0x1234, "cmove memory source");
        assert.equal(ex.read8(0x60008), 1, "setcc memory destination");
    }

    // Stage 5 coverage: 16-bit operands (`66`), address-size override (`67`),
    // LOCK, memory-form shifts and CPUID, all inside a hot compiled block.
    {
        const cov_base = 0x5000;
        ex.write8(0x60000, 0x44);
        ex.write8(0x60001, 0x33);
        ex.write8(0x60002, 0x22);
        ex.write8(0x60003, 0x11);
        const cov_program = [
            0x41, 0xBC, 0x58, 0x02, 0x00, 0x00, // mov r12d, 600
            0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, -1
            0x66, 0xB8, 0x34, 0x12, // mov ax, 0x1234
            0x66, 0x83, 0xC0, 0x01, // add ax, 1
            0x49, 0x89, 0xC5, // mov r13, rax
            0x66, 0xC7, 0x44, 0x24, 0x20, 0xCD, 0xAB, // mov word [rsp+0x20], 0xABCD
            0x66, 0x83, 0x44, 0x24, 0x20, 0x01, // add word [rsp+0x20], 1
            0xBB, 0x00, 0x00, 0x06, 0x00, // mov ebx, 0x60000
            0x67, 0x8B, 0x03, // mov eax, [ebx]
            0x41, 0x89, 0xC7, // mov r15d, eax
            0xF0, 0x48, 0x83, 0x44, 0x24, 0x28, 0x01, // lock add qword [rsp+0x28], 1
            0x48, 0xD1, 0x64, 0x24, 0x28, // shl qword [rsp+0x28], 1
            0x48, 0xD1, 0x6C, 0x24, 0x28, // shr qword [rsp+0x28], 1
            0x31, 0xC0, // xor eax, eax
            0x0F, 0xA2, // cpuid
            0x41, 0xFF, 0xCC, // dec r12d
            0x75, 0xBC, // jnz loop
            0xF4, // hlt
        ];
        for(let i = 0; i < cov_program.length; i++)
        {
            ex.write8(cov_base + i, cov_program[i]);
        }
        cpu.in_hlt[0] = 0;
        new DataView(buffer).setBigUint64(232, BigInt(cov_base), true);
        const compiled_before_cov = ex.jit64_compiled_count();
        let g = 0;
        while(!cpu.in_hlt[0] && g++ < 200000)
        {
            ex.main_loop();
        }
        assert.ok(
            ex.jit64_compiled_count() > compiled_before_cov,
            "the Stage 5 block was JIT-compiled",
        );

        const r13cov = new DataView(buffer).getBigUint64(200, true);
        const r15cov = new DataView(buffer).getBigUint64(216, true);
        assert.equal(r13cov, 0xFFFF_FFFF_FFFF_1235n, "16-bit arithmetic preserves the upper bits");
        assert.equal(r15cov, 0x11223344n, "32-bit addressing override load");
        assert.equal(u32[16 + 3] >>> 0, 0x756E6547, "cpuid leaf 0 ebx");
        assert.equal(u32[16 + 1] >>> 0, 0x6C65746E, "cpuid leaf 0 ecx");
        assert.equal(u32[16 + 2] >>> 0, 0x49656E69, "cpuid leaf 0 edx");
        const cov_word = read_guest(0x80020, 2);
        assert.equal(cov_word, 0xABCEn, "16-bit memory add");
        assert.equal(read_guest(0x80028, 8), 600n, "lock add and memory-form shifts");
    }

    // A 16-bit CL shift by more than the operand width must not underflow the
    // carry calculation in the JIT helper.
    {
        const shift_base = 0x6000;
        const shift_program = [
            0x41, 0xBC, 0x58, 0x02, 0x00, 0x00, // mov r12d, 600
            0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, -1
            0x66, 0xB8, 0x34, 0x12, // mov ax, 0x1234
            0xB9, 0x14, 0x00, 0x00, 0x00, // mov ecx, 20
            0x66, 0xD3, 0xE0, // shl ax, cl
            0x49, 0x89, 0xC6, // mov r14, rax
            0x41, 0xFF, 0xCC, // dec r12d
            0x75, 0xF5, // jnz loop
            0xF4, // hlt
        ];
        for(let i = 0; i < shift_program.length; i++)
        {
            ex.write8(shift_base + i, shift_program[i]);
        }
        cpu.in_hlt[0] = 0;
        new DataView(buffer).setBigUint64(232, BigInt(shift_base), true);
        let g = 0;
        while(!cpu.in_hlt[0] && g++ < 200000)
        {
            ex.main_loop();
        }
        const r14shift = new DataView(buffer).getBigUint64(208, true);
        assert.equal(
            r14shift,
            0xFFFF_FFFF_FFFF_0000n,
            "16-bit shift by more than the width preserves the upper bits",
        );
    }

    console.log("jit64 longmode: test passed (" + compiled + " compiled block(s))");
    process.exit(0);
});
