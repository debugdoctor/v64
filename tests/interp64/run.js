#!/usr/bin/env node

// Tests for the hand-written 64-bit interpreter (src/rust/cpu/interp64.rs).
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp64/run.js`

import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";

const BASE = 0x1000;
const SCRATCH = 0x2000;

const emulator = new v64({
    autostart: false,
    memory_size: 2 * 1024 * 1024,
    disable_jit: 1,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const buffer = ex.memory.buffer;
    const u32 = new Uint32Array(buffer);

    // Long mode is not entered yet, so run with paging off and flat segments
    cpu.cr[0] = 0;
    cpu.is_32[0] = 1;

    const reg64 = i =>
        (i < 8)
            ? (BigInt(u32[32 + i]) << 32n) | BigInt(u32[16 + i])
            : new BigUint64Array(buffer, 160, 8)[i - 8];

    const guest_read64 = address =>
        BigInt(ex.read32s(address) >>> 0) | (BigInt(ex.read32s(address + 4) >>> 0) << 32n);

    const guest_write64 = (address, value) => {
        ex.write8(address, Number(value & 0xFFn));
        ex.write8(address + 1, Number(value >> 8n & 0xFFn));
        ex.write8(address + 2, Number(value >> 16n & 0xFFn));
        ex.write8(address + 3, Number(value >> 24n & 0xFFn));
        ex.write8(address + 4, Number(value >> 32n & 0xFFn));
        ex.write8(address + 5, Number(value >> 40n & 0xFFn));
        ex.write8(address + 6, Number(value >> 48n & 0xFFn));
        ex.write8(address + 7, Number(value >> 56n & 0xFFn));
    };

    const load = (code, steps) => {
        for(let i = 0; i < code.length; i++)
        {
            ex.write8(BASE + i, code[i]);
        }
        cpu.instruction_pointer[0] = BASE;
        for(let i = 0; i < steps; i++)
        {
            ex.interp64_run_one();
        }
    };

    // Test 1: 32-bit default operand size, register operands
    {
        load([
            0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
            0xBB, 0x02, 0x00, 0x00, 0x00, // mov ebx, 2
            0x01, 0xD8,                   // add eax, ebx
            0xF4,                         // hlt
        ], 4);
        assert.equal(reg64(0) & 0xFFFF_FFFFn, 3n, "eax after add");
        assert.equal(reg64(3) & 0xFFFF_FFFFn, 2n, "ebx");
        assert.equal(cpu.instruction_pointer[0], BASE + 13, "eip");
        assert.equal(cpu.in_hlt[0], 1, "hlt");
    }

    // Test 2: 64-bit operands (REX.W), r8-r15 (REX.B/REX.R) and memory store
    {
        const a = 0x1122334455667788n;
        const b = 0x00AABBCCDDEEFF11n;
        const sum = a + b & 0xFFFF_FFFF_FFFF_FFFFn;

        const code = [];
        const push = (...bytes) => code.push(...bytes);
        push(0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11); // mov rax, a
        push(0x49, 0xB8, 0x11, 0xFF, 0xEE, 0xDD, 0xCC, 0xBB, 0xAA, 0x00); // mov r8, b
        push(0x4C, 0x01, 0xC0);                                           // add rax, r8
        push(0x48, 0xBB, 0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00); // mov rbx, SCRATCH
        push(0x48, 0x89, 0x03);                                           // mov [rbx], rax
        push(0xF4);                                                       // hlt
        load(code, 6);

        assert.equal(reg64(0), sum, "rax after 64-bit add");
        assert.equal(reg64(8), b, "r8");
        assert.equal(guest_read64(SCRATCH), sum, "memory store");
        assert.equal(cpu.instruction_pointer[0], BASE + code.length, "eip");
    }

    // Test 3: RIP-relative load (mov eax, [rip + disp32])
    {
        const value = 0xDEADBEEFn;
        guest_write64(0x1400, value);
        load([
            0x8B, 0x05, 0xFA, 0x03, 0x00, 0x00, // mov eax, [rip + 0x3fa] -> 0x1400
            0xF4,
        ], 1);
        assert.equal(reg64(0) & 0xFFFF_FFFFn, value, "rip-relative load");
    }

    // Test 4: 8-bit ALU (mov al/bl, add al, bl)
    {
        load([
            0xB0, 0x12, // mov al, 0x12
            0xB3, 0x34, // mov bl, 0x34
            0x00, 0xD8, // add al, bl
            0xF4,
        ], 4);
        assert.equal(reg64(0) & 0xFFn, 0x46n, "al after add");
        assert.equal(reg64(3) & 0xFFn, 0x34n, "bl");
    }

    // Test 5: xor eax, eax sets ZF
    {
        load([0x31, 0xC0, 0xF4], 2);
        assert.equal(reg64(0) & 0xFFFF_FFFFn, 0n, "eax after xor");
        assert.ok(cpu.flags[0] & 1 << 6, "ZF set");
    }

    // Test 6: sub + cmp + je (taken)
    {
        load([
            0xB8, 0x05, 0x00, 0x00, 0x00, // mov eax, 5
            0xB9, 0x03, 0x00, 0x00, 0x00, // mov ecx, 3
            0x29, 0xC8,                   // sub eax, ecx
            0x83, 0xF8, 0x02,             // cmp eax, 2
            0x74, 0x05,                   // je +5
            0xB8, 0xFF, 0x00, 0x00, 0x00, // mov eax, 0xff (skipped)
            0xF4,                         // hlt
        ], 6);
        assert.equal(reg64(0) & 0xFFFF_FFFFn, 2n, "eax after sub/je");
    }

    // Test 7: push/pop
    {
        load([
            0x48, 0xC7, 0xC0, 0x11, 0x00, 0x00, 0x00, // mov rax, 0x11
            0x50,                                     // push rax
            0x48, 0x31, 0xC0,                         // xor rax, rax
            0x5B,                                     // pop rbx
            0xF4,
        ], 5);
        assert.equal(reg64(3), 0x11n, "rbx popped");
        assert.equal(reg64(0), 0n, "rax zeroed");
    }

    // Test 8: call/ret
    {
        load([
            0xE8, 0x05, 0x00, 0x00, 0x00, // call +5 -> 0x100a
            0xB8, 0x2A, 0x00, 0x00, 0x00, // mov eax, 0x2a (after return)
            0xF4,                         // hlt
            0xB8, 0x07, 0x00, 0x00, 0x00, // func: mov eax, 7
            0xC3,                         // ret
        ], 5);
        assert.equal(reg64(0) & 0xFFFF_FFFFn, 0x2An, "eax after call/ret");
    }

    // Test 9: movzx / movsx
    {
        load([
            0xB0, 0x80,       // mov al, 0x80
            0x0F, 0xBE, 0xC8, // movsx ecx, al
            0xF4,
        ], 2);
        assert.equal(reg64(1) & 0xFFFF_FFFFn, 0xFFFF_FF80n, "movsx");

        load([
            0xB0, 0xFF,       // mov al, 0xff
            0x0F, 0xB6, 0xC8, // movzx ecx, al
            0xF4,
        ], 2);
        assert.equal(reg64(1) & 0xFFFF_FFFFn, 0xFFn, "movzx");
    }

    // Test 10: lea with SIB + disp32
    {
        load([
            0x48, 0x8D, 0x04, 0x25, 0x00, 0x30, 0x00, 0x00, // lea rax, [0x3000]
            0xF4,
        ], 1);
        assert.equal(reg64(0), 0x3000n, "lea");
    }

    // Test 11: run through the main loop in long mode, with 4-level paging
    {
        const write64 = (address, value) => {
            ex.write32(address, Number(value & 0xFFFF_FFFFn));
            ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
        };

        // Identity-map the first 1 GiB with 2 MiB pages
        const PML4 = 0x10000;
        const PDPT = 0x11000;
        const PD = 0x12000;
        write64(PML4, BigInt(PDPT) | 0x3n);
        write64(PDPT, BigInt(PD) | 0x3n);
        for(let i = 0; i < 512; i++)
        {
            write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);
        }

        const program = [
            0x48, 0xC7, 0xC0, 0x00, 0x00, 0x04, 0x00, // mov rax, 0x40000
            0x48, 0xC7, 0xC3, 0x42, 0x00, 0x00, 0x00, // mov rbx, 0x42
            0x48, 0x89, 0x18,                         // mov [rax], rbx
            0x48, 0x8B, 0x08,                         // mov rcx, [rax]
            0xF4,                                     // hlt
        ];
        for(let i = 0; i < program.length; i++)
        {
            ex.write8(BASE + i, program[i]);
        }
        cpu.instruction_pointer[0] = BASE;
        u32[16 + 4] = 0x80000; // rsp (low dword)
        cpu.flags[0] &= ~(1 << 9); // clear IF (no IDT set up yet)
        cpu.in_hlt[0] = 0;

        ex.enter_long_mode(PML4);
        ex.main_loop();

        assert.equal(reg64(1) & 0xFFFF_FFFFn, 0x42n, "rcx after paging store/load");
        assert.equal(guest_read64(0x40000), 0x42n, "guest memory through paging");
        assert.equal(cpu.in_hlt[0], 1, "halted");
    }

    // Test 12: long-mode exception handling (#UD via ud2, 64-bit IDT + iretq)
    {
        const write64 = (address, value) => {
            ex.write32(address, Number(value & 0xFFFF_FFFFn));
            ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
        };

        const PML4 = 0x10000;
        const PDPT = 0x11000;
        const PD = 0x12000;
        write64(PML4, BigInt(PDPT) | 0x3n);
        write64(PDPT, BigInt(PD) | 0x3n);
        for(let i = 0; i < 512; i++)
        {
            write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);
        }

        // 64-bit interrupt gate for vector 6 (#UD) -> 0x3000, selector 0x08
        const HANDLER = 0x3000;
        const IDT = 0x20000;
        const gate = (offset, selector, type) =>
            (offset & 0xFFFFn)
            | BigInt(selector) << 16n
            | BigInt(type) << 40n
            | 1n << 47n
            | (offset >> 16n & 0xFFFFn) << 48n;
        write64(IDT + 6 * 16, gate(BigInt(HANDLER), 0x08, 0xE));

        // Handler: add qword [rsp], 2 (skip the faulting ud2); store vector; iretq
        const handler = [
            0x48, 0x83, 0x04, 0x24, 0x02,                               // add qword [rsp], 2
            0x48, 0xC7, 0x04, 0x25, 0x00, 0x60, 0x00, 0x00, 0x06, 0x00, 0x00, 0x00, // mov qword [0x6000], 6
            0x48, 0xCF,                                                 // iretq
        ];
        for(let i = 0; i < handler.length; i++)
        {
            ex.write8(HANDLER + i, handler[i]);
        }

        // Program: ud2 (0F 0B), then hlt
        const program = [0x0F, 0x0B, 0xF4];
        for(let i = 0; i < program.length; i++)
        {
            ex.write8(BASE + i, program[i]);
        }

        cpu.idtr_offset[0] = IDT;
        cpu.idtr_size[0] = 0xFFF;
        cpu.instruction_pointer[0] = BASE;
        u32[16 + 4] = 0x80000;
        cpu.flags[0] &= ~(1 << 9);
        cpu.in_hlt[0] = 0;
        write64(0x6000, 0n);

        ex.enter_long_mode(PML4);
        ex.main_loop();

        assert.equal(guest_read64(0x6000), 6n, "#UD handler ran");
        assert.equal(cpu.in_hlt[0], 1, "halted after iretq");
    }

    // Test 13: memory operand with disp8 + immediate (ModRM/SIB ordering)
    {
        const program = [
            0x48, 0xC7, 0xC0, 0x00, 0x00, 0x04, 0x00, // mov rax, 0x40000
            0x48, 0xC7, 0xC3, 0x11, 0x00, 0x00, 0x00, // mov rbx, 0x11
            0x48, 0x89, 0x58, 0x10,                   // mov [rax+0x10], rbx
            0x48, 0x83, 0x40, 0x10, 0x29,             // add qword [rax+0x10], 0x29
            0x48, 0x8B, 0x48, 0x10,                   // mov rcx, [rax+0x10]
            0xF4,
        ];
        load(program, 6);
        assert.equal(reg64(1), 0x3An, "rcx after disp8 add");
        assert.equal(guest_read64(0x40010), 0x3An, "memory after disp8 add");
    }

    // Test 14: 64-bit virtual address above 4 GiB (kernel-style high address)
    {
        const write64 = (address, value) => {
            ex.write32(address, Number(value & 0xFFFF_FFFFn));
            ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
        };

        const PML4 = 0x10000;
        const PDPT = 0x11000;
        const PD = 0x12000;
        write64(PML4, BigInt(PDPT) | 0x3n);
        write64(PDPT, BigInt(PD) | 0x3n);
        for(let i = 0; i < 512; i++)
        {
            write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);
        }

        // Map 0xffffffff80000000 -> physical 0 with a 2 MiB page:
        //   PML4 index 511, PDPT index 510, PD index 0
        const PDPT_HI = 0x13000;
        const PD_HI = 0x14000;
        write64(PML4 + 511 * 8, BigInt(PDPT_HI) | 0x3n);
        write64(PDPT_HI + 510 * 8, BigInt(PD_HI) | 0x3n);
        write64(PD_HI, 0x83n); // 2 MiB page, present|rw|ps, physical 0

        // Program at physical 0: mov rax,0x1234; mov rbx,0x5678; add rax,rbx; hlt
        const program = [
            0x48, 0xC7, 0xC0, 0x34, 0x12, 0x00, 0x00,
            0x48, 0xC7, 0xC3, 0x78, 0x56, 0x00, 0x00,
            0x48, 0x01, 0xD8,
            0xF4,
        ];
        for(let i = 0; i < program.length; i++)
        {
            ex.write8(i, program[i]);
        }

        cpu.idtr_offset[0] = 0;
        cpu.flags[0] &= ~(1 << 9);
        cpu.in_hlt[0] = 0;
        u32[16 + 4] = 0x80000; // rsp (identity mapped)

        ex.enter_long_mode(PML4);
        // Set the 64-bit RIP directly (cannot be represented in instruction_pointer)
        new DataView(buffer).setBigUint64(232, 0xFFFF_FFFF_8000_0000n, true);
        ex.main_loop();

        assert.equal(reg64(0), 0x68ACn, "high-address program ran");
        assert.equal(cpu.in_hlt[0], 1, "halted");
    }

    // Test 15: 8/16-bit ADC/SBB/IMUL sign-extend from the operand width.
    // 0xFF is -1 as i8, not 255; 0xFFFF is -1 as i16, not 65535.
    {
        cpu.cr[0] = 0;
        cpu.in_hlt[0] = 0;
        load([
            0xB0, 0xFF,             // mov al, -1
            0xA8, 0x00,             // test al, 0 (CF=0)
            0x14, 0x01,             // adc al, 1
            0xF4,
        ], 4);
        assert.equal(reg64(0) & 0xFFn, 0n, "adc al, 1 result");
        assert.equal(cpu.flags[0] & 1, 1, "adc al, 1 CF");
        assert.equal(cpu.flags[0] & (1 << 11), 0, "adc al, 1 OF");

        load([
            0x30, 0xC0,             // xor al, al
            0x80, 0xD8, 0x01,       // sbb al, 1
            0xF4,
        ], 3);
        assert.equal(reg64(0) & 0xFFn, 0xFFn, "sbb al, 1 result");
        assert.equal(cpu.flags[0] & 1, 1, "sbb al, 1 CF");
        assert.equal(cpu.flags[0] & (1 << 11), 0, "sbb al, 1 OF");

        load([
            0xB8, 0xFF, 0xFF, 0x00, 0x00, // mov eax, 0xFFFF
            0x83, 0xC0, 0x00,             // add eax, 0 (CF=0)
            0x66, 0x15, 0x01, 0x00,       // adc ax, 1
            0xF4,
        ], 4);
        assert.equal(reg64(0) & 0xFFFFn, 0n, "adc ax, 1 result");
        assert.equal(cpu.flags[0] & 1, 1, "adc ax, 1 CF");
        assert.equal(cpu.flags[0] & (1 << 11), 0, "adc ax, 1 OF");

        load([
            0xB8, 0x00, 0x80, 0x00, 0x00, // mov eax, 0x8000
            0x66, 0x6B, 0xC0, 0x02,       // imul ax, ax, 2
            0xF4,
        ], 3);
        assert.equal(reg64(0) & 0xFFFFn, 0n, "imul ax, 2 result");
        assert.equal(cpu.flags[0] & 1, 1, "imul ax, 2 CF");
        assert.equal(cpu.flags[0] & (1 << 11), 1 << 11, "imul ax, 2 OF");
    }

    // Test 16: #PF on a canonical address above 4 GiB keeps the full CR2.
    {
        const write64 = (address, value) => {
            ex.write32(address, Number(value & 0xFFFF_FFFFn));
            ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
        };

        const PML4 = 0x10000;
        const PDPT = 0x11000;
        const PD = 0x12000;
        write64(PML4, BigInt(PDPT) | 0x3n);
        write64(PDPT, BigInt(PD) | 0x3n);
        for(let i = 0; i < 512; i++)
        {
            write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);
        }

        const FAULT = 0xFFFF_8000_0000_1000n;
        const HANDLER = 0x3000;
        const IDT = 0x20000;
        const gate = (offset, selector, type) =>
            (offset & 0xFFFFn)
            | BigInt(selector) << 16n
            | BigInt(type) << 40n
            | 1n << 47n
            | (offset >> 16n & 0xFFFFn) << 48n;
        write64(IDT + 14 * 16, gate(BigInt(HANDLER), 0x08, 0xE));

        const handler = [
            0x0F, 0x20, 0xD0,                                           // mov rax, cr2
            0x48, 0x89, 0x04, 0x25, 0x00, 0x60, 0x00, 0x00,             // mov [0x6000], rax
            0x48, 0x8B, 0x04, 0x24,                                     // mov rax, [rsp]
            0x48, 0x89, 0x04, 0x25, 0x08, 0x60, 0x00, 0x00,             // mov [0x6008], rax
            0xF4,
        ];
        for(let i = 0; i < handler.length; i++)
        {
            ex.write8(HANDLER + i, handler[i]);
        }

        const program = [
            0x48, 0xB8,                                                 // mov rax, imm64
            0x00, 0x10, 0x00, 0x00, 0x00, 0x80, 0xFF, 0xFF,
            0x48, 0x8B, 0x00,                                           // mov rax, [rax]
            0xF4,
        ];
        for(let i = 0; i < program.length; i++)
        {
            ex.write8(BASE + i, program[i]);
        }

        cpu.idtr_offset[0] = IDT;
        cpu.idtr_size[0] = 0xFFF;
        cpu.instruction_pointer[0] = BASE;
        u32[16 + 4] = 0x80000;
        u32[32 + 4] = 0;
        cpu.flags[0] &= ~(1 << 9);
        cpu.in_hlt[0] = 0;
        cpu.cpl[0] = 0;
        write64(0x6000, 0n);
        write64(0x6008, 0xFFFF_FFFF_FFFF_FFFFn);

        ex.enter_long_mode(PML4);
        ex.main_loop();

        assert.equal(guest_read64(0x6000), FAULT, "cr2 from #PF handler");
        assert.equal(guest_read64(0x6008), 0n, "#PF error code");
        assert.equal(new DataView(buffer).getBigUint64(1064, true), FAULT, "cr2 slot");
        assert.equal(cpu.cr[2] >>> 0, 0x1000, "cr2 low 32 bits");
        assert.equal(cpu.in_hlt[0], 1, "halted in #PF handler");
    }

    // Test 17: cpuid reports an Intel 64 CPU, and 32-bit results zero-extend.
    {
        cpu.cr[0] = 0;
        cpu.in_hlt[0] = 0;
        load([
            0xB8, 0x00, 0x00, 0x00, 0x00,       // mov eax, 0
            0x0F, 0xA2,                         // cpuid
            0xF4,
        ], 2);
        assert.equal(reg64(3), 0x756E6547n, "vendor ebx");
        assert.equal(reg64(2), 0x49656E69n, "vendor edx");
        assert.equal(reg64(1), 0x6C65746En, "vendor ecx");

        load([
            0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, -1
            0xB8, 0x01, 0x00, 0x00, 0x00,             // mov eax, 1
            0x0F, 0xA2,
            0xF4,
        ], 3);
        assert.equal(reg64(0) >> 32n, 0n, "cpuid zero-extends rax");
        assert.equal(Number(reg64(0) & 0xFFF_FFFFn), 0x806E9, "family 6 model 142 stepping 9");

        // Brand string, leaf 0x80000002: "Intel(R) Core(TM)" in EAX/EBX/ECX/EDX.
        load([
            0xB8, 0x02, 0x00, 0x00, 0x80,       // mov eax, 0x80000002
            0x0F, 0xA2,
            0xF4,
        ], 2);
        const brand = [reg64(0), reg64(3), reg64(1), reg64(2)]
            .map(r => {
                const v = Number(r & 0xFFFF_FFFFn);
                return String.fromCharCode(v & 0xFF, v >> 8 & 0xFF, v >> 16 & 0xFF, v >> 24 & 0xFF);
            })
            .join("");
        assert.equal(brand, "Intel(R) Core(TM", "brand string");

        load([
            0xB8, 0x01, 0x00, 0x00, 0x80,       // mov eax, 0x80000001
            0x0F, 0xA2,
            0xF4,
        ], 2);
        assert.equal(reg64(2) & (1n << 29n), 1n << 29n, "LM");
        assert.equal(reg64(2) & (1n << 20n), 0n, "NX not claimed");
        assert.equal(reg64(2) & (1n << 11n), 0n, "SYSCALL not claimed");
    }

    console.log("interp64: all tests passed");
    process.exit(0);
});
