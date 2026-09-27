#!/usr/bin/env node

// Regression test for a page-crossing code fetch in the 64-bit interpreter:
// fetch16/fetch32 read 2/4 bytes from one translated address, so a value
// starting at page end - 1 takes its tail from the next physical page (busybox
// hit this with its RIP-relative store at +0x48FFC).
// cf. Intel SDM Vol. 1, 3.2.1.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp64/code_page_boundary.js`

import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";

const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const PT = 0x13000;
const PHYS_A = 0x180000;   // backs virtual page 0x2000
const PHYS_B = 0x1A0000;   // backs virtual page 0x3000 (not adjacent to PHYS_A)

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
    const u32 = new Uint32Array(ex.memory.buffer);

    const write64 = (address, value) => {
        ex.write32(address, Number(value & 0xFFFF_FFFFn));
        ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
    };
    const w = (address, bytes) => {
        for(let i = 0; i < bytes.length; i++)
        {
            ex.write8(address + i, bytes[i]);
        }
    };

    // Identity map 2 MiB with 4 KiB pages, then remap 0x2000/0x3000 elsewhere.
    write64(PML4, BigInt(PDPT) | 3n);
    write64(PDPT, BigInt(PD) | 3n);
    write64(PD, BigInt(PT) | 3n);
    for(let i = 0; i < 512; i++)
    {
        write64(PT + i * 8, BigInt(i) * 0x1000n | 3n);
    }
    write64(PT + (0x2000 >> 12) * 8, BigInt(PHYS_A) | 3n);
    write64(PT + (0x3000 >> 12) * 8, BigInt(PHYS_B) | 3n);

    // Anything after PHYS_A is unrelated; make a stray wide read obvious.
    for(let i = 0; i < 0x1000; i++)
    {
        ex.write8(PHYS_A + 0x1000 + i, 0xAA);
    }

    cpu.cr[0] = 0;
    cpu.is_32[0] = 1;

    const runAt = (vaddr, steps) =>
    {
        cpu.instruction_pointer[0] = vaddr;
        u32[16 + 4] = 0x80000; // rsp
        u32[32 + 4] = 0;
        cpu.flags[0] = 0x2;
        cpu.in_hlt[0] = 0;
        cpu.cpl[0] = 0;
        ex.enter_long_mode(PML4);
        for(let i = 0; i < steps; i++)
        {
            ex.interp64_run_one();
        }
    };

    // (1) `mov eax, imm32` with the imm32 at page end - 1 (opcode at 0x2FFE).
    {
        ex.write8(PHYS_A + 0xFFE, 0xB8);
        ex.write8(PHYS_A + 0xFFF, 0x78);
        w(PHYS_B, [0x56, 0x34, 0x12, 0xF4]);
        runAt(0x2FFE, 1);
        assert.equal(u32[16] >>> 0, 0x1234_5678,
            "mov eax, imm32 with the immediate starting at page end - 1");
    }

    // (2) `movb disp32(%rip), %al` with the disp32 at page end - 1 (opcode at
    //     0x2FFD). Target = next_rip (0x3003) + disp32.
    {
        const disp = 0x120;
        w(PHYS_A + 0xFFD, [0x88, 0x05]);
        ex.write8(PHYS_A + 0xFFF, disp & 0xFF);
        w(PHYS_B, [disp >> 8 & 0xFF, disp >> 16 & 0xFF, disp >> 24 & 0xFF, 0xF4]);
        ex.write8(PHYS_B + (0x3003 - 0x3000 + disp), 0x00);
        u32[16] = 0x5A; // al
        runAt(0x2FFD, 1);
        assert.equal(ex.read8(PHYS_B + (0x3003 - 0x3000 + disp)), 0x5A,
            "movb disp32(%rip), %al with the disp32 starting at page end - 1");
    }

    console.log("interp64 code page boundary: all tests passed");
    process.exit(0);
});
