#!/usr/bin/env node

// Regression test for the 64-bit interpreter's decode cache: a block cached in
// one physical page must be re-decoded after the guest writes that page. The
// cache used to memcmp the block bytes on every hit; it now compares a
// per-page write counter, so a write that misses the counter would run stale
// instructions.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp64/smc.js`

import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";

const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const PT = 0x13000;
const CODE = 0x2000;

const emulator = new v64({
    autostart: false,
    memory_size: 2 * 1024 * 1024,
    disable_jit: true,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const u32 = new Uint32Array(ex.memory.buffer);

    const w64 = (address, value) => {
        ex.write32(address, Number(value & 0xFFFF_FFFFn));
        ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
    };
    const w = (address, bytes) => {
        for(let i = 0; i < bytes.length; i++) ex.write8(address + i, bytes[i]);
    };

    // Identity map the first 2 MiB with 4 KiB pages.
    w64(PML4, BigInt(PDPT) | 3n);
    w64(PDPT, BigInt(PD) | 3n);
    w64(PD, BigInt(PT) | 3n);
    for(let i = 0; i < 512; i++) w64(PT + i * 8, BigInt(i) * 0x1000n | 3n);

    // mov ecx,3 / loop: mov eax,1 / mov byte [0x2006],2 / dec ecx / jnz loop / hlt
    // The first pass rewrites the immediate of `mov eax,1` to 2, so a correct
    // re-decode leaves eax == 2; a stale cached block leaves it at 1.
    // `mov [disp32], imm8` needs a SIB byte: mod=00/rm=101 is RIP-relative.
    w(CODE, [
        0xB9, 0x03, 0x00, 0x00, 0x00,
        0xB8, 0x01, 0x00, 0x00, 0x00,
        0xC6, 0x04, 0x25, 0x06, 0x20, 0x00, 0x00, 0x02,
        0xFF, 0xC9,
        0x75, 0xEF,
        0xF4,
    ]);

    cpu.cr[0] = 0;
    cpu.is_32[0] = 1;
    cpu.instruction_pointer[0] = CODE;
    u32[16 + 4] = 0x80000; // rsp
    u32[32 + 4] = 0;
    cpu.flags[0] = 0x2;
    cpu.in_hlt[0] = 0;
    cpu.cpl[0] = 0;
    ex.enter_long_mode(PML4);

    let guard = 0;
    while(!cpu.in_hlt[0] && guard++ < 100000) ex.main_loop();

    assert.equal(cpu.in_hlt[0], 1, "the program halted");
    assert.equal(u32[16] >>> 0, 2, "self-modified `mov eax, imm32` is re-decoded");

    console.log("interp64 smc: all tests passed");
    process.exit(0);
});
