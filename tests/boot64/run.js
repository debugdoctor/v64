#!/usr/bin/env node

// Test for the direct 64-bit boot path (src/kernel.js load_kernel64,
// src/rust/cpu/interp64.rs boot64).
//
// Builds a fake bzImage whose protected-mode entry runs `mov rax, 0x1234; hlt`,
// boots it and checks that the kernel ran in long mode with the boot loader's
// boot_params in RSI.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/boot64/run.js`

import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";

const emulator = new v64({
    autostart: false,
    memory_size: 4 * 1024 * 1024,
    disable_jit: 1,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const buffer = ex.memory.buffer;
    const u32 = new Uint32Array(buffer);

    const reg64 = i =>
        (i < 8)
            ? (BigInt(u32[32 + i]) << 32n) | BigInt(u32[16 + i])
            : new BigUint64Array(buffer, 160, 8)[i - 8];

    // Fake bzImage: 4 setup sectors, header, protected-mode kernel at
    // (setup_sects + 1) * 512, entry at +0x200.
    const setup_sects = 4;
    const prot_start = (setup_sects + 1) * 512;
    const kernel_code = [
        0x48, 0xB8, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rax, 0x1234
        0x48, 0x89, 0x34, 0x25, 0x00, 0xE0, 0x00, 0x00,             // mov [0xE000], rsi
        0xF4,                                                       // hlt
    ];
    const size = (prot_start + 0x200 + kernel_code.length + 3) & ~3;
    const bzimage = new Uint8Array(size);
    bzimage[0x1F1] = setup_sects;
    bzimage[0x1FE] = 0x55;
    bzimage[0x1FF] = 0xAA; // boot flag 0xAA55
    bzimage[0x201] = 0x40; // setup header size
    bzimage[0x202] = 0x48;
    bzimage[0x203] = 0x64;
    bzimage[0x204] = 0x72;
    bzimage[0x205] = 0x53; // "HdrS"
    bzimage[0x206] = 0x0C;
    bzimage[0x207] = 0x02; // protocol 0x020c (64-bit)
    bzimage.set(kernel_code, prot_start + 0x200);

    cpu.boot_kernel64(bzimage.buffer, undefined, "console=ttyS0");

    // boot_params should be set up and identity-mapped at 0x10000.
    const cmdline_ptr =
        ex.read8(0x10000 + 0x228) |
        ex.read8(0x10000 + 0x229) << 8 |
        ex.read8(0x10000 + 0x22A) << 16 |
        ex.read8(0x10000 + 0x22B) << 24;
    assert.equal(cmdline_ptr >>> 0, 0x80000, "cmdline ptr in boot_params");

    let guard = 0;
    while(!cpu.in_hlt[0] && guard++ < 1000)
    {
        ex.main_loop();
    }

    const guest_read64 = a => {
        let value = 0n;
        for(let i = 0; i < 8; i++) value |= BigInt(ex.read8(a + i) >>> 0) << BigInt(i * 8);
        return value;
    };

    assert.equal(reg64(0), 0x1234n, "kernel ran in long mode");
    assert.equal(cpu.in_hlt[0], 1, "halted");
    assert.equal(guest_read64(0xE000), 0x10000n, "RSI = boot_params");

    console.log("boot64: test passed");
    process.exit(0);
});
