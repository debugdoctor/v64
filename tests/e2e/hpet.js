#!/usr/bin/env node

// A 64-bit kernel uses the HPET as a clocksource.
//
// The kernel reads the capabilities, enables the main counter and waits for it
// to advance. The ACPI HPET table (built for direct boot) points at 0xFED00000.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/e2e/hpet.js`

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { v64 } from "../../src/main.js";

const HPET = 0xFED00000;
const IO_APIC = 0xFEC00000;
const PD2 = 0x5000;
const PDPT = 0x2000;

const kernel = [
    0xBB, 0x00, 0x00, 0xD0, 0xFE,                   // mov ebx, HPET
    0x67, 0x48, 0x8B, 0x03,                         // mov rax, [ebx] (capabilities)
    0x48, 0x89, 0x04, 0x25, 0x00, 0xE0, 0x00, 0x00, // mov [0xE000], rax
    0xB8, 0x01, 0x00, 0x00, 0x00,                   // mov eax, 1
    0x67, 0x89, 0x43, 0x10,                         // mov [ebx+0x10], eax (enable counter)
    0x67, 0x48, 0x8B, 0x83, 0xF0, 0x00, 0x00, 0x00, // mov rax, [ebx+0xF0] (main counter)
    0x48, 0x89, 0x04, 0x25, 0x08, 0xE0, 0x00, 0x00, // mov [0xE008], rax
    0x67, 0x48, 0x8B, 0x93, 0xF0, 0x00, 0x00, 0x00, // poll: mov rdx, [ebx+0xF0]
    0x48, 0x39, 0xC2,                               // cmp rdx, rax
    0x74, 0xF3,                                     // je poll
    0x48, 0x89, 0x14, 0x25, 0x10, 0xE0, 0x00, 0x00, // mov [0xE010], rdx
    0xF4,                                           // hlt
];

function bzImage(body)
{
    const setupSects = 4;
    const protStart = (setupSects + 1) * 512;
    const image = new Uint8Array((protStart + 0x200 + body.length + 511) & ~511);
    image[0x1F1] = setupSects;
    image[0x1FE] = 0x55;
    image[0x1FF] = 0xAA;
    image[0x201] = 0x40;
    image.set([0x48, 0x64, 0x72, 0x53], 0x202);
    image[0x206] = 0x0C;
    image[0x207] = 0x02;
    image[0x238] = 0xFF;
    image.set(body, protStart + 0x200);
    return image;
}

const imagePath = path.join(os.tmpdir(), "v64-e2e-hpet.img");
fs.writeFileSync(imagePath, bzImage(kernel));

const emulator = new v64({
    autostart: false,
    memory_size: 8 * 1024 * 1024,
    disable_jit: 1,
    log_level: 0,
    bzimage: { url: imagePath, async: false },
    cmdline: "",
    direct_boot: true,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;

    const write64 = (address, value) => {
        ex.write32(address, Number(value & 0xFFFF_FFFFn));
        ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
    };
    const read64 = address => {
        let value = 0n;
        for(let i = 0; i < 8; i++) value |= BigInt(ex.read8(address + i) >>> 0) << BigInt(i * 8);
        return value;
    };

    // Map the IO-APIC/HPET 2 MiB page.
    write64(PDPT + 3 * 8, BigInt(PD2) | 0x7n);
    const pdIndex = (IO_APIC >> 21) & 0x1FF;
    write64(PD2 + pdIndex * 8, BigInt(IO_APIC) | 0x87n);
    if(ex.full_clear_tlb) ex.full_clear_tlb();

    const deadline = Date.now() + 15000;
    while(!cpu.in_hlt[0] && Date.now() < deadline) ex.main_loop();

    assert.equal(cpu.in_hlt[0], 1, "the kernel halted once the counter advanced");

    const caps = read64(0xE000);
    assert.equal(Number(caps >> 32n), 10_000_000, "counter period is 10 ns (100 MHz)");
    assert.equal(Number(caps >> 8n & 0x1Fn), 2, "three timers");
    assert.equal(Number(caps >> 13n & 1n), 1, "64-bit counter");
    assert.equal(Number(caps >> 15n & 1n), 1, "legacy route capable");

    const first = read64(0xE008);
    const second = read64(0xE010);
    assert.ok(second > first, "the main counter advanced (" + first + " -> " + second + ")");

    console.log("e2e hpet: test passed");
    process.exit(0);
});
