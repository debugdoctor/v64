#!/usr/bin/env node

// Direct 64-bit boot: ACPI tables for CPU enumeration.
//
// A firmware-less boot has no ACPI tables, so the loader synthesizes RSDP,
// RSDT, XSDT and MADT and publishes the RSDP address in boot_params. A
// long-mode kernel then walks them, exactly as Linux does, to find the local
// and I/O APIC.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/e2e/acpi.js`

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { v64 } from "../../src/main.js";

const RSDP = 0xF0000;
const XSDT = 0xF0080;
const MADT = 0xF0100;
const LOCAL_APIC = 0xFEE00000;
const IO_APIC = 0xFEC00000;

const kernel = [
    0x48, 0x89, 0xF3,                               // mov rbx, rsi (boot_params)
    0x48, 0x8B, 0x43, 0x70,                         // mov rax, [rbx+0x70] (acpi_rsdp_addr)
    0x48, 0x89, 0x04, 0x25, 0x00, 0xE0, 0x00, 0x00, // mov [0xE000], rax
    0x8B, 0x08,                                     // mov ecx, [rax]
    0x89, 0x0C, 0x25, 0x08, 0xE0, 0x00, 0x00,       // mov [0xE008], ecx
    0x8B, 0x48, 0x04,                               // mov ecx, [rax+4]
    0x89, 0x0C, 0x25, 0x0C, 0xE0, 0x00, 0x00,       // mov [0xE00C], ecx
    0x31, 0xC9,                                     // xor ecx, ecx
    0x31, 0xD2,                                     // xor edx, edx
    0x44, 0x0F, 0xB6, 0x04, 0x08,                   // sum: movzx r8d, byte [rax+rcx]
    0x44, 0x01, 0xC2,                               // add edx, r8d
    0xFF, 0xC1,                                     // inc ecx
    0x83, 0xF9, 0x14,                               // cmp ecx, 20
    0x72, 0xF1,                                     // jb sum
    0x81, 0xE2, 0xFF, 0x00, 0x00, 0x00,             // and edx, 0xFF
    0x89, 0x14, 0x25, 0x10, 0xE0, 0x00, 0x00,       // mov [0xE010], edx
    0x4C, 0x8B, 0x48, 0x18,                         // mov r9, [rax+0x18] (XSDT)
    0x4C, 0x89, 0x0C, 0x25, 0x18, 0xE0, 0x00, 0x00, // mov [0xE018], r9
    0x41, 0x8B, 0x09,                               // mov ecx, [r9]
    0x89, 0x0C, 0x25, 0x20, 0xE0, 0x00, 0x00,       // mov [0xE020], ecx
    0x4D, 0x8B, 0x51, 0x24,                         // mov r10, [r9+0x24] (first table)
    0x4C, 0x89, 0x14, 0x25, 0x28, 0xE0, 0x00, 0x00, // mov [0xE028], r10
    0x41, 0x8B, 0x0A,                               // mov ecx, [r10]
    0x89, 0x0C, 0x25, 0x30, 0xE0, 0x00, 0x00,       // mov [0xE030], ecx
    0x41, 0x8B, 0x4A, 0x24,                         // mov ecx, [r10+0x24] (local APIC)
    0x89, 0x0C, 0x25, 0x34, 0xE0, 0x00, 0x00,       // mov [0xE034], ecx
    0x41, 0x8B, 0x4A, 0x28,                         // mov ecx, [r10+0x28] (flags)
    0x89, 0x0C, 0x25, 0x38, 0xE0, 0x00, 0x00,       // mov [0xE038], ecx
    0x41, 0x0F, 0xB6, 0x4A, 0x2C,                   // movzx ecx, byte [r10+0x2C] (entry type)
    0x89, 0x0C, 0x25, 0x3C, 0xE0, 0x00, 0x00,       // mov [0xE03C], ecx
    0x41, 0x8B, 0x4A, 0x30,                         // mov ecx, [r10+0x30] (LAPIC flags)
    0x89, 0x0C, 0x25, 0x40, 0xE0, 0x00, 0x00,       // mov [0xE040], ecx
    0x41, 0x8B, 0x4A, 0x38,                         // mov ecx, [r10+0x38] (I/O APIC address)
    0x89, 0x0C, 0x25, 0x44, 0xE0, 0x00, 0x00,       // mov [0xE044], ecx
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

const imagePath = path.join(os.tmpdir(), "v64-e2e-acpi.img");
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

    const read32 = address =>
        (ex.read8(address) | ex.read8(address + 1) << 8 | ex.read8(address + 2) << 16 | ex.read8(address + 3) << 24) >>> 0;
    const read64 = address => {
        let value = 0n;
        for(let i = 0; i < 8; i++) value |= BigInt(ex.read8(address + i) >>> 0) << BigInt(i * 8);
        return value;
    };

    let guard = 0;
    while(!cpu.in_hlt[0] && guard++ < 1000) ex.main_loop();
    assert.equal(cpu.in_hlt[0], 1, "the kernel halted");

    assert.equal(read64(0xE000), BigInt(RSDP), "boot_params.acpi_rsdp_addr");
    assert.equal(read32(0xE008), 0x20445352, "RSDP signature \"RSD \"");
    assert.equal(read32(0xE00C), 0x20525450, "RSDP signature \"PTR \"");
    assert.equal(read32(0xE010), 0, "RSDP checksum");
    assert.equal(read64(0xE018), BigInt(XSDT), "XSDT address");
    assert.equal(read32(0xE020), 0x54445358, "XSDT signature");
    assert.equal(read64(0xE028), BigInt(MADT), "MADT address");
    assert.equal(read32(0xE030), 0x43495041, "MADT signature \"APIC\"");
    assert.equal(read32(0xE034), LOCAL_APIC, "MADT local APIC address");
    assert.equal(read32(0xE038), 1, "MADT flags");
    assert.equal(read32(0xE03C), 0, "first entry is a Processor Local APIC");
    assert.equal(read32(0xE040), 1, "the local APIC is enabled");
    assert.equal(read32(0xE044), IO_APIC, "I/O APIC address");

    // The FADT and DSDT Linux needs to actually use ACPI.
    const FADT = 0xF0400;
    const DSDT = 0xF0300;
    assert.equal(read32(FADT), 0x50434146, "FADT signature \"FACP\"");
    assert.equal(read32(FADT + 0x28), DSDT, "FADT DSDT pointer");
    assert.equal(read32(FADT + 0x70), 1 << 20, "FADT flags: hardware-reduced ACPI");
    assert.equal(read64(FADT + 0x8C), BigInt(DSDT), "FADT X_DSDT");
    assert.equal(read32(DSDT), 0x54445344, "DSDT signature");

    console.log("e2e acpi: test passed");
    process.exit(0);
});
