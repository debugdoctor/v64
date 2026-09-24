#!/usr/bin/env node

// A 64-bit kernel brings up the local APIC and takes its timer interrupt.
//
// Direct boot starts the kernel; the host maps the APIC MMIO page and installs
// an IDT gate. The kernel enables the APIC through IA32_APIC_BASE, reads the
// version register, programs the LVT timer and waits for three ticks, which
// the handler acknowledges through the APIC EOI register.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/e2e/apic.js`

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { v64 } from "../../src/main.js";

const APIC = 0xFEE00000;
const HANDLER = 0x9000;
const IDT = 0x8000;
const PD2 = 0x5000;
const VECTOR = 0x40;
const TICKS = 3;

// PML4/PDPT/PD as set up by load_kernel64.
const PML4 = 0x1000;
const PDPT = 0x2000;

const kernel = [
    0xB9, 0x1B, 0x00, 0x00, 0x00,                   // mov ecx, 0x1B (IA32_APIC_BASE)
    0xB8, 0x00, 0x08, 0xE0, 0xFE,                   // mov eax, 0xFEE00800
    0x31, 0xD2,                                     // xor edx, edx
    0x0F, 0x30,                                     // wrmsr
    0xBB, 0x00, 0x00, 0xE0, 0xFE,                   // mov ebx, APIC
    0x67, 0x8B, 0x43, 0x20,                         // mov eax, [ebx+0x20] (ID)
    0x89, 0x04, 0x25, 0x00, 0x60, 0x00, 0x00,       // mov [0x6000], eax
    0x67, 0x8B, 0x43, 0x30,                         // mov eax, [ebx+0x30] (version)
    0x89, 0x04, 0x25, 0x04, 0x60, 0x00, 0x00,       // mov [0x6004], eax
    0xB8, 0xFF, 0x01, 0x00, 0x00,                   // mov eax, 0x1FF (SVR: enabled | 0xFF)
    0x67, 0x89, 0x83, 0xF0, 0x00, 0x00, 0x00,       // mov [ebx+0xF0], eax
    0x31, 0xC0,                                     // xor eax, eax
    0x67, 0x89, 0x83, 0x80, 0x00, 0x00, 0x00,       // mov [ebx+0x80], eax (TPR)
    0xB8, 0x40, 0x00, 0x02, 0x00,                   // mov eax, VECTOR | periodic (1<<17)
    0x67, 0x89, 0x83, 0x20, 0x03, 0x00, 0x00,       // mov [ebx+0x320], eax (LVT timer)
    0xB8, 0x03, 0x00, 0x00, 0x00,                   // mov eax, 3
    0x67, 0x89, 0x83, 0xE0, 0x03, 0x00, 0x00,       // mov [ebx+0x3E0], eax (divide)
    0xB8, 0x98, 0x3A, 0x00, 0x00,                   // mov eax, 15000
    0x67, 0x89, 0x83, 0x80, 0x03, 0x00, 0x00,       // mov [ebx+0x380], eax (initial count)
    0xC7, 0x04, 0x25, 0x10, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov dword [0x6010], 0
    0xFB,                                           // sti
    0x8B, 0x04, 0x25, 0x10, 0x60, 0x00, 0x00,       // poll: mov eax, [0x6010]
    0x83, 0xF8, TICKS,                              // cmp eax, TICKS
    0x72, 0xF4,                                     // jb poll
    0xFA,                                           // cli
    0xF4,                                           // hlt
];

const handler = [
    0xFF, 0x04, 0x25, 0x10, 0x60, 0x00, 0x00,       // inc dword [0x6010]
    0xBB, 0x00, 0x00, 0xE0, 0xFE,                   // mov ebx, APIC
    0x31, 0xC0,                                     // xor eax, eax
    0x67, 0x89, 0x83, 0xB0, 0x00, 0x00, 0x00,       // mov [ebx+0xB0], eax (EOI)
    0x48, 0xCF,                                     // iretq
];

function bz_image(body)
{
    const setup_sects = 4;
    const prot_start = (setup_sects + 1) * 512;
    const image = new Uint8Array((prot_start + 0x200 + body.length + 511) & ~511);
    image[0x1F1] = setup_sects;
    image[0x1FE] = 0x55;
    image[0x1FF] = 0xAA;
    image[0x201] = 0x40;
    image.set([0x48, 0x64, 0x72, 0x53], 0x202);
    image[0x206] = 0x0C;
    image[0x207] = 0x02;
    image[0x238] = 0xFF;
    image.set(body, prot_start + 0x200);
    return image;
}

const image_path = path.join(os.tmpdir(), "v64-e2e-apic.img");
fs.writeFileSync(image_path, bz_image(kernel));

const emulator = new v64({
    autostart: false,
    memory_size: 8 * 1024 * 1024,
    disable_jit: 1,
    log_level: 0,
    bzimage: { url: image_path, async: false },
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
    const read32 = address =>
        (ex.read8(address) | ex.read8(address + 1) << 8 | ex.read8(address + 2) << 16 | ex.read8(address + 3) << 24) >>> 0;

    // Map the APIC MMIO page into the boot page tables.
    const pd_index = (APIC >> 21) & 0x1FF;
    write64(PDPT + 3 * 8, BigInt(PD2) | 0x7n);
    write64(PD2 + pd_index * 8, BigInt(APIC) | 0x87n);

    for(let i = 0; i < handler.length; i++) ex.write8(HANDLER + i, handler[i]);
    const gate = (offset, selector, type) => {
        const o = BigInt(offset);
        return (o & 0xFFFFn) | BigInt(selector) << 16n | BigInt(type) << 40n | 1n << 47n | (o >> 16n & 0xFFFFn) << 48n;
    };
    write64(IDT + VECTOR * 16, gate(HANDLER, 0x08, 0xE));
    cpu.idtr_offset[0] = IDT;
    cpu.idtr_size[0] = 0xFFF;
    if(ex.full_clear_tlb) ex.full_clear_tlb();

    const deadline = Date.now() + 15000;
    while(!cpu.in_hlt[0] && Date.now() < deadline) ex.main_loop();

    assert.equal(cpu.in_hlt[0], 1, "the kernel halted after the ticks");
    assert.equal((read32(0x6004) & 0xFF), 0x14, "APIC version register");
    assert.equal(read32(0x6010), TICKS, "the APIC timer interrupt fired three times");

    console.log("e2e apic: test passed");
    process.exit(0);
});
