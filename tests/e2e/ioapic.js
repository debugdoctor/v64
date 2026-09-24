#!/usr/bin/env node

// A 64-bit kernel takes a device interrupt through the I/O APIC.
//
// The kernel enables the local APIC, programs I/O APIC redirection entry 0 to
// deliver the 8254 timer (ISA IRQ0) to a vector, and waits for three ticks.
// The interrupt is raised by the device layer, routed through the I/O APIC to
// the local APIC, and acknowledged there.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/e2e/ioapic.js`

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { v64 } from "../../src/main.js";

const LOCAL_APIC = 0xFEE00000;
const IO_APIC = 0xFEC00000;
const HANDLER = 0x9000;
const IDT = 0x8000;
const PD2 = 0x5000;
const VECTOR = 0x50;
const TICKS = 3;

// PML4/PDPT as set up by load_kernel64.
const PDPT = 0x2000;

const kernel = [
    0xB9, 0x1B, 0x00, 0x00, 0x00,                   // mov ecx, 0x1B (IA32_APIC_BASE)
    0xB8, 0x00, 0x08, 0xE0, 0xFE,                   // mov eax, 0xFEE00800
    0x31, 0xD2,                                     // xor edx, edx
    0x0F, 0x30,                                     // wrmsr
    0xBB, 0x00, 0x00, 0xE0, 0xFE,                   // mov ebx, LOCAL_APIC
    0xB8, 0xFF, 0x01, 0x00, 0x00,                   // mov eax, 0x1FF (SVR)
    0x67, 0x89, 0x83, 0xF0, 0x00, 0x00, 0x00,       // mov [ebx+0xF0], eax
    0xBE, 0x00, 0x00, 0xC0, 0xFE,                   // mov esi, IO_APIC
    0xB8, 0x10, 0x00, 0x00, 0x00,                   // mov eax, 0x10 (IOREGSEL = IRQ0 low)
    0x67, 0x89, 0x06,                               // mov [esi], eax
    0xB8, 0x50, 0x00, 0x00, 0x00,                   // mov eax, VECTOR (edge, unmasked)
    0x67, 0x89, 0x46, 0x10,                         // mov [esi+0x10], eax
    0xB8, 0x11, 0x00, 0x00, 0x00,                   // mov eax, 0x11 (IOREGSEL = IRQ0 high)
    0x67, 0x89, 0x06,                               // mov [esi], eax
    0x31, 0xC0,                                     // xor eax, eax
    0x67, 0x89, 0x46, 0x10,                         // mov [esi+0x10], eax (destination 0)
    0xB0, 0x36, 0xE6, 0x43,                         // PIT ch0, mode 3
    0xB0, 0x9C, 0xE6, 0x40,                         // divisor low
    0xB0, 0x2E, 0xE6, 0x40,                         // divisor high
    0xC7, 0x04, 0x25, 0x00, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov dword [0x6000], 0
    0xFB,                                           // sti
    0x8B, 0x04, 0x25, 0x00, 0x60, 0x00, 0x00,       // poll: mov eax, [0x6000]
    0x83, 0xF8, TICKS,                              // cmp eax, TICKS
    0x72, 0xF4,                                     // jb poll
    0xFA,                                           // cli
    0xF4,                                           // hlt
];

const handler = [
    0xFF, 0x04, 0x25, 0x00, 0x60, 0x00, 0x00,       // inc dword [0x6000]
    0xBB, 0x00, 0x00, 0xE0, 0xFE,                   // mov ebx, LOCAL_APIC
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

const image_path = path.join(os.tmpdir(), "v64-e2e-ioapic.img");
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

    // Map both APIC pages into the boot page tables.
    write64(PDPT + 3 * 8, BigInt(PD2) | 0x7n);
    for(const address of [LOCAL_APIC, IO_APIC])
    {
        const pd_index = (address >> 21) & 0x1FF;
        write64(PD2 + pd_index * 8, BigInt(address) | 0x87n);
    }

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
    assert.equal(read32(0x6000), TICKS, "the I/O APIC delivered the timer interrupt three times");

    console.log("e2e ioapic: test passed");
    process.exit(0);
});
