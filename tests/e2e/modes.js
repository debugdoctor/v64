#!/usr/bin/env node

// Roadmap: a guest walks real mode, protected mode and long mode itself.
//
// Reset leaves the CPU in real mode at the firmware entry. The payload there
// sets CR0.PE and far-jumps into 32-bit protected mode, arms EFER.LME, enables
// paging and far-jumps into 64-bit code, which writes a line to COM1. The test
// reads that line. It does not call the direct-boot entry.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/e2e/modes.js`

import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";

const MESSAGE = "long mode\n";
const RESET = 0xFFFF0;

const emulator = new v64({
    autostart: false,
    memory_size: 8 * 1024 * 1024,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () =>
{
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const serial = [];
    emulator.add_listener("serial0-output-byte", byte => serial.push(byte));

    // Real mode at F000:FFF0. The operand-size prefix makes the control
    // register write and the far jump 32-bit, which is how firmware leaves
    // real mode.
    const real = [
        0xFA,                               // cli
        0x66, 0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, CR0.PE
        0x0F, 0x22, 0xC0,                   // mov cr0, eax
        0x66, 0xEA, 0x00, 0x20, 0x00, 0x00, // jmp far 0008:00002000
        0x08, 0x00,
    ];

    // Protected mode at 0x2000. Arm long mode, point CR3 at a PML4, enable PAE
    // and then paging. The paging write is the transition.
    const prot = [
        0xB9, 0x80, 0x00, 0x00, 0xC0,       // mov ecx, EFER
        0xB8, 0x00, 0x01, 0x00, 0x00,       // mov eax, EFER.LME
        0x31, 0xD2,                         // xor edx, edx
        0x0F, 0x30,                         // wrmsr
        0xB8, 0x00, 0x30, 0x00, 0x00,       // mov eax, pml4
        0x0F, 0x22, 0xD8,                   // mov cr3, eax
        0xB8, 0x20, 0x00, 0x00, 0x00,       // mov eax, CR4.PAE
        0x0F, 0x22, 0xE0,                   // mov cr4, eax
        0xB8, 0x01, 0x00, 0x00, 0x80,       // mov eax, PE|PG
        0x0F, 0x22, 0xC0,                   // mov cr0, eax
        0xFF, 0x2D, 0x00, 0x00, 0x00, 0x00, // jmp far [rip+0]
        0x40, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // target 0x2040
        0x18, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // selector 0x18
    ];

    const long = [];
    for(const byte of Buffer.from(MESSAGE))
    {
        long.push(
            0xBA, 0xFD, 0x03, 0x00, 0x00,   // mov edx, 0x3FD
            0xEC,                           // in al, dx
            0xA8, 0x20,                     // test al, 0x20
            0x74, 0xFB,                     // jz wait
            0xBA, 0xF8, 0x03, 0x00, 0x00,   // mov edx, 0x3F8
            0xB0, byte,                     // mov al, byte
            0xEE,                           // out dx, al
        );
    }
    long.push(0xF4);

    const write = (address, bytes) => {
        for(let i = 0; i < bytes.length; i++) ex.write8(address + i, bytes[i]);
    };
    const write64 = (address, value) => {
        for(let i = 0; i < 8; i++) ex.write8(address + i, Number(value >> BigInt(8 * i) & 0xFFn));
    };

    // GDT: null, 32-bit code (0x08), 32-bit data (0x10), 64-bit code (0x18).
    const gdt = [
        0, 0, 0, 0, 0, 0, 0, 0,
        0xFF, 0xFF, 0, 0, 0, 0x9A, 0xCF, 0,
        0xFF, 0xFF, 0, 0, 0, 0x92, 0xCF, 0,
        0xFF, 0xFF, 0, 0, 0, 0x9A, 0xAF, 0,
        0xFF, 0xFF, 0, 0, 0, 0x92, 0xCF, 0,
    ];
    write(0x1000, gdt);
    cpu.gdtr_size[0] = gdt.length - 1;
    cpu.gdtr_offset[0] = 0x1000;

    // Identity-map the first 2 MiB so the long-mode code is reachable.
    write64(0x3000, 0x4007n);
    write64(0x4000, 0x5007n);
    write64(0x5000, 0x87n);

    write(RESET, real);
    write(0x2000, prot);
    write(0x2040, long);

    let steps = 0;
    while(!cpu.in_hlt[0] && steps++ < 100000)
    {
        ex.main_loop();
    }

    const text = Buffer.from(serial).toString("utf8");
    assert.equal(cpu.in_hlt[0], 1, "halted after reaching long mode");
    assert.equal(text, MESSAGE, "COM1 after real, protected and long mode");
    console.log("e2e modes: test passed");
    process.exit(0);
});
