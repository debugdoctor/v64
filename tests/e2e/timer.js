#!/usr/bin/env node

// A 64-bit kernel takes timer interrupts.
//
// Direct boot starts the kernel, which programs the 8259 and the 8254 for
// 100 Hz, enables interrupts and waits for three ticks. Each tick is handled
// in long mode, acknowledged and returned from with iretq; the handler prints
// one character on COM1. The test reads COM1 and the tick counter.
//
// The IDT is installed from the host because the kernel would normally use
// LIDT (see tests/interp64/system.js); everything after that is the guest.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/e2e/timer.js`

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { v64 } from "../../src/main.js";

const TICKS = 3;
const HANDLER = 0x5000;
const IDT = 0x8000;
const VECTOR = 0x20;

const kernel = [
    0xB0, 0x11, 0xE6, 0x20, // mov al,0x11; out 0x20,al   (ICW1)
    0xB0, 0x20, 0xE6, 0x21, // mov al,0x20; out 0x21,al   (ICW2: vector 0x20)
    0xB0, 0x04, 0xE6, 0x21, // mov al,0x04; out 0x21,al   (ICW3: slave on IRQ2)
    0xB0, 0x01, 0xE6, 0x21, // mov al,0x01; out 0x21,al   (ICW4)
    0xB0, 0xFE, 0xE6, 0x21, // mov al,0xFE; out 0x21,al   (unmask IRQ0)
    0xB0, 0x36, 0xE6, 0x43, // mov al,0x36; out 0x43,al   (PIT ch0, mode 3)
    0xB0, 0x9C, 0xE6, 0x40, // mov al,0x9C; out 0x40,al   (divisor low)
    0xB0, 0x2E, 0xE6, 0x40, // mov al,0x2E; out 0x40,al   (divisor high)
    0xC7, 0x04, 0x25, 0x00, 0x60, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov dword [0x6000],0
    0xFB,                   // sti
    0x8B, 0x04, 0x25, 0x00, 0x60, 0x00, 0x00, // poll: mov eax,[0x6000]
    0x83, 0xF8, TICKS,      // cmp eax, TICKS
    0x72, 0xF4,             // jb poll
    0xFA,                   // cli
    0xF4,                   // hlt
];

const handler = [
    0xFF, 0x04, 0x25, 0x00, 0x60, 0x00, 0x00, // inc dword [0x6000]
    0xBA, 0xFD, 0x03, 0x00, 0x00,             // mov edx, 0x3FD
    0xEC,                                     // wait: in al, dx
    0xA8, 0x20,                               // test al, 0x20
    0x74, 0xFB,                               // jz wait
    0xBA, 0xF8, 0x03, 0x00, 0x00,             // mov edx, 0x3F8
    0xB0, 0x54,                               // mov al, 'T'
    0xEE,                                     // out dx, al
    0xB0, 0x20,                               // mov al, 0x20
    0xE6, 0x20,                               // out 0x20, al   (EOI)
    0x48, 0xCF,                               // iretq
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

const image_path = path.join(os.tmpdir(), "v64-e2e-timer.img");
fs.writeFileSync(image_path, bz_image(kernel));

const serial = [];
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

emulator.add_listener("serial0-output-byte", byte => serial.push(byte));

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;

    // The handler and the interrupt gate.
    for(let i = 0; i < handler.length; i++)
    {
        ex.write8(HANDLER + i, handler[i]);
    }
    const gate = (offset, selector, type) =>
        BigInt(offset & 0xFFFF)
        | BigInt(selector) << 16n
        | BigInt(type) << 40n
        | 1n << 47n
        | BigInt(offset >> 16 & 0xFFFF) << 48n;
    const entry = gate(HANDLER, 0x08, 0xE);
    ex.write32(IDT + VECTOR * 16, Number(entry & 0xFFFF_FFFFn));
    ex.write32(IDT + VECTOR * 16 + 4, Number(entry >> 32n & 0xFFFF_FFFFn));
    cpu.idtr_offset[0] = IDT;
    cpu.idtr_size[0] = 0xFFF;

    const deadline = Date.now() + 15000;
    while(!cpu.in_hlt[0] && Date.now() < deadline)
    {
        ex.main_loop();
    }

    const counter = ex.read8(0x6000) | ex.read8(0x6001) << 8 | ex.read8(0x6002) << 16 | ex.read8(0x6003) << 24;
    const text = Buffer.from(serial).toString("utf8");

    assert.equal(cpu.in_hlt[0], 1, "the kernel halted after the ticks");
    assert.equal(counter >>> 0, TICKS, "the handler ran once per tick");
    assert.equal(text, "T".repeat(TICKS), "each tick printed one character");

    console.log("e2e timer: test passed");
    process.exit(0);
});
