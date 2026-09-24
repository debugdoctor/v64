#!/usr/bin/env node

// Roadmap M1: a hand-written 64-bit kernel, started by the Linux 64-bit boot
// protocol, prints on the serial console.
//
// The emulator loads a bzImage, enters long mode at the kernel entry, and
// bytes written to COM1 show up on serial0. The test reads that text and
// nothing else: no registers, no page tables, no JIT counters.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/e2e/serial.js`

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { v64 } from "../../src/main.js";

const CMDLINE = "console=ttyS0 root=/dev/ram0";

// 8250 at 0x3F8, polled the way early Linux prints a line: wait until the
// transmitter holding register is empty, then write the byte.
function uartByte(value)
{
    return [
        0xBA, 0xFD, 0x03, 0x00, 0x00, // mov edx, 0x3FD
        0xEC,                         // in al, dx
        0xA8, 0x20,                   // test al, 0x20
        0x74, 0xFB,                   // jz .wait
        0xBA, 0xF8, 0x03, 0x00, 0x00, // mov edx, 0x3F8
        0xB0, value,                  // mov al, byte
        0xEE,                         // out dx, al
    ];
}

// Same poll, but the byte is the one the command line walk already loaded.
function uartAl()
{
    return [
        0xBA, 0xFD, 0x03, 0x00, 0x00, // mov edx, 0x3FD
        0xEC,                         // in al, dx
        0xA8, 0x20,                   // test al, 0x20
        0x74, 0xFB,                   // jz .wait
        0x8A, 0x1E,                   // mov bl, [rsi]
        0xBA, 0xF8, 0x03, 0x00, 0x00, // mov edx, 0x3F8
        0x88, 0xD8,                   // mov al, bl
        0xEE,                         // out dx, al
    ];
}

const code = [];
const labels = {};
const patches = [];
const emit = (...bytes) => code.push(...bytes);
const label = name => { labels[name] = code.length; };
const jz = name => { patches.push({ at: code.length, name, near: false }); emit(0x74, 0); };
const jmp = name => { patches.push({ at: code.length, name, near: true }); emit(0xE9, 0, 0, 0, 0); };

emit(...[..."v64 serial\n"].flatMap(c => uartByte(c.charCodeAt(0))));
emit(0x48, 0x89, 0xF0);                   // mov rax, rsi (boot_params)
emit(0x8B, 0x80, 0x28, 0x02, 0x00, 0x00); // mov eax, [rax + 0x228]
emit(0x48, 0x89, 0xC6);                   // mov rsi, rax
label("next");
emit(0x8A, 0x06);                         // mov al, [rsi]
emit(0x84, 0xC0);                         // test al, al
jz("done");
emit(...uartAl());
emit(0x48, 0xFF, 0xC6);                   // inc rsi
jmp("next");
label("done");
emit(...uartByte(0x0A));
emit(0xF4);                               // hlt

for(const patch of patches)
{
    const from = patch.at + (patch.near ? 5 : 2);
    const displacement = labels[patch.name] - from;
    if(patch.near)
    {
        code[patch.at + 1] = displacement & 0xFF;
        code[patch.at + 2] = displacement >> 8 & 0xFF;
        code[patch.at + 3] = displacement >> 16 & 0xFF;
        code[patch.at + 4] = displacement >> 24 & 0xFF;
    }
    else
    {
        code[patch.at + 1] = displacement & 0xFF;
    }
}

function bzImage(body)
{
    const setupSects = 4;
    const protStart = (setupSects + 1) * 512;
    const image = new Uint8Array((protStart + 0x200 + body.length + 511) & ~511);
    image[0x1F1] = setupSects;
    image[0x1FE] = 0x55;
    image[0x1FF] = 0xAA;
    image[0x201] = 0x40;
    image.set([0x48, 0x64, 0x72, 0x53], 0x202); // "HdrS"
    image[0x206] = 0x0C;
    image[0x207] = 0x02; // protocol 0x020c
    image[0x238] = 0xFF; // cmdline_size
    image.set(body, protStart + 0x200);
    return image;
}

const imagePath = path.join(os.tmpdir(), "v64-e2e-serial.img");
fs.writeFileSync(imagePath, bzImage(code));

const serial = [];
const emulator = new v64({
    autostart: false,
    memory_size: 8 * 1024 * 1024,
    disable_jit: 1,
    log_level: 0,
    bzimage: { url: imagePath, async: false },
    cmdline: CMDLINE,
    direct_boot: true,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("serial0-output-byte", byte => serial.push(byte));

emulator.add_listener("emulator-loaded", () =>
{
    const cpu = emulator.v86.cpu;
    let steps = 0;
    while(!cpu.in_hlt[0] && steps++ < 100000)
    {
        cpu.wm.exports.main_loop();
    }

    const text = Buffer.from(serial).toString("utf8");
    assert.equal(cpu.in_hlt[0], 1, "kernel halted after printing");
    assert.equal(text, "v64 serial\n" + CMDLINE + "\n");

    console.log("e2e serial: test passed");
    console.log(text.trimEnd());
    process.exit(0);
});
