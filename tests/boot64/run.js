#!/usr/bin/env node

// Direct 64-bit boot, observed from the outside.
//
// A bzImage that speaks the Linux 64-bit boot protocol is loaded through the
// public direct_boot option. Its entry point writes one line to COM1 and
// halts. The test reads that line from the serial console; it does not look
// at registers or guest memory.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/boot64/run.js`

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { v64 } from "../../src/main.js";

const LINE = "boot64\n";

function uart(value)
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

const code = [...LINE].flatMap(c => uart(c.charCodeAt(0)));
code.push(0xF4);

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
    image[0x238] = 0xFF;
    image.set(body, protStart + 0x200);
    return image;
}

const imagePath = path.join(os.tmpdir(), "v64-boot64.img");
fs.writeFileSync(imagePath, bzImage(code));

const serial = [];
const emulator = new v64({
    autostart: false,
    memory_size: 8 * 1024 * 1024,
    disable_jit: 1,
    log_level: 0,
    bzimage: { url: imagePath, async: false },
    cmdline: "console=ttyS0",
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
    assert.equal(cpu.in_hlt[0], 1, "kernel halted");
    assert.equal(text, LINE, "COM1 after a direct 64-bit boot");

    console.log("boot64: test passed");
    process.exit(0);
});
