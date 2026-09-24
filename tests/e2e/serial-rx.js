#!/usr/bin/env node

// Direct 64-bit boot: the serial console receives as well as sends.
//
// The kernel polls COM1 for received bytes and echoes each one. The host feeds
// it a line through the JavaScript API; the echo must come back unchanged.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/e2e/serial-rx.js`

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { v64 } from "../../src/main.js";

const MESSAGE = "hello";

const kernel = [
    0xB9, 0x05, 0x00, 0x00, 0x00,             // mov ecx, 5
    0xBA, 0xFD, 0x03, 0x00, 0x00,             // poll: mov edx, 0x3FD
    0xEC,                                     // in al, dx
    0xA8, 0x01,                               // test al, 1
    0x74, 0xF6,                               // jz poll
    0xBA, 0xF8, 0x03, 0x00, 0x00,             // mov edx, 0x3F8
    0xEC,                                     // in al, dx
    0x88, 0xC3,                               // mov bl, al
    0xBA, 0xFD, 0x03, 0x00, 0x00,             // txwait: mov edx, 0x3FD
    0xEC,                                     // in al, dx
    0xA8, 0x20,                               // test al, 0x20
    0x74, 0xF6,                               // jz txwait
    0xBA, 0xF8, 0x03, 0x00, 0x00,             // mov edx, 0x3F8
    0x88, 0xD8,                               // mov al, bl
    0xEE,                                     // out dx, al
    0xFF, 0xC9,                               // dec ecx
    0x75, 0xD8,                               // jnz poll
    0xF4,                                     // hlt
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

const image_path = path.join(os.tmpdir(), "v64-e2e-serial-rx.img");
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
    emulator.serial0_send(MESSAGE);

    const deadline = Date.now() + 10000;
    while(!cpu.in_hlt[0] && Date.now() < deadline)
    {
        cpu.wm.exports.main_loop();
    }

    assert.equal(cpu.in_hlt[0], 1, "the kernel halted after echoing");
    assert.equal(Buffer.from(serial).toString("utf8"), MESSAGE, "the echo came back");

    console.log("e2e serial-rx: test passed");
    process.exit(0);
});
