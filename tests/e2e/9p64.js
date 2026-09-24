#!/usr/bin/env node

// Long-mode 9p.
//
// A 64-bit kernel (direct boot) brings up the virtio-9p device and speaks
// 9P2000.L over its virtqueue: Tversion, Tattach, Twalk, Tlopen, Tread,
// Tclunk. The host side is the real in-memory 9p server (`Virtio9p` + `FS`),
// so a file created on the host is read back by the guest.
//
// The kernel is assembled from 9p64.asm, so the sequence is the documented
// virtio PCI + 9P2000.L one, not a copy of the emulator's implementation.
//
// Requires a debug wasm build (`make build/v64-debug.wasm`) and nasm.
// Run with: `node tests/e2e/9p64.js`

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { v64 } from "../../src/main.js";

const SOURCE = new URL("./9p64.asm", import.meta.url);
const FILE_SIZE = 256;

// The reply type of each transaction, in order:
// Rversion, Rattach, Rwalk, Rlopen, Rread, Rclunk.
const REPLY_TYPES = [101, 105, 111, 13, 117, 121];

function assemble()
{
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "v64-9p64-"));
    const binary = path.join(dir, "9p64.bin");
    try
    {
        execFileSync("nasm", ["-f", "bin", "-o", binary, SOURCE.pathname], { stdio: "inherit" });
        return new Uint8Array(fs.readFileSync(binary));
    }
    finally
    {
        fs.rmSync(dir, { recursive: true, force: true });
    }
}

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

const file = new Uint8Array(FILE_SIZE);
for(let i = 0; i < file.length; i++) file[i] = (i * 7 + 3) & 0xFF;

const kernel_path = path.join(os.tmpdir(), "v64-e2e-9p64-kernel.img");
fs.writeFileSync(kernel_path, bz_image(assemble()));

const emulator = new v64({
    autostart: false,
    memory_size: 8 * 1024 * 1024,
    disable_jit: 1,
    log_level: 0,
    bzimage: { url: kernel_path, async: false },
    filesystem: {},
    cmdline: "",
    direct_boot: true,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", async () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;

    const read32 = address =>
        (ex.read8(address) | ex.read8(address + 1) << 8 | ex.read8(address + 2) << 16 | ex.read8(address + 3) << 24) >>> 0;

    // Put a file in the 9p filesystem for the guest to read.
    await emulator.create_file("hello.txt", file);

    // The 9p replies are asynchronous, so yield to the event loop while the
    // guest spins waiting for the used ring.
    let guard = 0;
    while(!cpu.in_hlt[0] && guard++ < 2000000)
    {
        ex.main_loop();
        await new Promise(resolve => setImmediate(resolve));
    }
    assert.equal(cpu.in_hlt[0], 1, "the kernel halted");

    assert.notEqual(read32(0xE000), 0xFFFF_FFFF, "the virtio-9p PCI device was found");

    const types = REPLY_TYPES.map((_, i) => ex.read8(0xE010 + i));
    assert.deepEqual(types, REPLY_TYPES, "each 9P request got its reply");

    assert.equal(read32(0xE020), FILE_SIZE, "Rread returned the file size");
    const data = cpu.read_blob(0x4100B, FILE_SIZE);
    assert.deepEqual(Array.from(data), Array.from(file), "the read data matches the file");

    console.log("e2e 9p64: test passed");
    process.exit(0);
});
