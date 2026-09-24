#!/usr/bin/env node

// Long-mode virtio-blk.
//
// A 64-bit kernel (direct boot) brings up the virtio-blk device and reads a
// sector through a virtqueue, following the virtio 1.x specification:
// find the device, walk its PCI capability list to the common/notification/
// device configuration regions, read the capacity, negotiate VERSION_1, set up
// queue 0 and submit a VIRTIO_BLK_T_IN request for sector 0.
//
// The kernel is assembled from virtio_blk64.asm, so the sequence is the
// documented PCI + virtio one, not a copy of the emulator's implementation.
//
// Requires a debug wasm build (`make build/v64-debug.wasm`) and nasm.
// Run with: `node tests/e2e/virtio_blk64.js`

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { v64 } from "../../src/main.js";

const SOURCE = new URL("./virtio_blk64.asm", import.meta.url);
const SECTORS = 8;

function assemble()
{
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "v64-virtio-blk64-"));
    const binary = path.join(dir, "virtio_blk64.bin");
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

const disk = new Uint8Array(512 * SECTORS);
for(let i = 0; i < disk.length; i++) disk[i] = (i * 5 + 1) & 0xFF;
const diskPath = path.join(os.tmpdir(), "v64-e2e-virtio-blk64.img");
fs.writeFileSync(diskPath, disk);

const kernelPath = path.join(os.tmpdir(), "v64-e2e-virtio-blk64-kernel.img");
fs.writeFileSync(kernelPath, bzImage(assemble()));

const emulator = new v64({
    autostart: false,
    memory_size: 8 * 1024 * 1024,
    disable_jit: 1,
    log_level: 0,
    bzimage: { url: kernelPath, async: false },
    virtio_blk: { url: diskPath, async: false },
    cmdline: "",
    direct_boot: true,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;

    const read32 = address =>
        (ex.read8(address) | ex.read8(address + 1) << 8 | ex.read8(address + 2) << 16 | ex.read8(address + 3) << 24) >>> 0;

    let guard = 0;
    while(!cpu.in_hlt[0] && guard++ < 100000) ex.main_loop();
    assert.equal(cpu.in_hlt[0], 1, "the kernel halted");

    // The device was found and its capacity read from the device configuration.
    assert.notEqual(read32(0xE000), 0xFFFF_FFFF, "the virtio-blk PCI device was found");
    assert.equal(read32(0xE004), SECTORS, "the device-config capacity (low) matches the disk");
    assert.equal(read32(0xE008), 0, "the device-config capacity (high)");

    // The device consumed the request and reported success.
    assert.equal(ex.read16(0x31002), 1, "the used ring advanced");
    assert.equal(ex.read8(0x32300), 0, "the request completed with status OK");

    // The data buffer holds sector 0 of the disk.
    const data = cpu.read_blob(0x32100, 512);
    assert.deepEqual(Array.from(data), Array.from(disk.subarray(0, 512)),
        "the read data matches sector 0");

    console.log("e2e virtio-blk64: test passed");
    process.exit(0);
});
