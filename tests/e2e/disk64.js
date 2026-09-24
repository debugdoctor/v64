#!/usr/bin/env node

// Long-mode disk I/O.
//
// A 64-bit kernel (direct boot) reads one sector from the primary IDE channel
// over ATA PIO (28-bit LBA) and stores it in guest memory. The test attaches a
// disk image with a known pattern and checks the sector that came back.
//
// The kernel is assembled from disk64.asm, so the register sequence is the
// documented ATA one, not a copy of the emulator's implementation.
//
// Requires a debug wasm build (`make build/v64-debug.wasm`) and nasm.
// Run with: `node tests/e2e/disk64.js`

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { v64 } from "../../src/main.js";

const SOURCE = new URL("./disk64.asm", import.meta.url);

function assemble()
{
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "v64-disk64-"));
    const binary = path.join(dir, "disk64.bin");
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

// A four-sector disk with a deterministic pattern.
const disk = new Uint8Array(512 * 4);
for(let i = 0; i < disk.length; i++) disk[i] = (i * 7 + 3) & 0xFF;
const disk_path = path.join(os.tmpdir(), "v64-e2e-disk64.img");
fs.writeFileSync(disk_path, disk);

const kernel_path = path.join(os.tmpdir(), "v64-e2e-disk64-kernel.img");
fs.writeFileSync(kernel_path, bz_image(assemble()));

const emulator = new v64({
    autostart: false,
    memory_size: 8 * 1024 * 1024,
    disable_jit: 1,
    log_level: 0,
    bzimage: { url: kernel_path, async: false },
    hda: { url: disk_path, async: false },
    cmdline: "",
    direct_boot: true,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;

    let guard = 0;
    while(!cpu.in_hlt[0] && guard++ < 200000) ex.main_loop();
    assert.equal(cpu.in_hlt[0], 1, "the kernel halted");

    const sector = new Uint8Array(512);
    for(let i = 0; i < sector.length; i++) sector[i] = ex.read8(0xE000 + i);
    assert.deepEqual(Array.from(sector), Array.from(disk.subarray(0, 512)), "the sector read back matches the disk");

    console.log("e2e disk64: test passed");
    process.exit(0);
});
