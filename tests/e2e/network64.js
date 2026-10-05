#!/usr/bin/env node

// Long-mode networking.
//
// A 64-bit kernel (direct boot) enumerates PCI using mechanism #1 to find the
// network controller, then drives the DP8390/NE2000 through its register file
// and remote DMA to transmit one frame. The test observes the frame on the
// emulator bus, exactly where a host network adapter would pick it up.
//
// The kernel is assembled from network64.asm, so the hardware sequence is the
// documented one (PCI config space, DP8390 registers), not a copy of the
// emulator's implementation.
//
// Requires a debug wasm build (`make build/v64-debug.wasm`) and nasm.
// Run with: `node tests/e2e/network64.js`

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { v64 } from "../../src/main.js";

const SOURCE = new URL("./network64.asm", import.meta.url);
const VENDOR_RTL8029 = 0x10EC;
const DEVICE_RTL8029 = 0x8029;

function assemble()
{
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), "v64-network64-"));
    const binary = path.join(dir, "network64.bin");
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

const image_path = path.join(os.tmpdir(), "v64-e2e-network64.img");
fs.writeFileSync(image_path, bz_image(assemble()));

const frames = [];
const emulator = new v64({
    autostart: false,
    memory_size: 8 * 1024 * 1024,
    disable_jit: true,
    log_level: 0,
    bzimage: { url: image_path, async: false },
    cmdline: "",
    direct_boot: true,
    net_device: { type: "ne2k" },
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("net0-send", data => frames.push(Buffer.from(data)));

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;

    const read32 = address =>
        (ex.read8(address) | ex.read8(address + 1) << 8 | ex.read8(address + 2) << 16 | ex.read8(address + 3) << 24) >>> 0;

    let guard = 0;
    while(!cpu.in_hlt[0] && guard++ < 100000) ex.main_loop();
    assert.equal(cpu.in_hlt[0], 1, "the kernel halted");

    // PCI: a network-class device was found and its BAR is an I/O BAR.
    const device = read32(0xE000);
    assert.notEqual(device, 0xFFFF_FFFF, "a network-class PCI device was found");
    const ids = read32(0xE004);
    assert.equal(ids & 0xFFFF, VENDOR_RTL8029, "PCI vendor id (Realtek)");
    assert.equal(ids >>> 16, DEVICE_RTL8029, "PCI device id (RTL8029)");
    assert.notEqual(read32(0xE008), 0, "the NIC has an I/O BAR");

    // The DP8390 transmit reached the bus with the expected frame.
    assert.equal(frames.length, 1, "exactly one frame was transmitted");
    const frame = frames[0];
    assert.equal(frame.length, 60, "frame length");
    assert.deepEqual(
        Array.from(frame.subarray(0, 6)),
        [0x02, 0x00, 0x00, 0x00, 0x00, 0x01],
        "destination MAC",
    );
    assert.deepEqual(
        Array.from(frame.subarray(6, 12)),
        [0x02, 0x00, 0x00, 0x00, 0x00, 0x02],
        "source MAC",
    );
    assert.deepEqual(Array.from(frame.subarray(12, 14)), [0x08, 0x00], "ethertype");
    assert.equal(frame.subarray(14, 18).toString("latin1"), "v64!", "payload");

    console.log("e2e network64: test passed");
    process.exit(0);
});
