#!/usr/bin/env node

// The hard goal: a real 64-bit Linux kernel plus initramfs reaches a shell.
//
// The image is not distributed with the repository, so the test skips unless
// one is provided (see tests/images/README.md):
//
//   LINUX64_IMAGE=tests/images/vmlinuz-virt \
//   LINUX64_INITRD=tests/images/initramfs.cpio.gz \
//       node tests/e2e/linux.js
//
// It boots through the 64-bit boot protocol and watches COM1 for the kernel
// banner and a shell prompt.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { v64 } from "../../src/main.js";

const ROOT = fileURLToPath(new URL("../..", import.meta.url));
const TIMEOUT_MS = +process.env.LINUX64_TIMEOUT_MS || 120000;
const CMDLINE = process.env.LINUX64_CMDLINE || "console=ttyS0 earlyprintk=serial";

function first_existing(candidates)
{
    for(const candidate of candidates)
    {
        if(candidate && fs.existsSync(candidate)) return candidate;
    }
    return null;
}

const image = first_existing([
    process.env.LINUX64_IMAGE,
    path.join(ROOT, "tests/images/bzImage"),
    path.join(ROOT, "tests/images/vmlinuz-virt"),
    path.join(ROOT, "tests/images/vmlinuz"),
]);

const initrd = first_existing([
    process.env.LINUX64_INITRD,
    path.join(ROOT, "tests/images/initramfs-virt"),
    path.join(ROOT, "tests/images/initramfs.cpio.gz"),
    path.join(ROOT, "tests/images/rootfs.cpio"),
    path.join(ROOT, "tests/images/initrd"),
]);

if(!image || !initrd)
{
    const missing = [!image && "kernel (LINUX64_IMAGE)", !initrd && "initrd (LINUX64_INITRD)"].filter(Boolean);
    console.log("e2e linux: missing " + missing.join(" and ") + ", test skipped");
    console.log("  put them in tests/images/ (see tests/images/README.md)");
    process.exit(0);
}

const serial = [];
const emulator = new v64({
    autostart: false,
    memory_size: 512 * 1024 * 1024,
    disable_jit: 0,
    log_level: 0,
    bzimage: { url: image, async: false },
    initrd: initrd ? { url: initrd, async: false } : undefined,
    cmdline: CMDLINE,
    direct_boot: true,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("serial0-output-byte", byte => serial.push(byte));

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;

    const deadline = Date.now() + TIMEOUT_MS;
    while(!cpu.in_hlt[0] && Date.now() < deadline)
    {
        ex.main_loop();
        const text = Buffer.from(serial).toString("utf8");
        if(/login:|[#$] $|~ #/.test(text)) break;
    }

    const text = Buffer.from(serial).toString("utf8");
    console.log(text.slice(-2000));

    assert.ok(text.includes("Linux version"), "the kernel banner appeared on COM1");
    assert.ok(/login:|[#$] $|~ #/.test(text), "the kernel reached a shell");

    console.log("e2e linux: test passed");
    process.exit(0);
});
