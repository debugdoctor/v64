#!/usr/bin/env node

// The hard goal, initramfs flavour: a real 64-bit kernel unpacks an initramfs
// and runs its /init to a shell.
//
// The initramfs is generated here as a newc cpio archive; a static busybox is
// needed for /init to be a script. See tests/images/README.md:
//
//   LINUX64_IMAGE=tests/images/vmlinuz-virt \
//   LINUX64_BUSYBOX=tests/images/busybox \
//       node tests/e2e/initramfs.js
//
// Requires a debug wasm build: `make build/v64-debug.wasm`

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { v64 } from "../../src/main.js";

const ROOT = fileURLToPath(new URL("../..", import.meta.url));
const MARKER = "v64-INITRAMFS-OK";
const TIMEOUT_MS = +process.env.LINUX64_TIMEOUT_MS || 120000;

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
const busybox = first_existing([
    process.env.LINUX64_BUSYBOX,
    path.join(ROOT, "tests/images/busybox"),
]);

if(!image || !busybox)
{
    const missing = [!image && "kernel (LINUX64_IMAGE)", !busybox && "busybox (LINUX64_BUSYBOX)"].filter(Boolean);
    console.log("e2e initramfs: missing " + missing.join(" and ") + ", test skipped");
    console.log("  put them in tests/images/ (see tests/images/README.md)");
    process.exit(0);
}

// Minimal newc cpio writer.
function cpio(entries)
{
    const chunks = [];
    let ino = 1;
    const add = (name, mode, data) =>
    {
        const name_bytes = Buffer.from(name + "\0");
        const data_buf = Buffer.from(data);
        const header = Buffer.alloc(110);
        const field = value => value.toString(16).padStart(8, "0");
        header.write("070701", 0, "ascii");
        const values = [ino++, mode, 0, 0, 1, 0, data_buf.length, 0, 0, 0, 0, name_bytes.length, 0];
        let offset = 6;
        for(const value of values)
        {
            header.write(field(value), offset, "ascii");
            offset += 8;
        }
        chunks.push(header, name_bytes);
        const name_pad = (4 - (header.length + name_bytes.length) % 4) % 4;
        if(name_pad) chunks.push(Buffer.alloc(name_pad));
        chunks.push(data_buf);
        const data_pad = (4 - data_buf.length % 4) % 4;
        if(data_pad) chunks.push(Buffer.alloc(data_pad));
    };

    add("bin", 0o040755, new Uint8Array(0));
    add("bin/busybox", 0o100755, fs.readFileSync(busybox));
    add("bin/sh", 0o120777, Buffer.from("busybox\0"));
    add("init", 0o100755, Buffer.from("#!/bin/sh\necho " + MARKER + "\nexec sh\n\0"));
    add("TRAILER!!!", 0, new Uint8Array(0));
    return Buffer.concat(chunks);
}

const initramfs_path = path.join(os.tmpdir(), "v64-e2e-initramfs.cpio");
fs.writeFileSync(initramfs_path, cpio());

const serial = [];
const emulator = new v64({
    autostart: false,
    memory_size: 512 * 1024 * 1024,
    disable_jit: 0,
    log_level: 0,
    bzimage: { url: image, async: false },
    initrd: { url: initramfs_path, async: false },
    cmdline: "console=ttyS0 rdinit=/init",
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
        if(Buffer.from(serial).toString("utf8").includes(MARKER)) break;
    }

    const text = Buffer.from(serial).toString("utf8");
    console.log(text.slice(-2000));
    assert.ok(text.includes(MARKER), "the initramfs /init ran and printed its marker");

    console.log("e2e initramfs: test passed");
    process.exit(0);
});
