#!/usr/bin/env node

// Fast reproduction: pack the Alpine minirootfs into an initramfs (so the
// guest runs Alpine's own busybox 1.37, the same `top` as the live ISO),
// boot it and run `top` until it faults. The kernel's trap line is dumped
// after top exits.
//
//   node tests/e2e/repro-top-minirootfs.js

import { execFileSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { v64 } from "../../src/main.js";

const ROOT = fileURLToPath(new URL("../..", import.meta.url));
const KERNEL = process.env.LINUX64_IMAGE || path.join(ROOT, "images/vmlinuz-virt");
const TARBALL = path.join(ROOT, "images/alpine-minirootfs-3.24.2-x86_64.tar.gz");
const EXTRACT = path.join(ROOT, "build/minirootfs");
const CPIO = path.join(ROOT, "build/alpine-minirootfs.cpio");
const TIMEOUT_MS = +process.env.MINIROOTFS_TIMEOUT_MS || 240000;

const DEFAULT_INIT = `#!/bin/sh
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
export HOME=/root
export TERM=linux
mount -t proc proc /proc
mount -t sysfs sys /sys
mount -t devtmpfs dev /dev 2>/dev/null
mount -t tmpfs tmpfs /tmp 2>/dev/null
echo 8 > /proc/sys/kernel/printk 2>/dev/null
echo 0 > /proc/sys/kernel/randomize_va_space 2>/dev/null
cd /
echo V64-READY
top
echo "V64-TOP-EXIT rc=$?"
echo V64-DMESG
dmesg | tail -n 40
echo V64-DONE
exec /bin/sh
`;

const INIT = process.env.INIT_OVERRIDE ||
    (process.env.INIT_OVERRIDE_FILE ? fs.readFileSync(process.env.INIT_OVERRIDE_FILE, "utf8") : DEFAULT_INIT);

function extract()
{
    if(fs.existsSync(path.join(EXTRACT, "bin/busybox")) && !process.env.REBUILD)
    {
        return;
    }
    fs.rmSync(EXTRACT, { recursive: true, force: true });
    fs.mkdirSync(EXTRACT, { recursive: true });
    execFileSync("tar", ["-xzf", TARBALL, "-C", EXTRACT], { stdio: "inherit" });
}

function cpio_write(chunks, name, mode, data)
{
    const name_buf = Buffer.from(name + "\0");
    const data_buf = Buffer.from(data);
    const header = Buffer.alloc(110);
    header.write("070701", 0, "ascii");
    const field = value => value.toString(16).padStart(8, "0");
    const values = [1, mode, 0, 0, 1, 0, data_buf.length, 0, 0, 0, 0, name_buf.length, 0];
    let offset = 6;
    for(const value of values)
    {
        header.write(field(value), offset, "ascii");
        offset += 8;
    }
    chunks.push(header, name_buf);
    const name_pad = (4 - (header.length + name_buf.length) % 4) % 4;
    if(name_pad) chunks.push(Buffer.alloc(name_pad));
    chunks.push(data_buf);
    const data_pad = (4 - data_buf.length % 4) % 4;
    if(data_pad) chunks.push(Buffer.alloc(data_pad));
}

function build_cpio()
{
    if(fs.existsSync(CPIO) && !process.env.REBUILD)
    {
        return;
    }
    const chunks = [];
    const add = (rel, abs) =>
    {
        const stat = fs.lstatSync(abs);
        const perm = stat.mode & 0o7777;
        if(stat.isDirectory())
        {
            cpio_write(chunks, rel, 0o040000 | perm, Buffer.alloc(0));
        }
        else if(stat.isSymbolicLink())
        {
            cpio_write(chunks, rel, 0o120000 | perm, fs.readlinkSync(abs));
        }
        else if(stat.isFile())
        {
            cpio_write(chunks, rel, 0o100000 | perm, fs.readFileSync(abs));
        }
        // device/fifo/socket nodes are dropped; devtmpfs covers /dev
    };

    add("init", write_init());

    const walk = (rel) =>
    {
        const abs = path.join(EXTRACT, rel);
        for(const entry of fs.readdirSync(abs))
        {
            const child_rel = rel ? rel + "/" + entry : entry;
            const child_abs = path.join(EXTRACT, child_rel);
            add(child_rel, child_abs);
            if(fs.lstatSync(child_abs).isDirectory())
            {
                walk(child_rel);
            }
        }
    };
    walk("");

    cpio_write(chunks, "TRAILER!!!", 0, Buffer.alloc(0));
    fs.writeFileSync(CPIO, Buffer.concat(chunks));
}

function write_init()
{
    const init_path = path.join(EXTRACT, "init");
    fs.writeFileSync(init_path, INIT, { mode: 0o755 });
    return init_path;
}

extract();
build_cpio();

const serial = [];
const emulator = new v64({
    autostart: false,
    memory_size: 512 * 1024 * 1024,
    disable_jit: +(process.env.DISABLE_JIT || 0),
    log_level: 0,
    bzimage: { url: KERNEL, async: false },
    initrd: { url: CPIO, async: false },
    cmdline: process.env.LINUX64_CMDLINE || "console=ttyS0 rdinit=/init",
    direct_boot: true,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("serial0-output-byte", byte => serial.push(byte));

emulator.add_listener("emulator-loaded", () =>
{
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const deadline = Date.now() + TIMEOUT_MS;

    let last_report = 0;
    while(Date.now() < deadline)
    {
        ex.main_loop();
        const out = Buffer.from(serial).toString("utf8");
        if(out.includes("V64-DONE"))
        {
            break;
        }
        if(process.env.PROGRESS && Date.now() - last_report > 15000)
        {
            last_report = Date.now();
            process.stderr.write("[progress " + Math.round((Date.now() - (deadline - TIMEOUT_MS)) / 1000) +
                "s hlt=" + cpu.in_hlt[0] + " insn=" + emulator.get_instruction_counter() +
                "] " + JSON.stringify(out.slice(-160)) + "\n");
        }
    }

    const out = Buffer.from(serial).toString("utf8");
    const interesting = out.match(/V64-READY[\s\S]*$/);
    console.log(interesting ? interesting[0].slice(-6000) : out.slice(-4000));

    const crashed = /Segmentation fault/.test(out);
    console.log("\n==== top " + (crashed ? "SEGFAULTED" : "ran without a fault") + " ====");
    process.exit(crashed ? 1 : 0);
});
