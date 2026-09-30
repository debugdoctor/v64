#!/usr/bin/env node

// Alpine live-ISO boot throughput: boots the official ISO over serial and
// reports wall time, retired guest instructions and the marker reached.
//
// Needs images/alpine-virt-*.iso (see images/README.md):
//   SECONDS=300 WASM_PATH=build/v64.wasm node tests/e2e/alpine-perf.js
//   INTERP=1 SECONDS=300 node tests/e2e/alpine-perf.js
//
// Prints a RESULT line; exits 0 if the marker appeared.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { v64 } from "../../src/main.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const ISO = process.env.ALPINE_ISO || path.join(ROOT, "images/alpine-virt-3.24.2-x86_64.iso");
const SECONDS = +(process.env.SECONDS || 600);
const MARKER = process.env.MARKER || "login:";

if(!fs.existsSync(ISO))
{
    console.log("alpine-perf: no ISO at " + ISO + " (see images/README.md)");
    process.exit(0);
}

const BOOT_ENTRY = "APPEND modules=loop,squashfs,sd-mod,usb-storage quiet";
// sr-mod loads the CD driver up front; loop/squashfs come from the initramfs.
const BOOT_ENTRY_PATCHED = "APPEND modules=sr-mod console=ttyS0";

const load_patched_iso = () =>
{
    const bytes = new Uint8Array(fs.readFileSync(ISO));
    const entry = new TextEncoder().encode(BOOT_ENTRY);
    let at = -1;
    for(let from = 0; (from = bytes.indexOf(entry[0], from)) !== -1 && from < bytes.length - entry.length; from++)
    {
        let match = true;
        for(let i = 1; i < entry.length; i++)
        {
            if(bytes[from + i] !== entry[i]) { match = false; break; }
        }
        if(match) { at = from; break; }
    }
    if(at === -1)
    {
        console.warn("alpine-perf: boot entry not found, booting the ISO unmodified");
    }
    else
    {
        bytes.set(new TextEncoder().encode(BOOT_ENTRY_PATCHED.padEnd(entry.length)), at);
    }
    return bytes.buffer;
};

const emulator = new v64({
    autostart: false,
    wasm_path: process.env.WASM_PATH || path.join(ROOT, "build/v64.wasm"),
    memory_size: 1024 * 1024 * 1024,
    bios: { url: path.join(ROOT, "bios/bochs-bios.bin"), async: false },
    cdrom: { buffer: load_patched_iso() },
    boot_order: 0x213,
    disable_jit: process.env.INTERP ? 1 : 0,
    log_level: 0,
});

let serial = "";
emulator.add_listener("serial0-output-byte", b => { serial += String.fromCharCode(b); });

emulator.add_listener("emulator-loaded", async () =>
{
    const ex = emulator.v86.cpu.wm.exports;
    if(process.env.INTERP) ex.jit64_set_enabled(0);
    if(process.env.INTERP_CACHE === "0") ex.interp64_set_fetch_cache(0);
    if(process.env.INTERP_MEM_SINGLE === "0") ex.interp64_set_mem_single(0);
    const counter = () => new Uint32Array(ex.memory.buffer)[166];

    const start = Date.now();
    let instructions = 0;
    let last = counter();
    let next = start + 30000;
    let reached = false;
    while(Date.now() - start < SECONDS * 1000)
    {
        ex.main_loop();
        await new Promise(resolve => setImmediate(resolve));
        if(Date.now() >= next)
        {
            next += 30000;
            const now = counter();
            instructions += (now - last) >>> 0; // 32-bit counter, accumulate deltas
            last = now;
            console.error(Math.round((Date.now() - start) / 1000) + "s: " + serial.slice(-120).replace(/\n/g, " "));
        }
        if(serial.includes(MARKER) || /Kernel panic/.test(serial))
        {
            reached = serial.includes(MARKER);
            break;
        }
    }
    const elapsed = (Date.now() - start) / 1000;
    instructions += (counter() - last) >>> 0;

    if(!reached && serial.includes(MARKER)) reached = true;
    if(process.env.PERF_STATS === "1")
    {
        const s = i => ex.jit64_stat(i);
        console.error(
            "jit64 stats: run_hits=" + s(0) + " run_misses=" + s(1) + " interp=" + s(2) +
            " compiles=" + s(3) + " fails=" + s(4) + " blocks=" + s(5) + " faults=" + s(6),
        );
    }
    console.log(
        "RESULT wasm=" + (process.env.WASM_PATH || "build/v64.wasm") +
        " jit=" + (process.env.INTERP ? "0" : "1") +
        " seconds=" + elapsed.toFixed(1) +
        " instructions=" + instructions +
        " mips=" + (instructions / elapsed / 1e6).toFixed(1) +
        " reached=" + (reached ? MARKER : "timeout"),
    );
    process.exit(reached ? 0 : 1);
});
