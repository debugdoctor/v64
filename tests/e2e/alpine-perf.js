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
    if(process.env.BLOCK_PROLOGUE !== undefined) ex.jit64_set_block_prologue(parseInt(process.env.BLOCK_PROLOGUE, 10));
    if(process.env.EXIT_STATS === "1") ex.jit64_set_exit_stats(1);
    if(process.env.SUPERBLOCKS !== undefined) ex.jit64_set_superblocks(parseInt(process.env.SUPERBLOCKS, 10));
    if(process.env.BLOCK_LIMIT !== undefined) ex.jit64_set_block_limit(parseInt(process.env.BLOCK_LIMIT, 10));
    if(process.env.INLINE_MEMORY === "0") ex.jit64_set_inline_memory(0);
    if(process.env.INLINE_WRITE === "0") ex.jit64_set_inline_write(0);
    if(process.env.ENTRY_CACHE === "0") ex.jit64_set_entry_cache(0);
    if(process.env.HOT_CACHE === "0") ex.jit64_set_hot_cache(0);
    if(process.env.SMC_BAIL !== undefined) ex.jit64_set_smc_bail(parseInt(process.env.SMC_BAIL, 10));
    if(process.env.SELFCHECK === "1") ex.jit64_set_selfcheck(1);
    if(process.env.SELFCHECK_MIN !== undefined) ex.jit64_set_selfcheck_min(parseInt(process.env.SELFCHECK_MIN, 10));
    if(process.env.SELFCHECK_REPEAT !== undefined) ex.jit64_set_selfcheck_repeat(parseInt(process.env.SELFCHECK_REPEAT, 10));
    if(process.env.BAIL_RIP)
    {
        const [lo, hi] = process.env.BAIL_RIP.split(",").map(x => BigInt(x));
        ex.jit64_set_bail_rip(lo, hi);
    }
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
        const c = emulator.v86.cpu;
        console.error(
            "jit64 stats: run_hits=" + s(0) + " run_misses=" + s(1) + " interp=" + s(2) +
            " compiles=" + s(3) + " fails=" + s(4) + " blocks=" + s(5) + " faults=" + s(6) +
            " no_translate=" + s(7) + " too_big=" + s(8) + " no_index=" + s(9) +
            " registered=" + s(10) + " replaced=" + s(11) +
            " forget(evict/stale/smc/invlpg)=" + s(12) + "/" + s(13) + "/" + s(14) + "/" + s(15) +
            " chain_hops=" + s(16),
        );
        console.error(
            "jit64 memory slow path: reads=" + s(17) + " writes=" + s(18) + " cross_page=" + s(19),
        );
        const names = ["fallthrough", "jmp", "jcc_taken", "jcc_not", "call", "ret", "hlt", "exc"];
        console.error("jit64 exit reasons: " +
            names.map((n, i) => n + "=" + ex.jit64_exit_reason(i)).join(" "));
        const nb = ex.jit64_big_block_count();
        const big = [];
        for(let i = 0; i < nb; i++)
        {
            big.push("0x" + ex.jit64_big_block_rip(i).toString(16) + ":" + ex.jit64_big_block_len(i));
        }
        console.error("jit64 big blocks (" + nb + "): " + big.join(" "));
        {
            const bl = ex.jit64_big_block_bytes_len();
            let s = "first big block bytes (" + bl + "):";
            for(let i = 0; i < bl; i++) s += " " + ex.jit64_big_block_bytes(i).toString(16).padStart(2, "0");
            console.error("jit64 " + s);
        }
        console.error("jit64 big block span: 0x" + ex.jit64_big_block_lo().toString(16) +
            " .. 0x" + ex.jit64_big_block_hi().toString(16) +
            " last_locals=" + ex.jit64_last_big_block_locals());
        const sc = i => ex.jit64_selfcheck_info(i);
        if(sc(0) > 0 || sc(10) > 0)
        {
            console.error("jit64 selfcheck: mismatches=" + sc(0) +
                " runs=" + sc(10) + " interp_instrs=" + sc(11) +
                " block=0x" + sc(1).toString(16) + "..0x" + sc(2).toString(16) +
                " jit_rip=0x" + sc(3).toString(16) + " interp_rip=0x" + sc(4).toString(16) +
                " jit_flags=0x" + sc(5).toString(16) + " interp_flags=0x" + sc(6).toString(16) +
                " reg=" + sc(7) + " jit=0x" + sc(8).toString(16) + " interp=0x" + sc(9).toString(16) +
                " mem_mismatch=" + sc(29) + " mem_phys=0x" + sc(30).toString(16) +
                " mem_jit=0x" + sc(31).toString(16) + " mem_interp=0x" + sc(32).toString(16) +
                " mem_width=" + sc(33) + " skipped_fault=" + sc(34) +
                " pre_reg=0x" + sc(35).toString(16) + " retired=" + sc(36) +
                " bytes=" + (() => {
                    let s2 = "";
                    const bl = sc(12);
                    const words = [];
                    for(let i = 0; i < 16; i++) words.push(sc(13 + i));
                    for(let i = 0; i < bl; i++)
                    {
                        const w = words[Math.floor(i / 8)];
                        s2 += " " + Number((w >> BigInt(8 * (i % 8))) & 0xFFn).toString(16).padStart(2, "0");
                    }
                    return s2;
                })());
        }
        console.error(
            "jit64 compile time: ms=" + (c.jit64_compile_ms || 0).toFixed(1) +
            " count=" + (c.jit64_compile_count || 0) +
            " per_block_us=" + ((c.jit64_compile_ms || 0) / (c.jit64_compile_count || 1) * 1000).toFixed(1) +
            " module_ms=" + (c.jit64_module_ms || 0).toFixed(1) +
            " instance_ms=" + (c.jit64_instance_ms || 0).toFixed(1) +
            " errors=" + (c.jit64_compile_errors || 0),
        );
    }
    if(process.env.PERF_FAIL_OPS === "1")
    {
        const ops = [];
        for(let i = 0; i < 256; i++)
        {
            const c = ex.jit64_fail_op(i);
            if(c) ops.push([i, c]);
        }
        ops.sort((a, b) => b[1] - a[1]);
        console.error("jit64 fail opcodes: " +
            ops.slice(0, 24).map(([o, c]) => "0x" + o.toString(16) + "=" + c).join(" "));
        const iops = [];
        for(let i = 0; i < 256; i++)
        {
            const c = ex.interp64_opcode_count(i);
            if(c) iops.push([i, c]);
        }
        iops.sort((a, b) => b[1] - a[1]);
        console.error("interp opcodes: " +
            iops.slice(0, 24).map(([o, c]) => "0x" + o.toString(16) + "=" + c).join(" "));
        const fops = [];
        for(let i = 0; i < 256; i++)
        {
            const c = ex.interp64_opcode0f_count(i);
            if(c) fops.push([i, c]);
        }
        fops.sort((a, b) => b[1] - a[1]);
        console.error("interp 0f opcodes: " +
            fops.slice(0, 24).map(([o, c]) => "0x" + o.toString(16) + "=" + c).join(" "));
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
