#!/usr/bin/env node

// End-to-end test for the long-mode JIT.
//
// The guest program is real x86-64 assembly (e2e.asm), assembled with nasm at
// test time. It is run twice: once entirely interpreted and once hot enough to
// be compiled by jit64. Both runs must match results derived independently of
// the emulator, so the test does not just compare the JIT against the
// interpreter it was written next to.
//
// Requires a debug wasm build (`make build/v64-debug.wasm`) and nasm.
// Run with: `node tests/jit64/e2e.js`

import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { v64 } from "../../src/main.js";

const BASE = 0x1000;
const LOOPS = 600;
const SOURCE = new URL("./e2e.asm", import.meta.url);

function assemble()
{
    const dir = mkdtempSync(path.join(tmpdir(), "jit64-e2e-"));
    const binary = path.join(dir, "e2e.bin");
    try
    {
        execFileSync("nasm", ["-f", "bin", "-o", binary, SOURCE.pathname], { stdio: "inherit" });
        return readFileSync(binary);
    }
    finally
    {
        rmSync(dir, { recursive: true, force: true });
    }
}

const program = assemble();

const emulator = new v64({
    autostart: false,
    memory_size: 2 * 1024 * 1024,
    disable_jit: 0,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    // The wasm memory can grow, so always read the current buffer.
    const u32 = () => new Uint32Array(cpu.wasm_memory.buffer);
    const guest = () => new DataView(cpu.wasm_memory.buffer);

    const write64 = (address, value) => {
        ex.write32(address, Number(value & 0xFFFF_FFFFn));
        ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
    };
    // Guest physical memory, read through the CPU rather than the wasm buffer.
    const readGuest = (address, bytes) => {
        let value = 0n;
        for(let i = bytes - 1; i >= 0; i--)
        {
            value = value << 8n | BigInt(ex.read8(address + i));
        }
        return value;
    };
    const reg64 = index => {
        const u = u32();
        return (BigInt(u[32 + index]) << 32n) | BigInt(u[16 + index]);
    };
    const extReg = index => guest().getBigUint64(160 + 8 * index, true);

    // Identity-map the first 1 GiB with 2 MiB pages.
    const PML4 = 0x10000;
    const PDPT = 0x11000;
    const PD = 0x12000;
    const PT = 0x13000;
    write64(PML4, BigInt(PDPT) | 0x3n);
    write64(PDPT, BigInt(PD) | 0x3n);
    write64(PD, BigInt(PT) | 0x3n);
    for(let i = 0; i < 512; i++)
    {
        write64(PT + i * 8, BigInt(i) * 0x1000n | 0x3n);
    }
    for(let i = 1; i < 512; i++)
    {
        write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);
    }

    for(let i = 0; i < program.length; i++)
    {
        ex.write8(BASE + i, program[i]);
    }
    // The dword the program loads through a 32-bit address.
    write64(0x60000, 0x11223344n);

    const run = jitEnabled => {
        ex.jit64_set_enabled(jitEnabled ? 1 : 0);
        ex.jit64_clear_cache();
        for(let i = 0; i < 16; i++) ex.write8(0x80020 + i, 0);
        cpu.instruction_pointer[0] = BASE;
        u32()[16 + 4] = 0x80000; // rsp
        cpu.flags[0] &= ~(1 << 9); // cli
        cpu.in_hlt[0] = 0;
        ex.enter_long_mode(PML4);

        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 1000000)
        {
            ex.main_loop();
        }
        assert.equal(cpu.in_hlt[0], 1, "the program halted");
        return {
            compiled: ex.jit64_compiled_count(),
            r8: extReg(0),
            r9: extReg(1),
            r10: extReg(2),
            r13: extReg(5),
            r14: extReg(6),
            r15: extReg(7),
            word: readGuest(0x80020, 2),
            qword: readGuest(0x80028, 8),
        };
    };

    const interpreted = run(false);
    const compiled = run(true);

    assert.equal(interpreted.compiled, 0, "the interpreter run compiled nothing");
    assert.ok(compiled.compiled > 0, "the JIT compiled the hot loop");

    // Derived by hand from the Intel semantics, not from either run.
    const expected = {
        r13: 0xFFFF_FFFF_FFFF_1235n, // 16-bit add keeps the upper bits
        r15: 0x11223344n,            // loaded through a 32-bit address
        word: 0xABCEn,               // 16-bit memory add
        qword: BigInt(LOOPS),        // lock add, then shl 1 / shr 1 cancel
        r14: 0xFFFF_FFFF_FFFF_0000n, // 16-bit shift by 20 clears the low half
    };

    for(const result of [interpreted, compiled])
    {
        assert.equal(result.r13, expected.r13, "16-bit arithmetic");
        assert.equal(result.r15, expected.r15, "32-bit addressing override");
        assert.equal(result.word, expected.word, "16-bit memory add");
        assert.equal(result.qword, expected.qword, "lock add and memory shifts");
        assert.equal(result.r14, expected.r14, "16-bit shift past the width");
        // CPUID leaf 0: "GenuineIntel" split across ebx, edx, ecx.
        assert.equal(result.r8, 0x756E6547n, "cpuid vendor ebx");
        assert.equal(result.r10, 0x49656E69n, "cpuid vendor edx");
        assert.equal(result.r9, 0x6C65746En, "cpuid vendor ecx");
    }

    console.log("jit64 e2e: passed (" + compiled.compiled + " compiled block(s))");
    process.exit(0);
});
