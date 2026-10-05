#!/usr/bin/env node

// Verifies the correctness hole the per-page write counter opens.
//
// The 64-bit decode cache validates a block by comparing one counter for the
// block's physical page, and that counter only moves when a write funnels
// through jit64::invalidate_physical_page. The JIT's inline stores do not: they
// write straight into the wasm memory. So a page holding a decode-cache block
// must be flagged TLB64_HAS_CODE, which is what the JIT's generated store guard
// tests before taking the inline path -- otherwise a self-modifying guest could
// be served a stale block.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp64/decode-cache-page-flag.js`

import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";

const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const PT = 0x13000;
const CODE = 0x2000;
const SCRATCH = 0x80000;

const emulator = new v64({
    autostart: false,
    memory_size: 8 * 1024 * 1024,
    disable_jit: true,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () =>
{
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const u32 = new Uint32Array(ex.memory.buffer);

    const w = (address, bytes) => {
        for(let i = 0; i < bytes.length; i++) ex.write8(address + i, bytes[i]);
    };
    const w64 = (address, value) => {
        for(let i = 0; i < 8; i++) {
            ex.write8(address + i, Number(value >> BigInt(8 * i) & 0xFFn));
        }
    };

    // Identity-map the low 1 MiB so the code page is writable RAM, not MMIO.
    w64(PML4, BigInt(PDPT) | 3n);
    w64(PDPT, BigInt(PD) | 3n);
    w64(PD, BigInt(PT) | 3n);
    for(let i = 0; i < 512; i++) w64(PT + i * 8, BigInt(i) * 0x1000n | 3n);

    // The interpreter's batch loop only runs with jit64 off, which is also the
    // configuration in which the decode cache is populated.
    ex.set_cpu_config(0 /* JIT_DISABLE */, 1 | (1 << 1));

    // mov eax,1 / mov ebx,3 / mov ecx,N / loop: add eax,ebx; xor ebx,eax;
    // shl ebx,1; sub ecx,1 / jnz loop / hlt -- a basic block the decoder covers,
    // so a cache entry really does get built for this page.
    const N = 20000;
    w(CODE, [
        0xB8, 0x01, 0x00, 0x00, 0x00,
        0xBB, 0x03, 0x00, 0x00, 0x00,
        0xB9, N & 0xFF, N >> 8 & 0xFF, N >> 16 & 0xFF, N >> 24 & 0xFF,
        0x01, 0xD8,
        0x31, 0xC3,
        0xD1, 0xE3,
        0x83, 0xE9, 0x01,
        0x75, 0xF5,
        0xF4,
    ]);

    cpu.cr[0] = 0;
    cpu.is_32[0] = 1;
    cpu.instruction_pointer[0] = CODE;
    u32[16 + 4] = SCRATCH; // rsp
    u32[32 + 4] = 0;
    cpu.flags[0] = 0x2;
    cpu.in_hlt[0] = 0;
    cpu.cpl[0] = 0;
    ex.enter_long_mode(PML4);

    const page = CODE >>> 12;
    assert.equal(
        ex.page_code_state_for_test(page) >>> 0, 0,
        "code page carries neither mark before the cache is used"
    );

    // interp64_run is the loop that consults the decode cache.
    ex.interp64_run(1000000);

    assert.equal(cpu.in_hlt[0], 1, "guest ran to hlt");
    assert.equal(
        ex.page_code_state_for_test(page) & 1, 1,
        "page holding a decode-cache block is marked, so the JIT's inline stores take the slow path and bump the page write counter"
    );
    assert.equal(
        ex.page_blocks_inline_writes_for_test(page) >>> 0, 1,
        "and TLB64_HAS_CODE would therefore be set for that page"
    );

    console.log("interp64 decode-cache page flag: test passed");
    process.exit(0);
});