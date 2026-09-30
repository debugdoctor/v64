#!/usr/bin/env node

// Long-mode interpreter: REP MOVS/STOS semantics, including the bulk fast path.
//
// Covers forward/backward direction, overlapping copies, page crossings, the
// 8/16/32-bit element sizes and the RCX/RSI/RDI updates. Expected memory is
// computed with a byte-by-byte simulation of the x86 semantics.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp64/rep.js`

import { v64 } from "../../src/main.js";

const BASE = 0x1000;
const SRC = 0x10000;
const DST = 0x20000;
const SRC2 = 0x30000;
const DST2 = 0x40000;
const FILL = 0x50000;

const imm64 = value => {
    const out = [];
    for(let i = 0; i < 8; i++) out.push(Number(value >> BigInt(8 * i) & 0xFFn));
    return out;
};
const mov_rsi = v => [0x48, 0xBE, ...imm64(BigInt(v))];
const mov_rdi = v => [0x48, 0xBF, ...imm64(BigInt(v))];
const mov_rcx = v => [0x48, 0xB9, ...imm64(BigInt(v))];
const mov_rax = v => [0x48, 0xB8, ...imm64(BigInt(v))];

const emulator = new v64({
    autostart: false,
    memory_size: 2 * 1024 * 1024,
    disable_jit: 1,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const u32 = new Uint32Array(ex.memory.buffer);
    const ext = new BigUint64Array(ex.memory.buffer, 160, 8);

    const failures = [];
    let active_test = "";
    const note = message => failures.push(active_test + ": " + message);
    const expect = (got, want, message) => {
        if(got !== want) note(message + " [want " + want + ", got " + got + "]");
    };
    const expect_bytes = (start, want, message) => {
        for(let i = 0; i < want.length; i++)
        {
            if(ex.read8(start + i) !== want[i])
            {
                note(message + " byte " + i + " [want " + want[i] + ", got " + ex.read8(start + i) + "]");
                return;
            }
        }
    };
    const reg64 = i => (i < 8)
        ? (BigInt(u32[32 + i]) << 32n) | BigInt(u32[16 + i])
        : ext[i - 8];
    const set_reg64 = (i, raw) => {
        const value = BigInt(raw);
        if(i < 8)
        {
            u32[16 + i] = Number(value & 0xFFFF_FFFFn);
            u32[32 + i] = Number(value >> 32n & 0xFFFF_FFFFn);
        }
        else
        {
            ext[i - 8] = value;
        }
    };

    const run = code => {
        for(let i = 0; i < code.length; i++) ex.write8(BASE + i, code[i]);
        ex.write8(BASE + code.length, 0xF4);
        cpu.instruction_pointer[0] = BASE;
        cpu.in_hlt[0] = 0;
        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 10000) ex.interp64_run_one();
        if(!cpu.in_hlt[0]) note("did not halt");
    };

    const reset = () => {
        for(let i = 0; i < 16; i++) set_reg64(i, 0n);
        cpu.cr[0] = 0;
        cpu.is_32[0] = 1;
        cpu.cpl[0] = 0;
        cpu.flags[0] = 0x2;
        cpu.in_hlt[0] = 0;
    };

    const fill = (start, bytes) => {
        for(let i = 0; i < bytes; i++) ex.write8(start + i, i + 1 & 0xFF);
    };
    const read = (start, bytes) => Array.from({ length: bytes }, (_, i) => ex.read8(start + i));

    // ---- rep movsb, forward ----
    active_test = "rep movsb forward";
    reset();
    fill(SRC, 64);
    run([...mov_rsi(SRC), ...mov_rdi(DST), ...mov_rcx(64), 0xF3, 0xA4]);
    expect_bytes(DST, read(SRC, 64), "copied bytes");
    expect(reg64(1), 0n, "rcx");
    expect(reg64(6), BigInt(SRC + 64), "rsi");
    expect(reg64(7), BigInt(DST + 64), "rdi");

    // ---- rep movsb, backward (std) ----
    active_test = "rep movsb backward";
    reset();
    fill(SRC, 64);
    run([0xFD, ...mov_rsi(SRC + 63), ...mov_rdi(DST + 63), ...mov_rcx(64), 0xF3, 0xA4]);
    expect_bytes(DST, read(SRC, 64), "copied bytes");
    expect(reg64(1), 0n, "rcx");
    expect(reg64(6), BigInt(SRC - 1), "rsi");
    expect(reg64(7), BigInt(DST - 1), "rdi");

    // ---- rep movsb, overlapping forward: element-by-element semantics ----
    active_test = "rep movsb overlapping forward";
    reset();
    fill(SRC, 32);
    const overlap_want = [];
    for(let i = 0; i < 16; i++) overlap_want.push(i < 2 ? ex.read8(SRC + i) : overlap_want[i - 2]);
    run([...mov_rsi(SRC), ...mov_rdi(SRC + 2), ...mov_rcx(16), 0xF3, 0xA4]);
    expect_bytes(SRC + 2, overlap_want, "overlapping copy");

    // ---- rep movsb, page crossing ----
    active_test = "rep movsb page crossing";
    reset();
    fill(SRC + 0xFF0, 64);
    run([...mov_rsi(SRC + 0xFF0), ...mov_rdi(DST + 0xFF0), ...mov_rcx(64), 0xF3, 0xA4]);
    expect_bytes(DST + 0xFF0, read(SRC + 0xFF0, 64), "cross-page copy");

    // ---- rep movsw / movsd ----
    active_test = "rep movsw";
    reset();
    fill(SRC2, 32);
    run([...mov_rsi(SRC2), ...mov_rdi(DST2), ...mov_rcx(16), 0xF3, 0x66, 0xA5]);
    expect_bytes(DST2, read(SRC2, 32), "movsw bytes");
    expect(reg64(1), 0n, "rcx");

    active_test = "rep movsd";
    reset();
    fill(SRC2, 32);
    run([...mov_rsi(SRC2), ...mov_rdi(DST2), ...mov_rcx(8), 0xF3, 0xA5]);
    expect_bytes(DST2, read(SRC2, 32), "movsd bytes");
    expect(reg64(1), 0n, "rcx");

    // ---- rep stosb / stosd ----
    active_test = "rep stosb";
    reset();
    run([...mov_rdi(FILL), ...mov_rcx(100), ...mov_rax(0xAB), 0xF3, 0xAA]);
    expect_bytes(FILL, new Array(100).fill(0xAB), "filled bytes");
    expect(reg64(1), 0n, "rcx");
    expect(reg64(7), BigInt(FILL + 100), "rdi");

    active_test = "rep stosd";
    reset();
    run([...mov_rdi(FILL), ...mov_rcx(25), ...mov_rax(0x11223344), 0xF3, 0xAB]);
    const dwords = [];
    for(let i = 0; i < 25; i++) dwords.push(0x44, 0x33, 0x22, 0x11);
    expect_bytes(FILL, dwords, "filled dwords");
    expect(reg64(1), 0n, "rcx");
    expect(reg64(7), BigInt(FILL + 100), "rdi");

    active_test = "rep stosq";
    reset();
    run([...mov_rdi(FILL), ...mov_rcx(25), ...mov_rax(0x1122334455667788n), 0xF3, 0x48, 0xAB]);
    const qwords = [];
    for(let i = 0; i < 25; i++) qwords.push(0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11);
    expect_bytes(FILL, qwords, "filled qwords");
    expect(reg64(1), 0n, "rcx");
    expect(reg64(7), BigInt(FILL + 200), "rdi");

    active_test = "rep stosw";
    reset();
    run([...mov_rdi(FILL), ...mov_rcx(25), ...mov_rax(0x1234), 0xF3, 0x66, 0xAB]);
    const words = [];
    for(let i = 0; i < 25; i++) words.push(0x34, 0x12);
    expect_bytes(FILL, words, "filled words");

    // ---- rcx = 0 is a no-op ----
    active_test = "rep movsb rcx=0";
    reset();
    fill(SRC, 16);
    const dst_before = read(DST, 16);
    run([...mov_rsi(SRC), ...mov_rdi(DST), ...mov_rcx(0), 0xF3, 0xA4]);
    expect_bytes(DST, dst_before, "nothing copied");
    expect(reg64(6), BigInt(SRC), "rsi");
    expect(reg64(7), BigInt(DST), "rdi");

    // ---- large copy spanning several pages ----
    active_test = "rep movsb multi-page";
    reset();
    fill(SRC, 0x3000);
    run([...mov_rsi(SRC), ...mov_rdi(DST), ...mov_rcx(0x3000), 0xF3, 0xA4]);
    expect_bytes(DST, read(SRC, 0x3000), "multi-page copy");
    expect(reg64(1), 0n, "rcx");

    if(failures.length)
    {
        for(const failure of failures.slice(0, 20)) console.log("FAIL " + failure);
        console.log("interp64 rep: " + failures.length + " failure(s)");
        process.exit(1);
    }
    console.log("interp64 rep: all tests passed");
    process.exit(0);
});
