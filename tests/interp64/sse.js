#!/usr/bin/env node

// Long-mode SSE / SSE2.
//
// The expected values come from the Intel SDM and were cross-checked on an
// x86-64 host. The file also checks the CPUID promise: if leaf 1 advertises
// SSE/SSE2, then SSE instructions must actually execute. Right now they do
// not, which is the point of this test.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp64/sse.js`

import { v64 } from "../../src/main.js";

const BASE = 0x1000;
const HANDLER = 0x5000;
const IDT = 0x8000;
const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const SCRATCH = 0x3000;
const MARKER = 0x6000;

const emulator = new v64({
    autostart: false,
    memory_size: 4 * 1024 * 1024,
    disable_jit: 1,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const buffer = ex.memory.buffer;
    const u32 = new Uint32Array(buffer);
    const ext = new BigUint64Array(buffer, 160, 8);

    const failures = [];
    let activeTest = "";
    const fmt = value => typeof value === "bigint" ? "0x" + value.toString(16) : String(value);
    const note = message => failures.push(activeTest + ": " + message);
    const expect = (got, want, message) => {
        if(got !== want) note(message + " [want " + fmt(want) + ", got " + fmt(got) + "]");
    };

    const write64 = (address, value) => {
        ex.write32(address, Number(value & 0xFFFF_FFFFn));
        ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
    };
    const read32 = address =>
        (ex.read8(address) | ex.read8(address + 1) << 8 | ex.read8(address + 2) << 16 | ex.read8(address + 3) << 24) >>> 0;
    const gate = (offset, selector, type) => {
        const o = BigInt(offset);
        return (o & 0xFFFFn)
            | BigInt(selector) << 16n
            | BigInt(type) << 40n
            | 1n << 47n
            | (o >> 16n & 0xFFFFn) << 48n;
    };

    const handler = [
        0x48, 0xC7, 0x04, 0x25, 0x00, 0x60, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // mov qword [MARKER], 1
        0xF4,
    ];

    const buildPageTables = () => {
        write64(PML4, BigInt(PDPT) | 0x3n);
        write64(PDPT, BigInt(PD) | 0x3n);
        for(let i = 0; i < 512; i++)
        {
            write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);
        }
    };

    const resetRegs = () => {
        for(let i = 0; i < 16; i++)
        {
            u32[16 + i] = 0;
            u32[32 + i] = 0;
            if(i < 8) ext[i] = 0n;
        }
    };
    const setReg = (i, value) => {
        const v = BigInt(value);
        if(i < 8)
        {
            u32[16 + i] = Number(v & 0xFFFF_FFFFn);
            u32[32 + i] = Number(v >> 32n & 0xFFFF_FFFFn);
        }
        else
        {
            ext[i - 8] = v;
        }
    };
    const reg64 = i =>
        (i < 8)
            ? (BigInt(u32[32 + i]) << 32n) | BigInt(u32[16 + i])
            : ext[i - 8];

    // Run `code` in long mode; returns true if it took #UD.
    const runCode = code => {
        buildPageTables();
        for(let i = 0; i < code.length; i++) ex.write8(BASE + i, code[i]);
        ex.write8(BASE + code.length, 0xF4);
        for(let i = 0; i < handler.length; i++) ex.write8(HANDLER + i, handler[i]);
        write64(IDT + 6 * 16, gate(HANDLER, 0x08, 0xE));
        write64(MARKER, 0n);
        cpu.idtr_offset[0] = IDT;
        cpu.idtr_size[0] = 0xFFF;
        cpu.instruction_pointer[0] = BASE;
        u32[16 + 4] = 0x80000;
        u32[32 + 4] = 0;
        cpu.in_hlt[0] = 0;
        cpu.cpl[0] = 0;
        ex.enter_long_mode(PML4);
        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 1000) ex.interp64_run_one();
        if(!cpu.in_hlt[0]) note("did not halt at 0x" + (cpu.instruction_pointer[0] >>> 0).toString(16));
        return read32(MARKER) !== 0;
    };

    const sse = (name, init, code, checks) => {
        activeTest = name;
        resetRegs();
        for(const [i, value] of Object.entries(init)) setReg(+i, value);
        expect(runCode(code), false, "not #UD");
        checks();
    };

    // movd moves a dword in and out of an XMM register.
    sse("movd round trip", { 0: 0x11223344 }, [0x66, 0x0F, 0x6E, 0xC0, 0x66, 0x0F, 0x7E, 0xC0], () => {
        expect(reg64(0), 0x11223344n, "round trip");
    });

    sse("pxor", {}, [0x66, 0x0F, 0xEF, 0xC0, 0x66, 0x0F, 0x7E, 0xC0], () => {
        expect(reg64(0), 0n, "xmm0 zeroed");
    });

    sse("pcmpeqd", {}, [0x66, 0x0F, 0x76, 0xC0, 0x66, 0x48, 0x0F, 0x7E, 0xC0], () => {
        expect(reg64(0), 0xFFFF_FFFF_FFFF_FFFFn, "all ones");
    });

    sse("paddd", { 0: 1, 3: 2 }, [
        0x66, 0x0F, 0x6E, 0xC0, 0x66, 0x0F, 0x6E, 0xCB, 0x66, 0x0F, 0xFE, 0xC1, 0x66, 0x0F, 0x7E, 0xC0,
    ], () => expect(reg64(0), 3n, "lane sum"));

    sse("psubd", { 0: 5, 3: 2 }, [
        0x66, 0x0F, 0x6E, 0xC0, 0x66, 0x0F, 0x6E, 0xCB, 0x66, 0x0F, 0xFA, 0xC1, 0x66, 0x0F, 0x7E, 0xC0,
    ], () => expect(reg64(0), 3n, "lane difference"));

    sse("pand/por/pxor", { 0: 0xF0, 3: 0x0F }, [
        0x66, 0x0F, 0x6E, 0xC0, 0x66, 0x0F, 0x6E, 0xCB,
        0x66, 0x0F, 0xDB, 0xC1, 0x66, 0x0F, 0xEB, 0xC1, 0x66, 0x0F, 0xEF, 0xC1,
        0x66, 0x0F, 0x7E, 0xC0,
    ], () => expect(reg64(0), 0n, "boolean chain"));

    sse("addss", { 0: 0x3F800000, 3: 0x40000000 }, [
        0x66, 0x0F, 0x6E, 0xC0, 0x66, 0x0F, 0x6E, 0xCB, 0xF3, 0x0F, 0x58, 0xC1, 0x66, 0x0F, 0x7E, 0xC0,
    ], () => expect(reg64(0), 0x40400000n, "1.0f + 2.0f"));

    sse("addsd", { 0: 0x3FF0000000000000, 3: 0x4000000000000000 }, [
        0x66, 0x48, 0x0F, 0x6E, 0xC0, 0x66, 0x48, 0x0F, 0x6E, 0xCB, 0xF2, 0x0F, 0x58, 0xC1, 0x66, 0x48, 0x0F, 0x7E, 0xC0,
    ], () => expect(reg64(0), 0x4008000000000000n, "1.0 + 2.0"));

    sse("mulss", { 0: 0x40400000, 3: 0x40800000 }, [
        0x66, 0x0F, 0x6E, 0xC0, 0x66, 0x0F, 0x6E, 0xCB, 0xF3, 0x0F, 0x59, 0xC1, 0x66, 0x0F, 0x7E, 0xC0,
    ], () => expect(reg64(0), 0x41400000n, "3.0f * 4.0f"));

    sse("movmskps", {}, [0x66, 0x0F, 0x76, 0xC0, 0x0F, 0x50, 0xC0], () => {
        expect(reg64(0), 0xFn, "sign mask");
    });

    sse("movdqa copy", { 3: 0xCAFEBABE }, [
        0x66, 0x0F, 0x6E, 0xCB, 0x66, 0x0F, 0x6F, 0xC1, 0x66, 0x0F, 0x7E, 0xC0,
    ], () => expect(reg64(0), 0xCAFEBABEn, "copied lane"));

    sse("cvtsi2ss", { 0: 3 }, [0xF3, 0x0F, 0x2A, 0xC0, 0x66, 0x0F, 0x7E, 0xC0], () => {
        expect(reg64(0), 0x40400000n, "int to float");
    });

    sse("cvttss2si", { 0: 0x40400000 }, [0x66, 0x0F, 0x6E, 0xC0, 0xF3, 0x0F, 0x2C, 0xC0], () => {
        expect(reg64(0), 3n, "float to int");
    });

    sse("pslld imm", { 0: 1 }, [
        0x66, 0x0F, 0x6E, 0xC0, 0x66, 0x0F, 0x72, 0xF0, 0x04, 0x66, 0x0F, 0x7E, 0xC0,
    ], () => expect(reg64(0), 0x10n, "shifted lane"));

    sse("psrld imm", { 0: 0x100 }, [
        0x66, 0x0F, 0x6E, 0xC0, 0x66, 0x0F, 0x72, 0xD0, 0x04, 0x66, 0x0F, 0x7E, 0xC0,
    ], () => expect(reg64(0), 0x10n, "shifted lane"));

    // MXCSR round trip through memory.
    activeTest = "ldmxcsr/stmxcsr";
    ex.write32(SCRATCH, 0x1F80);
    expect(runCode([
        0x0F, 0xAE, 0x14, 0x25, 0x00, 0x30, 0x00, 0x00, // ldmxcsr [0x3000]
        0x0F, 0xAE, 0x1C, 0x25, 0x04, 0x30, 0x00, 0x00, // stmxcsr [0x3004]
    ]), false, "not #UD");
    expect(read32(SCRATCH + 4), 0x1F80, "MXCSR round trip");

    // FXSAVE writes MXCSR at offset 24 of the 512-byte area.
    activeTest = "fxsave";
    ex.write32(SCRATCH, 0x1F80);
    expect(runCode([
        0x0F, 0xAE, 0x14, 0x25, 0x00, 0x30, 0x00, 0x00, // ldmxcsr [0x3000]
        0x0F, 0xAE, 0x04, 0x25, 0x00, 0x31, 0x00, 0x00, // fxsave [0x3100]
    ]), false, "not #UD");
    expect(read32(0x3100 + 24), 0x1F80, "saved MXCSR");

    activeTest = "fxrstor";
    expect(runCode([
        0x0F, 0xAE, 0x0C, 0x25, 0x00, 0x31, 0x00, 0x00, // fxrstor [0x3100]
    ]), false, "not #UD");

    // movdqu through memory.
    activeTest = "movdqu memory";
    resetRegs();
    setReg(0, 0x11223344);
    expect(runCode([
        0x66, 0x0F, 0x6E, 0xC0,                         // movd xmm0, eax
        0xF3, 0x0F, 0x7F, 0x04, 0x25, 0x00, 0x30, 0x00, 0x00, // movdqu [0x3000], xmm0
        0xF3, 0x0F, 0x6F, 0x0C, 0x25, 0x00, 0x30, 0x00, 0x00, // movdqu xmm1, [0x3000]
        0x66, 0x0F, 0x7E, 0xC8,                         // movd eax, xmm1
    ]), false, "not #UD");
    expect(read32(SCRATCH), 0x11223344, "stored lane");
    expect(reg64(0), 0x11223344n, "loaded lane");

    // CPUID promises SSE/SSE2, so the instructions above must work.
    activeTest = "CPUID vs SSE";
    resetRegs();
    runCode([0xB8, 0x01, 0x00, 0x00, 0x00, 0x0F, 0xA2]); // mov eax,1; cpuid
    const edx = u32[16 + 2] >>> 0;
    expect(edx & (1 << 25), 1 << 25, "CPUID leaf 1 advertises SSE");
    expect(edx & (1 << 26), 1 << 26, "CPUID leaf 1 advertises SSE2");

    if(failures.length)
    {
        for(const failure of failures)
        {
            console.log("FAIL " + failure);
        }
        console.log("interp64 sse: " + failures.length + " failure(s)");
        process.exit(1);
    }

    console.log("interp64 sse: all tests passed");
    process.exit(0);
});
