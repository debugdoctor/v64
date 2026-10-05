#!/usr/bin/env node

// CR4 control bits.
//
// CR4 must report PAE/PSE in long mode, accept the bits a 64-bit kernel sets
// (PGE, OSFXSR, OSXMMEXCPT, FSGSBASE, SMEP, SMAP, ...), reject reserved bits
// with #GP, and make FSGSBASE actually work once enabled.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp64/cr4.js`

import { v64 } from "../../src/main.js";

const BASE = 0x1000;
const HANDLER = 0x5000;
const IDT = 0x8000;
const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const MARKER = 0x6000;

const emulator = new v64({
    autostart: false,
    memory_size: 4 * 1024 * 1024,
    disable_jit: true,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const buffer = ex.memory.buffer;
    const u32 = new Uint32Array(buffer);

    const failures = [];
    let active_test = "";
    const note = message => failures.push(active_test + ": " + message);
    const expect = (got, want, message) => {
        if(got !== want) note(message + " [want " + want + ", got " + got + "]");
    };

    const write64 = (address, value) => {
        ex.write32(address, Number(value & 0xFFFF_FFFFn));
        ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
    };
    const read32 = address =>
        (ex.read8(address) | ex.read8(address + 1) << 8 | ex.read8(address + 2) << 16 | ex.read8(address + 3) << 24) >>> 0;
    const gate = (offset, selector, type) => {
        const o = BigInt(offset);
        return (o & 0xFFFFn) | BigInt(selector) << 16n | BigInt(type) << 40n | 1n << 47n | (o >> 16n & 0xFFFFn) << 48n;
    };

    const handler = [
        0x48, 0xC7, 0x04, 0x25, 0x00, 0x60, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        0xF4,
    ];

    const build_page_tables = () => {
        write64(PML4, BigInt(PDPT) | 0x3n);
        write64(PDPT, BigInt(PD) | 0x3n);
        for(let i = 0; i < 512; i++) write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);
    };

    const run_code = code => {
        try
        {
            build_page_tables();
            for(let i = 0; i < code.length; i++) ex.write8(BASE + i, code[i]);
            ex.write8(BASE + code.length, 0xF4);
            for(let i = 0; i < handler.length; i++) ex.write8(HANDLER + i, handler[i]);
            for(const vector of [6, 13]) write64(IDT + vector * 16, gate(HANDLER, 0x08, 0xE));
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
            return { faulted: read32(MARKER) !== 0, panicked: false };
        }
        catch(error)
        {
            return { faulted: false, panicked: true, error: String(error && error.message || error) };
        }
    };

    const imm32 = value => [value & 0xFF, value >> 8 & 0xFF, value >> 16 & 0xFF, value >> 24 & 0xFF];
    const rbx = () => (BigInt(u32[32 + 3]) << 32n) | BigInt(u32[16 + 3]);

    // Long mode reports PAE and PSE.
    active_test = "CR4 PAE/PSE";
    run_code([0x0F, 0x20, 0xE0]); // mov rax, cr4
    const cr4 = (BigInt(u32[32]) << 32n) | BigInt(u32[16]);
    expect(cr4 & (1n << 5n), 1n << 5n, "PAE");
    expect(cr4 & (1n << 4n), 1n << 4n, "PSE");

    // Bits a kernel sets must survive a write/read round trip.
    const bits = [
        ["PGE", 1 << 7],
        ["OSFXSR", 1 << 9],
        ["OSXMMEXCPT", 1 << 10],
        ["UMIP", 1 << 11],
        ["FSGSBASE", 1 << 16],
        ["OSXSAVE", 1 << 18],
        ["SMEP", 1 << 20],
        ["SMAP", 1 << 21],
    ];
    for(const [name, bit] of bits)
    {
        active_test = "CR4 " + name;
        const original = cpu.cr[4];
        const result = run_code([
            0x0F, 0x20, 0xE0,             // mov rax, cr4
            0x48, 0x0D, ...imm32(bit),    // or rax, bit
            0x0F, 0x22, 0xE0,             // mov cr4, rax
            0x0F, 0x20, 0xE0,             // mov rax, cr4
            0x48, 0x89, 0xC3,             // mov rbx, rax
        ]);
        cpu.cr[4] = original;
        if(result.panicked) note("panicked (" + result.error + ")");
        else if(result.faulted) note("faulted");
        else expect(rbx() & BigInt(bit), BigInt(bit), name + " set");
    }

    // A reserved bit is a #GP.
    active_test = "CR4 reserved bit";
    const reserved = run_code([
        0x0F, 0x20, 0xE0,                                           // mov rax, cr4
        0x48, 0xBB, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, // mov rbx, 1<<63
        0x48, 0x09, 0xD8,                                           // or rax, rbx
        0x0F, 0x22, 0xE0,                                           // mov cr4, rax
    ]);
    expect(reserved.panicked, false, "panicked instead of #GP");
    expect(reserved.faulted, true, "#GP");

    // With FSGSBASE enabled, WRFSBASE/RDFSBASE round-trip.
    active_test = "FSGSBASE round trip";
    {
        const original = cpu.cr[4];
        const result = run_code([
            0x0F, 0x20, 0xE0,                         // mov rax, cr4
            0x48, 0x0D, ...imm32(1 << 16),            // or rax, FSGSBASE
            0x0F, 0x22, 0xE0,                         // mov cr4, rax
            0x48, 0xB8, 0x34, 0x12, 0x00, 0x00, 0xFF, 0x7F, 0x00, 0x00, // mov rax, 0x7FFF00001234
            0xF3, 0x48, 0x0F, 0xAE, 0xD0,             // wrfsbase rax
            0xF3, 0x48, 0x0F, 0xAE, 0xC3,             // rdfsbase rbx
        ]);
        cpu.cr[4] = original;
        if(result.panicked) note("panicked (" + result.error + ")");
        else if(result.faulted) note("faulted");
        else expect(rbx(), 0x7FFF_0000_1234n, "FS base");
    }

    if(failures.length)
    {
        for(const failure of failures) console.log("FAIL " + failure);
        console.log("interp64 cr4: " + failures.length + " failure(s)");
        process.exit(1);
    }
    console.log("interp64 cr4: all tests passed");
    process.exit(0);
});
