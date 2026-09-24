#!/usr/bin/env node

// Long-mode system instructions.
//
// Descriptor-table loads (LGDT/LIDT/SGDT/SIDT), CR/DR moves, INVLPG, CLTS,
// fences, PAUSE, WBINVD/INVD, CLFLUSH, RDTSCP and XGETBV. Each instruction is
// run in long mode with a #UD/#GP handler installed, so an undecoded opcode is
// reported as such instead of silently doing nothing.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp64/system.js`

import { v64 } from "../../src/main.js";

const BASE = 0x1000;
const HANDLER = 0x5000;
const IDT = 0x8000;
const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const DESC = 0x3000;
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
    const read64 = address => {
        let value = 0n;
        for(let i = 0; i < 8; i++) value |= BigInt(ex.read8(address + i) >>> 0) << BigInt(i * 8);
        return value;
    };
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
        0xF4,                                                                   // hlt
    ];

    const buildPageTables = () => {
        write64(PML4, BigInt(PDPT) | 0x3n);
        write64(PDPT, BigInt(PD) | 0x3n);
        for(let i = 0; i < 512; i++)
        {
            write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);
        }
    };

    // Run `code` in long mode. Returns true if it took #UD or #GP.
    const runCode = code => {
        buildPageTables();
        for(let i = 0; i < code.length; i++) ex.write8(BASE + i, code[i]);
        ex.write8(BASE + code.length, 0xF4);
        for(let i = 0; i < handler.length; i++) ex.write8(HANDLER + i, handler[i]);
        for(const vector of [6, 13])
        {
            write64(IDT + vector * 16, gate(HANDLER, 0x08, 0xE));
        }
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
        return read64(MARKER) !== 0n;
    };

    // LGDT loads the 10-byte descriptor at DESC (2-byte limit, 8-byte base).
    activeTest = "lgdt";
    ex.write8(DESC + 0, 0x27);
    ex.write8(DESC + 1, 0x00);
    write64(DESC + 2, 0x1234n);
    expect(runCode([0x0F, 0x01, 0x14, 0x25, 0x00, 0x30, 0x00, 0x00]), false, "not #UD");
    expect(cpu.gdtr_size[0] >>> 0, 0x27, "GDTR limit");
    expect(cpu.gdtr_offset[0] >>> 0, 0x1234, "GDTR base");

    // SGDT writes the current GDTR back out.
    activeTest = "sgdt";
    cpu.gdtr_size[0] = 0x27;
    cpu.gdtr_offset[0] = 0x1234;
    expect(runCode([0x0F, 0x01, 0x04, 0x25, 0x00, 0x30, 0x00, 0x00]), false, "not #UD");
    expect(ex.read8(DESC) | ex.read8(DESC + 1) << 8, 0x27, "stored limit");
    expect(read64(DESC + 2), 0x1234n, "stored base");

    // LIDT loads the IDTR; restore the test's own IDT afterwards.
    activeTest = "lidt";
    ex.write8(DESC + 0, 0xFF);
    ex.write8(DESC + 1, 0x0F);
    write64(DESC + 2, 0x20000n);
    expect(runCode([0x0F, 0x01, 0x1C, 0x25, 0x00, 0x30, 0x00, 0x00]), false, "not #UD");
    expect(cpu.idtr_size[0] >>> 0, 0xFFF, "IDTR limit");
    expect(cpu.idtr_offset[0] >>> 0, 0x20000, "IDTR base");

    // SIDT writes the current IDTR back out.
    activeTest = "sidt";
    cpu.idtr_offset[0] = IDT;
    cpu.idtr_size[0] = 0xFFF;
    expect(runCode([0x0F, 0x01, 0x0C, 0x25, 0x00, 0x30, 0x00, 0x00]), false, "not #UD");
    expect(ex.read8(DESC) | ex.read8(DESC + 1) << 8, 0xFFF, "stored limit");
    expect(read64(DESC + 2), BigInt(IDT), "stored base");

    // CLTS clears CR0.TS.
    activeTest = "clts";
    cpu.cr[0] |= 1 << 3;
    expect(runCode([0x0F, 0x06]), false, "not #UD");
    expect(cpu.cr[0] & (1 << 3), 0, "TS cleared");

    // CR reads reflect the CPU state.
    activeTest = "mov rax, cr0";
    expect(runCode([0x0F, 0x20, 0xC0]), false, "not #UD");
    expect((BigInt(u32[32]) << 32n) | BigInt(u32[16]), BigInt(cpu.cr[0] >>> 0), "CR0");

    activeTest = "mov rax, cr4";
    expect(runCode([0x0F, 0x20, 0xE0]), false, "not #UD");
    expect((BigInt(u32[32]) << 32n) | BigInt(u32[16]), BigInt(cpu.cr[4] >>> 0), "CR4");

    // A CR3 write followed by a read gives back the same value.
    activeTest = "mov cr3, rax";
    {
        const code = [
            0x48, 0xB8, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rax, PML4
            0x0F, 0x22, 0xD8,                                           // mov cr3, rax
            0x0F, 0x20, 0xD8,                                           // mov rax, cr3
            0x48, 0x89, 0xC3,                                           // mov rbx, rax
        ];
        expect(runCode(code), false, "not #UD");
        expect((BigInt(u32[32 + 3]) << 32n) | BigInt(u32[16 + 3]), BigInt(PML4), "CR3 round trip");
    }

    // INVLPG and the cache/fence instructions are no-ops that must not fault.
    activeTest = "invlpg";
    expect(runCode([0x0F, 0x01, 0x3C, 0x25, 0x00, 0x30, 0x00, 0x00]), false, "not #UD");
    activeTest = "lfence";
    expect(runCode([0x0F, 0xAE, 0xE8]), false, "not #UD");
    activeTest = "mfence";
    expect(runCode([0x0F, 0xAE, 0xF0]), false, "not #UD");
    activeTest = "sfence";
    expect(runCode([0x0F, 0xAE, 0xF8]), false, "not #UD");
    activeTest = "pause";
    expect(runCode([0xF3, 0x90]), false, "not #UD");
    activeTest = "wbinvd";
    expect(runCode([0x0F, 0x09]), false, "not #UD");
    activeTest = "invd";
    expect(runCode([0x0F, 0x08]), false, "not #UD");
    activeTest = "clflush";
    expect(runCode([0x0F, 0xAE, 0x3C, 0x25, 0x00, 0x30, 0x00, 0x00]), false, "not #UD");

    // RDTSCP also returns TSC_AUX in ECX (0 after reset).
    activeTest = "rdtscp";
    expect(runCode([0x0F, 0x01, 0xF9]), false, "not #UD");
    expect(BigInt(u32[16 + 1]), 0n, "ECX = TSC_AUX");

    // XGETBV returns XCR0, whose bit 0 (x87) is always set.
    activeTest = "xgetbv";
    expect(runCode([0x0F, 0x01, 0xD0]), false, "not #UD");
    expect(BigInt(u32[16]) & 1n, 1n, "XCR0.x87");

    // Debug registers are readable at ring 0.
    activeTest = "mov rax, dr0";
    expect(runCode([0x0F, 0x21, 0xC0]), false, "not #UD");

    // Segment register moves (8C / 8E).
    activeTest = "mov ax, cs";
    expect(runCode([0x66, 0x8C, 0xC8]), false, "not #UD");
    activeTest = "mov ax, ss";
    expect(runCode([0x66, 0x8C, 0xD0]), false, "not #UD");

    if(failures.length)
    {
        for(const failure of failures)
        {
            console.log("FAIL " + failure);
        }
        console.log("interp64 system: " + failures.length + " failure(s)");
        process.exit(1);
    }

    console.log("interp64 system: all tests passed");
    process.exit(0);
});
