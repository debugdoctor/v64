#!/usr/bin/env node

// Long-mode MSRs.
//
// The long-mode MSRs round-trip. The MSRs a 64-bit Linux kernel reads early
// (APIC_BASE, PAT, MTRR, SYSENTER, TSC_AUX, ...) are checked too, and a
// reserved MSR must raise #GP rather than silently returning something.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp64/msr.js`

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
    const note = message => failures.push(activeTest + ": " + message);
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

    const buildPageTables = () => {
        write64(PML4, BigInt(PDPT) | 0x3n);
        write64(PDPT, BigInt(PD) | 0x3n);
        for(let i = 0; i < 512; i++) write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);
    };

    const runCode = code => {
        try
        {
            buildPageTables();
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
    const rdmsr = msr => {
        runCode([0xB9, ...imm32(msr), 0x0F, 0x32]);
        return (BigInt(u32[16 + 2]) << 32n) | BigInt(u32[16]);
    };
    const roundTrip = (name, msr, value) => {
        activeTest = "MSR " + name + " (0x" + msr.toString(16) + ")";
        const code = [
            0xB9, ...imm32(msr),
            0xB8, ...imm32(Number(value & 0xFFFF_FFFFn)),
            0xBA, ...imm32(Number(value >> 32n & 0xFFFF_FFFFn)),
            0x0F, 0x30, // wrmsr
            0xB9, ...imm32(msr),
            0x0F, 0x32, // rdmsr
        ];
        const result = runCode(code);
        if(result.panicked)
        {
            note("panicked (" + result.error + ")");
            return;
        }
        if(result.faulted)
        {
            note("faulted (#GP/#UD)");
            return;
        }
        const read = (BigInt(u32[16 + 2]) << 32n) | BigInt(u32[16]);
        expect(read, value, "round trip");
    };

    // The long-mode MSRs.
    roundTrip("EFER", 0xC0000080, 0x501n);
    roundTrip("STAR", 0xC0000081, 0x0020_0010_0000_0000n);
    roundTrip("LSTAR", 0xC0000082, 0xFFFF_8000_0000_1000n);
    roundTrip("SFMASK", 0xC0000084, 0x200n);
    roundTrip("FS_BASE", 0xC0000100, 0x0000_7FFF_1234_5000n);
    roundTrip("GS_BASE", 0xC0000101, 0x0000_7FFF_0000_2000n);
    roundTrip("KERNEL_GS_BASE", 0xC0000102, 0xFFFF_8000_1111_0000n);

    // MSRs a 64-bit kernel touches during early bring-up.
    roundTrip("APIC_BASE", 0x1B, 0xFEE0_0800n);
    roundTrip("SYSENTER_CS", 0x174, 0x0010n);
    roundTrip("SYSENTER_ESP", 0x175, 0xFFFF_8000_0000_9000n);
    roundTrip("SYSENTER_EIP", 0x176, 0xFFFF_8000_0000_2000n);
    roundTrip("PAT", 0x277, 0x0007_0406_0007_0406n);
    roundTrip("MTRR_DEF_TYPE", 0x2FF, 0x800n);
    roundTrip("TSC_AUX", 0xC0000103, 0x1234_5678n);

    // A reserved MSR must #GP.
    activeTest = "reserved MSR";
    const reserved = runCode([0xB9, 0x83, 0x00, 0x00, 0xC0, 0x0F, 0x32]);
    expect(reserved.panicked, false, "panicked instead of #GP");
    expect(reserved.faulted, true, "rdmsr 0xC0000083 #GP");

    if(failures.length)
    {
        for(const failure of failures) console.log("FAIL " + failure);
        console.log("interp64 msr: " + failures.length + " failure(s)");
        process.exit(1);
    }
    console.log("interp64 msr: all tests passed");
    process.exit(0);
});
