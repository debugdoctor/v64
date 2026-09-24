#!/usr/bin/env node

// Long mode: TSS and the IST stack.
//
// A kernel installs a GDT with a 64-bit TSS, loads TR, and gives an interrupt
// gate an IST index. On the interrupt the CPU must switch to TSS.IST1 before
// pushing the frame. The kernel records RSP in the handler and TR afterwards.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp64/ist.js`

import { v64 } from "../../src/main.js";

const BASE = 0x1000;
const GDT = 0x3000;
const TSS = 0x4000;
const GDTR = 0x5000;
const UD_HANDLER = 0x6200;
const HANDLER = 0x6100;
const IDT = 0x8000;
const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const MARKER = 0x6000;
const IST_STACK = 0x90000;
const SAVED_RSP = 0x7000;
const SAVED_TR = 0x7008;

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
    const note = message => failures.push(message);
    const expect = (got, want, message) => {
        if(got !== want) note(message + " [want " + want + ", got " + got + "]");
    };
    const write16 = (address, value) => {
        ex.write8(address, value & 0xFF);
        ex.write8(address + 1, value >> 8 & 0xFF);
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
    const gate = (offset, selector, type, ist) => {
        const o = BigInt(offset);
        return (o & 0xFFFFn)
            | BigInt(selector) << 16n
            | BigInt(ist) << 32n
            | BigInt(type) << 40n
            | 1n << 47n
            | (o >> 16n & 0xFFFFn) << 48n;
    };

    // Identity-map the first 1 GiB with 2 MiB pages.
    write64(PML4, BigInt(PDPT) | 0x3n);
    write64(PDPT, BigInt(PD) | 0x3n);
    for(let i = 0; i < 512; i++) write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);

    // GDT: null, 64-bit code, 64-bit data, 64-bit TSS.
    write64(GDT + 0, 0n);
    write64(GDT + 8, 0x00AF9A000000FFFFn);
    write64(GDT + 16, 0x00CF92000000FFFFn);
    write64(GDT + 24, 0x0000890040000067n); // TSS: type 0x9, base 0x4000, limit 0x67
    write64(GDT + 32, 0n);

    // The TSS, with IST1 pointing at a dedicated stack.
    for(let i = 0; i < 0x68; i++) ex.write8(TSS + i, 0);
    write64(TSS + 0x24, BigInt(IST_STACK));

    write16(GDTR, 0x27);
    write64(GDTR + 2, BigInt(GDT));

    // #UD handler (marker) and the vector 0x80 handler (records RSP).
    const ud_handler = [0x48, 0xC7, 0x04, 0x25, 0x00, 0x60, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0xF4];
    const handler = [0x48, 0x89, 0x24, 0x25, 0x00, 0x70, 0x00, 0x00, 0xF4]; // mov [SAVED_RSP], rsp
    for(let i = 0; i < ud_handler.length; i++) ex.write8(UD_HANDLER + i, ud_handler[i]);
    for(let i = 0; i < handler.length; i++) ex.write8(HANDLER + i, handler[i]);
    write64(IDT + 6 * 16, gate(UD_HANDLER, 0x08, 0xE, 0));
    write64(IDT + 0x80 * 16, gate(HANDLER, 0x08, 0xE, 1));

    // The kernel: load the GDT, load TR, read it back, take the interrupt.
    const program = [
        0x0F, 0x01, 0x14, 0x25, 0x00, 0x50, 0x00, 0x00, // lgdt [GDTR]
        0x66, 0xB8, 0x18, 0x00,                         // mov ax, 0x18
        0x0F, 0x00, 0xD8,                               // ltr ax
        0x66, 0x0F, 0x00, 0xC8,                         // str ax
        0x66, 0x89, 0x04, 0x25, 0x08, 0x70, 0x00, 0x00, // mov [SAVED_TR], ax
        0xCD, 0x80,                                     // int 0x80
        0xF4,                                           // hlt
    ];
    for(let i = 0; i < program.length; i++) ex.write8(BASE + i, program[i]);

    write64(MARKER, 0n);
    write64(SAVED_RSP, 0n);
    write64(SAVED_TR, 0n);
    cpu.idtr_offset[0] = IDT;
    cpu.idtr_size[0] = 0xFFF;
    cpu.instruction_pointer[0] = BASE;
    u32[16 + 4] = 0x80000;
    u32[32 + 4] = 0;
    cpu.flags[0] &= ~(1 << 9);
    cpu.in_hlt[0] = 0;
    cpu.cpl[0] = 0;
    ex.enter_long_mode(PML4);

    let guard = 0;
    while(!cpu.in_hlt[0] && guard++ < 1000) ex.interp64_run_one();
    if(!cpu.in_hlt[0]) note("did not halt at 0x" + (cpu.instruction_pointer[0] >>> 0).toString(16));

    expect(read64(MARKER), 0n, "no #UD");
    expect(read64(SAVED_RSP), BigInt(IST_STACK - 5 * 8), "handler ran on TSS.IST1");
    expect(read64(SAVED_TR) & 0xFFFFn, 0x18n, "TR selector");

    if(failures.length)
    {
        for(const failure of failures) console.log("FAIL " + failure);
        console.log("interp64 ist: " + failures.length + " failure(s)");
        process.exit(1);
    }
    console.log("interp64 ist: all tests passed");
    process.exit(0);
});
