#!/usr/bin/env node

// CPUID must not advertise what the long-mode CPU cannot execute.
//
// For every feature bit CPUID sets, the matching instruction is run; if it
// takes #UD the promise is false. Features Linux needs but that are correctly
// not claimed (NX, RDTSCP) are checked too, as is the physical address width,
// which must stay within the 32-bit physical space.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp64/cpuid.js`

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
    const advertised = [];
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
        build_page_tables();
        for(let i = 0; i < code.length; i++) ex.write8(BASE + i, code[i]);
        ex.write8(BASE + code.length, 0xF4);
        for(let i = 0; i < handler.length; i++) ex.write8(HANDLER + i, handler[i]);
        write64(IDT + 6 * 16, gate(HANDLER, 0x08, 0xE));
        write64(IDT + 13 * 16, gate(HANDLER, 0x08, 0xE));
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

    const imm32 = value => [value & 0xFF, value >> 8 & 0xFF, value >> 16 & 0xFF, value >> 24 & 0xFF];
    const cpuid = (leaf, sub) => {
        run_code([0xB8, ...imm32(leaf), 0xB9, ...imm32(sub), 0x0F, 0xA2]);
        return { eax: u32[16], ebx: u32[16 + 3], ecx: u32[16 + 1], edx: u32[16 + 2] };
    };

    // If a feature bit is advertised, the instruction must not #UD.
    const must_work = (name, leaf, sub, reg, bit, code) => {
        const value = cpuid(leaf, sub)[reg] >>> 0;
        if((value & (1 << bit)) === 0) return;
        advertised.push(name);
        active_test = "CPUID advertises " + name;
        expect(run_code(code), false, "advertised but #UD");
    };

    // Leaf 0: vendor and max level.
    active_test = "leaf 0";
    const leaf0 = cpuid(0, 0);
    expect(leaf0.ebx >>> 0, 0x756E6547, "vendor ebx");
    expect(leaf0.edx >>> 0, 0x49656E69, "vendor edx");
    expect(leaf0.ecx >>> 0, 0x6C65746E, "vendor ecx");
    expect(leaf0.eax >>> 0 >= 7, true, "max level covers leaf 7");

    // Leaf 1 features with an instruction attached.
    must_work("FPU", 1, 0, "edx", 0, [0xD9, 0xE8]);                          // fld1
    must_work("TSC", 1, 0, "edx", 4, [0x0F, 0x31]);                          // rdtsc
    must_work("MSR", 1, 0, "edx", 5, [0xB9, 0x80, 0x00, 0x00, 0xC0, 0x0F, 0x32]); // rdmsr EFER
    must_work("CMOV", 1, 0, "edx", 15, [0x0F, 0x44, 0xC3]);                  // cmove eax,ebx
    must_work("MMX", 1, 0, "edx", 23, [0x0F, 0x6E, 0xC0]);                   // movd mm0,eax
    must_work("FXSR", 1, 0, "edx", 24, [0x0F, 0xAE, 0x04, 0x25, 0x00, 0x30, 0x00, 0x00]); // fxsave
    must_work("SSE", 1, 0, "edx", 25, [0x0F, 0x28, 0xC1]);                   // movaps xmm0,xmm1
    must_work("SSE2", 1, 0, "edx", 26, [0x66, 0x0F, 0x6F, 0xC1]);            // movdqa
    must_work("SSE3", 1, 0, "ecx", 0, [0xF2, 0x0F, 0xD0, 0xC1]);             // addsubps
    must_work("POPCNT", 1, 0, "ecx", 23, [0xF3, 0x0F, 0xB8, 0xC3]);          // popcnt eax,ebx
    must_work("RDRAND", 1, 0, "ecx", 30, [0x0F, 0xC7, 0xF0]);                // rdrand eax

    // Leaf 7: enhanced REP MOVSB.
    must_work("ERMS", 7, 0, "ebx", 9, [0xB9, 0x00, 0x00, 0x00, 0x00, 0xF3, 0xA4]); // rep movsb

    // Extended leaf: LM/SYSCALL yes, NX/RDTSCP no.
    active_test = "leaf 0x80000001";
    const ext = cpuid(0x80000001, 0);
    expect((ext.edx >>> 0) & (1 << 29), 1 << 29, "LM");
    expect((ext.edx >>> 0) & (1 << 11), 1 << 11, "SYSCALL");
    expect((ext.edx >>> 0) & (1 << 20), 0, "NX not claimed");
    expect((ext.edx >>> 0) & (1 << 27), 0, "RDTSCP not claimed");

    // OSXSAVE must only be claimed if XGETBV works.
    must_work("OSXSAVE", 1, 0, "ecx", 27, [0x0F, 0x01, 0xD0]);               // xgetbv

    // Physical address width must fit the 32-bit physical space.
    active_test = "leaf 0x80000008";
    const addr = cpuid(0x80000008, 0);
    const phys_bits = (addr.eax >>> 0) & 0xFF;
    const virt_bits = ((addr.eax >>> 0) >> 8) & 0xFF;
    expect(phys_bits >= 32 && phys_bits <= 52, true, "physical address bits (got " + phys_bits + ")");
    expect(virt_bits >= 48, true, "virtual address bits (got " + virt_bits + ")");

    if(advertised.length) console.log("advertised features checked: " + advertised.join(", "));

    if(failures.length)
    {
        for(const failure of failures) console.log("FAIL " + failure);
        console.log("interp64 cpuid: " + failures.length + " failure(s)");
        process.exit(1);
    }
    console.log("interp64 cpuid: all tests passed");
    process.exit(0);
});
