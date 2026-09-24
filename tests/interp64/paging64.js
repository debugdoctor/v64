#!/usr/bin/env node

// Long-mode 4-level paging: 1 GiB and 2 MiB huge pages, 4 KiB pages, page
// protection faults and canonical 64-bit virtual addresses above 4 GiB.
//
// Expected behaviour is the Intel SDM's: the page-walk indices, the PS bit,
// the R/W bit and the #PF error code bits (P|W|U) are all dictated by the
// architecture, not read back from the emulator.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp64/paging64.js`

import { v64 } from "../../src/main.js";

const BASE = 0x1000;
const HANDLER = 0x7000;
const IDT = 0x8000;
const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const PT = 0x13000;
const PDPT_HI = 0x14000;
const PD_HI = 0x15000;
const PD2 = 0x16000;

const SCRATCH_CR2 = 0x6000;
const SCRATCH_ERR = 0x6008;
const SCRATCH_RAN = 0x6010;

const emulator = new v64({
    autostart: false,
    memory_size: 8 * 1024 * 1024,
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
    let active_test = "";
    const fmt = value => typeof value === "bigint" ? "0x" + value.toString(16) : String(value);
    const note = message => failures.push(active_test + ": " + message);
    const expect = (got, want, message) => {
        if(got !== want) note(message + " [want " + fmt(want) + ", got " + fmt(got) + "]");
    };

    const write64 = (address, value) => {
        ex.write32(address, Number(value & 0xFFFF_FFFFn));
        ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
    };
    const read64 = address =>
        BigInt(ex.read8(address)) |
        BigInt(ex.read8(address + 1)) << 8n |
        BigInt(ex.read8(address + 2)) << 16n |
        BigInt(ex.read8(address + 3)) << 24n |
        BigInt(ex.read8(address + 4)) << 32n |
        BigInt(ex.read8(address + 5)) << 40n |
        BigInt(ex.read8(address + 6)) << 48n |
        BigInt(ex.read8(address + 7)) << 56n;
    const reg64 = i =>
        (i < 8)
            ? (BigInt(u32[32 + i]) << 32n) | BigInt(u32[16 + i])
            : new BigUint64Array(buffer, 160, 8)[i - 8];

    const gate = (offset, selector, type) =>
        (offset & 0xFFFFn)
        | BigInt(selector) << 16n
        | BigInt(type) << 40n
        | 1n << 47n
        | (offset >> 16n & 0xFFFFn) << 48n;

    // Identity-map the first 1 GiB: 4 KiB pages for the first 2 MiB (so single
    // pages can be remapped), 2 MiB pages for the rest.
    const build_page_tables = () => {
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
    };

    // A long-mode interrupt handler for #PF (14) and #GP (13): record that it
    // ran, CR2 and the error code, then halt.
    const handler = [
        0x48, 0xC7, 0x04, 0x25, 0x10, 0x60, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // mov qword [0x6010], 1
        0x0F, 0x20, 0xD0,                                                       // mov rax, cr2
        0x48, 0x89, 0x04, 0x25, 0x00, 0x60, 0x00, 0x00,                         // mov [0x6000], rax
        0x48, 0x8B, 0x04, 0x24,                                                 // mov rax, [rsp]
        0x48, 0x89, 0x04, 0x25, 0x08, 0x60, 0x00, 0x00,                         // mov [0x6008], rax
        0xF4,                                                                   // hlt
    ];
    for(let i = 0; i < handler.length; i++)
    {
        ex.write8(HANDLER + i, handler[i]);
    }
    write64(IDT + 13 * 16, gate(BigInt(HANDLER), 0x08, 0xE));
    write64(IDT + 14 * 16, gate(BigInt(HANDLER), 0x08, 0xE));

    const boot = (program, setup) => {
        build_page_tables();
        if(setup) setup();
        for(let i = 0; i < program.length; i++)
        {
            ex.write8(BASE + i, program[i]);
        }
        write64(SCRATCH_CR2, 0xFFFF_FFFF_FFFF_FFFFn);
        write64(SCRATCH_ERR, 0xFFFF_FFFF_FFFF_FFFFn);
        write64(SCRATCH_RAN, 0n);

        cpu.idtr_offset[0] = IDT;
        cpu.idtr_size[0] = 0xFFF;
        cpu.instruction_pointer[0] = BASE;
        u32[16 + 4] = 0x80000; // rsp
        u32[32 + 4] = 0;
        cpu.flags[0] &= ~(1 << 9); // cli
        cpu.in_hlt[0] = 0;
        cpu.cpl[0] = 0;

        ex.enter_long_mode(PML4);
        cpu.cr[0] |= 1 << 16; // CR0.WP: enforce read-only pages for supervisor writes
        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 100000)
        {
            ex.main_loop();
        }
        if(!cpu.in_hlt[0])
        {
            note("did not halt at 0x" + (cpu.instruction_pointer[0] >>> 0).toString(16));
        }
    };

    // 1 GiB huge page: PDPT[1].PS maps 0x40000000..0x7FFFFFFF, here to PA 0.
    active_test = "1 GiB PDPT huge page";
    boot([
        0x48, 0xB8, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x00, 0x00, // mov rax, 0x40000000
        0xBB, 0x34, 0x12, 0x00, 0x00,                               // mov ebx, 0x1234
        0x48, 0x89, 0x18,                                           // mov [rax], rbx
        0x48, 0x8B, 0x08,                                           // mov rcx, [rax]
        0xF4,
    ], () => write64(PDPT + 1 * 8, 0x83n));
    expect(reg64(1), 0x1234n, "read back through the huge page");
    expect(read64(0), 0x1234n, "aliases to PA 0");

    // 2 MiB page reached through PDPT[2] -> PD2, physical 0x200000.
    active_test = "2 MiB page at 0x80000000";
    boot([
        0x48, 0xB8, 0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, // mov rax, 0x80000000
        0xBB, 0x78, 0x56, 0x00, 0x00,                               // mov ebx, 0x5678
        0x48, 0x89, 0x18,                                           // mov [rax], rbx
        0x48, 0x8B, 0x08,                                           // mov rcx, [rax]
        0xF4,
    ], () => {
        write64(PDPT + 2 * 8, BigInt(PD2) | 0x3n);
        write64(PD2, 0x200000n | 0x83n);
    });
    expect(reg64(1), 0x5678n, "read back through the 2 MiB page");
    expect(read64(0x200000), 0x5678n, "landed at physical 0x200000");

    // 4 KiB page: PT[2] remaps VA 0x2000 to PA 0x40000.
    active_test = "4 KiB non-identity page";
    boot([
        0x48, 0xC7, 0xC0, 0x00, 0x20, 0x00, 0x00, // mov rax, 0x2000
        0xBB, 0xAB, 0x00, 0x00, 0x00,             // mov ebx, 0xAB
        0x48, 0x89, 0x18,                         // mov [rax], rbx
        0x48, 0x8B, 0x08,                         // mov rcx, [rax]
        0xF4,
    ], () => write64(PT + 2 * 8, 0x40000n | 0x3n));
    expect(reg64(1), 0xABn, "read back through the 4 KiB page");
    expect(read64(0x40000), 0xABn, "landed at physical 0x40000");
    expect(read64(0x2000), 0n, "the old physical page is untouched");

    // A present but read-only page can be read and faults on write, with the
    // #PF error code P|W|U = 1|1|0.
    active_test = "read-only page";
    boot([
        0x48, 0xC7, 0xC0, 0x00, 0x30, 0x00, 0x00, // mov rax, 0x3000
        0x48, 0x8B, 0x08,                         // mov rcx, [rax]  (readable)
        0xBB, 0x01, 0x00, 0x00, 0x00,             // mov ebx, 1
        0x48, 0x89, 0x18,                         // mov [rax], rbx  (faults)
        0xF4,
    ], () => write64(PT + 3 * 8, 0x50000n | 0x1n));
    expect(read64(SCRATCH_RAN), 1n, "#PF handler ran");
    expect(read64(SCRATCH_CR2), 0x3000n, "CR2 holds the faulting address");
    expect(read64(SCRATCH_ERR), 0x3n, "error code P=1 W=1 U=0");

    // A not-present page faults with error code 0.
    active_test = "not-present page";
    boot([
        0x48, 0xC7, 0xC0, 0x00, 0x40, 0x00, 0x00, // mov rax, 0x4000
        0x48, 0x8B, 0x08,                         // mov rcx, [rax]
        0xF4,
    ], () => write64(PT + 4 * 8, 0n));
    expect(read64(SCRATCH_RAN), 1n, "#PF handler ran");
    expect(read64(SCRATCH_CR2), 0x4000n, "CR2 holds the faulting address");
    expect(read64(SCRATCH_ERR), 0n, "error code P=0");

    // A canonical address above 4 GiB, mapped down to physical 0.
    active_test = "canonical address above 4 GiB";
    boot([
        0x48, 0xB8, 0x00, 0x00, 0x00, 0x80, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, 0xFFFFFFFF80000000
        0xBB, 0xAD, 0xDE, 0x00, 0x00,                               // mov ebx, 0xDEAD
        0x48, 0x89, 0x18,                                           // mov [rax], rbx
        0x48, 0x8B, 0x08,                                           // mov rcx, [rax]
        0xF4,
    ], () => {
        write64(PML4 + 511 * 8, BigInt(PDPT_HI) | 0x3n);
        write64(PDPT_HI + 510 * 8, BigInt(PD_HI) | 0x3n);
        write64(PD_HI, 0x83n);
    });
    expect(reg64(1), 0xDEADn, "read back through the high address");
    expect(read64(0), 0xDEADn, "high address aliases to PA 0");

    // A non-canonical address is a #GP, not a #PF.
    active_test = "non-canonical address";
    boot([
        0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, // mov rax, 0x0001000000000000
        0x48, 0x8B, 0x08,                                           // mov rcx, [rax]
        0xF4,
    ]);
    expect(read64(SCRATCH_RAN), 1n, "#GP handler ran");

    if(failures.length)
    {
        for(const failure of failures)
        {
            console.log("FAIL " + failure);
        }
        console.log("interp64 paging64: " + failures.length + " failure(s)");
        process.exit(1);
    }

    console.log("interp64 paging64: all tests passed");
    process.exit(0);
});
