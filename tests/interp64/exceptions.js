#!/usr/bin/env node

// Long-mode exceptions and interrupts.
//
// Every case installs a gate, triggers a vector and checks the frame the CPU
// pushed: the order of the error code, RIP, CS and RFLAGS, the #PF error code
// bits, gate type (interrupt vs trap) and iretq. The layout is the one in the
// Intel SDM, not whatever the handler happens to observe.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp64/exceptions.js`

import { v64 } from "../../src/main.js";

const BASE = 0x1000;
const HANDLER = 0x5000;
const IDT = 0x8000;
const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const PT = 0x13000;

const MARKER = 0x6000;
const SLOT = [0x6008, 0x6010, 0x6018, 0x6020, 0x6028, 0x6030];
const CURRENT_FLAGS = 0x6038;

const F_IF = 1 << 9;

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
        0x48, 0x8B, 0x04, 0x24,                                                 // mov rax, [rsp]
        0x48, 0x89, 0x04, 0x25, 0x08, 0x60, 0x00, 0x00,                         // mov [SLOT0], rax
        0x48, 0x8B, 0x44, 0x24, 0x08,                                           // mov rax, [rsp+8]
        0x48, 0x89, 0x04, 0x25, 0x10, 0x60, 0x00, 0x00,                         // mov [SLOT1], rax
        0x48, 0x8B, 0x44, 0x24, 0x10,                                           // mov rax, [rsp+0x10]
        0x48, 0x89, 0x04, 0x25, 0x18, 0x60, 0x00, 0x00,                         // mov [SLOT2], rax
        0x48, 0x8B, 0x44, 0x24, 0x18,                                           // mov rax, [rsp+0x18]
        0x48, 0x89, 0x04, 0x25, 0x20, 0x60, 0x00, 0x00,                         // mov [SLOT3], rax
        0x48, 0x8B, 0x44, 0x24, 0x20,                                           // mov rax, [rsp+0x20]
        0x48, 0x89, 0x04, 0x25, 0x28, 0x60, 0x00, 0x00,                         // mov [SLOT4], rax
        0x48, 0x8B, 0x44, 0x24, 0x28,                                           // mov rax, [rsp+0x28]
        0x48, 0x89, 0x04, 0x25, 0x30, 0x60, 0x00, 0x00,                         // mov [SLOT5], rax
        0x9C,                                                                   // pushfq
        0x58,                                                                   // pop rax
        0x48, 0x89, 0x04, 0x25, 0x38, 0x60, 0x00, 0x00,                         // mov [CURRENT_FLAGS], rax
        0xF4,                                                                   // hlt
    ];

    // Identity-map the first 1 GiB: 4 KiB pages for the first 2 MiB (so a
    // single page can be made not-present), 2 MiB pages above that. The user
    // bit lets the ring-3 case execute from these pages.
    const buildPageTables = () => {
        write64(PML4, BigInt(PDPT) | 0x7n);
        write64(PDPT, BigInt(PD) | 0x7n);
        write64(PD, BigInt(PT) | 0x7n);
        for(let i = 0; i < 512; i++)
        {
            write64(PT + i * 8, BigInt(i) * 0x1000n | 0x7n);
        }
        for(let i = 1; i < 512; i++)
        {
            write64(PD + i * 8, BigInt(i) * 0x200000n | 0x87n);
        }
    };

    // Install the handler and gates for `vectors`, then run `program`.
    const boot = (program, { vectors, type = 0xE, cpl = 0, setup } = {}) => {
        buildPageTables();
        if(setup) setup();
        for(let i = 0; i < program.length; i++)
        {
            ex.write8(BASE + i, program[i]);
        }
        for(let i = 0; i < handler.length; i++)
        {
            ex.write8(HANDLER + i, handler[i]);
        }
        for(const vector of vectors)
        {
            write64(IDT + vector * 16, gate(HANDLER, 0x08, type));
        }
        write64(MARKER, 0n);
        for(const slot of SLOT) write64(slot, 0n);
        write64(CURRENT_FLAGS, 0n);

        cpu.idtr_offset[0] = IDT;
        cpu.idtr_size[0] = 0xFFF;
        cpu.instruction_pointer[0] = BASE;
        u32[16 + 4] = 0x80000; // rsp
        u32[32 + 4] = 0;
        cpu.in_hlt[0] = 0;
        cpu.cpl[0] = 0;

        ex.enter_long_mode(PML4);
        cpu.cpl[0] = cpl;
        cpu.flags[0] = (cpu.flags[0] | F_IF) >>> 0;

        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 1000)
        {
            ex.interp64_run_one();
        }
        if(!cpu.in_hlt[0])
        {
            note("did not halt at 0x" + (cpu.instruction_pointer[0] >>> 0).toString(16));
        }
    };

    // #BP (int3): no error code, so the frame starts with RIP.
    activeTest = "int3 (#BP)";
    boot([0xCC, 0xF4], { vectors: [3] });
    expect(read64(MARKER), 1n, "handler ran");
    expect(read64(SLOT[0]), BigInt(BASE + 1), "saved RIP is after int3");

    // #UD (ud2): a fault, so the saved RIP is the faulting instruction.
    activeTest = "ud2 (#UD)";
    boot([0x0F, 0x0B, 0xF4], { vectors: [6] });
    expect(read64(MARKER), 1n, "handler ran");
    expect(read64(SLOT[0]), BigInt(BASE), "saved RIP is the faulting ud2");

    // Vector 0 through a software interrupt (hardware #DE needs `div`, which
    // the interpreter does not decode yet).
    activeTest = "int 0";
    boot([0xCD, 0x00, 0xF4], { vectors: [0] });
    expect(read64(MARKER), 1n, "handler ran");
    expect(read64(SLOT[0]), BigInt(BASE + 2), "saved RIP is after int");

    // A software interrupt (int n) has no error code either.
    activeTest = "int 0x80";
    boot([0xCD, 0x80, 0xF4], { vectors: [0x80] });
    expect(read64(MARKER), 1n, "handler ran");
    expect(read64(SLOT[0]), BigInt(BASE + 2), "saved RIP is after int");

    // #PF pushes an error code, so the frame starts with it and RIP is second.
    activeTest = "#PF error code";
    boot(
        [0x48, 0xC7, 0xC0, 0x00, 0x40, 0x00, 0x00, 0x48, 0x8B, 0x08, 0xF4], // mov rax,0x4000; mov rcx,[rax]; hlt
        { vectors: [14], setup: () => write64(PT + 4 * 8, 0n) },
    );
    expect(read64(MARKER), 1n, "handler ran");
    expect(read64(SLOT[0]), 0n, "error code P=0");
    expect(read64(SLOT[1]), BigInt(BASE + 7), "saved RIP is the faulting instruction");

    // #GP pushes an error code too. Trigger it from ring 3 with a privileged
    // instruction.
    activeTest = "#GP from ring 3";
    boot([0x0F, 0x20, 0xC0, 0xF4], { vectors: [13], cpl: 3 }); // mov rax, cr0
    expect(read64(MARKER), 1n, "handler ran");
    expect(read64(SLOT[0]), 0n, "error code 0");

    // An interrupt gate clears IF in the handler...
    activeTest = "interrupt gate clears IF";
    boot([0xCD, 0x81, 0xF4], { vectors: [0x81], type: 0xE });
    expect(read64(MARKER), 1n, "handler ran");
    expect(read64(CURRENT_FLAGS) & BigInt(F_IF), 0n, "IF cleared");

    // ...a trap gate does not.
    activeTest = "trap gate keeps IF";
    boot([0xCD, 0x82, 0xF4], { vectors: [0x82], type: 0xF });
    expect(read64(MARKER), 1n, "handler ran");
    expect(read64(CURRENT_FLAGS) & BigInt(F_IF), BigInt(F_IF), "IF kept");

    // iretq returns to the instruction after `int` and restores RFLAGS.
    {
        activeTest = "iretq";
        const returnHandler = [
            0x48, 0xC7, 0x04, 0x25, 0x00, 0x60, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // mov qword [MARKER], 1
            0x48, 0xCF,                                                             // iretq
        ];
        buildPageTables();
        const program = [0xCD, 0x80, 0x48, 0xC7, 0xC0, 0x42, 0x00, 0x00, 0x00, 0xF4]; // int 0x80; mov rax,0x42; hlt
        for(let i = 0; i < program.length; i++) ex.write8(BASE + i, program[i]);
        for(let i = 0; i < returnHandler.length; i++) ex.write8(HANDLER + i, returnHandler[i]);
        write64(IDT + 0x80 * 16, gate(HANDLER, 0x08, 0xE));
        write64(MARKER, 0n);
        cpu.idtr_offset[0] = IDT;
        cpu.idtr_size[0] = 0xFFF;
        cpu.instruction_pointer[0] = BASE;
        u32[16 + 4] = 0x80000;
        u32[32 + 4] = 0;
        cpu.in_hlt[0] = 0;
        cpu.cpl[0] = 0;
        ex.enter_long_mode(PML4);
        cpu.flags[0] = (cpu.flags[0] | F_IF) >>> 0;
        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 1000) ex.interp64_run_one();
        expect(cpu.in_hlt[0], 1, "halted after iretq");
        expect(read64(MARKER), 1n, "handler ran");
        expect((BigInt(u32[32]) << 32n) | BigInt(u32[16]), 0x42n, "execution resumed after int");
        expect((cpu.flags[0] >>> 0) & F_IF, F_IF, "iretq restored IF");
    }

    // A fault whose gate is absent escalates to #DF (vector 8).
    activeTest = "double fault";
    boot(
        [0x48, 0xC7, 0xC0, 0x00, 0x40, 0x00, 0x00, 0x48, 0x8B, 0x08, 0xF4],
        { vectors: [8], setup: () => write64(PT + 4 * 8, 0n) },
    );
    expect(read64(MARKER), 1n, "#DF handler ran");

    if(failures.length)
    {
        for(const failure of failures)
        {
            console.log("FAIL " + failure);
        }
        console.log("interp64 exceptions: " + failures.length + " failure(s)");
        process.exit(1);
    }

    console.log("interp64 exceptions: all tests passed");
    process.exit(0);
});
