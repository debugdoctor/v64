#!/usr/bin/env node

// The wasm32 / table32 contract.
//
// The emulator runs 64-bit guests, but the wasm module it runs on must stay
// 32-bit: a 32-bit linear memory (no memory64) and a 32-bit funcref table
// (no table64). Guest physical addresses therefore stay below 4 GiB, while
// 64-bit virtual addresses above 4 GiB are mapped onto that 32-bit physical
// space by the page tables. This test checks both halves of that contract.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/wasm32/platform.js`

import { v64 } from "../../src/main.js";

const BASE = 0x1000;
const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const PDPT_HI = 0x13000;
const PD_HI = 0x14000;
const HIGH_VA = 0xFFFF_FFFF_8000_0000n;

const emulator = new v64({
    autostart: false,
    memory_size: 8 * 1024 * 1024,
    disable_jit: 0,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

// Minimal WebAssembly reader, just enough to see the declared memory and table
// limits. A limits flag with bit 2 set means memory64 / table64.
function leb(bytes, p)
{
    let value = 0, shift = 0, byte;
    do
    {
        byte = bytes[p++];
        value += (byte & 0x7F) * 2 ** shift;
        shift += 7;
    } while(byte & 0x80);
    return [value, p];
}

function parse_limits(bytes, p, kind, found)
{
    let flags;
    [flags, p] = leb(bytes, p);
    found.push({ kind, flags, is64: !!(flags & 0x04) });
    let min;
    [min, p] = leb(bytes, p);
    if(flags & 0x01)
    {
        let max;
        [max, p] = leb(bytes, p);
    }
    return p;
}

function inspect_module(source)
{
    const bytes = source instanceof Uint8Array ? source : new Uint8Array(source);
    if(bytes[0] !== 0x00 || bytes[1] !== 0x61 || bytes[2] !== 0x73 || bytes[3] !== 0x6D)
    {
        throw new Error("not a wasm module");
    }

    const found = [];
    let p = 8;
    while(p < bytes.length)
    {
        const id = bytes[p++];
        let size;
        [size, p] = leb(bytes, p);
        const end = p + size;

        if(id === 2) // import section
        {
            let count;
            [count, p] = leb(bytes, p);
            for(let i = 0; i < count; i++)
            {
                let length;
                [length, p] = leb(bytes, p);
                p += length; // module
                [length, p] = leb(bytes, p);
                p += length; // name
                const kind = bytes[p++];
                if(kind === 0)
                {
                    let type;
                    [type, p] = leb(bytes, p);
                }
                else if(kind === 1)
                {
                    p++; // element type
                    p = parse_limits(bytes, p, "table", found);
                }
                else if(kind === 2)
                {
                    p = parse_limits(bytes, p, "memory", found);
                }
                else if(kind === 3)
                {
                    p += 2; // value type + mutability
                }
                else if(kind === 4)
                {
                    p++; // attribute
                    let type;
                    [type, p] = leb(bytes, p);
                }
            }
        }
        else if(id === 4) // table section
        {
            let count;
            [count, p] = leb(bytes, p);
            for(let i = 0; i < count; i++)
            {
                p++; // element type
                p = parse_limits(bytes, p, "table", found);
            }
        }
        else if(id === 5) // memory section
        {
            let count;
            [count, p] = leb(bytes, p);
            for(let i = 0; i < count; i++)
            {
                p = parse_limits(bytes, p, "memory", found);
            }
        }

        p = end;
    }
    return found;
}

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

    // 1. The built module declares 32-bit memory and a 32-bit table.
    const declared = inspect_module(emulator.wasm_source);
    const memories = declared.filter(entry => entry.kind === "memory");
    const tables = declared.filter(entry => entry.kind === "table");
    expect(memories.length > 0, true, "the module declares a memory");
    expect(tables.length > 0, true, "the module declares a function table");
    expect(memories.every(entry => !entry.is64), true, "no memory64");
    expect(tables.every(entry => !entry.is64), true, "no table64");

    // 2. The live resources are 32-bit too.
    expect(cpu.wasm_memory instanceof WebAssembly.Memory, true, "wasm_memory is a WebAssembly.Memory");
    expect(cpu.wm.wasm_table instanceof WebAssembly.Table, true, "wasm_table is a WebAssembly.Table");
    expect(buffer.byteLength < 2 ** 32, true, "linear memory fits in 32 bits");
    expect(cpu.wm.wasm_table.length < 2 ** 32, true, "function table fits in 32 bits");

    // 3. Guest physical addresses stay below 4 GiB even though the CPU has
    //    64-bit virtual addresses.
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

    const run = (program, guardMax) => {
        for(let i = 0; i < program.length; i++)
        {
            ex.write8(BASE + i, program[i]);
        }
        u32[16 + 4] = 0x80000; // rsp
        u32[32 + 4] = 0;
        cpu.flags[0] &= ~(1 << 9);
        cpu.in_hlt[0] = 0;
        ex.enter_long_mode(PML4);
        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < guardMax)
        {
            ex.main_loop();
        }
        return cpu.in_hlt[0] === 1;
    };

    // Identity-map the first 1 GiB with 2 MiB pages, then map the high canonical
    // address onto physical 0.
    write64(PML4, BigInt(PDPT) | 0x3n);
    write64(PDPT, BigInt(PD) | 0x3n);
    for(let i = 0; i < 512; i++)
    {
        write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);
    }
    write64(PML4 + 511 * 8, BigInt(PDPT_HI) | 0x3n);
    write64(PDPT_HI + 510 * 8, BigInt(PD_HI) | 0x3n);
    write64(PD_HI, 0x83n);

    // A 64-bit virtual address above 4 GiB, resolved to physical memory below
    // 4 GiB. 64-bit arithmetic runs in between.
    cpu.instruction_pointer[0] = BASE;
    const high_program = [
        0x48, 0xB8, 0x00, 0x00, 0x00, 0x80, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, HIGH_VA
        0x48, 0xBA, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, // mov rdx, 1<<32
        0x48, 0x01, 0xD0,                                           // add rax, rdx
        0x48, 0xB8, 0x00, 0x00, 0x00, 0x80, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, HIGH_VA
        0xBB, 0xAD, 0xDE, 0x00, 0x00,                               // mov ebx, 0xDEAD
        0x48, 0x89, 0x18,                                           // mov [rax], rbx
        0x48, 0x8B, 0x08,                                           // mov rcx, [rax]
        0xF4,
    ];
    const high_halted = run(high_program, 100000);
    expect(high_halted, true, "the high-address program halted");
    expect(reg64(1), 0xDEADn, "read back through the high address");
    expect(reg64(2), 0x1_0000_0000n, "64-bit add above 4 GiB");
    expect(read64(0), 0xDEADn, "the high address landed below 4 GiB");

    // A hot 64-bit loop goes through the shared 32-bit function table.
    cpu.instruction_pointer[0] = 0x2000;
    const loop_program = [
        0x48, 0x31, 0xC0,                         // xor rax, rax
        0x48, 0xC7, 0xC1, 0x58, 0x02, 0x00, 0x00, // mov rcx, 600
        0x48, 0x83, 0xC0, 0x01,                   // add rax, 1
        0x48, 0x83, 0xE9, 0x01,                   // sub rcx, 1
        0x75, 0xF4,                               // jnz loop
        0xF4,
    ];
    for(let i = 0; i < loop_program.length; i++)
    {
        ex.write8(0x2000 + i, loop_program[i]);
    }
    ex.jit64_clear_cache();
    ex.enter_long_mode(PML4);
    cpu.in_hlt[0] = 0;
    let guard = 0;
    while(!cpu.in_hlt[0] && guard++ < 100000)
    {
        ex.main_loop();
    }
    expect(cpu.in_hlt[0], 1, "the loop halted");
    expect(reg64(0), 600n, "the loop counted to 600");
    expect(ex.jit64_compiled_count() > 0, true, "the loop was JIT-compiled through the shared table");

    if(failures.length)
    {
        for(const failure of failures)
        {
            console.log("FAIL " + failure);
        }
        console.log("wasm32 platform: " + failures.length + " failure(s)");
        process.exit(1);
    }

    console.log("wasm32 platform: all tests passed");
    process.exit(0);
});
