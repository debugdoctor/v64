#!/usr/bin/env node

// Long-mode interpreter: REX, r8-r15 and addressing modes.
//
// The programs use LEA so the addressing result is visible without touching
// memory. The expected values come from the Intel SDM manual and were
// cross-checked on an x86-64 host before being written down.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp64/addressing.js`

import { v64 } from "../../src/main.js";

const BASE = 0x1000;

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
    const buffer = ex.memory.buffer;
    const u32 = new Uint32Array(buffer);
    const ext = new BigUint64Array(buffer, 160, 8);

    const failures = [];
    let active_test = "";
    const fmt = value => typeof value === "bigint" ? "0x" + value.toString(16) : String(value);
    const note = message => failures.push(active_test + ": " + message);
    const expect = (got, want, message) => {
        if(got !== want) note(message + " [want " + fmt(want) + ", got " + fmt(got) + "]");
    };

    const reg64 = i =>
        (i < 8)
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

    const reset = () => {
        for(let i = 0; i < 16; i++) set_reg64(i, 0n);
        cpu.cr[0] = 0;
        cpu.is_32[0] = 1;
        cpu.cpl[0] = 0;
        cpu.flags[0] = 0x2;
        cpu.in_hlt[0] = 0;
    };

    const run = code => {
        for(let i = 0; i < code.length; i++)
        {
            ex.write8(BASE + i, code[i]);
        }
        ex.write8(BASE + code.length, 0xF4);
        cpu.instruction_pointer[0] = BASE;
        cpu.in_hlt[0] = 0;
        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 500)
        {
            ex.interp64_run_one();
        }
        if(!cpu.in_hlt[0])
        {
            note("did not halt (#UD?) at 0x" + (cpu.instruction_pointer[0] >>> 0).toString(16));
        }
    };

    // lea rax, [rbx + rcx*4 + 0x10]
    active_test = "lea base+index*4+disp8";
    reset();
    set_reg64(3, 0x1000);
    set_reg64(1, 3);
    run([0x48, 0x8D, 0x44, 0x8B, 0x10]);
    expect(reg64(0), 0x101Cn, "sum with scale 4");

    // lea r8, [r9 + r10*8] - REX.R and REX.B select extended registers
    active_test = "lea r8,[r9+r10*8]";
    reset();
    set_reg64(9, 0x2000);
    set_reg64(10, 0x100);
    run([0x4F, 0x8D, 0x04, 0xD1]);
    expect(reg64(8), 0x2800n, "extended base and index");

    // lea rax, [rbx*2 + 0x1234] - index without a base
    active_test = "lea rax,[rbx*2+disp32]";
    reset();
    set_reg64(3, 0x1000);
    run([0x48, 0x8D, 0x84, 0x1B, 0x34, 0x12, 0x00, 0x00]);
    expect(reg64(0), 0x3234n, "index-only SIB");

    // lea eax, [ebx + ecx] - 67 override truncates the address to 32 bits and
    // the 32-bit destination zero-extends into RAX
    active_test = "lea eax,[ebx+ecx]";
    reset();
    set_reg64(3, 0x1_0000_1234n);
    set_reg64(1, 0x1000);
    run([0x67, 0x8D, 0x04, 0x0B]);
    expect(reg64(0), 0x2234n, "32-bit address size");

    // lea rax, [r13] - r13/rbp need an explicit disp8
    active_test = "lea rax,[r13]";
    reset();
    set_reg64(13, 0x1234);
    run([0x49, 0x8D, 0x45, 0x00]);
    expect(reg64(0), 0x1234n, "extended base with disp8");

    // lea rax, [rbx + rcx] - plain base+index
    active_test = "lea rax,[rbx+rcx]";
    reset();
    set_reg64(3, 0x1000);
    set_reg64(1, 0x234);
    run([0x48, 0x8D, 0x04, 0x0B]);
    expect(reg64(0), 0x1234n, "base+index");

    // lea rax, [rip + 0x1234]: the instruction is 7 bytes and starts at BASE.
    active_test = "lea rax,[rip+disp32]";
    reset();
    run([0x48, 0x8D, 0x05, 0x34, 0x12, 0x00, 0x00]);
    expect(reg64(0), BigInt(BASE + 7 + 0x1234), "RIP-relative");

    // REX selects r8b instead of ah, so 8-bit extended registers work.
    active_test = "mov/add r8b";
    reset();
    run([0x41, 0xB0, 0x11, 0x41, 0x80, 0xC0, 0x01]); // mov r8b,0x11; add r8b,1
    expect(reg64(8), 0x12n, "r8b");

    // A 32-bit write to an extended register clears its top half.
    active_test = "mov r8d, r9d";
    reset();
    set_reg64(8, 0xFFFF_FFFF_FFFF_FFFFn);
    set_reg64(9, 0x1_1122_3344n);
    run([0x45, 0x89, 0xC8]); // mov r8d, r9d
    expect(reg64(8), 0x1122_3344n, "32-bit move zero-extends");

    // A full 64-bit immediate into an extended register.
    active_test = "mov r8, imm64";
    reset();
    run([0x49, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11]);
    expect(reg64(8), 0x1122_3344_5566_7788n, "r8 imm64");

    if(failures.length)
    {
        for(const failure of failures)
        {
            console.log("FAIL " + failure);
        }
        console.log("interp64 addressing: " + failures.length + " failure(s)");
        process.exit(1);
    }

    console.log("interp64 addressing: all tests passed");
    process.exit(0);
});
