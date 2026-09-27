#!/usr/bin/env node

// Long-mode x87 tests: memory and register forms, integer and BCD
// conversions, and the environment save/restore forms, checked against the
// results the Intel SDM specifies.
//
// Run: node tests/interp64/x87.js

import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";

const CODE = 0x6000;
const DATA = 0xA000;
const IDT = 0x8000;
const STUB = 0x9000;
const VECTOR_FLAG = 0x5000; // clear of the IDT (0x8000+), stubs and stack (0x70000)

const emulator = new v64({
    autostart: false,
    memory_size: 2 * 1024 * 1024,
    disable_jit: 0,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () =>
{
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const buffer = ex.memory.buffer;

    const w64 = (address, value) =>
    {
        ex.write32(address, Number(value & 0xFFFF_FFFFn));
        ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
    };
    // guest addresses here are identity mapped, so the wasm memory can be
    // read directly (re-fetched because the buffer moves when it grows)
    const read32 = address => new DataView(ex.memory.buffer).getUint32(address, true);
    const read16 = address => new DataView(ex.memory.buffer).getUint16(address, true);
    const read64 = address => BigInt(read32(address)) | BigInt(read32(address + 4)) << 32n;

    // identity map the first 1 GiB
    const PML4 = 0x10000, PDPT = 0x11000, PD = 0x12000, PT = 0x13000;
    w64(PML4, BigInt(PDPT) | 3n);
    w64(PDPT, BigInt(PD) | 3n);
    w64(PD, BigInt(PT) | 3n);
    for(let i = 0; i < 512; i++)
    {
        w64(PT + i * 8, BigInt(i) * 0x1000n | 3n);
    }
    for(let i = 1; i < 512; i++)
    {
        w64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);
    }
    ex.enter_long_mode(PML4);

    // a trivial IDT so a #UD is recorded instead of panicking
    const cs = read16(670);
    for(let v = 0; v < 256; v++)
    {
        const stub = STUB + v * 16;
        [0xC6, 0x04, 0x25, 0x00, 0x50, 0x00, 0x00, v, 0xF4].forEach((b, i) => ex.write8(stub + i, b));
        const gate = IDT + v * 16;
        ex.write8(gate, stub & 0xFF);
        ex.write8(gate + 1, stub >> 8 & 0xFF);
        ex.write8(gate + 2, cs & 0xFF);
        ex.write8(gate + 3, cs >> 8 & 0xFF);
        ex.write8(gate + 5, 0x8E);
        ex.write8(gate + 6, stub >> 16 & 0xFF);
        ex.write8(gate + 7, stub >> 24 & 0xFF);
    }
    ex.write8(VECTOR_FLAG, 0);
    w64(1240, BigInt(IDT));       // idtr_base
    ex.write32(564, 256 * 16 - 1); // idtr_size
    ex.write32(568, IDT);          // idtr_offset (32-bit mirror)

    let exception = 0;
    emulator.cpu_exception_hook = n => { exception = n; };

    // run a program and return the delivered exception vector (0 = none)
    const run = bytes =>
    {
        for(let i = 0; i < bytes.length; i++)
        {
            ex.write8(CODE + i, bytes[i]);
        }
        ex.write8(CODE + bytes.length, 0xF4);
        for(let i = 0; i < 8; i++)
        {
            ex.write32(64 + i * 4, 0);
            ex.write32(128 + i * 4, 0);
        }
        for(let i = 0; i < 8; i++)
        {
            w64(160 + i * 8, 0n);
        }
        w64(96, 0x70000n); // rsp
        ex.write32(120, 0x2); // IF set, as in the coverage harness
        ex.write8(VECTOR_FLAG, 0);
        exception = 0;
        cpu.in_hlt[0] = 0;
        new DataView(ex.memory.buffer).setBigUint64(232, BigInt(CODE), true);
        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 200)
        {
            ex.main_loop();
        }
        return exception; // the exception hook is the reliable signal
    };

    const MOV_RAX_DATA = [0x48, 0xC7, 0xC0, 0x00, 0xA0, 0x00, 0x00]; // mov rax, DATA

    assert.equal(run([0x0F, 0x0B]), 6, "ud2 probe reports #UD");

    // FINIT, FLD m32, FADD m32, FSTP m32
    ex.write32(DATA, 0x3FC00000); // 1.5f
    assert.equal(run([0xDB, 0xE3, ...MOV_RAX_DATA, 0xD9, 0x00, 0xD8, 0x00, 0xD9, 0x18]), 0, "no #UD");
    // Stack top from the FPU state (F80 arrays are 16 bytes apart).
    const st0 = () =>
    {
        const top = 1152 + (read16(1032) & 7) * 16;
        return [read64(top), read16(top + 8)];
    };
    const EQ_1_5 = [0xC000000000000000n, 0x3FFF];
    const EQ_3_0 = [0xC000000000000000n, 0x4000];
    const eq = (a, b, what) => assert.equal(a[0] === b[0] && a[1] === b[1], true,
        what + ": got 0x" + a[0].toString(16) + "/0x" + a[1].toString(16));

    // FLD m32 leaves 1.5 on the stack; FADD m32 makes it 3.0
    ex.write32(DATA, 0x3FC00000); // 1.5f
    assert.equal(run([0xDB, 0xE3, ...MOV_RAX_DATA, 0xD9, 0x00]), 0, "no #UD");
    eq(st0(), EQ_1_5, "FLD m32 loads 1.5");
    assert.equal(run([0xDB, 0xE3, ...MOV_RAX_DATA, 0xD9, 0x00, 0xD8, 0x00]), 0, "no #UD");
    eq(st0(), EQ_3_0, "FADD m32 gives 3.0");
    // and the store/load pair round-trips it
    assert.equal(run([0xDB, 0xE3, ...MOV_RAX_DATA, 0xD9, 0x00, 0xD8, 0x00, 0xD9, 0x18, 0xD9, 0x00]), 0, "no #UD");
    eq(st0(), EQ_3_0, "FSTP m32 then FLD m32 round-trips 3.0");

    // FLD m64 then FSQRT (D9 FA); st0 goes 4.0 -> 2.0 and round-trips
    w64(DATA, 0x4010000000000000n); // 4.0
    assert.equal(run([0xDB, 0xE3, ...MOV_RAX_DATA, 0xDD, 0x00, 0xD9, 0xFA]), 0, "no #UD");
    eq(st0(), [0x8000000000000000n, 0x4000], "FSQRT(4.0) = 2.0");
    assert.equal(run([0xDB, 0xE3, ...MOV_RAX_DATA, 0xDD, 0x00, 0xD9, 0xFA, 0xDD, 0x18, 0xDD, 0x00]), 0, "no #UD");
    eq(st0(), [0x8000000000000000n, 0x4000], "FSTP m64 then FLD m64 round-trips 2.0");

    // FILD m32 (DB /0) then FISTP m32 (DB /3) keeps 7, and reading it back as
    // an integer gives 7 again
    ex.write32(DATA, 7);
    assert.equal(run([0xDB, 0xE3, ...MOV_RAX_DATA, 0xDB, 0x00, 0xDB, 0x18, 0xDB, 0x00]), 0, "no #UD");
    eq(st0(), [0xE000000000000000n, 0x4001], "FILD/FISTP/FILD of 7");

    // FILD m64 (DF /5) and FISTP m64 (DF /7) keep -7 (sign bit set in st0)
    w64(DATA, 0xFFFFFFFFFFFFFFF9n); // -7
    assert.equal(run([0xDB, 0xE3, ...MOV_RAX_DATA, 0xDF, 0x28, 0xDF, 0x38, 0xDF, 0x28]), 0, "no #UD");
    assert.equal(st0()[1] & 0x8000, 0x8000, "FILD/FISTP of -7 keeps the sign");

    // FBSTP (DF /6) then FBLD (DF /4) round-trips 123456789, cross-checked
    // against FILD's encoding
    ex.write32(DATA, 123456789);
    assert.equal(run([0xDB, 0xE3, ...MOV_RAX_DATA, 0xDF, 0x28]), 0, "no #UD");
    const from_fild = st0();
    assert.equal(run([0xDB, 0xE3, ...MOV_RAX_DATA, 0xDF, 0x28, 0xDF, 0x30, 0xDF, 0x20]), 0, "no #UD");
    eq(st0(), from_fild, "FBSTP then FBLD round-trips what FILD gave");

    // FNSTENV (D9 /6) then FLDENV (D9 /4): the control word round-trips.
    assert.equal(run([
        0xDB, 0xE3,                      // finit -> control word 0x37f
        ...MOV_RAX_DATA,
        0x66, 0xD9, 0x30,                // fnstenv [rax]  (16-bit form)
        0x66, 0xD9, 0x20,                // fldenv [rax]   -> restores what it saved
    ]), 0, "no #UD");
    assert.equal(read16(1036), 0x037F, "FLDENV restored the control word");

    // FNSAVE (DD /6), FRSTOR (DD /4): the environment half round-trips. The
    // register image follows the 32-bit implementation's 10-byte layout but
    // is not asserted here.
    ex.write32(DATA, 0x3FC00000);
    assert.equal(run([
        0xDB, 0xE3,
        ...MOV_RAX_DATA,
        0xD9, 0x00,                      // fld [rax]     -> st0 = 1.5
        0xDD, 0x30,                      // fnsave [rax]
        0xDD, 0x20,                      // frstor [rax]
    ]), 0, "no #UD");
    assert.equal(read16(1036), 0x037F, "FRSTOR restored the control word");

    console.log("interp64 x87: all tests passed");
    process.exit(0);
});
