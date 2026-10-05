#!/usr/bin/env node

// xchg with rAX, including the 0x90 + REX.B encoding (`49 90` = xchg r8, rax).
// jit64 used to decode 0x90 as NOP unconditionally, which dropped the exchange
// in compiled blocks; busybox's xrealloc_vector relies on `xchg r8, rax`, so
// `top` corrupted its process-list pointer and segfaulted after a few
// refreshes. Each program is run enough times to be JIT-compiled.
import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";
import { set_cpu_config } from "../../src/config.js";

const emulator = new v64({
    autostart: false,
    memory_size: 2 * 1024 * 1024,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const view = new DataView(ex.memory.buffer);
    const BASE = 0x1000;
    const write64 = (a, v) => {
        ex.write32(a, Number(v & 0xFFFFFFFFn));
        ex.write32(a + 4, Number(v >> 32n));
    };
    write64(0x10000, 0x11003n);
    write64(0x11000, 0x12003n);
    write64(0x12000, 0x83n);
    const imm64 = v => Array.from({ length: 8 }, (_, i) => Number(v >> BigInt(i * 8) & 255n));
    const reg64 = i =>
        (i < 8)
            ? BigInt(view.getUint32(64 + i * 4, true)) | BigInt(view.getUint32(128 + i * 4, true)) << 32n
            : BigInt(view.getUint32(160 + (i - 8) * 8, true)) | BigInt(view.getUint32(164 + (i - 8) * 8, true)) << 32n;

    const a = 0x1122334455667788n;
    const b = 0x99AABBCCDDEEFF00n;

    const run = (name, target, mov_op, xchg_op) => {
        const code = [
            ...mov_op, ...imm64(a), // mov <target>, a
            0x48, 0xB8, ...imm64(b), // mov rax, b
            ...xchg_op,              // xchg rax, <target>
            0xF4,
        ];
        ex.jit64_clear_cache();
        set_cpu_config(ex, "JIT_DISABLE", 0);
        cpu.mem8.set(code, BASE);
        for(let i = 0; i < 700; i++)
        {
            cpu.instruction_pointer[0] = BASE;
            ex.enter_long_mode(0x10000);
            cpu.flags[0] = 2;
            cpu.in_hlt[0] = 0;
            ex.run_exact_instructions(100);
        }
        assert.ok(ex.jit64_compiled_count() > 0, name + ": exercised compiled code");
        assert.equal(reg64(0), a, name + ": rax received the other register");
        assert.equal(reg64(target), b, name + ": the other register received rax");
    };

    // 0x90 + REX.B is xchg r8..r15, rax; 0x91..0x97 is xchg rCX..rDI, rax.
    run("xchg r8, rax", 8, [0x49, 0xB8], [0x49, 0x90]);
    run("xchg r9, rax", 9, [0x49, 0xB9], [0x49, 0x91]);
    run("xchg r12, rax", 12, [0x49, 0xBC], [0x49, 0x94]);
    run("xchg rcx, rax", 1, [0x48, 0xB9], [0x48, 0x91]);
    run("xchg rdx, rax", 2, [0x48, 0xBA], [0x48, 0x92]);

    // Plain 0x90 is still a NOP.
    ex.jit64_clear_cache();
    cpu.mem8.set([
        0x48, 0xB8, ...imm64(a), // mov rax, a
        0x90,                    // nop
        0xF4,
    ], BASE);
    for(let i = 0; i < 700; i++)
    {
        cpu.instruction_pointer[0] = BASE;
        ex.enter_long_mode(0x10000);
        cpu.flags[0] = 2;
        cpu.in_hlt[0] = 0;
        ex.run_exact_instructions(100);
    }
    assert.equal(reg64(0), a, "nop preserves rax");

    console.log("jit64 xchg: test passed");
    emulator.destroy();
});
