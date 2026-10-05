#!/usr/bin/env node

// VEX instructions with no ModRM byte must be decoded with the correct length.
// VZEROUPPER/VZEROALL (VEX.128/256.0F.WIG 77) have no ModRM; jit64 used to read
// the following byte as a ModRM, making the instruction one byte too long. In
// glibc's AVX2 strchr the byte after `vzeroupper` is the one-byte `ret` of the
// return-NULL path, so the compiled block skipped the return and ran on into
// unrelated code. That corrupted control flow and segfaulted TinyCorePure64
// when a shell read a file larger than a page.
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
    const RETTGT = BASE + 0x100;
    const STACK = 0x1F000;
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

    const good = 0x600DC0DEn;
    const bad = 0x0BAD0BADn;

    const run = (name, vex) => {
        // FUNC: <vex>; ret; then a fall-through body that must not run.
        ex.jit64_clear_cache();
        set_cpu_config(ex, "JIT_DISABLE", 0);
        cpu.mem8.set([
            ...vex,                       // vzeroupper / vzeroall
            0xC3,                         // ret
            0x48, 0xB8, ...imm64(bad),    // mov rax, bad
            0xF4,                         // hlt
        ], BASE);
        cpu.mem8.set([0x48, 0xB8, ...imm64(good), 0xF4], RETTGT);
        write64(STACK, BigInt(RETTGT));
        for(let i = 0; i < 700; i++)
        {
            cpu.instruction_pointer[0] = BASE;
            view.setUint32(64 + 4 * 4, STACK, true);
            view.setUint32(128 + 4 * 4, 0, true);
            ex.enter_long_mode(0x10000);
            cpu.flags[0] = 2;
            cpu.in_hlt[0] = 0;
            ex.run_exact_instructions(100);
        }
        assert.ok(ex.jit64_compiled_count() > 0, name + ": exercised compiled code");
        assert.equal(reg64(0), good, name + ": the ret returned to the caller");
    };

    run("vzeroupper", [0xC5, 0xF8, 0x77]);
    run("vzeroall", [0xC5, 0xFC, 0x77]);

    console.log("jit64 vex length: test passed");
    emulator.destroy();
});
