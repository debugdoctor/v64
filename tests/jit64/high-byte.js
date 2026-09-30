#!/usr/bin/env node

// High-byte ALU writeback, including kernfs_create_dir_ns's `or dh, 0x40`
// which adds S_IFDIR to sysfs directory modes.
import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";

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
    const seed = 0x11223344000001EDn;
    const write64 = (address, value) => {
        ex.write32(address, Number(value & 0xFFFFFFFFn));
        ex.write32(address + 4, Number(value >> 32n));
    };
    write64(0x10000, 0x11003n);
    write64(0x11000, 0x12003n);
    write64(0x12000, 0x83n);
    const imm64 = value => Array.from({ length: 8 }, (_, i) => Number(value >> BigInt(i * 8) & 255n));

    const run = (name, r, operation, high) => {
        const body = [0x48, 0xB8 + r, ...imm64(seed), ...operation];
        const code = [
            0x41, 0xBF, 0x58, 0x02, 0, 0,    // mov r15d, 600
            ...body,
            0x41, 0xFF, 0xCF,                // dec r15d
            0x75, (-(body.length + 5)) & 255, // jnz body
            0xF4,
        ];
        cpu.mem8.set(code, BASE);
        ex.jit64_clear_cache();
        cpu.instruction_pointer[0] = BASE;
        ex.enter_long_mode(0x10000);
        cpu.flags[0] = 2;
        cpu.in_hlt[0] = 0;
        view.setUint32(80, 0x90000, true);
        view.setUint32(144, 0, true);
        ex.jit64_set_enabled(1);
        ex.run_exact_instructions(10000);
        assert.equal(cpu.in_hlt[0], 1, `${name}: loop halted`);
        assert.ok(ex.jit64_compiled_count() > 0, `${name}: exercised compiled code`);
        const actual = BigInt(view.getUint32(64 + r * 4, true))
            | BigInt(view.getUint32(128 + r * 4, true)) << 32n;
        const expected = (seed & ~0xFF00n) | BigInt(high) << 8n;
        assert.equal(actual, expected, `${name}: high byte changed, other bits preserved`);
    };

    run("or dh, 0x40 (S_IFDIR)", 2, [0x80, 0xCE, 0x40], 0x41);
    run("or dh, bl", 2, [0xBB, 0x40, 0, 0, 0, 0x08, 0xDE], 0x41);
    ex.write8(0x6000, 0x40);
    run("or dh, [rsi]", 2, [0xBE, 0, 0x60, 0, 0, 0x0A, 0x36], 0x41);
    run("add dh, dl (same parent)", 2, [0x00, 0xD6], 0xEE);
    for(let r = 0; r < 4; r++)
    {
        const names = ["ah", "ch", "dh", "bh"];
        const rm = r + 4;
        run(`or ${names[r]}, 0x40`, r, [0x80, 0xC8 | rm, 0x40], 0x41);
        run(`add ${names[r]}, 3`, r, [0x80, 0xC0 | rm, 3], 4);
        run(`not ${names[r]}`, r, [0xF6, 0xD0 | rm], 0xFE);
        run(`neg ${names[r]}`, r, [0xF6, 0xD8 | rm], 0xFF);
        run(`cmp ${names[r]}, 0x40`, r, [0x80, 0xF8 | rm, 0x40], 1);
        run(`test ${names[r]}, 0x40`, r, [0xF6, 0xC0 | rm, 0x40], 1);
        run(`mov ${names[r]}, 0x40`, r, [0xB0 + rm, 0x40], 0x40);
    }
    console.log("jit64 high byte: arithmetic writes update AH/CH/DH/BH and preserve parent bits");
    emulator.destroy();
});
