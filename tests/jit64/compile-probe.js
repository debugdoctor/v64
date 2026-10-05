#!/usr/bin/env node

// Hotness accounting runs after an interpreted instruction. A CR3 switch may
// unmap that instruction: attempting compilation must not inject a guest #PF.
import assert from "node:assert/strict";
import * as main from "../../src/main.js";
import { set_cpu_config } from "../../src/config.js";
const v64 = main.v64 || main.v64;

const emulator = new v64({
    autostart: false,
    memory_size: 4 * 1024 * 1024,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const view = new DataView(cpu.wasm_memory.buffer);
    const write64 = (address, value) => {
        ex.write32(address, Number(value & 0xFFFFFFFFn));
        ex.write32(address + 4, Number(value >> 32n));
    };
    for(const base of [0x10000, 0x20000])
    {
        write64(base, BigInt(base + 0x1000) | 3n);
        write64(base + 0x1000, BigInt(base + 0x2000) | 3n);
        write64(base + 0x2000, BigInt(base + 0x3000) | 3n);
        for(let page = 0; page < 512; page++)
        {
            write64(base + 0x3000 + page * 8, BigInt(page * 0x1000) | 3n);
        }
    }
    write64(0x23000 + 8, 0n); // New CR3 does not map the old instruction.
    cpu.mem8.set([0x0F, 0x22, 0xD8], 0x1000); // mov cr3, rax
    // A valid #PF gate makes accidental delivery observable without a crash.
    write64(0x8000 + 14 * 16, 0x00008E0000107000n);
    write64(0x8000 + 14 * 16 + 8, 0n);
    cpu.idtr_offset[0] = 0x8000;
    cpu.idtr_size[0] = 0xFFF;
    set_cpu_config(ex, "JIT_DISABLE", 0);

    for(let iteration = 0; iteration < 501; iteration++)
    {
        cpu.instruction_pointer[0] = 0x1000;
        ex.enter_long_mode(0x10000);
        view.setUint32(64, 0x20000, true); // RAX = new CR3
        view.setUint32(128, 0, true);
        view.setUint32(64 + 4 * 4, 0x9000, true);
        view.setUint32(128 + 4 * 4, 0, true);
        view.setBigUint64(1064, 0x12345678n, true); // CR2 sentinel
        cpu.cpl[0] = 0;
        cpu.flags[0] = 2;
        cpu.in_hlt[0] = 0;
        ex.run_exact_instructions(1);
        assert.equal(cpu.cr[3] >>> 0, 0x20000, `CR3 at iteration ${iteration}`);
        assert.equal(view.getBigUint64(232, true), 0x1003n, `RIP at iteration ${iteration}`);
        assert.equal(view.getBigUint64(1064, true), 0x12345678n, `CR2 at iteration ${iteration}`);
        assert.equal(view.getUint32(80, true), 0x9000, `RSP at iteration ${iteration}`);
        assert.equal(cpu.flags[0], 2, `flags at iteration ${iteration}`);
    }
    // A real instruction-fetch #PF also changes RIP before hotness accounting.
    // A compilation probe of the missing page must not deliver a second #PF
    // at the handler entry, before its CR3/GS entry prologue has executed.
    ex.jit64_clear_cache();
    cpu.mem8[0x1000] = 0x90; // nop
    for(let iteration = 0; iteration < 500; iteration++)
    {
        cpu.instruction_pointer[0] = 0x1000;
        ex.enter_long_mode(0x10000);
        view.setUint32(80, 0x9000, true);
        view.setUint32(144, 0, true);
        cpu.cpl[0] = 0;
        cpu.flags[0] = 2;
        cpu.in_hlt[0] = 0;
        if(iteration === 499) write64(0x13000 + 8, 0n);
        ex.run_exact_instructions(1);
        if(iteration < 499)
        {
            assert.equal(view.getBigUint64(232, true), 0x1001n);
        }
    }
    assert.equal(view.getBigUint64(232, true), 0x7000n, "real #PF enters the handler");
    assert.equal(view.getBigUint64(1064, true), 0x1000n, "CR2 holds the real fetch fault");
    assert.equal(view.getUint32(80, true), 0x9000 - 48, "only one #PF frame is pushed");
    const frame = cpu.mem8.byteOffset + 0x9000 - 48;
    assert.equal(view.getBigUint64(frame + 8, true), 0x1000n, "frame saves the faulting instruction, not the handler");
    console.log("jit64 compile probe: unmapped hot instructions do not inject or duplicate #PF");
    emulator.destroy();
});
