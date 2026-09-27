#!/usr/bin/env node

// Diagnostics must never inject guest faults or set page-table A/D bits.
import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";

const emulator = new v64({
    autostart: false,
    memory_size: 4 * 1024 * 1024,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const write64 = (address, value) => {
        ex.write32(address, Number(value & 0xFFFFFFFFn));
        ex.write32(address + 4, Number(value >> 32n));
    };
    write64(0x10000, 0x11003n);
    write64(0x11000, 0x12003n);
    write64(0x12000, 0x13003n);
    write64(0x13000, 0x20003n);
    ex.enter_long_mode(0x10000);

    // Include CR2, RIP, RSP, exception state and the page tables in snapshots.
    const snapshot = () => ({
        state: new Uint8Array(cpu.wasm_memory.buffer, 0, 4096).slice(),
        tables: cpu.mem8.slice(0x10000, 0x14000),
    });
    const before = snapshot();
    const mapped = BigInt.asUintN(64, ex.jit64_debug_translate(0x123n));
    assert.equal(mapped, BigInt(cpu.mem8.byteOffset + 0x20123));
    for(const address of [0x1000n, 0xFFFFFFFF81000000n, 0x800000000000n])
    {
        assert.equal(BigInt.asUintN(64, ex.jit64_debug_translate(address)), 0xFFFFFFFFFFFFFFFFn);
        assert.deepEqual(snapshot(), before, `diagnostic probe changed guest state at ${address.toString(16)}`);
    }
    console.log("jit64 debug probe: mapped, unmapped and non-canonical addresses leave guest state unchanged");
    emulator.destroy();
});
