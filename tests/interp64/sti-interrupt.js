#!/usr/bin/env node

// Pending IRQs must be serviced when IF opens: after STI's one-instruction
// shadow, or immediately after POPFQ/IRETQ, before the next CLI region.
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
    write64(0x10000, 0x11003n);
    write64(0x11000, 0x12003n);
    write64(0x12000, 0x83n);
    write64(0x8000 + 0x20 * 16, 0x00008E0000107000n);
    write64(0x8000 + 0x20 * 16 + 8, 0n);
    cpu.idtr_offset[0] = 0x8000;
    cpu.idtr_size[0] = 0xFFF;
    // sti; nop; cli; hlt. Handler sets a distinct register, then halts.
    cpu.mem8.set([0xFB, 0x90, 0xFA, 0xF4], 0x1000);
    cpu.mem8.set([0xBB, 0x78, 0x56, 0x34, 0x12, 0xF4], 0x7000);
    // PIC ports are handled in Rust, before the JavaScript I/O table.
    const out = (port, value) => ex.jit64_out(BigInt(port), BigInt(value), 8);
    out(0x20, 0x11);
    out(0x21, 0x20);
    out(0x21, 4);
    out(0x21, 1);
    out(0x21, 0xFE);
    out(0xA1, 0xFF);
    cpu.instruction_pointer[0] = 0x1000;
    ex.enter_long_mode(0x10000);
    view.setUint32(80, 0x9000, true);
    view.setUint32(144, 0, true);
    cpu.flags[0] = 2;
    cpu.cpl[0] = 0;
    cpu.in_hlt[0] = 0;
    set_cpu_config(ex, "JIT_DISABLE", 0);
    cpu.device_lower_irq(0);
    cpu.device_raise_irq(0);
    const pic = new Uint8Array(cpu.wasm_memory.buffer, cpu.get_pic_addr_master(), 12);
    assert.equal(pic[0] & pic[3], 1, "IRQ0 is pending and unmasked");
    ex.run_exact_instructions(1);
    assert.equal(view.getBigUint64(232, true), 0x1001n, "STI itself does not deliver IRQ");
    assert.equal(cpu.flags[0] & 0x200, 0x200, "STI enables IF");
    ex.run_exact_instructions(1);
    assert.equal(view.getBigUint64(232, true), 0x7000n, "shadow expiry delivers IRQ before CLI");
    ex.run_fixed_cycles(1);
    assert.equal(view.getUint32(76, true), 0x12345678, "pending IRQ enters handler before CLI");
    const frame = cpu.mem8.byteOffset + 0x9000 - 40;
    assert.equal(view.getBigUint64(frame, true), 0x1002n, "IRQ resumes after the shadow NOP");
    assert.equal(view.getBigUint64(frame + 16, true) & 0x200n, 0x200n, "saved IF is enabled");

    // Warm a real compiled block at the instruction following STI. Its jump
    // must not execute within the shadow: only the first NOP is protected.
    cpu.mem8.set([0x90, 0xEB, 0xFD], 0x4000); // nop; jmp 0x4000
    cpu.flags[0] = 2;
    out(0x20, 0x20); // EOI for the previous IRQ
    const compiles = ex.jit64_stat(3);
    for(let iteration = 0; iteration < 500; iteration++)
    {
        view.setBigUint64(232, 0x4000n, true);
        cpu.in_hlt[0] = 0;
        ex.run_exact_instructions(1);
    }
    assert.ok(ex.jit64_stat(3) > compiles, "following NOP/jump block is hot enough to compile");
    cpu.mem8[0x3FFF] = 0xFB;
    view.setBigUint64(232, 0x3FFFn, true);
    view.setUint32(80, 0x9000, true);
    view.setUint32(144, 0, true);
    cpu.flags[0] = 2;
    cpu.in_hlt[0] = 0;
    cpu.device_lower_irq(0);
    cpu.device_raise_irq(0);
    ex.run_exact_instructions(2);
    assert.equal(view.getBigUint64(232, true), 0x7000n, "IRQ delivered even when the shadow instruction has a cached block");
    assert.equal(view.getBigUint64(frame, true), 0x4001n, "only the first instruction of that block ran");

    // POPFQ and IRETQ open an interrupt window without STI's shadow. Servicing
    // them only at batch boundaries lets a recurring IRQ0 starve lower IRQs.
    for(const instruction of [0x9D, 0xCF])
    {
        cpu.flags[0] = 2;
        out(0x20, 0x20);
        cpu.mem8.set([instruction, 0xFA, 0xF4], 0x5000);
        view.setBigUint64(232, 0x5000n, true);
        view.setUint32(80, 0x8800, true);
        view.setUint32(144, 0, true);
        cpu.in_hlt[0] = 0;
        if(instruction === 0x9D)
        {
            write64(0x8800, 0x202n);
        }
        else
        {
            for(const [offset, value] of [[0, 0x6000n], [8, 0x10n], [16, 0x202n], [24, 0x9000n], [32, 0x18n]])
            {
                write64(0x8800 + offset, value);
            }
        }
        cpu.device_lower_irq(0);
        cpu.device_raise_irq(0);
        ex.run_exact_instructions(1);
        assert.equal(view.getBigUint64(232, true), 0x7000n,
            `${instruction === 0x9D ? "POPFQ" : "IRETQ"} services pending IRQ on enabling IF`);
        const expected_sp = instruction === 0x9D ? 0x8808 : 0x9000;
        const irq_frame = cpu.mem8.byteOffset + expected_sp - 40;
        assert.equal(view.getBigUint64(irq_frame, true), instruction === 0x9D ? 0x5001n : 0x6000n,
            "IRQ saves the architectural return RIP");
    }
    console.log("interp64 interrupt windows: STI shadow (including cached JIT), POPFQ and IRETQ passed");
    emulator.destroy();
});
