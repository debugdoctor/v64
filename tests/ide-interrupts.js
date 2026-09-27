#!/usr/bin/env node

// Exercise ATA interrupt masking and the observable command-completion state.
import assert from "node:assert/strict";
import { v64 } from "../src/main.js";

const emulator = new v64({
    autostart: false,
    memory_size: 4 * 1024 * 1024,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
    cdrom: { buffer: new ArrayBuffer(2048) },
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const channel = cpu.devices.ide.secondary;
    const drive = channel.master;
    const events = [];
    cpu.device_raise_irq = irq => {
        events.push({ event: "raise", irq, status: drive.status_reg, reason: drive.sector_count_reg & 7 });
    };
    cpu.device_lower_irq = irq => events.push({ event: "lower", irq });
    const control = value => cpu.io.port_write8(0x376, value);
    const command = value => cpu.io.port_write8(0x177, value);

    control(0xA);
    events.length = 0;
    channel.push_irq();
    assert.equal(events.length, 0, "nIEN must suppress INTRQ");
    assert.equal(channel.irq_pending, true);
    control(0x8);
    assert.equal(events.at(-1).event, "raise", "clearing nIEN must expose pending INTRQ");
    assert.equal(events.at(-1).irq, 15);

    cpu.io.port_read8(0x376);
    assert.equal(channel.irq_pending, true, "alternate status must not acknowledge INTRQ");
    cpu.io.port_read8(0x177);
    assert.equal(channel.irq_pending, false, "regular status acknowledges INTRQ");
    const raises = events.filter(e => e.event === "raise").length;
    control(0xA);
    control(0x8);
    assert.equal(events.filter(e => e.event === "raise").length, raises,
        "acknowledged INTRQ must not reappear when nIEN changes");

    events.length = 0;
    command(0xA1); // IDENTIFY PACKET DEVICE completes synchronously.
    assert.deepEqual(events.map(e => e.event), ["lower", "raise"],
        "command issue must clear old INTRQ before generating its own INTRQ");
    cpu.io.port_read8(0x177);

    events.length = 0;
    command(0xA0);
    // TEST UNIT READY is a twelve-byte all-zero CDB, with no data phase.
    for(let i = 0; i < 3; i++) cpu.io.port_write32(0x170, 0);
    const completion = events.find(e => e.event === "raise");
    assert.ok(completion, "TEST UNIT READY must generate completion INTRQ");
    assert.equal(completion.reason, 3, "CoD/IO must describe command completion before INTRQ");
    assert.equal(completion.status & 0x88, 0, "BSY/DRQ must be clear before completion INTRQ");

    const snapshot = value => {
        if(value && typeof value.get_state === "function") return snapshot(value.get_state());
        if(Array.isArray(value)) return value.map(snapshot);
        return value;
    };
    const saved = snapshot(channel);
    channel.irq_pending = false;
    channel.set_state(saved);
    assert.equal(channel.irq_pending, true, "pending INTRQ must survive save/restore");
    const legacy = saved.slice(0, 13);
    channel.set_state(legacy);
    assert.equal(channel.irq_pending, false, "legacy snapshots have no pending-INTRQ field");
    control(0xE);
    assert.equal(channel.irq_pending, false, "software reset clears pending INTRQ");

    console.log("IDE interrupts: masking, acknowledgement, completion ordering and snapshots passed");
    emulator.destroy();
});
