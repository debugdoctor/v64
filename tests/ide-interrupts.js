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
    assert.equal(events.at(-1).event, "raise",
        "clearing nIEN must assert the still-pending INTRQ (level signal)");
    assert.equal(events.at(-1).irq, 15);
    cpu.io.port_read8(0x177);
    assert.equal(channel.irq_pending, false, "status read acknowledges INTRQ");

    cpu.io.port_read8(0x376);
    assert.equal(channel.irq_pending, false, "alternate status must not create INTRQ");

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

    // GET EVENT STATUS NOTIFICATION (MMC-3 6.13): the media-change query the
    // Linux cdrom layer depends on. It must not fail with CHECK CONDITION, and
    // the descriptor has to satisfy sr_get_events' validation.
    const packet = cdb => {
        command(0xA0);
        for(let i = 0; i < 3; i++)
        {
            cpu.io.port_write32(0x170, cdb[4 * i] | cdb[4 * i + 1] << 8 | cdb[4 * i + 2] << 16 | cdb[4 * i + 3] << 24);
        }
    };
    const response = () => {
        const bytes = [];
        for(let i = 0; i < 2; i++)
        {
            const word = cpu.io.port_read32(0x170) >>> 0;
            bytes.push(word & 0xFF, word >>> 8 & 0xFF, word >>> 16 & 0xFF, word >>> 24 & 0xFF);
        }
        return bytes;
    };
    control(0x8);
    events.length = 0;
    packet([0x4A, 1, 0, 0, 1 << 4, 0, 0, 0, 8, 0, 0, 0]);
    assert.equal(events.find(e => e.event === "raise").status & 0x41, 0x40,
        "GET EVENT STATUS NOTIFICATION must complete, not report CHECK CONDITION");
    const descriptor = response();
    assert.equal(descriptor[0] << 8 | descriptor[1], 4, "event header data_len is 4, big endian");
    assert.equal(descriptor[2] & 7, 4, "notification class must be media");
    assert.equal(descriptor[2] & 0x80, 0, "NEA must be clear: an event is reported");
    assert.equal(descriptor[3], 0x10, "supported event classes must announce media events");
    assert.equal(descriptor[4], 0, "media event code must be no-change");
    assert.equal(descriptor[5] & 2, 2, "media must be reported present");
    cpu.io.port_read8(0x177);

    events.length = 0;
    packet([0x4A, 0, 0, 0, 1 << 4, 0, 0, 0, 8, 0, 0, 0]);
    assert.equal(events.find(e => e.event === "raise").status & 0x41, 0x41,
        "non-polled GET EVENT STATUS NOTIFICATION is not supported");
    cpu.io.port_read8(0x177);

    // A single ATAPI data command must raise one interrupt when its data is
    // ready and one on completion. A third, duplicate assertion is delivered
    // after libata has already completed the command, which it reports as
    // "lost interrupt" before freezing the port.
    control(0x8);
    cpu.io.port_write8(0x171, 0);      // no DMA
    cpu.io.port_write8(0x174, 0xFE);   // byte count limit
    cpu.io.port_write8(0x175, 0xFF);
    events.length = 0;
    packet([0x28, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]);  // READ(10), one sector
    assert.equal(events.filter(e => e.event === "raise").length, 1,
        "data-ready must raise exactly one interrupt");
    for(let i = 0; i < 2048 / 4; i++) cpu.io.port_read32(0x170);
    assert.equal(events.filter(e => e.event === "raise").length, 2,
        "completion must add exactly one interrupt");
    cpu.io.port_read8(0x177);

    console.log("IDE interrupts: masking, acknowledgement, completion ordering and snapshots passed");
    emulator.destroy();
});
