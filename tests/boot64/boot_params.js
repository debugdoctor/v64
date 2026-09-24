#!/usr/bin/env node

// Direct 64-bit boot: the boot_params contract.
//
// A fake bzImage is started through the Linux x86 boot protocol. The loader
// must build boot_params (copied setup header, command line, memory map) and
// enter the kernel in long mode with RSI pointing at it. The field offsets are
// the ones in the kernel's struct boot_params, not values read from the loader.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/boot64/boot_params.js`

import { v64 } from "../../src/main.js";

const BOOT_PARAMS = 0x10000;
const CMDLINE = "console=ttyS0 root=/dev/ram0";

const emulator = new v64({
    autostart: false,
    memory_size: 4 * 1024 * 1024,
    disable_jit: 1,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const buffer = ex.memory.buffer;
    const u32 = new Uint32Array(buffer);

    const failures = [];
    const note = message => failures.push(message);
    const expect = (got, want, message) => {
        if(got !== want) note(message + " [want " + want + ", got " + got + "]");
    };

    const read8 = address => ex.read8(address) >>> 0;
    const read16 = address => read8(address) | read8(address + 1) << 8;
    const read32 = address => (read8(address) | read8(address + 1) << 8 | read8(address + 2) << 16 | read8(address + 3) << 24) >>> 0;
    const read64 = address => {
        let value = 0n;
        for(let i = 0; i < 8; i++) value |= BigInt(read8(address + i)) << BigInt(i * 8);
        return value;
    };
    const readString = (address, length) => {
        let text = "";
        for(let i = 0; i < length; i++)
        {
            const byte = read8(address + i);
            if(byte === 0) break;
            text += String.fromCharCode(byte);
        }
        return text;
    };
    const reg64 = i =>
        (i < 8)
            ? (BigInt(u32[32 + i]) << 32n) | BigInt(u32[16 + i])
            : new BigUint64Array(buffer, 160, 8)[i - 8];

    // Fake bzImage: 4 setup sectors, a header and a protected-mode kernel that
    // records RSI and halts.
    const setup_sects = 4;
    const prot_start = (setup_sects + 1) * 512;
    const kernel_code = [
        0x48, 0xB8, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // mov rax, 0x1234
        0x48, 0x89, 0x34, 0x25, 0x00, 0xE0, 0x00, 0x00,             // mov [0xE000], rsi
        0xF4,                                                       // hlt
    ];
    const size = (prot_start + 0x200 + kernel_code.length + 3) & ~3;
    const bzimage = new Uint8Array(size);
    bzimage[0x1F1] = setup_sects;
    bzimage[0x1FE] = 0x55;
    bzimage[0x1FF] = 0xAA; // boot flag
    bzimage[0x201] = 0x40;
    bzimage.set([0x48, 0x64, 0x72, 0x53], 0x202); // "HdrS"
    bzimage[0x206] = 0x0C;
    bzimage[0x207] = 0x02; // protocol 0x020c
    bzimage[0x238] = 0xFF; // cmdline_size
    bzimage.set(kernel_code, prot_start + 0x200);

    cpu.boot_kernel64(bzimage.buffer, undefined, CMDLINE);

    let guard = 0;
    while(!cpu.in_hlt[0] && guard++ < 1000)
    {
        ex.main_loop();
    }

    expect(cpu.in_hlt[0], 1, "the kernel halted");
    expect(reg64(0), 0x1234n, "the kernel ran in long mode");
    expect(read64(0xE000), BigInt(BOOT_PARAMS), "RSI = boot_params");

    // struct setup_header, embedded at boot_params + 0x1F1.
    expect(read16(BOOT_PARAMS + 0x1FE), 0xAA55, "hdr.boot_flag");
    expect(readString(BOOT_PARAMS + 0x202, 4), "HdrS", "hdr.header");
    expect(read16(BOOT_PARAMS + 0x206), 0x020C, "hdr.version");
    expect(read8(BOOT_PARAMS + 0x1F1), setup_sects, "hdr.setup_sects");
    expect(read32(BOOT_PARAMS + 0x238), 0xFF, "hdr.cmdline_size");

    // The command line.
    expect(read32(BOOT_PARAMS + 0x228), 0x80000, "hdr.cmd_line_ptr");
    expect(readString(0x80000, CMDLINE.length + 1), CMDLINE, "the command line was copied");

    // struct boot_params memory map.
    const entries = read8(BOOT_PARAMS + 0x1E8);
    expect(entries > 0, true, "e820_entries is set");
    expect(read64(BOOT_PARAMS + 0x2D0), 0n, "e820 RAM entry starts at 0");
    expect(read32(BOOT_PARAMS + 0x2D0 + 16), 1, "e820 RAM entry has type 1");

    if(failures.length)
    {
        for(const failure of failures)
        {
            console.log("FAIL " + failure);
        }
        console.log("boot64 boot_params: " + failures.length + " failure(s)");
        process.exit(1);
    }

    console.log("boot64 boot_params: all tests passed");
    process.exit(0);
});
