#!/usr/bin/env node

// Direct 64-bit boot: the initrd and the entry state.
//
// The loader must place the initrd, publish ramdisk_image/ramdisk_size and
// code32_start in boot_params, and enter the kernel with the 64-bit selectors,
// paging on and a usable stack. The kernel records what it sees; the test
// reads it back.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/boot64/initrd.js`

import { v64 } from "../../src/main.js";

const BOOT_PARAMS = 0x10000;
const INITRD_ADDRESS = 64 << 20;
const KERNEL_ADDRESS = 0x100000;

const emulator = new v64({
    autostart: false,
    memory_size: 96 * 1024 * 1024,
    disable_jit: 1,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;

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

    // The kernel records ramdisk_image/size, code32_start, CR0, CR3 and RSP at
    // 0xE000. The selectors are read from the host (mov Sreg is a separate gap).
    const kernel_code = [
        0x48, 0x89, 0xF3,                                           // mov rbx, rsi
        0x8B, 0x83, 0x18, 0x02, 0x00, 0x00,                         // mov eax, [rbx+0x218]
        0x48, 0x89, 0x04, 0x25, 0x00, 0xE0, 0x00, 0x00,             // mov [0xE000], rax
        0x8B, 0x83, 0x1C, 0x02, 0x00, 0x00,                         // mov eax, [rbx+0x21C]
        0x48, 0x89, 0x04, 0x25, 0x08, 0xE0, 0x00, 0x00,             // mov [0xE008], rax
        0x8B, 0x83, 0x14, 0x02, 0x00, 0x00,                         // mov eax, [rbx+0x214]
        0x48, 0x89, 0x04, 0x25, 0x10, 0xE0, 0x00, 0x00,             // mov [0xE010], rax
        0x0F, 0x20, 0xC0,                                           // mov rax, cr0
        0x48, 0x89, 0x04, 0x25, 0x20, 0xE0, 0x00, 0x00,             // mov [0xE020], rax
        0x0F, 0x20, 0xD8,                                           // mov rax, cr3
        0x48, 0x89, 0x04, 0x25, 0x28, 0xE0, 0x00, 0x00,             // mov [0xE028], rax
        0x48, 0x89, 0x24, 0x25, 0x30, 0xE0, 0x00, 0x00,             // mov [0xE030], rsp
        0xF4,                                                       // hlt
    ];

    const setup_sects = 4;
    const prot_start = (setup_sects + 1) * 512;
    const size = (prot_start + 0x200 + kernel_code.length + 3) & ~3;
    const bzimage = new Uint8Array(size);
    bzimage[0x1F1] = setup_sects;
    bzimage[0x1FE] = 0x55;
    bzimage[0x1FF] = 0xAA;
    bzimage[0x201] = 0x40;
    bzimage.set([0x48, 0x64, 0x72, 0x53], 0x202); // "HdrS"
    bzimage[0x206] = 0x0C;
    bzimage[0x207] = 0x02;
    bzimage[0x238] = 0xFF;
    bzimage.set(kernel_code, prot_start + 0x200);

    const initrd = new Uint8Array(0x1234);
    for(let i = 0; i < initrd.length; i++) initrd[i] = (i * 7 + 3) & 0xFF;

    cpu.boot_kernel64(bzimage.buffer, initrd.buffer, "console=ttyS0");

    let guard = 0;
    while(!cpu.in_hlt[0] && guard++ < 1000) ex.main_loop();

    expect(cpu.in_hlt[0], 1, "the kernel halted");
    expect(read32(BOOT_PARAMS + 0x218), INITRD_ADDRESS, "hdr.ramdisk_image");
    expect(read32(BOOT_PARAMS + 0x21C), initrd.length, "hdr.ramdisk_size");
    expect(read32(BOOT_PARAMS + 0x214), KERNEL_ADDRESS, "hdr.code32_start");
    expect(cpu.sreg[1], 0x10, "entry CS");
    expect(cpu.sreg[2], 0x18, "entry SS");
    expect(read64(0xE020) & 0x8000_0000n, 0x8000_0000n, "CR0.PG");
    expect(read64(0xE028), 0x1000n, "CR3 points at the PML4");
    expect(read64(0xE030) !== 0n, true, "RSP is set");

    let initrdOk = true;
    for(let i = 0; i < initrd.length; i++)
    {
        if(read8(INITRD_ADDRESS + i) !== initrd[i])
        {
            initrdOk = false;
            break;
        }
    }
    expect(initrdOk, true, "the initrd bytes are in guest memory");

    if(failures.length)
    {
        for(const failure of failures) console.log("FAIL " + failure);
        console.log("boot64 initrd: " + failures.length + " failure(s)");
        process.exit(1);
    }
    console.log("boot64 initrd: all tests passed");
    process.exit(0);
});
