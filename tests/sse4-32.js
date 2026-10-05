#!/usr/bin/env node

// 32-bit protected mode: SSSE3/SSE4.1/SSE4.2/AES reach the shared simd_instr
// layer through the 32-bit interpreter (and the 32-bit JIT calls the same
// functions). Boots a flat 32-bit payload from the reset vector and checks the
// results in RAM.
//
// Run with: `node tests/sse4-32.js`

import { v64 } from "../src/main.js";

const RESET = 0xFFFF0;
const PROT = 0x2000;
const GDT = 0x1400;
const FLAGS_OUT = 0x3000;
const MUL_OUT = 0x3010;
const CRC_OUT = 0x3020;
const IDX_OUT = 0x3030;

const imm32 = v => [v & 0xFF, v >> 8 & 0xFF, v >> 16 & 0xFF, v >> 24 & 0xFF];
const mov_eax = v => [0xB8, ...imm32(v)];
const mov_ebx = v => [0xBB, ...imm32(v)];
const movd_xmm_eax = r => [0x66, 0x0F, 0x6E, 0xC0 | (r << 3)]; // movd xmm_r, eax
const movd_eax_xmm = r => [0x66, 0x0F, 0x7E, 0xC0 | (r << 3)]; // movd eax, xmm_r
const store_eax = addr => [0xA3, ...imm32(addr)];

const GDT_BYTES = [
    0, 0, 0, 0, 0, 0, 0, 0,
    0xFF, 0xFF, 0, 0, 0, 0x9A, 0xCF, 0,
    0xFF, 0xFF, 0, 0, 0, 0x92, 0xCF, 0,
];
const REAL_STUB = [
    0xFA,
    0x66, 0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, CR0.PE
    0x0F, 0x22, 0xC0,                   // mov cr0, eax
    0x66, 0xEA, ...imm32(PROT), 0x08, 0x00, // jmp far 0008:2000
];
const PROT_PREFIX = [
    0x66, 0xB8, 0x10, 0x00, // mov ax, 0x10
    0x8E, 0xD8,             // mov ds, ax
    0x8E, 0xC0,             // mov es, ax
    0x8E, 0xD0,             // mov ss, ax
    0xBC, 0x00, 0x80, 0x00, 0x00, // mov esp, 0x8000
];

const PAYLOAD = [
    // xmm0 = 0x0f0f0f0f, xmm1 = 0xf0f0f0f0
    ...mov_eax(0x0f0f0f0f), ...movd_xmm_eax(0),
    ...mov_eax(0xF0F0F0F0), ...movd_xmm_eax(1),
    // ptest xmm0, xmm1
    0x66, 0x0F, 0x38, 0x17, 0xC1,
    0x9C, 0x58, // pushf; pop eax
    ...store_eax(FLAGS_OUT),
    // pmulld xmm0, xmm1
    0x66, 0x0F, 0x38, 0x40, 0xC1,
    ...movd_eax_xmm(0),
    ...store_eax(MUL_OUT),
    // crc32 eax, bl
    ...mov_eax(0xFFFFFFFF),
    ...mov_ebx(0x31), // '1'
    0xF2, 0x0F, 0x38, 0xF0, 0xC3,
    ...store_eax(CRC_OUT),
    // xmm0 = "abcd", xmm1 = "abce"; pcmpistri xmm0, xmm1, equal each + negative
    ...mov_eax(0x64636261), ...movd_xmm_eax(0),
    ...mov_eax(0x65636261), ...movd_xmm_eax(1),
    0x66, 0x0F, 0x3A, 0x63, 0xC1, 0x18,
    0x89, 0xC8, // mov eax, ecx
    ...store_eax(IDX_OUT),
    0xF4, // hlt
];

const emulator = new v64({
    autostart: false,
    memory_size: 8 * 1024 * 1024,
    disable_jit: true,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const write = (address, bytes) => {
        for(let i = 0; i < bytes.length; i++) ex.write8(address + i, bytes[i]);
    };

    write(GDT, GDT_BYTES);
    cpu.gdtr_size[0] = GDT_BYTES.length - 1;
    cpu.gdtr_offset[0] = GDT;
    write(RESET, REAL_STUB);
    write(PROT, [...PROT_PREFIX, ...PAYLOAD]);

    let guard = 0;
    while(!cpu.in_hlt[0] && guard++ < 100000) ex.main_loop();
    if(!cpu.in_hlt[0]) throw new Error("did not halt");

    const failures = [];
    const expect = (got, want, m) => {
        if((got >>> 0) !== (want >>> 0)) failures.push(m + " [want " + (want >>> 0).toString(16) + ", got " + (got >>> 0).toString(16) + "]");
    };

    // ptest: AND == 0 -> ZF; ANDN != 0 -> CF clear
    const flags = ex.read32s(FLAGS_OUT);
    expect(flags & 0x40, 0x40, "ptest ZF");
    expect(flags & 0x01, 0, "ptest CF");
    // pmulld low lane
    const product = (BigInt(0x0f0f0f0f) * BigInt(0xF0F0F0F0)) & 0xFFFFFFFFn;
    expect(ex.read32s(MUL_OUT), Number(product), "pmulld");
    // crc32c raw of '1'
    expect(ex.read32s(CRC_OUT), 0x6F0A661C, "crc32");
    // pcmpistri equal each + negative -> index 3
    expect(ex.read32s(IDX_OUT), 3, "pcmpistri index");

    if(failures.length)
    {
        for(const f of failures) console.log("FAIL " + f);
        console.log("sse4-32: " + failures.length + " failure(s)");
        process.exit(1);
    }
    console.log("sse4-32: all tests passed");
    process.exit(0);
});
