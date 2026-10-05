#!/usr/bin/env node

// 32-bit protected mode: VEX-encoded AVX reaches the shared simd_instr layer
// through the 32-bit interpreter. Boots a flat 32-bit payload from the reset
// vector and checks the results in RAM.
//
// Run with: `node tests/avx-32.js`

import { v64 } from "../src/main.js";

const RESET = 0xFFFF0;
const PROT = 0x2000;
const GDT = 0x1400;
const ADD_OUT = 0x3000;
const INT_OUT = 0x3010;
const XOR_OUT = 0x3020;
const CVT_OUT = 0x3030;
const CMP_OUT = 0x3040;
const FMA_OUT = 0x3050;
const PERM_OUT = 0x3060;

const imm32 = v => [v & 0xFF, v >> 8 & 0xFF, v >> 16 & 0xFF, v >> 24 & 0xFF];
const mov_eax = v => [0xB8, ...imm32(v)];
const movd_xmm_eax = r => [0x66, 0x0F, 0x6E, 0xC0 | (r << 3)];
const movd_eax_xmm = r => [0x66, 0x0F, 0x7E, 0xC0 | (r << 3)];
const store_eax = addr => [0xA3, ...imm32(addr)];
const vex2 = (vvvv, l, pp) => [0xC5, 0x80 | ((~vvvv & 0xF) << 3) | (l ? 4 : 0) | pp];
const vex3 = (map, vvvv, l, pp, w) => [0xC4, 0xE0 | map, (w ? 0x80 : 0) | ((~vvvv & 0xF) << 3) | (l ? 4 : 0) | pp];

const GDT_BYTES = [
    0, 0, 0, 0, 0, 0, 0, 0,
    0xFF, 0xFF, 0, 0, 0, 0x9A, 0xCF, 0,
    0xFF, 0xFF, 0, 0, 0, 0x92, 0xCF, 0,
];
const REAL_STUB = [
    0xFA,
    0x66, 0xB8, 0x01, 0x00, 0x00, 0x00,
    0x0F, 0x22, 0xC0,
    0x66, 0xEA, ...imm32(PROT), 0x08, 0x00,
];
const PROT_PREFIX = [
    0x66, 0xB8, 0x10, 0x00,
    0x8E, 0xD8,
    0x8E, 0xC0,
    0x8E, 0xD0,
    0xBC, 0x00, 0x80, 0x00, 0x00,
];

const PAYLOAD = [
    // xmm0 = 1.0f, xmm1 = 2.0f
    ...mov_eax(0x3F800000), ...movd_xmm_eax(0),
    ...mov_eax(0x40000000), ...movd_xmm_eax(1),
    // vaddps xmm0, xmm0, xmm1
    ...vex2(0, 0, 0), 0x58, 0xC1,
    ...movd_eax_xmm(0), ...store_eax(ADD_OUT),
    // reload xmm0 = 1 and xmm1 = 2 (integers), then vpaddd xmm0, xmm0, xmm1
    ...mov_eax(1), ...movd_xmm_eax(0),
    ...mov_eax(2), ...movd_xmm_eax(1),
    ...vex2(0, 0, 1), 0xFE, 0xC1,
    ...movd_eax_xmm(0), ...store_eax(INT_OUT),
    // vxorps xmm0, xmm0, xmm0
    ...vex2(0, 0, 0), 0x57, 0xC0,
    ...movd_eax_xmm(0), ...store_eax(XOR_OUT),
    // xmm1 = 2 (integer); vcvtss2sd xmm0, xmm1 -> 2.0 double (low qword)
    ...mov_eax(2), ...movd_xmm_eax(1),
    ...vex2(0, 0, 0), 0x5B, 0xC1, // vcvtdq2ps xmm0, xmm1
    ...movd_eax_xmm(0), ...store_eax(CVT_OUT),
    // xmm0 = 1.0f, xmm1 = 2.0f; vucomiss xmm0, xmm1 -> CF set
    ...mov_eax(0x3F800000), ...movd_xmm_eax(0),
    ...mov_eax(0x40000000), ...movd_xmm_eax(1),
    ...vex2(0, 0, 0), 0x2E, 0xC1,
    0x9C, 0x58, // pushf; pop eax
    ...store_eax(CMP_OUT),
    // xmm1 = [1,2,3,4]; vpermilps xmm0, xmm1, 0x39 -> lane0 = xmm1[1] = 2
    ...mov_eax(1), 0x66, 0x0F, 0x3A, 0x22, 0xC8, 0, // pinsrd xmm1, eax, 0
    ...mov_eax(2), 0x66, 0x0F, 0x3A, 0x22, 0xC8, 1,
    ...mov_eax(3), 0x66, 0x0F, 0x3A, 0x22, 0xC8, 2,
    ...mov_eax(4), 0x66, 0x0F, 0x3A, 0x22, 0xC8, 3,
    ...vex3(3, 0, 0, 1, false), 0x04, 0xC1, 0x39, // vpermilps xmm0, xmm1, 0x39
    ...movd_eax_xmm(0), ...store_eax(PERM_OUT),
    // xmm0 = 2.0, xmm1 = 3.0, xmm2 = 1.0; vfmadd213ps xmm0, xmm1, xmm2 -> 7.0
    ...mov_eax(0x40000000), ...movd_xmm_eax(0),
    ...mov_eax(0x40400000), ...movd_xmm_eax(1),
    ...mov_eax(0x3F800000), ...movd_xmm_eax(2),
    ...vex3(2, 1, 0, 1, false), 0xA8, 0xC2,
    ...movd_eax_xmm(0), ...store_eax(FMA_OUT),
    // vzeroupper
    0xC5, 0xF8, 0x77,
    0xF4,
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
    expect(ex.read32s(ADD_OUT), 0x40400000, "vaddps"); // 3.0
    expect(ex.read32s(INT_OUT), 3, "vpaddd");
    expect(ex.read32s(XOR_OUT), 0, "vxorps");
    expect(ex.read32s(CVT_OUT), 0x40000000, "vcvtdq2ps"); // 2.0
    expect(ex.read32s(CMP_OUT) & 1, 1, "vucomiss CF"); // 1.0 < 2.0
    expect(ex.read32s(FMA_OUT), 0x40E00000, "vfmadd213ps"); // 7.0
    expect(ex.read32s(PERM_OUT), 2, "vpermilps");

    if(failures.length)
    {
        for(const f of failures) console.log("FAIL " + f);
        console.log("avx-32: " + failures.length + " failure(s)");
        process.exit(1);
    }
    console.log("avx-32: all tests passed");
    process.exit(0);
});
