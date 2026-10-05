#!/usr/bin/env node

// Long-mode interpreter: AVX (VEX-encoded) semantics, including 256-bit YMM
// state (loaded/stored through memory so the test can observe the upper half).
//
// Run with: `node tests/interp64/avx.js`

import { v64 } from "../../src/main.js";

const BASE = 0x1000;
const DATA = 0x20000;

const emulator = new v64({
    autostart: false,
    memory_size: 2 * 1024 * 1024,
    disable_jit: true,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const u32 = new Uint32Array(ex.memory.buffer);
    const ext = new BigUint64Array(ex.memory.buffer, 160, 8);
    // 64-bit RIP (global pointer at wasm offset 232), for `interp64_run`.
    const rip64 = new BigUint64Array(ex.memory.buffer, 232, 1);

    const failures = [];
    let active = "";
    const note = m => failures.push(active + ": " + m);
    const expect = (got, want, m) => {
        if(got !== want) note(m + " [want " + want + ", got " + got + "]");
    };
    const expect_bytes = (start, want, m) => {
        for(let i = 0; i < want.length; i++)
        {
            if(ex.read8(start + i) !== want[i])
            {
                note(m + " byte " + i + " [want " + want[i] + ", got " + ex.read8(start + i) + "]");
                return;
            }
        }
    };
    const set_reg64 = (i, raw) => {
        const v = BigInt(raw);
        if(i < 8)
        {
            u32[16 + i] = Number(v & 0xFFFF_FFFFn);
            u32[32 + i] = Number(v >> 32n & 0xFFFF_FFFFn);
        }
        else ext[i - 8] = v;
    };

    const run = code => {
        for(let i = 0; i < code.length; i++) ex.write8(BASE + i, code[i]);
        ex.write8(BASE + code.length, 0xF4);
        cpu.instruction_pointer[0] = BASE;
        cpu.in_hlt[0] = 0;
        cpu.is_32[0] = 1;
        cpu.cr[0] = 0;
        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 100) ex.interp64_run_one();
        if(!cpu.in_hlt[0]) note("did not halt");
    };
    // Same, but through the decode cache (`interp64_run`): the block is decoded
    // by jit64 into Instrs and the cached AVX entry is executed in place.
    const run_cached = code => {
        for(let i = 0; i < code.length; i++) ex.write8(BASE + i, code[i]);
        ex.write8(BASE + code.length, 0xF4);
        cpu.instruction_pointer[0] = BASE;
        rip64[0] = BigInt(BASE);
        cpu.in_hlt[0] = 0;
        cpu.is_32[0] = 1;
        cpu.cr[0] = 0;
        ex.interp64_run(100);
        if(!cpu.in_hlt[0]) note("did not halt (cached)");
    };

    // VEX 2-byte prefix: R (0 = no extend), vvvv, L, pp.
    const vex2 = (r_ext, vvvv, l, pp) =>
        [0xC5, (r_ext ? 0 : 0x80) | ((~vvvv & 0xF) << 3) | (l ? 4 : 0) | pp];
    const vex3 = (map, vvvv, l, pp, w) =>
        [0xC4, 0xE0 | map, (w ? 0x80 : 0) | ((~vvvv & 0xF) << 3) | (l ? 4 : 0) | pp];

    const f32 = v => {
        const b = new ArrayBuffer(4);
        new DataView(b).setFloat32(0, v, true);
        return Array.from(new Uint8Array(b));
    };
    const f32_at = (addr, v) => { for(const b of f32(v)) ex.write8(addr++, b); };

    // ---- VADDPS xmm (128-bit) ----
    active = "vaddps xmm";
    // xmm1 = [1,2,3,4], xmm2 = [10,20,30,40]
    for(let i = 0; i < 4; i++) f32_at(DATA + i * 4, i + 1);
    for(let i = 0; i < 4; i++) f32_at(DATA + 16 + i * 4, (i + 1) * 10);
    set_reg64(0, DATA);
    run([...vex2(0, 0, 0, 0), 0x28, 0x08]);
    set_reg64(0, DATA + 16);
    run([...vex2(0, 0, 0, 0), 0x28, 0x10]); // vmovaps xmm2, [rax]
    run([...vex2(0, 1, 0, 0), 0x58, 0xC2]); // vaddps xmm0, xmm1, xmm2
    set_reg64(0, DATA);
    run([...vex2(0, 0, 0, 0), 0x29, 0x00]); // vmovaps [rax], xmm0
    const sum = [11, 22, 33, 44];
    for(let i = 0; i < 4; i++) expect(ex.read32s(DATA + i * 4), new DataView(new Uint8Array(f32(sum[i])).buffer).getInt32(0, true), "lane " + i);

    // ---- VADDPS ymm (256-bit) through memory ----
    active = "vaddps ymm";
    for(let i = 0; i < 8; i++) f32_at(DATA + 0x100 + i * 4, i + 1);
    for(let i = 0; i < 8; i++) f32_at(DATA + 0x200 + i * 4, (i + 1) * 2);
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 0, 1, 0), 0x28, 0x08]); // vmovaps ymm1, [rax]
    set_reg64(0, DATA + 0x200);
    run([...vex2(0, 0, 1, 0), 0x28, 0x10]); // vmovaps ymm2, [rax]
    run([...vex2(0, 1, 1, 0), 0x58, 0xC2]); // vaddps ymm0, ymm1, ymm2
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 0), 0x29, 0x00]); // vmovaps [rax], ymm0
    for(let i = 0; i < 8; i++)
    {
        const want = (i + 1) + (i + 1) * 2;
        expect(ex.read32s(DATA + 0x300 + i * 4), new DataView(new Uint8Array(f32(want)).buffer).getInt32(0, true), "lane " + i);
    }

    // ---- VXORPS ymm ----
    active = "vxorps ymm";
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 0, 1, 0), 0x28, 0x08]);
    run([...vex2(0, 0, 1, 0), 0x28, 0x10]);
    run([...vex2(0, 1, 1, 0), 0x57, 0xC2]); // vxorps ymm0, ymm1, ymm2
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 0), 0x29, 0x00]);
    expect_bytes(DATA + 0x300, new Array(32).fill(0), "xor self");

    // ---- VPADDD ymm (AVX2) ----
    active = "vpaddd ymm";
    for(let i = 0; i < 8; i++) ex.write32(DATA + 0x100 + i * 4, 0x01010101);
    for(let i = 0; i < 8; i++) ex.write32(DATA + 0x200 + i * 4, 0x02020202);
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 0, 1, 1), 0x6F, 0x08]); // vmovdqa ymm1, [rax]
    set_reg64(0, DATA + 0x200);
    run([...vex2(0, 0, 1, 1), 0x6F, 0x10]);
    run([...vex2(0, 1, 1, 1), 0xFE, 0xC2]); // vpaddd ymm0, ymm1, ymm2
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 1), 0x7F, 0x00]);
    for(let i = 0; i < 8; i++) expect(ex.read32s(DATA + 0x300 + i * 4) >>> 0, 0x03030303, "dword " + i);

    // ---- VPSHUFD ----
    active = "vpshufd";
    for(let i = 0; i < 4; i++) ex.write32(DATA + 0x100 + i * 4, i + 1);
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 0, 0, 1), 0x6F, 0x08]); // vmovdqa xmm1, [rax]
    run([...vex2(0, 1, 0, 1), 0x70, 0xC1, 0b00_11_10_01]); // vpshufd xmm0, xmm1, 0x39
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 0, 1), 0x7F, 0x00]);
    expect(ex.read32s(DATA + 0x300 + 0) >>> 0, 2, "shuf lane 0");
    expect(ex.read32s(DATA + 0x300 + 4) >>> 0, 3, "shuf lane 1");
    expect(ex.read32s(DATA + 0x300 + 8) >>> 0, 4, "shuf lane 2");
    expect(ex.read32s(DATA + 0x300 + 12) >>> 0, 1, "shuf lane 3");

    // ---- VZEROALL ----
    active = "vzeroall";
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 0, 1, 0), 0x28, 0x08]); // load ymm1
    run([0xC5, 0xFC, 0x77]); // vzeroall
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 0), 0x29, 0x08]); // vmovaps [rax], ymm1
    expect_bytes(DATA + 0x300, new Array(32).fill(0), "vzeroall");

    // ---- VPERMD (AVX2) ----
    active = "vpermd";
    for(let i = 0; i < 8; i++) ex.write32(DATA + 0x100 + i * 4, i);       // src values
    for(let i = 0; i < 8; i++) ex.write32(DATA + 0x200 + i * 4, 7 - i);   // indices
    set_reg64(0, DATA + 0x200);
    run([...vex2(0, 0, 1, 1), 0x6F, 0x08]); // ymm1 = indices
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 0, 1, 1), 0x6F, 0x10]); // ymm2 = values
    run([...vex3(2, 1, 1, 1, false), 0x36, 0xC2]); // vpermd ymm0, ymm1, ymm2
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 1), 0x7F, 0x00]);
    for(let i = 0; i < 8; i++) expect(ex.read32s(DATA + 0x300 + i * 4) >>> 0, 7 - i, "permd " + i);

    // ---- VPBROADCASTD (AVX2) ----
    active = "vpbroadcastd";
    ex.write32(DATA + 0x100, 0xDEADBEEF);
    set_reg64(0, DATA + 0x100);
    run([...vex3(2, 0, 1, 1, false), 0x58, 0x00]); // vpbroadcastd ymm0, [rax]
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 1), 0x7F, 0x00]);
    for(let i = 0; i < 8; i++) expect(ex.read32s(DATA + 0x300 + i * 4) >>> 0, 0xDEADBEEF, "bcast " + i);

    // ---- VINSERTI128 (AVX2) ----
    active = "vinserti128";
    for(let i = 0; i < 8; i++) ex.write32(DATA + 0x100 + i * 4, 0x11111111 + i);
    ex.write32(DATA + 0x200, 0x22222222);
    ex.write32(DATA + 0x204, 0x33333333);
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 0, 1, 1), 0x6F, 0x08]); // ymm1
    set_reg64(0, DATA + 0x200);
    run([...vex2(0, 0, 0, 1), 0x6F, 0x10]); // xmm2
    run([...vex3(3, 1, 1, 1, false), 0x38, 0xC2, 0x01]); // vinserti128 ymm0, ymm1, xmm2, 1
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 1), 0x7F, 0x00]);
    expect(ex.read32s(DATA + 0x300) >>> 0, 0x11111111, "insert lo");
    expect(ex.read32s(DATA + 0x310) >>> 0, 0x22222222, "insert hi0");
    expect(ex.read32s(DATA + 0x314) >>> 0, 0x33333333, "insert hi1");

    // ---- VFMADD213PS (FMA) ----
    active = "vfmadd213ps";
    for(let i = 0; i < 8; i++) f32_at(DATA + 0x100 + i * 4, 2);
    for(let i = 0; i < 8; i++) f32_at(DATA + 0x200 + i * 4, 3);
    for(let i = 0; i < 8; i++) f32_at(DATA + 0x400 + i * 4, 1);
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 0, 1, 0), 0x28, 0x00]); // ymm0 = 2
    set_reg64(0, DATA + 0x200);
    run([...vex2(0, 0, 1, 0), 0x28, 0x10]); // ymm2 = 3 (rm)
    set_reg64(0, DATA + 0x400);
    run([...vex2(0, 0, 1, 0), 0x28, 0x18]); // ymm3 = 1 (vvvv)
    // vfmadd213ps ymm0, ymm3, ymm2 -> ymm0 = ymm3*ymm0 + ymm2 = 5
    run([...vex3(2, 3, 1, 1, false), 0xA8, 0xC2]);
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 0), 0x29, 0x00]);
    for(let i = 0; i < 8; i++)
    {
        const want = new DataView(new Uint8Array(f32(5)).buffer).getInt32(0, true);
        expect(ex.read32s(DATA + 0x300 + i * 4), want, "fma " + i);
    }

    // ---- VPERMILPS (imm) ----
    active = "vpermilps imm";
    for(let i = 0; i < 4; i++) f32_at(DATA + 0x100 + i * 4, i);
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 0, 0, 0), 0x28, 0x08]); // xmm1 = [0,1,2,3]
    run([...vex3(3, 0, 0, 1, false), 0x04, 0xC1, 0x39]); // vpermilps xmm0, xmm1, 0x39
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 0, 0), 0x29, 0x00]);
    for(let i = 0; i < 4; i++)
    {
        const want = new DataView(new Uint8Array(f32([1, 2, 3, 0][i])).buffer).getInt32(0, true);
        expect(ex.read32s(DATA + 0x300 + i * 4), want, "permil " + i);
    }

    // ---- VINSERTPS ----
    active = "vinsertps";
    for(let i = 0; i < 4; i++) ex.write32(DATA + 0x100 + i * 4, i === 0 ? 0 : (0x11111111 * i));
    for(let i = 0; i < 4; i++) ex.write32(DATA + 0x200 + i * 4, 0x44444444 * (i + 1));
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 0, 0, 0), 0x28, 0x08]); // xmm1 = base
    set_reg64(0, DATA + 0x200);
    run([...vex2(0, 0, 0, 0), 0x28, 0x10]); // xmm2 = source
    // vinsertps xmm0, xmm1, xmm2, src lane 2 -> dst lane 0, zero lane 2
    run([...vex3(3, 1, 0, 1, false), 0x21, 0xC2, (2 << 6) | (0 << 4) | 0b0100]);
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 0, 0), 0x29, 0x00]);
    expect(ex.read32s(DATA + 0x300) >>> 0, 0xCCCCCCCC, "insertps lane0");
    expect(ex.read32s(DATA + 0x300 + 4) >>> 0, 0x11111111, "insertps lane1");
    expect(ex.read32s(DATA + 0x300 + 8) >>> 0, 0, "insertps lane2");

    // ---- VGATHERDPS (AVX2) ----
    active = "vgatherdps";
    for(let i = 0; i < 8; i++) ex.write32(DATA + i * 4, 1000 + i);      // data
    for(let i = 0; i < 8; i++) ex.write32(DATA + 0x100 + i * 4, i);     // indices
    for(let i = 0; i < 8; i++) ex.write32(DATA + 0x200 + i * 4, 0x80000000); // mask
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 0, 1, 1), 0x6F, 0x08]); // ymm1 = indices
    set_reg64(0, DATA + 0x200);
    run([...vex2(0, 0, 1, 1), 0x6F, 0x10]); // ymm2 = mask
    set_reg64(0, DATA);
    // vgatherdps ymm0, [rax + ymm1*4], ymm2
    run([0xC4, 0xE2, 0x6D, 0x92, 0x04, 0x88]);
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 0), 0x29, 0x00]);
    for(let i = 0; i < 8; i++) expect(ex.read32s(DATA + 0x300 + i * 4) >>> 0, 1000 + i, "gather " + i);

    // ---- VPSLLVD (AVX2) ----
    active = "vpsllvd";
    for(let i = 0; i < 8; i++) ex.write32(DATA + 0x100 + i * 4, 1 << i); // values
    for(let i = 0; i < 8; i++) ex.write32(DATA + 0x200 + i * 4, 1);      // counts
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 0, 1, 1), 0x6F, 0x08]); // ymm1
    set_reg64(0, DATA + 0x200);
    run([...vex2(0, 0, 1, 1), 0x6F, 0x10]); // ymm2
    run([...vex3(2, 1, 1, 1, false), 0x47, 0xC2]); // vpsllvd ymm0, ymm1, ymm2
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 1), 0x7F, 0x00]);
    for(let i = 0; i < 8; i++) expect(ex.read32s(DATA + 0x300 + i * 4) >>> 0, 1 << (i + 1), "vpsllvd " + i);

    // ---- VPERMILPS (variable) ----
    active = "vpermilps variable";
    for(let i = 0; i < 8; i++) ex.write32(DATA + 0x100 + i * 4, i);
    for(let i = 0; i < 8; i++) ex.write32(DATA + 0x200 + i * 4, 7 - i);
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 0, 1, 1), 0x6F, 0x08]); // ymm1 = data
    set_reg64(0, DATA + 0x200);
    run([...vex2(0, 0, 1, 1), 0x6F, 0x10]); // ymm2 = control
    run([...vex3(2, 1, 1, 1, false), 0x0C, 0xC2]); // vpermilps ymm0, ymm1, ymm2
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 1), 0x7F, 0x00]);
    for(let i = 0; i < 8; i++) expect(ex.read32s(DATA + 0x300 + i * 4) >>> 0, (i & 4) + 3 - (i & 3), "vpermil " + i);

    // ---- VMOVDQU ymm + VPCMPEQB/VPMOVMSKB (the glibc AVX2 strlen shape) ----
    active = "vmovdqu strlen";
    for(let i = 0; i < 32; i++) ex.write8(DATA + 0x100 + i, 0x41);
    ex.write8(DATA + 0x102, 0); // null at index 2
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 7, 1, 1), 0xEF, 0xFF]); // vpxor ymm7, ymm7, ymm7
    run([...vex2(0, 0, 1, 2), 0x6F, 0x00]); // vmovdqu ymm0, [rax]
    run([...vex2(0, 7, 1, 1), 0x74, 0xC0]); // vpcmpeqb ymm0, ymm7, ymm0
    run([...vex2(0, 0, 1, 1), 0xD7, 0xC0]); // vpmovmskb eax, ymm0
    expect(u32[16] >>> 0, 1 << 2, "strlen mask");

    // 128-bit VMOVDQU must load, not store.
    active = "vmovdqu xmm";
    for(let i = 0; i < 16; i++) ex.write8(DATA + 0x200 + i, 0x10 + i);
    set_reg64(0, DATA + 0x200);
    run([...vex2(0, 0, 0, 2), 0x6F, 0x00]); // vmovdqu xmm0, [rax]
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 0, 2), 0x7F, 0x00]); // vmovdqu [rax], xmm0
    for(let i = 0; i < 16; i++) expect(ex.read8(DATA + 0x300 + i), 0x10 + i, "movdqu byte " + i);

    // ---- BMI1/BMI2 ----
    // Intel form -> dst=ModRM.reg, src=rm, other=VEX.vvvv (or the reverse).
    active = "bmi";
    set_reg64(3, 0x80000000); set_reg64(1, 1);            // ebx, ecx
    run([...vex3(2, 1, 0, 3, false), 0xF7, 0xC3]);        // shrx eax, ebx, ecx
    expect(u32[16] >>> 0, 0x40000000, "shrx");
    set_reg64(3, 1); set_reg64(1, 4);
    run([...vex3(2, 1, 0, 1, false), 0xF7, 0xC3]);        // shlx eax, ebx, ecx
    expect(u32[16] >>> 0, 16, "shlx");
    set_reg64(3, 0x80000000); set_reg64(1, 4);
    run([...vex3(2, 1, 0, 2, false), 0xF7, 0xC3]);        // sarx eax, ebx, ecx
    expect(u32[16] >>> 0, 0xF8000000, "sarx");
    set_reg64(3, 0x0F0F); set_reg64(1, 0x00FF);
    run([...vex3(2, 3, 0, 0, false), 0xF2, 0xC1]);        // andn eax, ebx, ecx
    expect(u32[16] >>> 0, 0x00F0, "andn");
    set_reg64(3, 0x12345678); set_reg64(1, (8 << 8) | 4);
    run([...vex3(2, 1, 0, 0, false), 0xF7, 0xC3]);        // bextr eax, ebx, ecx
    expect(u32[16] >>> 0, 0x67, "bextr");
    set_reg64(3, 0xFFFFFFFF); set_reg64(1, 8);
    run([...vex3(2, 1, 0, 0, false), 0xF5, 0xC3]);        // bzhi eax, ebx, ecx
    expect(u32[16] >>> 0, 0xFF, "bzhi");
    set_reg64(3, 0b11); set_reg64(1, 0b1010);
    run([...vex3(2, 3, 0, 3, false), 0xF5, 0xC1]);        // pdep eax, ebx, ecx
    expect(u32[16] >>> 0, 0b1010, "pdep");
    set_reg64(3, 0b1010); set_reg64(1, 0b1010);
    run([...vex3(2, 3, 0, 2, false), 0xF5, 0xC1]);        // pext eax, ebx, ecx
    expect(u32[16] >>> 0, 0b11, "pext");
    set_reg64(3, 0b1100);
    run([...vex3(2, 0, 0, 0, false), 0xF3, 0xCB]);        // blsr eax, ebx
    expect(u32[16] >>> 0, 0b1000, "blsr");
    set_reg64(3, 0x100);
    run([0xF3, 0x0F, 0xBC, 0xC3]);                        // tzcnt eax, ebx
    expect(u32[16] >>> 0, 8, "tzcnt");
    run([0xF3, 0x0F, 0xBD, 0xC3]);                        // lzcnt eax, ebx
    expect(u32[16] >>> 0, 23, "lzcnt");
    set_reg64(3, 0x12345678);
    run([...vex3(3, 0, 0, 3, false), 0xF0, 0xC3, 8]);     // rorx eax, ebx, 8
    expect(u32[16] >>> 0, 0x78123456, "rorx");
    set_reg64(2, 0x10000); set_reg64(3, 0x10000);
    run([...vex3(2, 0, 0, 3, false), 0xF6, 0xCB]);        // mulx ecx, eax, ebx
    expect(u32[16] >>> 0, 0, "mulx lo");
    expect(u32[16 + 1] >>> 0, 1, "mulx hi");

    // ---- VPMINUB + VPCMPEQB/VPMOVMSKB over 32 bytes (glibc strcpy shape) ----
    active = "vpminub";
    for(let i = 0; i < 32; i++) ex.write8(DATA + 0x100 + i, i);
    for(let i = 0; i < 32; i++) ex.write8(DATA + 0x200 + i, 31 - i);
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 0, 1, 1), 0x6F, 0x08]);
    set_reg64(0, DATA + 0x200);
    run([...vex2(0, 0, 1, 1), 0x6F, 0x10]);
    run([...vex2(0, 1, 1, 1), 0xDA, 0xC2]); // vpminub ymm0, ymm1, ymm2
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 1), 0x7F, 0x00]);
    for(let i = 0; i < 32; i++) expect(ex.read8(DATA + 0x300 + i), Math.min(i, 31 - i), "minub " + i);

    // ---- AVX2 high half + unaligned 256-bit load/store ----
    active = "avx2 high/unaligned";
    for(let i = 0; i < 32; i++) ex.write8(DATA + 0x100 + i, 0x41);
    ex.write8(DATA + 0x114, 0); // null at index 20 (high half)
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 7, 1, 1), 0xEF, 0xFF]);
    run([...vex2(0, 0, 1, 2), 0x6F, 0x00]);
    run([...vex2(0, 7, 1, 1), 0x74, 0xC0]);
    run([...vex2(0, 0, 1, 1), 0xD7, 0xC0]);
    expect(u32[16] >>> 0, 1 << 20, "strlen high half");
    for(let i = 0; i < 40; i++) ex.write8(DATA + 0x200 + i, i + 1);
    set_reg64(0, DATA + 0x201);
    run([...vex2(0, 0, 1, 2), 0x6F, 0x00]); // vmovdqu ymm0, [rax+1]
    set_reg64(0, DATA + 0x301);
    run([...vex2(0, 0, 1, 2), 0x7F, 0x00]); // vmovdqu [rax+1], ymm0
    for(let i = 0; i < 32; i++) expect(ex.read8(DATA + 0x301 + i), i + 2, "unaligned " + i);

    // ---- VMOVD / VMOVQ (VEX 6E/7E/D6), used by glibc's AVX2 memchr/strcpy ----
    active = "vmovd/vmovq";
    set_reg64(6, 0xDEADBEEF);
    run([...vex2(0, 0, 0, 1), 0x6E, 0xC6]); // vmovd xmm0, esi
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 0, 1), 0x7E, 0x00]); // vmovd [rax], xmm0
    expect(ex.read32s(DATA + 0x300) >>> 0, 0xDEADBEEF, "vmovd");
    set_reg64(6, 0x1122334455667788n);
    run([...vex3(1, 0, 0, 1, true), 0x6E, 0xC6]); // vmovq xmm0, rsi (W1)
    set_reg64(0, DATA + 0x300);
    run([...vex3(1, 0, 0, 1, true), 0xD6, 0x00]); // vmovq [rax], xmm0
    for(let i = 0; i < 8; i++) expect(ex.read8(DATA + 0x300 + i), [0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11][i], "vmovq " + i);

    // ---- VPBROADCASTB (used by glibc's AVX2 strlen) ----
    active = "vpbroadcastb";
    ex.write8(DATA + 0x100, 0xAB);
    set_reg64(0, DATA + 0x100);
    run([...vex3(2, 0, 1, 1, false), 0x78, 0x00]); // vpbroadcastb ymm0, [rax]
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 1), 0x7F, 0x00]);
    for(let i = 0; i < 32; i++) expect(ex.read8(DATA + 0x300 + i), 0xAB, "bcastb " + i);

    // ---- VEX SIB index extension (the X bit) ----
    active = "vex sib index";
    for(let i = 0; i < 16; i++) ex.write8(DATA + 0x400 + i, 0x30 + i);
    set_reg64(0, DATA + 0x400); // rax
    set_reg64(9, 0);            // r9 (extended index)
    run([0xC4, 0xA1, 0x7A, 0x6F, 0x04, 0x08]); // vmovdqu xmm0, [rax + r9]
    set_reg64(0, DATA + 0x500);
    run([0xC4, 0xA1, 0x7A, 0x7F, 0x04, 0x08]); // vmovdqu [rax + r9], xmm0
    for(let i = 0; i < 16; i++) expect(ex.read8(DATA + 0x500 + i), 0x30 + i, "vex x bit " + i);

    // ---- high registers as VEX.vvvv (glibc uses ymm8-15) ----
    active = "high vvvv";
    for(let i = 0; i < 32; i++) ex.write8(DATA + 0x200 + i, 0x11);
    for(const src of [8, 10, 15])
    {
        set_reg64(0, DATA + 0x200);
        run([...vex2(src >= 8 ? 1 : 0, 0, 1, 2), 0x6F, (src & 7) << 3]); // ymm[src] = [rax]
        set_reg64(0, DATA + 0x200);
        run([...vex2(0, 0, 1, 2), 0x6F, 0x00]); // ymm0 = 0x11
        run([...vex2(0, src, 1, 1), 0x74, 0xC0]); // vpcmpeqb ymm0, ymm[src], ymm0
        run([...vex2(0, 0, 1, 1), 0xD7, 0xC0]); // vpmovmskb eax, ymm0
        expect(u32[16] >>> 0, 0xFFFFFFFF, "high vvvv " + src);
    }

    // ---- all 16 YMM registers: high half round-trip ----
    active = "ymm all regs";
    for(let r = 0; r < 16; r++)
    {
        for(let i = 0; i < 32; i++) ex.write8(DATA + 0x100 + i, (i < 16 ? 0x80 : 0x00) ^ r);
        set_reg64(0, DATA + 0x100);
        run([...vex2(r >= 8 ? 1 : 0, 0, 1, 2), 0x6F, (r & 7) << 3]); // vmovdqu ymm[r], [rax]
        set_reg64(0, DATA + 0x300);
        run([...vex2(r >= 8 ? 1 : 0, 0, 1, 2), 0x7F, (r & 7) << 3]); // vmovdqu [rax], ymm[r]
        for(let i = 0; i < 32; i++)
        {
            expect(ex.read8(DATA + 0x300 + i), (i < 16 ? 0x80 : 0x00) ^ r, "ymm reg " + r + " byte " + i);
        }
        run([0xC4, (r >= 8 ? 0xC1 : 0xE1), 0x7D, 0xD7, 0xC0 | (r & 7)]); // vpmovmskb eax, ymm[r]
        expect(u32[16] >>> 0, 0xFFFF, "ymm reg movmsk " + r);
    }

    // ---- TZCNT/LZCNT edge cases (used by glibc's AVX2 paths) ----
    active = "tzcnt/lzcnt";
    set_reg64(3, 0);
    run([0xF3, 0x0F, 0xBC, 0xC3]);
    expect(u32[16] >>> 0, 32, "tzcnt 0");
    set_reg64(3, 0x80000000);
    run([0xF3, 0x0F, 0xBC, 0xC3]);
    expect(u32[16] >>> 0, 31, "tzcnt 31");
    set_reg64(3, 0);
    run([0xF3, 0x0F, 0xBD, 0xC3]);
    expect(u32[16] >>> 0, 32, "lzcnt 0");
    set_reg64(3, 1);
    run([0xF3, 0x0F, 0xBD, 0xC3]);
    expect(u32[16] >>> 0, 31, "lzcnt 31");

    // ---- vpcmpeqb with a VEX.B-extended memory base ([r8]) ----
    active = "vex mem base";
    for(let i = 0; i < 32; i++) ex.write8(DATA + 0x100 + i, i === 5 ? 0 : 0x41);
    set_reg64(8, DATA + 0x100);
    run([...vex2(0, 7, 1, 1), 0xEF, 0xFF]); // vpxor ymm7, ymm7, ymm7
    run([0xC4, 0xC1, 0x45, 0x74, 0x00]); // vpcmpeqb ymm0, ymm7, [r8]
    run([...vex2(0, 0, 1, 1), 0xD7, 0xC0]); // vpmovmskb eax, ymm0
    expect(u32[16] >>> 0, 1 << 5, "vex mem base");

    // ---- VPMOVMSKB with a high source register (rm extended by VEX.B) ----
    active = "vpmovmskb high";
    for(let i = 0; i < 32; i++) ex.write8(DATA + 0x100 + i, i < 16 ? 0x80 : 0x00);
    set_reg64(0, DATA + 0x100);
    run([...vex2(1, 0, 1, 2), 0x6F, 0x00]); // vmovdqu ymm8, [rax]
    run([0xC4, 0xC1, 0x7D, 0xD7, 0xC0]); // vpmovmskb eax, ymm8
    expect(u32[16] >>> 0, 0xFFFF, "vpmovmskb ymm8");

    // ---- VPANDN / VPOR / VPAND (used by glibc's AVX2 string functions) ----
    active = "vpandn/vpor/vpand";
    for(let i = 0; i < 32; i++) { ex.write8(DATA + 0x100 + i, 0x0F); ex.write8(DATA + 0x200 + i, 0x33); }
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 0, 1, 1), 0x6F, 0x08]); // ymm1 = 0x0F
    set_reg64(0, DATA + 0x200);
    run([...vex2(0, 0, 1, 1), 0x6F, 0x10]); // ymm2 = 0x33
    run([...vex2(0, 1, 1, 1), 0xDF, 0xC2]); // vpandn ymm0, ymm1, ymm2
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 1), 0x7F, 0x00]);
    for(let i = 0; i < 32; i++) expect(ex.read8(DATA + 0x300 + i), ~0x0F & 0x33, "vpandn " + i);
    run([...vex2(0, 1, 1, 1), 0xEB, 0xC2]); // vpor ymm0, ymm1, ymm2
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 1), 0x7F, 0x00]);
    for(let i = 0; i < 32; i++) expect(ex.read8(DATA + 0x300 + i), 0x0F | 0x33, "vpor " + i);
    run([...vex2(0, 1, 1, 1), 0xDB, 0xC2]); // vpand ymm0, ymm1, ymm2
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 1), 0x7F, 0x00]);
    for(let i = 0; i < 32; i++) expect(ex.read8(DATA + 0x300 + i), 0x0F & 0x33, "vpand " + i);

    // ---- randomized AVX2 null-finding (reference in JS) ----
    active = "avx2 random null";
    const rng = (() => { let s = 987654321; return () => (s = (s * 1103515245 + 12345) >>> 0) / 0x100000000; })();
    for(let t = 0; t < 100; t++)
    {
        const buf = [];
        for(let i = 0; i < 32; i++) buf.push(Math.floor(rng() * 256));
        buf[Math.floor(rng() * 32)] = 0;
        for(let i = 0; i < 32; i++) ex.write8(DATA + 0x100 + i, buf[i]);
        set_reg64(0, DATA + 0x100);
        run([...vex2(0, 0, 1, 2), 0x6F, 0x00]); // vmovdqu ymm0, [rax]
        run([...vex2(0, 7, 1, 1), 0xEF, 0xFF]); // vpxor ymm7, ymm7, ymm7
        run([...vex2(0, 7, 1, 1), 0x74, 0xC0]); // vpcmpeqb ymm0, ymm7, ymm0
        run([...vex2(0, 0, 1, 1), 0xD7, 0xC0]); // vpmovmskb eax, ymm0
        let ref = 0;
        for(let i = 0; i < 32; i++) if(buf[i] === 0) ref |= 1 << i;
        expect(u32[16] >>> 0, ref >>> 0, "random null " + t);
    }

    // ---- high registers (xmm8-15) ----
    active = "high regs";
    for(let i = 0; i < 32; i++) ex.write8(DATA + 0x100 + i, 0x20 + i);
    set_reg64(0, DATA + 0x100);
    run([...vex2(1, 0, 1, 2), 0x6F, 0x00]); // vmovdqu ymm8, [rax]
    set_reg64(0, DATA + 0x300);
    run([...vex2(1, 0, 1, 2), 0x7F, 0x00]); // vmovdqu [rax], ymm8
    for(let i = 0; i < 32; i++) expect(ex.read8(DATA + 0x300 + i), 0x20 + i, "high reg " + i);

    // ---- XSAVE/XRSTOR round-trip of the YMM state ----
    active = "xsave ymm";
    for(let i = 0; i < 32; i++) ex.write8(DATA + 0x100 + i, 0x11 + i);
    for(let i = 0; i < 32; i++) ex.write8(DATA + 0x200 + i, 0x99);
    set_reg64(0, DATA + 0x100);
    run([...vex2(0, 0, 1, 2), 0x6F, 0x00]); // ymm0 = [0x11..]
    set_reg64(0, 7); set_reg64(2, 0); set_reg64(1, DATA + 0x1000);
    run([0x0F, 0xAE, 0x21]); // xsave [rcx]
    set_reg64(0, DATA + 0x200);
    run([...vex2(0, 0, 1, 2), 0x6F, 0x00]); // ymm0 = [0x99..]
    set_reg64(0, 7); set_reg64(2, 0); set_reg64(1, DATA + 0x1000);
    run([0x0F, 0xAE, 0x29]); // xrstor [rcx]
    set_reg64(0, DATA + 0x300);
    run([...vex2(0, 0, 1, 2), 0x7F, 0x00]);
    for(let i = 0; i < 32; i++) expect(ex.read8(DATA + 0x300 + i), 0x11 + i, "xsave ymm " + i);

    // ---- Decode cache (jit64 decoder + shared AVX execution) ----
    active = "decode cache";
    for(let i = 0; i < 8; i++) ex.write32(DATA + 0x100 + i * 4, 0x01010101);
    for(let i = 0; i < 8; i++) ex.write32(DATA + 0x200 + i * 4, 0x02020202);
    set_reg64(0, DATA + 0x100);
    run_cached([...vex2(0, 0, 1, 1), 0x6F, 0x08]); // vmovdqa ymm1, [rax]
    set_reg64(0, DATA + 0x200);
    run_cached([...vex2(0, 0, 1, 1), 0x6F, 0x10]); // vmovdqa ymm2, [rax]
    run_cached([...vex2(0, 1, 1, 1), 0xFE, 0xC2]); // vpaddd ymm0, ymm1, ymm2
    set_reg64(0, DATA + 0x300);
    run_cached([...vex2(0, 0, 1, 1), 0x7F, 0x00]); // vmovdqa [rax], ymm0
    for(let i = 0; i < 8; i++) expect(ex.read32s(DATA + 0x300 + i * 4) >>> 0, 0x03030303, "cached dword " + i);

    if(failures.length)
    {
        for(const f of failures.slice(0, 30)) console.log("FAIL " + f);
        console.log("interp64 avx: " + failures.length + " failure(s)");
        process.exit(1);
    }
    console.log("interp64 avx: all tests passed");
    process.exit(0);
});
