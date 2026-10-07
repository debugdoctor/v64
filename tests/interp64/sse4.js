#!/usr/bin/env node

// Long-mode interpreter: SSSE3/SSE4.1/SSE4.2/AES-NI/PCLMULQDQ semantics.
//
// Drives interp64 one instruction at a time and compares the XMM register file
// against values computed here, including a full AES-128 encryption/decryption
// against the FIPS-197 Appendix B vector.
//
// Run with: `node tests/interp64/sse4.js`

import { v64 } from "../../src/main.js";

const BASE = 0x1000;
const XMM = 832;
const FLAGS = 120;

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
    const xmm = new Uint8Array(ex.memory.buffer, XMM, 128);
    const flags = new Int32Array(ex.memory.buffer, FLAGS, 1);

    const failures = [];
    let active = "";
    const note = m => failures.push(active + ": " + m);
    const expect = (got, want, m) => {
        if(got !== want) note(m + " [want " + want + ", got " + got + "]");
    };
    const expect_xmm = (r, want, m) => {
        for(let i = 0; i < 16; i++)
        {
            if(xmm[r * 16 + i] !== want[i])
            {
                note(m + " byte " + i + " [want " + want[i] + ", got " + xmm[r * 16 + i] + "]");
                return;
            }
        }
    };
    const set_xmm = (r, bytes) => { for(let i = 0; i < 16; i++) xmm[r * 16 + i] = bytes[i]; };
    const get_xmm = r => Array.from(xmm.slice(r * 16, r * 16 + 16));
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

    const sse38 = (op, dst, src) => [0x66, 0x0F, 0x38, op, 0xC0 | (dst << 3) | src];
    const sse3a = (op, dst, src, imm) => [0x66, 0x0F, 0x3A, op, 0xC0 | (dst << 3) | src, imm];
    const hex = s => s.match(/../g).map(x => parseInt(x, 16));

    // ---- PSHUFB ----
    active = "pshufb";
    set_xmm(0, hex("00112233445566778899aabbccddeeff"));
    set_xmm(1, hex("0f0e0d0c0b0a09080706050403020100"));
    run(sse38(0x00, 0, 1));
    expect_xmm(0, hex("ffeeddccbbaa99887766554433221100"), "pshufb reversed");

    // ---- PMOVSXBW / PMOVSXBD / PMOVSXWD ----
    active = "pmovsxbw";
    set_xmm(1, hex("80ff010280fe7f800000000000000000"));
    set_xmm(0, new Array(16).fill(0xCC));
    run(sse38(0x20, 0, 1));
    expect_xmm(0, hex("80ffffff0100020080fffeff7f0080ff"), "sign extend bytes");

    active = "pmovzxbd";
    set_xmm(1, hex("80ff0102000000000000000000000000"));
    set_xmm(0, new Array(16).fill(0xCC));
    run(sse38(0x31, 0, 1));
    expect_xmm(0, hex("80000000ff0000000100000002000000"), "zero extend bytes");

    // ---- PMINSD / PMINUD / PMAXSD / PMAXUD / PMINSB / PMINUW ----
    active = "pminsd";
    set_xmm(0, hex("00000080ffffffff0100000002000000"));
    set_xmm(1, hex("0100000000000000ffffffff03000000"));
    run(sse38(0x39, 0, 1));
    expect_xmm(0, hex("00000080ffffffffffffffff02000000"), "signed min dwords");

    active = "pminud";
    set_xmm(0, hex("00000080ffffffff0100000002000000"));
    set_xmm(1, hex("0100000000000000ffffffff03000000"));
    run(sse38(0x3B, 0, 1));
    expect_xmm(0, hex("01000000000000000100000002000000"), "unsigned min dwords");

    active = "pmaxsb";
    set_xmm(0, hex("80ff01027f8000000000000000000000"));
    set_xmm(1, hex("00000000000000000000000000000000"));
    run(sse38(0x3C, 0, 1));
    expect_xmm(0, hex("000001027f0000000000000000000000"), "signed max bytes");

    // ---- PMULLD / PMULDQ ----
    active = "pmulld";
    set_xmm(0, hex("01000000020000000300000004000000"));
    set_xmm(1, hex("05000000060000000700000008000000"));
    run(sse38(0x40, 0, 1));
    expect_xmm(0, hex("050000000c0000001500000020000000"), "pmulld");

    active = "pmuldq";
    set_xmm(0, hex("0100000000000000ffffffff00000000"));
    set_xmm(1, hex("02000000000000000300000000000000"));
    run(sse38(0x28, 0, 1));
    expect_xmm(0, hex("0200000000000000fdffffffffffffff"), "pmuldq even lanes");

    // ---- PCMPEQQ / PCMPGTQ ----
    active = "pcmpeqq";
    set_xmm(0, hex("01020304050607080000000000000000"));
    set_xmm(1, hex("0102030405060708ffffffffffffffff"));
    run(sse38(0x29, 0, 1));
    expect_xmm(0, hex("ffffffffffffffff0000000000000000"), "pcmpeqq");

    active = "pcmpgtq";
    set_xmm(0, hex("0100000000000000ffffffffffffffff"));
    set_xmm(1, hex("00000000000000000100000000000000"));
    run(sse38(0x37, 0, 1));
    expect_xmm(0, hex("ffffffffffffffff0000000000000000"), "pcmpgtq signed");

    // ---- PACKUSDW ----
    active = "packusdw";
    set_xmm(0, hex("00010000ffffffff0000010005000000"));
    set_xmm(1, hex("02000000000000000300000004000000"));
    run(sse38(0x2B, 0, 1));
    expect_xmm(0, hex("00010000ffff05000200000003000400"), "packusdw");

    // ---- PHMINPOSUW ----
    active = "phminposuw";
    set_xmm(1, hex("05000300010007000200040006000800"));
    run(sse38(0x41, 0, 1));
    expect_xmm(0, hex("01000200000000000000000000000000"), "min value/index");

    // ---- PTEST flags ----
    active = "ptest";
    set_xmm(0, hex("0f000000000000000000000000000000"));
    set_xmm(1, hex("f0000000000000000000000000000000"));
    flags[0] = 0;
    run(sse38(0x17, 0, 1));
    expect(flags[0] & 0x40, 0x40, "ZF set when AND is zero");
    expect(flags[0] & 0x01, 0, "CF clear when ANDN is nonzero");

    // ---- PBLENDVB (mask in XMM0) ----
    active = "pblendvb";
    set_xmm(0, hex("80017f02ff037f0480057f06ff077f08"));
    set_xmm(1, hex("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
    set_xmm(2, hex("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"));
    run(sse38(0x10, 2, 1));
    expect_xmm(2, hex("aabbbbbbaabbbbbbaabbbbbbaabbbbbb"), "blend by byte sign bit");

    // ---- ROUNDPS ----
    active = "roundps";
    set_xmm(1, hex("0000c03f000020400000c0bf000020c0")); // 1.5, 2.5, -1.5, -2.5
    set_xmm(0, new Array(16).fill(0));
    run(sse3a(0x08, 0, 1, 0x07)); // truncate (imm bit 2 selects the imm mode)
    expect_xmm(0, hex("0000803f00000040000080bf000000c0"), "roundps truncate");

    // ---- BLENDPS / PBLENDW ----
    active = "blendps";
    set_xmm(0, hex("00000000111111112222222233333333"));
    set_xmm(1, hex("aaaaaaaa bbbbbbbb cccccccc dddddddd".replace(/ /g, "")));
    run(sse3a(0x0C, 0, 1, 0b0101));
    expect_xmm(0, hex("aaaaaaaa11111111cccccccc33333333"), "blendps imm");

    // ---- PEXTRB / PINSRB ----
    active = "pextrb/pinsrb";
    set_xmm(1, hex("aabbccddeeff00112233445566778899"));
    set_reg64(0, 0);
    run([0x66, 0x0F, 0x3A, 0x14, 0xC8, 3]); // pextrb eax, xmm1, 3
    expect(u32[16] & 0xFF, 0xdd, "pextrb byte 3");
    set_xmm(2, new Array(16).fill(0));
    set_reg64(3, 0x5A);
    run([0x66, 0x0F, 0x3A, 0x20, 0xD3, 5]); // pinsrb xmm2, ebx, 5
    expect(xmm[2 * 16 + 5], 0x5A, "pinsrb byte 5");

    // ---- EXTRACTPS / INSERTPS ----
    active = "extractps/insertps";
    set_xmm(1, hex("11223344556677889900aabbccddeeff"));
    set_reg64(0, 0);
    run([0x66, 0x0F, 0x3A, 0x17, 0xC8, 2]);
    expect(u32[16], 0xbbaa0099, "extractps lane 2");
    set_xmm(2, hex("00000000111111112222222233333333"));
    set_xmm(3, hex("44444444555555556666666677777777"));
    run(sse3a(0x21, 2, 3, (2 << 6) | (0 << 4) | 0b0100)); // src lane 2 -> dst lane 0, zero lane 2
    expect_xmm(2, hex("66666666111111110000000033333333"), "insertps");

    // ---- DPPS ----
    active = "dpps";
    set_xmm(0, hex("0000803f000000400000404000008040")); // 1,2,3,4
    set_xmm(1, hex("0000803f0000803f0000803f0000803f")); // 1,1,1,1
    run(sse3a(0x40, 0, 1, 0xFF));
    expect_xmm(0, hex("00002041000020410000204100002041"), "dot product 10.0");

    // ---- PCLMULQDQ ----
    active = "pclmulqdq";
    set_xmm(0, hex("03000000000000000000000000000000"));
    set_xmm(1, hex("03000000000000000000000000000000"));
    run(sse3a(0x44, 0, 1, 0x00));
    expect_xmm(0, hex("05000000000000000000000000000000"), "3 clmul 3 = 5");

    // ---- CRC32 (CRC-32C known vector) ----
    active = "crc32";
    set_xmm(0, new Array(16).fill(0));
    // "123456789" in memory at BASE+0x100
    const digits = "123456789".split("").map(c => c.charCodeAt(0));
    for(let i = 0; i < digits.length; i++) ex.write8(BASE + 0x100 + i, digits[i]);
    set_reg64(0, 0xFFFFFFFF);
    set_reg64(3, BASE + 0x100);
    // crc32 eax, dword [rbx] (F2 0F 38 F1 /r) x4? process byte by byte via F0
    let code = [];
    for(let i = 0; i < 9; i++) code = code.concat([0xF2, 0x0F, 0x38, 0xF0, 0x03]); // crc32 eax, byte [rbx]
    code = code.concat([0x48, 0xFF, 0xC3]); // inc rbx (not needed but harmless)
    // re-run with rbx incrementing is complex; use direct address per byte instead.
    set_reg64(0, 0xFFFFFFFF);
    code = [];
    for(let i = 0; i < 9; i++)
    {
        set_reg64(3, BASE + 0x100 + i);
        // encode crc32 eax, byte [rbx]: F2 0F 38 F0 /r with rm=rbx (3), mod=00
        code.push([0xF2, 0x0F, 0x38, 0xF0, 0x03]);
    }
    set_reg64(0, 0xFFFFFFFF);
    for(let i = 0; i < 9; i++)
    {
        set_reg64(3, BASE + 0x100 + i);
        run([0xF2, 0x0F, 0x38, 0xF0, 0x03]);
    }
    expect(u32[16], 0x1CF96D7C, "crc32c raw");

    // ---- AES-128 FIPS-197 Appendix B ----
    const RK = [
        "000102030405060708090a0b0c0d0e0f",
        "d6aa74fdd2af72fadaa678f1d6ab76fe",
        "b692cf0b643dbdf1be9bc5006830b3fe",
        "b6ff744ed2c2c9bf6c590cbf0469bf41",
        "47f7f7bc95353e03f96c32bcfd058dfd",
        "3caaa3e8a99f9deb50f3af57adf622aa",
        "5e390f7df7a69296a7553dc10aa31f6b",
        "14f9701ae35fe28c440adf4d4ea9c026",
        "47438735a41c65b9e016baf4aebf7ad2",
        "549932d1f08557681093ed9cbe2c974e",
        "13111d7fe3944a17f307a78b4d2b30c5",
    ].map(hex);
    const plaintext = hex("00112233445566778899aabbccddeeff");
    const ciphertext = hex("69c4e0d86a7b0430d8cdb78070b4c55a");

    active = "aesenc";
    let state = plaintext.map((b, i) => b ^ RK[0][i]);
    set_xmm(0, state);
    for(let r = 1; r <= 9; r++)
    {
        set_xmm(1, RK[r]);
        run(sse38(0xDC, 0, 1)); // aesenc xmm0, xmm1
    }
    set_xmm(1, RK[10]);
    run(sse38(0xDD, 0, 1)); // aesenclast
    expect_xmm(0, ciphertext, "aes-128 encryption");

    active = "aesdec";
    state = ciphertext.map((b, i) => b ^ RK[10][i]);
    set_xmm(0, state);
    for(let r = 9; r >= 1; r--)
    {
        set_xmm(1, RK[r]);
        run(sse38(0xDB, 2, 1)); // aesimc xmm2, xmm1
        run(sse38(0xDE, 0, 2)); // aesdec xmm0, xmm2
    }
    set_xmm(1, RK[0]);
    run(sse38(0xDF, 0, 1)); // aesdeclast
    expect_xmm(0, plaintext, "aes-128 decryption");

    // ---- AESKEYGENASSIST ----
    //
    // Derived from the SDM definition, byte by byte. The previous golden value
    // was taken from whatever the implementation happened to produce, so it
    // agreed with a broken implementation: note that the low dword still
    // matches and only the RotWord/RCON halves differ, which is exactly the
    // part that was wrong. A stronger check would run the standard AES-NI key
    // expansion and compare against RK, but PSHUFD is not reachable from this
    // harness.
    active = "aeskeygenassist";
    set_xmm(1, RK[0]);
    run(sse3a(0xDF, 0, 1, 0x01));
    // SDM: DEST[31:0]=SubWord(X1); DEST[63:32]=SubWord(RotWord(X1)) XOR RCON;
    //      DEST[95:64]=SubWord(X3); DEST[127:96]=SubWord(RotWord(X3)) XOR RCON,
    // with X1 = SRC[63:32] and X3 = SRC[127:96].
    expect_xmm(0, hex("f26b6fc5" + "6a6fc5f2" + "fed7ab76" + "d6ab76fe"), "aeskeygenassist rcon=1");

    // ---- PCMPISTRI: equal each ----
    active = "pcmpistri equal each";
    set_xmm(1, hex("61626364000000000000000000000000")); // "abcd"
    set_xmm(2, hex("61626365000000000000000000000000")); // "abce"
    set_reg64(1, 0xFFFFFFFF);
    // pcmpistri xmm1, xmm2, equal each + negative polarity (imm = 0b11000)
    run(sse3a(0x63, 1, 2, 0b11000));
    expect(u32[16 + 1], 3, "index of first difference");

    if(failures.length)
    {
        for(const f of failures.slice(0, 40)) console.log("FAIL " + f);
        console.log("interp64 sse4: " + failures.length + " failure(s)");
        process.exit(1);
    }
    console.log("interp64 sse4: all tests passed");
    process.exit(0);
});
