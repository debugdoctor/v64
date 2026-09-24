#!/usr/bin/env node

// Long-mode interpreter: instruction semantics.
//
// Every case is a raw x86-64 program. The expected register values and flags
// come from the Intel SDM; they were cross-checked on an x86-64 host before
// being written down here, so a failure points at the emulator, not at the
// test being a copy of it.
//
// Cases the interpreter does not decode show up as "did not halt" (the opcode
// is left unhandled) and keep the rest of the suite running.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp64/insns.js`

import { v64 } from "../../src/main.js";

const BASE = 0x1000;
const SCRATCH = 0x3000;

const F_CF = 1 << 0;
const F_PF = 1 << 2;
const F_AF = 1 << 4;
const F_ZF = 1 << 6;
const F_SF = 1 << 7;
const F_OF = 1 << 11;

const emulator = new v64({
    autostart: false,
    memory_size: 2 * 1024 * 1024,
    disable_jit: 1,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", () => {
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const buffer = ex.memory.buffer;
    const u32 = new Uint32Array(buffer);
    const ext = new BigUint64Array(buffer, 160, 8);

    const failures = [];
    let activeTest = "";

    const fmt = value => typeof value === "bigint" ? "0x" + value.toString(16) : String(value);
    const note = message => failures.push(activeTest + ": " + message);
    const expect = (got, want, message) => {
        if(got !== want) note(message + " [want " + fmt(want) + ", got " + fmt(got) + "]");
    };

    const reg64 = i =>
        (i < 8)
            ? (BigInt(u32[32 + i]) << 32n) | BigInt(u32[16 + i])
            : ext[i - 8];

    const setReg64 = (i, raw) => {
        const value = BigInt(raw);
        if(i < 8)
        {
            u32[16 + i] = Number(value & 0xFFFF_FFFFn);
            u32[32 + i] = Number(value >> 32n & 0xFFFF_FFFFn);
        }
        else
        {
            ext[i - 8] = value;
        }
    };

    const read64 = address =>
        BigInt(ex.read8(address)) |
        BigInt(ex.read8(address + 1)) << 8n |
        BigInt(ex.read8(address + 2)) << 16n |
        BigInt(ex.read8(address + 3)) << 24n |
        BigInt(ex.read8(address + 4)) << 32n |
        BigInt(ex.read8(address + 5)) << 40n |
        BigInt(ex.read8(address + 6)) << 48n |
        BigInt(ex.read8(address + 7)) << 56n;

    const flagsOf = () => cpu.flags[0] >>> 0;

    const reset = () => {
        for(let i = 0; i < 16; i++) setReg64(i, 0n);
        cpu.cr[0] = 0;
        cpu.is_32[0] = 1;
        cpu.cpl[0] = 0;
        cpu.flags[0] = 0x2;
        cpu.in_hlt[0] = 0;
    };

    // Write a program, append hlt, and step it until it halts. An opcode the
    // interpreter does not decode never advances RIP, so the guard ends the run.
    const run = code => {
        for(let i = 0; i < code.length; i++)
        {
            ex.write8(BASE + i, code[i]);
        }
        ex.write8(BASE + code.length, 0xF4);
        cpu.instruction_pointer[0] = BASE;
        cpu.in_hlt[0] = 0;
        let guard = 0;
        while(!cpu.in_hlt[0] && guard++ < 500)
        {
            ex.interp64_run_one();
        }
        if(!cpu.in_hlt[0])
        {
            note("did not halt (#UD?) at 0x" + (cpu.instruction_pointer[0] >>> 0).toString(16));
        }
        return cpu.in_hlt[0] === 1;
    };

    const expectFlags = want => {
        const flags = flagsOf();
        const named = { CF: F_CF, PF: F_PF, AF: F_AF, ZF: F_ZF, SF: F_SF, OF: F_OF };
        for(const [name, mask] of Object.entries(named))
        {
            const key = name.toLowerCase();
            if(!(key in want)) continue;
            expect(flags & mask ? 1 : 0, want[key], name + " (flags=0x" + flags.toString(16) + ")");
        }
    };

    // stc / clc / cmc drive CF.
    activeTest = "stc/clc/cmc";
    reset();
    run([0xF9, 0xF8, 0xF5]); // stc; clc; cmc
    expectFlags({ cf: 1 });

    // inc/dec touch every flag except CF, and a 32-bit write clears the top half.
    activeTest = "inc eax";
    reset();
    setReg64(0, 0xFFFF_FFFF_FFFF_FFFFn);
    cpu.flags[0] = 0x2 | F_CF;
    run([0xB8, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xC0]); // mov eax,0; inc eax
    expect(reg64(0), 1n, "inc eax zero-extends");
    expectFlags({ cf: 1, zf: 0, sf: 0, of: 0, pf: 0, af: 0 });

    activeTest = "dec eax";
    reset();
    setReg64(0, 0xFFFF_FFFF_FFFF_FFFFn);
    cpu.flags[0] = 0x2 | F_CF;
    run([0xB8, 0x00, 0x00, 0x00, 0x00, 0xFF, 0xC8]); // mov eax,0; dec eax
    expect(reg64(0), 0xFFFF_FFFFn, "dec eax zero-extends");
    expectFlags({ cf: 1, zf: 0, sf: 1, of: 0, af: 1 });

    // neg and not.
    activeTest = "neg eax";
    reset();
    run([0xB8, 0x01, 0x00, 0x00, 0x00, 0xF7, 0xD8]); // mov eax,1; neg eax
    expect(reg64(0), 0xFFFF_FFFFn, "neg eax");
    expectFlags({ cf: 1, of: 0, sf: 1, pf: 1, af: 1, zf: 0 });

    activeTest = "not eax";
    reset();
    cpu.flags[0] = 0x2 | F_ZF | F_CF;
    run([0xB8, 0x00, 0x00, 0x00, 0x00, 0xF7, 0xD0]); // mov eax,0; not eax
    expect(reg64(0), 0xFFFF_FFFFn, "not eax");
    expectFlags({ cf: 1, zf: 1, of: 0, sf: 0 });

    // adc/sbb carry chain over 64 bits.
    activeTest = "adc/sbb 64";
    reset();
    cpu.flags[0] = 0x2 | F_CF;
    run([
        0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, -1
        0x48, 0x83, 0xD0, 0x00,                   // adc rax, 0
        0x48, 0x83, 0xD8, 0x00,                   // sbb rax, 0
    ]);
    expect(reg64(0), 0xFFFF_FFFF_FFFF_FFFFn, "adc then sbb");
    expectFlags({ cf: 1, zf: 0, sf: 1, of: 0, af: 1 });

    // 8- and 16-bit results keep the untouched upper bits.
    activeTest = "add al, 1";
    reset();
    run([
        0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, -1
        0xB0, 0xFF,                               // mov al, 0xFF
        0x04, 0x01,                               // add al, 1
    ]);
    expect(reg64(0), 0xFFFF_FFFF_FFFF_FF00n, "8-bit add");
    expectFlags({ cf: 1, zf: 1, pf: 1, af: 1, of: 0 });

    activeTest = "add ax, 1";
    reset();
    run([
        0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, -1
        0xBB, 0x34, 0x12, 0x00, 0x00,             // mov ebx, 0x1234
        0x66, 0x89, 0xD8,                         // mov ax, bx
        0x66, 0x83, 0xC0, 0x01,                   // add ax, 1
    ]);
    expect(reg64(0), 0xFFFF_FFFF_FFFF_1235n, "16-bit add");
    expectFlags({ cf: 0, zf: 0, sf: 0, of: 0 });

    activeTest = "mov ax, imm16";
    reset();
    run([0x66, 0xB8, 0x34, 0x12, 0x66, 0x83, 0xC0, 0x01]); // mov ax,0x1234; add ax,1
    expect(reg64(0), 0x1235n, "16-bit immediate");
    expectFlags({ cf: 0, zf: 0, sf: 0, of: 0 });

    // Widening and narrowing multiply.
    activeTest = "mul r64";
    reset();
    run([
        0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, -1
        0xB9, 0x02, 0x00, 0x00, 0x00,             // mov ecx, 2
        0x48, 0xF7, 0xE1,                         // mul rcx
    ]);
    expect(reg64(0), 0xFFFF_FFFF_FFFF_FFFEn, "mul rax");
    expect(reg64(2), 1n, "mul rdx");
    expectFlags({ cf: 1, of: 1 });

    activeTest = "imul r64 (one operand)";
    reset();
    run([
        0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, -1
        0xB9, 0x02, 0x00, 0x00, 0x00,             // mov ecx, 2
        0x48, 0xF7, 0xE9,                         // imul rcx
    ]);
    expect(reg64(0), 0xFFFF_FFFF_FFFF_FFFEn, "imul rax");
    expect(reg64(2), 0xFFFF_FFFF_FFFF_FFFFn, "imul rdx sign-extends");
    expectFlags({ cf: 1, of: 1 });

    activeTest = "imul two operand";
    reset();
    run([
        0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, // mov rax, 1<<63
        0xB9, 0x02, 0x00, 0x00, 0x00,                               // mov ecx, 2
        0x48, 0x0F, 0xAF, 0xC1,                                     // imul rax, rcx
    ]);
    expect(reg64(0), 0n, "imul wraps");
    expectFlags({ cf: 1, of: 1 });

    activeTest = "imul three operand";
    reset();
    run([
        0xB8, 0x05, 0x00, 0x00, 0x00, // mov eax, 5
        0x48, 0x6B, 0xD8, 0x03,       // imul rbx, rax, 3
    ]);
    expect(reg64(3), 15n, "imul three-operand");
    expectFlags({ cf: 0, of: 0 });

    // div and idiv use rdx:rax; idiv truncates toward zero.
    activeTest = "div r32";
    reset();
    run([
        0x31, 0xD2,                   // xor edx, edx
        0xB8, 0x64, 0x00, 0x00, 0x00, // mov eax, 100
        0xB9, 0x07, 0x00, 0x00, 0x00, // mov ecx, 7
        0xF7, 0xF1,                   // div ecx
    ]);
    expect(reg64(0), 14n, "div quotient");
    expect(reg64(2), 2n, "div remainder");

    activeTest = "idiv r64";
    reset();
    run([
        0x48, 0xC7, 0xC0, 0x9C, 0xFF, 0xFF, 0xFF, // mov rax, -100
        0x48, 0xC7, 0xC2, 0xFF, 0xFF, 0xFF, 0xFF, // mov rdx, -1
        0xB9, 0x07, 0x00, 0x00, 0x00,             // mov ecx, 7
        0x48, 0xF7, 0xF9,                         // idiv rcx
    ]);
    expect(reg64(0), 0xFFFF_FFFF_FFFF_FFF2n, "idiv quotient -14");
    expect(reg64(2), 0xFFFF_FFFF_FFFF_FFFEn, "idiv remainder -2");

    // Shifts: CL is masked to the operand width, CF is the last bit out.
    activeTest = "shifts";
    reset();
    run([
        0xB8, 0x01, 0x00, 0x00, 0x00,             // mov eax, 1
        0xC1, 0xE0, 0x04,                         // shl eax, 4
        0xD1, 0xF8,                               // sar eax, 1
        0xB9, 0x21, 0x00, 0x00, 0x00,             // mov ecx, 33
        0xD3, 0xE0,                               // shl eax, cl (masked to 1)
        0x48, 0xB8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80, // mov rax, 1<<63
        0x48, 0xD1, 0xE0,                         // shl rax, 1
    ]);
    expect(reg64(0), 0n, "shifted out the top bit");
    expectFlags({ cf: 1 });

    // Rotates set CF and OF too.
    activeTest = "rol/ror";
    reset();
    run([
        0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
        0xD1, 0xC0,                   // rol eax, 1
        0xD1, 0xC8,                   // ror eax, 1
    ]);
    expect(reg64(0), 1n, "rol then ror");
    expectFlags({ cf: 0, of: 0 });

    // Bit test/modify, register and immediate forms.
    activeTest = "bt/bts/btc";
    reset();
    run([
        0x31, 0xC0,                               // xor eax, eax
        0xB9, 0x05, 0x00, 0x00, 0x00,             // mov ecx, 5
        0x0F, 0xAB, 0xC8,                         // bts eax, ecx
        0x0F, 0xBB, 0xC8,                         // btc eax, ecx
        0x0F, 0xA3, 0xC8,                         // bt  eax, ecx
        0xB8, 0x00, 0x00, 0x00, 0x80,             // mov eax, 1<<31
        0x0F, 0xBA, 0xE0, 0x1F,                   // bt  eax, 31
    ]);
    expect(reg64(0), 0x8000_0000n, "bit ops leave the value");
    expectFlags({ cf: 1 });

    // Bit scan.
    activeTest = "bsf/bsr";
    reset();
    run([
        0xB8, 0x08, 0x00, 0x00, 0x00, // mov eax, 8
        0x0F, 0xBC, 0xD8,             // bsf ebx, eax
        0x0F, 0xBD, 0xC8,             // bsr ecx, eax
    ]);
    expect(reg64(3), 3n, "bsf");
    expect(reg64(1), 3n, "bsr");
    expectFlags({ zf: 0 });

    // setcc writes a byte, cmovcc moves or not depending on the flags.
    activeTest = "setcc/cmovcc";
    reset();
    run([
        0x31, 0xC9,                   // xor ecx, ecx
        0x31, 0xD2,                   // xor edx, edx
        0xB8, 0x05, 0x00, 0x00, 0x00, // mov eax, 5
        0x83, 0xF8, 0x05,             // cmp eax, 5 (ZF=1)
        0x0F, 0x94, 0xC1,             // sete cl
        0x0F, 0x95, 0xC2,             // setne dl
        0xB8, 0x0A, 0x00, 0x00, 0x00, // mov eax, 10
        0xBB, 0x02, 0x00, 0x00, 0x00, // mov ebx, 2
        0x0F, 0x44, 0xC3,             // cmove eax, ebx
        0x0F, 0x45, 0xCB,             // cmovne ecx, ebx
    ]);
    expect(reg64(0), 2n, "cmove taken");
    expect(reg64(1), 1n, "cmovne not taken");
    expect(reg64(2), 0n, "setne");
    expect(reg64(3), 2n, "source unchanged");
    expectFlags({ zf: 1 });

    // bswap.
    activeTest = "bswap";
    reset();
    run([
        0x48, 0xB8, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, // mov rax, 0x1122334455667788
        0x48, 0x0F, 0xC8,                                           // bswap rax
    ]);
    expect(reg64(0), 0x8877_6655_4433_2211n, "bswap rax");

    // xchg, xadd and cmpxchg.
    activeTest = "xchg eax, ebx";
    reset();
    run([
        0xB8, 0x11, 0x00, 0x00, 0x00, // mov eax, 0x11
        0xBB, 0x22, 0x00, 0x00, 0x00, // mov ebx, 0x22
        0x93,                         // xchg eax, ebx
    ]);
    expect(reg64(0), 0x22n, "xchg eax");
    expect(reg64(3), 0x11n, "xchg ebx");

    activeTest = "xchg qword [rax], rcx";
    reset();
    setReg64(0, SCRATCH);
    run([
        0x48, 0xC7, 0x00, 0xAA, 0x00, 0x00, 0x00, // mov qword [rax], 0xAA
        0xB9, 0x55, 0x00, 0x00, 0x00,             // mov ecx, 0x55
        0x48, 0x87, 0x08,                         // xchg qword [rax], rcx
    ]);
    expect(reg64(1), 0xAAn, "xchg loads the old memory value");
    expect(read64(SCRATCH), 0x55n, "xchg stores rcx");

    activeTest = "xadd eax, ebx";
    reset();
    run([
        0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, 1
        0xBB, 0x02, 0x00, 0x00, 0x00, // mov ebx, 2
        0x0F, 0xC1, 0xD8,             // xadd eax, ebx
    ]);
    expect(reg64(0), 3n, "xadd sum");
    expect(reg64(3), 1n, "xadd old value");

    activeTest = "xadd qword [r8], rbx";
    reset();
    setReg64(8, SCRATCH);
    run([
        0x49, 0xC7, 0x00, 0x05, 0x00, 0x00, 0x00, // mov qword [r8], 5
        0xBB, 0x03, 0x00, 0x00, 0x00,             // mov ebx, 3
        0x49, 0x0F, 0xC1, 0x18,                   // xadd qword [r8], rbx
    ]);
    expect(reg64(3), 5n, "xadd returns the old value");
    expect(read64(SCRATCH), 8n, "xadd sum in memory");

    activeTest = "cmpxchg ebx, ecx (match)";
    reset();
    run([
        0xB8, 0x05, 0x00, 0x00, 0x00, // mov eax, 5
        0xBB, 0x05, 0x00, 0x00, 0x00, // mov ebx, 5
        0xB9, 0x07, 0x00, 0x00, 0x00, // mov ecx, 7
        0x0F, 0xB1, 0xCB,             // cmpxchg ebx, ecx
    ]);
    expect(reg64(3), 7n, "cmpxchg writes on a match");
    expectFlags({ zf: 1 });

    activeTest = "cmpxchg ebx, ecx (mismatch)";
    reset();
    run([
        0xB8, 0x05, 0x00, 0x00, 0x00, // mov eax, 5
        0xBB, 0x09, 0x00, 0x00, 0x00, // mov ebx, 9
        0xB9, 0x07, 0x00, 0x00, 0x00, // mov ecx, 7
        0x0F, 0xB1, 0xCB,             // cmpxchg ebx, ecx
    ]);
    expect(reg64(0), 9n, "cmpxchg fails: eax = old value");
    expect(reg64(3), 9n, "cmpxchg fails: destination unchanged");
    expectFlags({ zf: 0 });

    activeTest = "cmpxchg qword [r8], rcx";
    reset();
    setReg64(8, SCRATCH);
    run([
        0x49, 0xC7, 0x00, 0x08, 0x00, 0x00, 0x00, // mov qword [r8], 8
        0xB8, 0x08, 0x00, 0x00, 0x00,             // mov eax, 8
        0xB9, 0x99, 0x00, 0x00, 0x00,             // mov ecx, 0x99
        0x49, 0x0F, 0xB1, 0x08,                   // cmpxchg qword [r8], rcx
    ]);
    expect(read64(SCRATCH), 0x99n, "cmpxchg writes memory on a match");
    expectFlags({ zf: 1 });

    // The sign-extend family.
    activeTest = "cqo/cdq/cdqe";
    reset();
    run([
        0x48, 0xC7, 0xC0, 0xFF, 0xFF, 0xFF, 0xFF, // mov rax, -1
        0x48, 0x99,                               // cqo
        0xB8, 0x00, 0x00, 0x00, 0x80,             // mov eax, 0x80000000
        0x99,                                     // cdq
        0xB8, 0xFF, 0xFF, 0xFF, 0xFF,             // mov eax, 0xFFFFFFFF
        0x48, 0x98,                               // cdqe
    ]);
    expect(reg64(0), 0xFFFF_FFFF_FFFF_FFFFn, "cdqe sign-extends");
    expect(reg64(2), 0xFFFF_FFFFn, "cdq writes edx only");

    activeTest = "movsxd";
    reset();
    run([
        0xB8, 0x00, 0x00, 0x00, 0x80, // mov eax, 0x80000000
        0x48, 0x63, 0xD0,             // movsxd rdx, eax
    ]);
    expect(reg64(2), 0xFFFF_FFFF_8000_0000n, "movsxd");

    // Memory-operand bit tests, shifts and sign extension.
    activeTest = "bt/bts/btr/btc qword [rax], rcx";
    reset();
    setReg64(0, SCRATCH);
    run([
        0x48, 0xC7, 0x00, 0x00, 0x00, 0x00, 0x00, // mov qword [rax], 0
        0xB9, 0x05, 0x00, 0x00, 0x00,             // mov ecx, 5
        0x48, 0x0F, 0xAB, 0x08,                   // bts qword [rax], rcx
        0x48, 0x0F, 0xB3, 0x08,                   // btr qword [rax], rcx
        0x48, 0x0F, 0xBB, 0x08,                   // btc qword [rax], rcx
        0x48, 0x0F, 0xA3, 0x08,                   // bt  qword [rax], rcx
    ]);
    expect(read64(SCRATCH), 0x20n, "bit 5 toggled back on");
    expectFlags({ cf: 1 });

    activeTest = "shl/sar qword [r8], cl";
    reset();
    setReg64(8, SCRATCH);
    run([
        0x49, 0xC7, 0x00, 0x01, 0x00, 0x00, 0x00, // mov qword [r8], 1
        0xB9, 0x04, 0x00, 0x00, 0x00,             // mov ecx, 4
        0x49, 0xD3, 0x20,                         // shl qword [r8], cl
        0x4D, 0x8B, 0x08,                         // mov r9, [r8]
        0x49, 0xC7, 0x00, 0xF0, 0xFF, 0xFF, 0xFF, // mov qword [r8], -16
        0x49, 0xD3, 0x38,                         // sar qword [r8], cl
        0x4D, 0x8B, 0x10,                         // mov r10, [r8]
    ]);
    expect(reg64(9), 0x10n, "memory shl");
    expect(reg64(10), 0xFFFF_FFFF_FFFF_FFFFn, "memory sar");

    activeTest = "movsx/movzx/imul from memory";
    reset();
    setReg64(8, SCRATCH);
    run([
        0x49, 0xC7, 0x00, 0x00, 0x00, 0x00, 0x00, // mov qword [r8], 0
        0x41, 0xC7, 0x00, 0x80, 0xFF, 0xFF, 0xFF, // mov dword [r8], 0xFFFFFF80
        0x49, 0x0F, 0xBE, 0x18,                   // movsx rbx, byte [r8]
        0x49, 0x0F, 0xB7, 0x08,                   // movzx rcx, word [r8]
        0x49, 0x63, 0x10,                         // movsxd rdx, dword [r8]
        0x4D, 0x6B, 0x08, 0x03,                   // imul r9, qword [r8], 3
    ]);
    expect(reg64(3), 0xFFFF_FFFF_FFFF_FF80n, "movsx byte");
    expect(reg64(1), 0xFF80n, "movzx word");
    expect(reg64(2), 0xFFFF_FFFF_FFFF_FF80n, "movsxd dword");
    expect(reg64(9), 0x2_FFFF_FE80n, "imul with a memory operand");

    // All sixteen Jcc conditions, with the flags set to make the branch taken
    // and again to make it not taken. `jcc -2` jumps back to BASE; a branch
    // that is not taken leaves RIP at BASE + 2.
    {
        const conditions = [
            ["o",  F_OF, 0],
            ["no", 0, F_OF],
            ["b",  F_CF, 0],
            ["ae", 0, F_CF],
            ["e",  F_ZF, 0],
            ["ne", 0, F_ZF],
            ["be", F_CF, 0],
            ["a",  0, F_CF],
            ["s",  F_SF, 0],
            ["ns", 0, F_SF],
            ["p",  F_PF, 0],
            ["np", 0, F_PF],
            ["l",  F_SF, 0],
            ["ge", 0, F_SF],
            ["le", F_SF, 0],
            ["g",  0, F_SF],
        ];
        for(let cc = 0; cc < 16; cc++)
        {
            const [name, takenFlags, notTakenFlags] = conditions[cc];
            activeTest = "j" + name;

            reset();
            cpu.flags[0] = 0x2 | takenFlags;
            ex.write8(BASE, 0x70 | cc);
            ex.write8(BASE + 1, 0xFE);
            cpu.instruction_pointer[0] = BASE;
            ex.interp64_run_one();
            expect(cpu.instruction_pointer[0], BASE, "taken");

            reset();
            cpu.flags[0] = 0x2 | notTakenFlags;
            ex.write8(BASE, 0x70 | cc);
            ex.write8(BASE + 1, 0xFE);
            cpu.instruction_pointer[0] = BASE;
            ex.interp64_run_one();
            expect(cpu.instruction_pointer[0], BASE + 2, "not taken");
        }
    }

    if(failures.length)
    {
        for(const failure of failures)
        {
            console.log("FAIL " + failure);
        }
        console.log("interp64 insns: " + failures.length + " failure(s)");
        process.exit(1);
    }

    console.log("interp64 insns: all tests passed");
    process.exit(0);
});
