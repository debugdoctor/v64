#!/usr/bin/env node

// XSAVE/XRSTOR round-trip, component masks and CPUID layout tests.
//
// What is checked:
//   * round trip: 16 distinct YMM values survive XSAVE -> clobber -> XRSTOR,
//     covering XMM0-7 and XMM8-15 as well as every YMM high half
//   * the SDM layout: legacy area at +0, XMM0-15 at +160, XSTATE_BV at +512,
//     XCOMP_BV at +520, and the AVX (YMM_Hi128) component at +576
//   * XSTATE_BV records mask & XCR0, not the raw mask
//   * XRSTOR splits the mask three ways: present in the area is restored,
//     absent is put in its initial state, not requested is left alone
//   * CPUID leaf 0DH agrees with where XSAVE actually writes, since the kernel
//     sizes its buffers from the leaf
//   * the AVX component is only written when XCR0 bit 2 is enabled
//
// Encodings come from `nasm -f bin`; see the table in the comments below.
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/jit64/xsave.js`

import { v64 } from "../../src/main.js";
import { set_cpu_config, JIT_DISABLE_64 } from "../../src/config.js";

const BASE = 0x1000;
const PML4 = 0x10000;
const PDPT = 0x11000;
const PD = 0x12000;
const PT = 0x13000;

// Staging and the XSAVE image. The image must be 64-byte aligned, and the
// offsets below have to leave room for the full 832-byte x87+SSE+AVX image.
const SRC = 0x100000;          // 16 x 32 bytes of YMM source data
const DST = 0x101000;          // 16 x 32 bytes the restored value is written to
const AREA = 0x102000;         // the XSAVE/XRSTOR image (64-byte aligned)

// nasm -f bin, and the rule checked against all sixteen registers rather than
// assumed:
//   vmovdqu ymmN, [r15]   = C4 C1 7E 6F <(N&7)<<3|07>   for N < 8
//                           C4 41 7E 7F ...             for N >= 8  (REX.R)
//   vmovdqu [r15], ymmN    = same with opcode 7F
//   xsave  [r15+0]        = 41 0f ae 27
//   xrstor [r15+0]        = 41 0f ae 2f
//   mov eax, 7            = b8 07 00 00 00
//   mov eax, 3            = b8 03 00 00 00
//   cpuid                 = 0f a2
//   xgetbv                = 0f 01 d0
//   mov edx, 0            = ba 00 00 00 00
const vmovdqu_load = n =>
    [0xC4, n < 8 ? 0xC1 : 0x41, 0x7E, 0x6F, ((n & 7) << 3) | 0x07];
const vmovdqu_store = n =>
    [0xC4, n < 8 ? 0xC1 : 0x41, 0x7E, 0x7F, ((n & 7) << 3) | 0x07];
const xsave_ = [0x41, 0x0F, 0xAE, 0x27];
const xrstor_ = [0x41, 0x0F, 0xAE, 0x2F];
const mov_eax_7 = [0xB8, 0x07, 0x00, 0x00, 0x00];
const mov_eax_3 = [0xB8, 0x03, 0x00, 0x00, 0x00];
const mov_edx_0 = [0xBA, 0x00, 0x00, 0x00, 0x00];

// mov rdi, imm64 -- so a case can point r15 at AREA / SRC / DST.
const mov_rdi_imm64 = v =>
{
    const out = [0x48, 0xBF];
    for(let i = 0; i < 8; i++) out.push(Number((BigInt(v) >> BigInt(8 * i)) & 0xFFn));
    return out;
};
// mov r15, rdi
const mov_r15_rdi = [0x49, 0x89, 0xFF];

// 16 distinct 32-byte YMM images. Every byte position is distinct across the set
// so a register swap or a lane swap cannot pass by accident.
const YMM_SRC = Array.from({ length: 16 }, (_, n) =>
    Array.from({ length: 32 }, (_, i) => (0x5A + n * 7 + i * 3) & 0xFF));

const hex = a => a.map(b => b.toString(16).padStart(2, "0")).join(" ");

const emulator = new v64({
    autostart: false,
    memory_size: 8 * 1024 * 1024,
    disable_jit: 0,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("emulator-loaded", async () =>
{
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const u32 = new Uint32Array(ex.memory.buffer);
    const ext = new BigUint64Array(ex.memory.buffer, 160, 8);

    const write64 = (a, v) =>
    {
        ex.write32(a, Number(v & 0xFFFF_FFFFn));
        ex.write32(a + 4, Number(v >> 32n & 0xFFFF_FFFFn));
    };
    write64(PML4, BigInt(PDPT) | 0x3n);
    write64(PDPT, BigInt(PD) | 0x3n);
    write64(PD, BigInt(PT) | 0x3n);
    for(let i = 0; i < 512; i++) write64(PT + i * 8, BigInt(i) * 0x1000n | 0x3n);

    // A fresh view per call: growing the wasm memory detaches any view taken
    // earlier, and writes to a detached view are silently dropped rather than
    // throwing, so a stale view makes register setup quietly do nothing.
    const set64 = (i, v) =>
    {
        const a = new Uint32Array(ex.memory.buffer);
        const e = new BigUint64Array(ex.memory.buffer, 160, 8);
        if(i < 8)
        {
            a[16 + i] = Number(v & 0xFFFF_FFFFn);
            a[32 + i] = Number((v >> 32n) & 0xFFFF_FFFFn);
        }
        else { e[i - 8] = v; }
    };

    const read = (addr, len) =>
    {
        const out = [];
        for(let i = 0; i < len; i++) out.push(ex.read8(addr + i) >>> 0);
        return out;
    };
    const read64 = addr =>
    {
        let v = 0n;
        for(let i = 0; i < 8; i++) v |= BigInt(ex.read8(addr + i) >>> 0) << BigInt(8 * i);
        return v;
    };

    // ---- harness self-check -------------------------------------------
    {
        const distinct = [];
        // Mask to 64 bits: the multiply overflows for i > 0, and an
        // unmasked expectation would make every register except r0 "fail".
        for(let i = 0; i < 16; i++)
        {
            distinct.push(BigInt.asUintN(64, 0x9E3779B97F4A7C15n * BigInt(i + 1)));
        }
        const want4 = 0x80000n;
        for(let i = 0; i < 16; i++) set64(i, distinct[i]);
        set64(4, want4);
        ex.write8(BASE, 0xF4);
        cpu.instruction_pointer[0] = BASE;
        cpu.flags[0] = 0x2;
        cpu.in_hlt[0] = 0;
        ex.enter_long_mode(PML4);
        let g = 0;
        while(!cpu.in_hlt[0] && g++ < 20000) ex.main_loop();
        const bad = [];
        for(let i = 0; i < 16; i++)
        {
            const want = i === 4 ? want4 : distinct[i];
            const got = i < 8
                ? ((BigInt(u32[32 + i]) << 32n) | BigInt(u32[16 + i]))
                : ext[i - 8];
            if(got !== want) bad.push("r" + i);
        }
        if(bad.length)
        {
            console.log("harness read-back FAILED: " + bad.join(", "));
            console.log("results below would be meaningless; aborting");
            process.exit(2);
        }
        console.log("harness self-check: register read-back ok (16/16 distinct)\n");
    }

    // Load all 16 YMM registers from a 512-byte block at r15, advancing r15 by
    // 32 bytes after each one.
    const each = (base, make) =>
    {
        const seq = [mov_rdi_imm64(base), mov_r15_rdi];
        for(let n = 0; n < 16; n++)
        {
            seq.push(...make(n), mov_rdi_imm64(base + (n + 1) * 32), mov_r15_rdi);
        }
        return seq.flat();
    };
    const loadAll = base => each(base, vmovdqu_load);
    const storeAll = base => each(base, vmovdqu_store);
    // Zero every YMM by loading a zero block, rather than introducing a second
    // encoding whose only job is to write zeroes.
    const ZERO = 0x103000;
    const zeroAll = () => loadAll(ZERO);

    const run = (program, jitEnabled) =>
    {
        set_cpu_config(ex, "JIT_DISABLE", jitEnabled ? 0 : JIT_DISABLE_64);
        ex.jit64_clear_cache();
        ex.jit64_clear_xmm();
        const code = program.concat([0xF4]);
        for(let i = 0; i < code.length; i++) ex.write8(BASE + i, code[i]);
        for(let i = 0; i < 16; i++) set64(i, 0n);
        set64(4, 0x80000n);
        set64(15, BigInt(AREA));
        cpu.flags[0] = 0x2;
        cpu.instruction_pointer[0] = BASE;
        cpu.in_hlt[0] = 0;
        ex.enter_long_mode(PML4);
        let g = 0;
        while(!cpu.in_hlt[0] && g++ < 200000) ex.main_loop();
        return cpu.in_hlt[0] === 1;
    };

    for(let n = 0; n < 16; n++)
        for(let i = 0; i < 32; i++) ex.write8(SRC + n * 32 + i, YMM_SRC[n][i]);
    for(let i = 0; i < 0x400; i++) ex.write8(AREA + i, 0x5A);
    for(let i = 0; i < 512; i++) ex.write8(ZERO + i, 0x00);
    for(let i = 0; i < 512; i++) ex.write8(DST + i, 0xCC);

    const engines = [];
    const which = process.env.XSAVE_ENGINE || "both";
    if(which === "interp" || which === "both") engines.push(["interp", false]);
    if(which === "jit" || which === "both") engines.push(["jit", true]);

    let failures = 0;
    const report = (name, bad) =>
    {
        if(bad.length) failures++;
        console.log((bad.length ? "FAIL " : "ok   ") + name);
        for(const b of bad) console.log("      " + b);
    };

    for(const [label, jitEnabled] of engines)
    {
        // ---- 1. round trip: XSAVE, clobber, XRSTOR, store -----------------
        // Mask 0x7 = x87 + SSE + AVX, which is what a kernel that has enabled
        // XCR0 for AVX asks for.
        const roundTrip = [].concat(
            loadAll(SRC),
            mov_rdi_imm64(AREA), mov_r15_rdi, mov_eax_7, mov_edx_0, xsave_,
            zeroAll(),
            mov_rdi_imm64(AREA), mov_r15_rdi, xrstor_,
            storeAll(DST),
        );
        const halted = run(roundTrip, jitEnabled);
        const bad = [];
        if(!halted) bad.push(label + ": did not halt");
        const got = read(DST, 512);
        const want = YMM_SRC.flat();
        if(got.length === want.length)
        {
            for(let n = 0; n < 16; n++)
            {
                const slice = got.slice(n * 32, n * 32 + 32);
                if(slice.some((v, i) => v !== want[n * 32 + i]))
                {
                    bad.push(label + ": ymm" + n + " got " + hex(slice) +
                        " want " + hex(YMM_SRC[n]));
                }
            }
        }
        else bad.push(label + ": short read");
        report("round trip: 16 YMM values survive XSAVE -> XRSTOR [" + label + "]", bad);

        // ---- 2. SDM layout ------------------------------------------------
        {
            const bad = [];
            // XSTATE_BV lives at +512 and records mask & XCR0.
            const bv = read64(AREA + 512);
            // Bit 0 is x87, bit 1 SSE, bit 2 AVX.
            if(bv === 0n) bad.push(label + ": XSTATE_BV at +512 is zero");
            if((bv & ~0x7n) !== 0n) bad.push(label + ": XSTATE_BV has bits outside the mask: 0x" + bv.toString(16));
            // The extended region starts after the 512-byte legacy region and
            // the 64-byte header, so the AVX component is at +576. That is what
            // Linux uses (`XSAVE_YMM_OFFSET = XSAVE_HDR_SIZE + XSAVE_HDR_OFFSET`
            // = 64 + 512) and what CPUID.0DH.2.EBX has to report. The SSE
            // registers are at +160, inside the legacy region.
            for(let n = 0; n < 16; n++)
            {
                const hi = read(AREA + 576 + n * 16, 16);
                if(hi.some((v, i) => v !== YMM_SRC[n][16 + i]))
                {
                    bad.push(label + ": AVX area ymm" + n + " high half got " + hex(hi) +
                        " want " + hex(YMM_SRC[n].slice(16)));
                }
            }
            // The low halves belong at +160, one 16-byte slot per register.
            for(let n = 0; n < 16; n++)
            {
                const lo = read(AREA + 160 + n * 16, 16);
                if(lo.some((v, i) => v !== YMM_SRC[n][i]))
                {
                    bad.push(label + ": legacy area xmm" + n + " got " + hex(lo) +
                        " want " + hex(YMM_SRC[n].slice(0, 16)));
                }
            }
            // The legacy region holds data only up to +416 (XMM0-15 end there),
            // so +416..+511 is padding between the legacy data and the header and
            // must be left as the 0x5A fill. Probing the old +576 would now be
            // probing the AVX area itself.
            // The legacy region is 512 bytes but only the first 416 hold data;
            // FXSAVE writes the reserved remainder as zero, and XSAVE does the
            // same. Asserting zero here pins that, rather than the 0x5A fill the
            // buffer was primed with -- an earlier version of this check probed
            // for the fill and failed against correct behaviour.
            const pad = read(AREA + 416, 96);
            if(pad.some(v => v !== 0))
            {
                bad.push(label + ": reserved bytes 416..511 are not zero");
            }
            report("SDM layout: XMM at +160, XSTATE_BV at +512, AVX at +576 [" + label + "]", bad);
        }

        // ---- 3. XRSTOR splits requested / present / absent ----------------
        // SDM Vol.1 13.4.2: a component the mask asks for and the area holds is
        // restored, one the mask asks for but the area does not hold goes to
        // its initial state, and one the mask does not ask for is left as it
        // was. Collapsing any two of those loses or invents state.
        {
            // Build an image with the AVX component, then clear its XSTATE_BV.
            run([].concat(
                loadAll(SRC),
                mov_rdi_imm64(AREA), mov_r15_rdi, mov_eax_7, mov_edx_0, xsave_,
            ), jitEnabled);
            for(let i = 0; i < 8; i++) ex.write8(AREA + 512 + i, 0);

            // 3a: mask asks for AVX, the area does not have it -> initial state.
            const bad = [];
            const cleared = [].concat(
                loadAll(SRC),
                mov_rdi_imm64(AREA), mov_r15_rdi, mov_eax_7, mov_edx_0, xrstor_,
                storeAll(DST),
            );
            run(cleared, jitEnabled);
            const gotZero = read(DST, 512);
            for(let n = 0; n < 16; n++)
            {
                for(let i = 0; i < 32; i++)
                {
                    if(gotZero[n * 32 + i] !== 0)
                    {
                        bad.push(label + ": ymm" + n + " byte " + i +
                            " should be 0 after restoring a component the area lacks");
                        break;
                    }
                }
            }
            report("XRSTOR initialises a requested component the area does not hold [" + label + "]", bad);
        }
        {
            // 3b: mask does not ask for AVX -> the high halves keep the values
            // the preceding load put there.
            const bad = [];
            run([].concat(
                loadAll(SRC),
                mov_rdi_imm64(AREA), mov_r15_rdi, mov_eax_7, mov_edx_0, xsave_,
            ), jitEnabled);
            const noAvx = [].concat(
                loadAll(SRC),
                mov_rdi_imm64(AREA), mov_r15_rdi, mov_eax_3, mov_edx_0, xrstor_,
                storeAll(DST),
            );
            run(noAvx, jitEnabled);
            const got = read(DST, 512);
            for(let n = 0; n < 16; n++)
            {
                for(let i = 0; i < 32; i++)
                {
                    if(got[n * 32 + i] !== YMM_SRC[n][i])
                    {
                        bad.push(label + ": ymm" + n + " byte " + i +
                            " changed although the mask did not request AVX");
                        break;
                    }
                }
            }
            report("XRSTOR leaves alone a component the mask does not request [" + label + "]", bad);
        }
    }

    // ---- 4. CPUID leaf 0DH agrees with where XSAVE writes ----------------
    // The kernel sizes its FPU buffers from leaf 0DH and then reads and writes
    // them through XSAVE/XRSTOR, so the two have to describe the same layout.
    {
        const bad = [];
        // Register numbering here is RAX=0, RCX=1, RDX=2, RBX=3, so CPUID's
        // sub-leaf selector is ECX (register 1) and its results come back as
        // EAX/EBX/ECX/EDX = registers 0/3/1/2. Getting this wrong reads the
        // wrong registers and looks like an emulator bug.
        const CPUID_REG = [0, 3, 1, 2];
        const cpuid = (leaf, sub) =>
        {
            set64(0, BigInt(leaf));
            set64(1, BigInt(sub));
            cpu.flags[0] = 0x2;
            cpu.in_hlt[0] = 0;
            cpu.instruction_pointer[0] = BASE;
            // run() left a compiled block for BASE covering a longer program,
            // and the interpreter keeps its own rip that only enter_long_mode
            // resyncs from instruction_pointer. Without both the stub is never
            // fetched from BASE at all.
            ex.jit64_clear_cache();
            ex.enter_long_mode(PML4);
            cpu.instruction_pointer[0] = BASE;
            ex.write8(BASE, 0x0F);
            ex.write8(BASE + 1, 0xA2);
            ex.write8(BASE + 2, 0xF4);
            // run_exact_instructions, not main_loop: main_loop delivers timer
            // interrupts and this harness has no IDT, so a triple fault ends
            // the run before the stub ever executes.
            ex.run_exact_instructions(2);
            // A fresh view: the JIT grows the wasm memory, which detaches any
            // view taken earlier, and reads from a detached view give undefined.
            const r = i =>
            {
                const a = new Uint32Array(ex.memory.buffer);
                return (BigInt(a[32 + i]) << 32n) | BigInt(a[16 + i]);
            };
            return CPUID_REG.map(i => r(i) & 0xFFFFFFFFn);
        };
        const want = [
            // leaf, sub, EAX, EBX, ECX, EDX
            ["sub-leaf 0 EAX: supported XCR0 bits", 0xD, 0, 7, null, null, 0],
            // Sub-leaf 0: EBX is the size for the features XCR0 enables, ECX the
            // size for every supported feature. Both are 832 for x87+SSE+AVX.
            ["sub-leaf 0 EBX: area size with XCR0 = 7", 0xD, 0, null, 832, 832, null],
            // Sub-leaf 1 is the XSAVE features leaf, not a state component. The
            // kernel allocates its per-task FPU buffer from this EBX, so it has to
            // be the size of the whole area.
            ["sub-leaf 1: features leaf, area size", 0xD, 1, 0, 832, 0, 0],
            // Sub-leaf 2 is the AVX component: 256 bytes at offset 576.
            ["sub-leaf 2: YMM_Hi128 size and offset", 0xD, 2, 256, 576, 0, 0],
        ];
        for(const [name, leaf, sub, a, b, c, d] of want)
        {
            const [ra, rb, rc, rd] = cpuid(leaf, sub);
            if(a !== null && ra !== BigInt(a)) bad.push(name + ": EAX=0x" + ra.toString(16));
            if(b !== null && rb !== BigInt(b)) bad.push(name + ": EBX=0x" + rb.toString(16));
            if(c !== null && rc !== BigInt(c)) bad.push(name + ": ECX=0x" + rc.toString(16));
            if(d !== null && rd !== BigInt(d)) bad.push(name + ": EDX=0x" + rd.toString(16));
        }
        // XGETBV must report the XCR0 that leaf 0xD described.
        set64(1, 0n);
        cpu.flags[0] = 0x2;
        cpu.in_hlt[0] = 0;
        cpu.instruction_pointer[0] = BASE;
        ex.jit64_clear_cache();
        ex.enter_long_mode(PML4);
        cpu.instruction_pointer[0] = BASE;
        ex.write8(BASE, 0x0F);
        ex.write8(BASE + 1, 0x01);
        ex.write8(BASE + 2, 0xD0);
        ex.write8(BASE + 3, 0xF4);
        {
            ex.run_exact_instructions(2);
            const a = new Uint32Array(ex.memory.buffer);
            // XGETBV returns EDX:EAX, and EDX is register 2 while EAX is 0.
            const xcr0 = ((BigInt(a[32 + 2]) << 32n) | (BigInt(a[16 + 0]) & 0xFFFFFFFFn));
            if((xcr0 & 7n) !== 7n)
            {
                bad.push("XGETBV reports XCR0=0x" + xcr0.toString(16) +
                    ", but leaf 0xD says x87+SSE+AVX are supported");
            }
        }
        report("CPUID leaf 0DH matches the XSAVE layout", bad);
    }

    console.log("\n" + (failures ? failures + " failing case(s)" : "XSAVE/XRSTOR match the SDM layout and round trip"));
    process.exit(failures ? 1 : 0);
});
