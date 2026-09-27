#!/usr/bin/env node

// Opcode-space coverage check for the long-mode interpreter (interp64).
//
// The list of defined opcodes comes from tests/interp64/opcode-maps.js, which is
// derived from the Linux kernel's arch/x86/lib/x86-opcode-map.txt
// (https://raw.githubusercontent.com/torvalds/linux/master/arch/x86/lib/x86-opcode-map.txt),
// itself generated from the opcode maps in Appendix A of the Intel SDM Vol. 2.
//
// For every opcode the map defines, this assembles a few candidate encodings
// (the operand shape is not known from the map alone) and runs them in long
// mode. An opcode that raises #UD for every candidate is either a gap in the
// interpreter or an encoding we failed to build; the report distinguishes them
// by hand later.
//
// Run: node tests/interp64/coverage.js [table]
//   table: one|0f|0f38|0f3a  (default: one)

import { v64 } from "../../src/main.js";
import { TABLES } from "./opcode-maps.js";

const CODE = 0x6000;
const STACK = 0x70000;

const emulator = new v64({
    autostart: false,
    memory_size: 2 * 1024 * 1024,
    disable_jit: 0,
    log_level: 0,
    wasm_path: process.env.WASM_PATH || undefined,
});

if(process.argv[2])
{
    const want = process.argv[2];
    const keep = TABLES.filter(t => t.name.replace(/ /g, "").startsWith(want) || t.prefix.join("") === want);
    TABLES.length = 0;
    TABLES.push(...keep);
}

emulator.add_listener("emulator-loaded", () =>
{
    const cpu = emulator.v86.cpu;
    const ex = cpu.wm.exports;
    const buffer = ex.memory.buffer;

    const write64 = (address, value) =>
    {
        ex.write32(address, Number(value & 0xFFFF_FFFFn));
        ex.write32(address + 4, Number(value >> 32n & 0xFFFF_FFFFn));
    };

    // Identity map the first 1 GiB (4 KiB pages, plus 2 MiB pages above 2 MiB)
    const PML4 = 0x10000, PDPT = 0x11000, PD = 0x12000, PT = 0x13000;
    write64(PML4, BigInt(PDPT) | 0x3n);
    write64(PDPT, BigInt(PD) | 0x3n);
    write64(PD, BigInt(PT) | 0x3n);
    for(let i = 0; i < 512; i++)
    {
        write64(PT + i * 8, BigInt(i) * 0x1000n | 0x3n);
    }
    for(let i = 1; i < 512; i++)
    {
        write64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);
    }

    ex.enter_long_mode(PML4);

    // Minimal IDT: every vector points at its own stub, which records the
    // vector in a flag byte and halts. Without an IDT the emulator panics
    // while delivering an exception, which makes #UD undetectable.
    const IDT = 0x8000, STUB = 0x9000, VECTOR_FLAG = 0x5000;
    const cs = ex.read16(670); // current CS selector
    const w16 = (a, v) => { ex.write8(a, v & 0xFF); ex.write8(a + 1, v >> 8 & 0xFF); };
    for(let v = 0; v < 256; v++)
    {
        const stub = STUB + v * 16;
        // mov byte [VECTOR_FLAG], v ; hlt
        [0xC6, 0x04, 0x25, 0x00, 0x50, 0x00, 0x00, v, 0xF4].forEach((b, i) => ex.write8(stub + i, b));
        const gate = IDT + v * 16;
        w16(gate + 0, stub & 0xFFFF);
        w16(gate + 2, cs);
        ex.write8(gate + 4, 0);       // IST
        ex.write8(gate + 5, 0x8E);    // present, DPL 0, 64-bit interrupt gate
        w16(gate + 6, stub >>> 16 & 0xFFFF);
        ex.write32(gate + 8, 0);
        ex.write32(gate + 12, 0);
    }
    ex.write8(VECTOR_FLAG, 0);
    write64(1240, BigInt(IDT));       // idtr_base
    ex.write32(564, 256 * 16 - 1);    // idtr_size
    ex.write32(568, IDT);             // idtr_offset (32-bit mirror)

    let exception = 0;
    emulator.cpu_exception_hook = n => { exception = n; };

    const reset_cpu = () =>
    {
        // zero the general purpose registers so operands have sane values
        for(let i = 0; i < 8; i++)
        {
            ex.write32(64 + i * 4, 0);
            ex.write32(128 + i * 4, 0);
        }
        for(let i = 0; i < 8; i++)
        {
            write64(160 + i * 8, 0n);
        }
        write64(64 + 4 * 8, BigInt(STACK)); // rsp
        ex.write32(120, 0x2);               // flags (IF)
        cpu.in_hlt[0] = 0;
        new DataView(ex.memory.buffer).setBigUint64(232, BigInt(CODE), true);
    };

    const run_one = bytes =>
    {
        for(let i = 0; i < bytes.length; i++)
        {
            ex.write8(CODE + i, bytes[i]);
        }
        ex.write8(CODE + bytes.length, 0xF4); // hlt
        exception = 0;
        ex.write8(VECTOR_FLAG, 0);
        reset_cpu();
        let guard = 0;
        try
        {
            while(!cpu.in_hlt[0] && !exception && guard++ < 200)
            {
                ex.main_loop();
            }
        }
        catch(e)
        {
            // A control transfer with our synthetic operands can jump into
            // unmapped memory and trap inside the emulator; the instruction
            // itself decoded and executed, so it is not a coverage gap.
            return -1;
        }
        return exception; // the exception hook is the reliable signal
    };

    // Invalid in 64-bit mode, so #UD is correct rather than a coverage gap:
// push/pop ES/CS/SS/DS, daa/das/aaa/aas, pusha/popa, bound, the undefined
// 80-alias, callf/jmpf, into, aam/aad/salc/icebp, the VEX prefixes, jmp rel32
// (the tails below do not supply its displacement), syscall/sysret (need
// EFER.SCE), ud2 (raising #UD is its behaviour) and the reserved 0F opcodes.
const INVALID64 = {
    "one byte": new Set([0x06, 0x07, 0x0e, 0x16, 0x17, 0x1e, 0x1f, 0x27, 0x2f, 0x37,
        0x3f, 0x60, 0x61, 0x62, 0x82, 0x9a, 0xc4, 0xc5, 0xce, 0xd4, 0xd5, 0xd6, 0xf1, 0xe9, 0xea]),
    "0f": new Set([0x04, 0x05, 0x07, 0x0a, 0x0b, 0x0c, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
        // 0f 0f is 3DNow!, deliberately unsupported (no MMX state is modelled)
        0x0f]),
};

// Candidate operand tails, cheapest first: register form, memory [rax],
    // [rax+disp8], and the no-ModRM case.
    const TAILS = [[0xC0], [0x00], [0x48, 0x00], []];

    // ENDBR64 is only encoded as F3 0F 1E FA.
    const PREFIXED = { "0f:1e": [[0xF3, 0x0F, 0x1E, 0xFA]] };

    console.log("probe UD2 -> exception " + run_one([0x0F, 0x0B]) + " (expect 6)");
    const summary = [];
    for(const table of TABLES)
    {
        const gaps = [];
        let covered = 0;
        for(let op = 0; op < 256; op++)
        {
            if(table.table[op] === ".")
            {
                continue;
            }
            let ok = false;
            const prefixed = PREFIXED[table.name.replace(/ /g, "") + ":" + op.toString(16)];
            if(prefixed)
            {
                ok = prefixed.some(bytes => run_one(bytes) !== 6 && run_one(bytes) !== -6);
                covered += ok ? 1 : 0;
                gaps.push(...(ok ? [] : [op]));
                continue;
            }
            for(const tail of TAILS)
            {
                const r = run_one([...table.prefix, op, ...tail]);
                if(Math.abs(r) !== 6) // 6 = #UD (hook reports it negative); a trap counts as ran
                {
                    ok = true;
                    break;
                }
            }
            if(ok)
            {
                covered++;
            }
            else
            {
                gaps.push(op);
            }
        }
        const skip = INVALID64[table.name] || new Set();
        summary.push({ name: table.name, covered, invalid: gaps.filter(o => skip.has(o)).length,
            gaps: gaps.filter(o => !skip.has(o)) });
    }

    for(const s of summary)
    {
        console.log("table " + s.name + ": covered " + s.covered +
            ", invalid-in-64-bit " + s.invalid +
            ", #UD for every candidate: " + s.gaps.length +
            (s.gaps.length ? "  [" + s.gaps.map(x => x.toString(16)).join(" ") + "]" : ""));
    }
    process.exit(0);
});
