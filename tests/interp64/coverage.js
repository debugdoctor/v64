#!/usr/bin/env node

// Coverage check for interp64: run every opcode from tests/interp64/opcode-maps.js
// (opcode data from the Linux kernel and Intel SDM Vol. 2 - see its header)
// in long mode and report the ones that raise #UD for all candidate encodings.
// A trap (privileged opcode / emulator panic) rebuilds the machine.
//
// node tests/interp64/coverage.js [table]   (default: all)

import { v64 } from "../../src/main.js";
import { TABLES } from "./opcode-maps.js";

const CODE = 0x6000;
const IDT = 0x8000;
const STUB = 0x9000;
const VECTOR_FLAG = 0x5000; // clear of the IDT, the stubs and the stack

const boot = () => new Promise(resolve =>
{
    const emulator = new v64({
        autostart: false,
        memory_size: 2 * 1024 * 1024,
        disable_jit: 0,
        log_level: 0,
        wasm_path: process.env.WASM_PATH || undefined,
    });

    emulator.add_listener("emulator-loaded", () =>
    {
        const cpu = emulator.v86.cpu;
        const ex = cpu.wm.exports;
        const view = () => new DataView(ex.memory.buffer);
        const rd32 = a => view().getUint32(a, true);
        const rd64 = a => BigInt(rd32(a)) | BigInt(rd32(a + 4)) << 32n;
        const wr64 = (a, v) =>
        {
            ex.write32(a, Number(v & 0xFFFF_FFFFn));
            ex.write32(a + 4, Number(v >> 32n & 0xFFFF_FFFFn));
        };

        // identity map the first 1 GiB
        const PML4 = 0x10000, PDPT = 0x11000, PD = 0x12000, PT = 0x13000;
        wr64(PML4, BigInt(PDPT) | 3n);
        wr64(PDPT, BigInt(PD) | 3n);
        wr64(PD, BigInt(PT) | 3n);
        for(let i = 0; i < 512; i++)
        {
            wr64(PT + i * 8, BigInt(i) * 0x1000n | 3n);
        }
        for(let i = 1; i < 512; i++)
        {
            wr64(PD + i * 8, BigInt(i) * 0x200000n | 0x83n);
        }
        ex.enter_long_mode(PML4);

        // every gate points at a stub that records the vector
        const cs = ex.read16(670);
        for(let v = 0; v < 256; v++)
        {
            const stub = STUB + v * 16;
            [0xC6, 0x04, 0x25, 0x00, 0x50, 0x00, 0x00, v, 0xF4].forEach((b, i) => ex.write8(stub + i, b));
            const gate = IDT + v * 16;
            ex.write8(gate, stub & 0xFF);
            ex.write8(gate + 1, stub >> 8 & 0xFF);
            ex.write8(gate + 2, cs & 0xFF);
            ex.write8(gate + 3, cs >> 8 & 0xFF);
            ex.write8(gate + 5, 0x8E);
            ex.write8(gate + 6, stub >> 16 & 0xFF);
            ex.write8(gate + 7, stub >> 24 & 0xFF);
        }
        ex.write8(VECTOR_FLAG, 0);

        // state that privileged opcodes under test may clobber
        const saved = {
            cr0: rd32(580), cr3: rd32(592), cr4: rd32(596),
            idtr_size: rd32(564), idtr_offset: rd32(568), idtr_base: rd64(1240),
        };

        let exception = 0;
        emulator.cpu_exception_hook = n => { exception = n; };

        resolve({
            // 0 = ran, a vector = delivered, -1 = trapped
            run(bytes)
            {
                for(let i = 0; i < bytes.length; i++)
                {
                    ex.write8(CODE + i, bytes[i]);
                }
                ex.write8(CODE + bytes.length, 0xF4);
                for(let i = 0; i < 8; i++)
                {
                    ex.write32(64 + i * 4, 0);
                    ex.write32(128 + i * 4, 0);
                    wr64(160 + i * 8, 0n);
                }
                wr64(96, 0x70000n); // rsp
                ex.write32(120, 0x2); // flags, IF set
                ex.write8(VECTOR_FLAG, 0);
                ex.write32(580, saved.cr0);
                ex.write32(592, saved.cr3);
                ex.write32(596, saved.cr4);
                ex.write32(564, saved.idtr_size);
                ex.write32(568, saved.idtr_offset);
                wr64(1240, saved.idtr_base);
                exception = 0;
                cpu.in_hlt[0] = 0;
                new DataView(ex.memory.buffer).setBigUint64(232, BigInt(CODE), true);
                try
                {
                    let guard = 0;
                    while(!cpu.in_hlt[0] && !exception && guard++ < 200)
                    {
                        ex.main_loop();
                    }
                }
                catch(e)
                {
                    return -1;
                }
                return exception;
            },
        });
    });
});

// operand tails tried per opcode: reg form, [rax], [rax+disp8], no ModRM
const TAILS = [[0xC0], [0x00], [0x48, 0x00], []];

// ENDBR64 is only encoded as F3 0F 1E FA
const PREFIXED = { "0f:1e": [[0xF3, 0x0F, 0x1E, 0xFA]] };

// invalid in 64-bit mode, so #UD is correct: ES/CS/SS/DS + BCD/string aliases,
// pusha/popa, bound, callf/jmpf, into, VEX prefixes, jmp rel32 (no displacement
// in the tails), syscall/sysret (EFER.SCE unset), ud2, reserved 0F opcodes
const INVALID64 = {
    "one byte": [0x06, 0x07, 0x0e, 0x16, 0x17, 0x1e, 0x1f, 0x27, 0x2f, 0x37, 0x3f,
        0x60, 0x61, 0x62, 0x82, 0x9a, 0xc4, 0xc5, 0xce, 0xd4, 0xd5, 0xd6, 0xf1, 0xe9, 0xea],
    "0f": [0x04, 0x05, 0x07, 0x0a, 0x0b, 0x0c, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x0f],
};

const main = async () =>
{
    const want = process.argv[2];
    const tables = want
        ? TABLES.filter(t => t.name.replace(/ /g, "") === want)
        : TABLES;

    let machine = await boot();
    let rebuilds = 0;
    const report = [];

    for(const table of tables)
    {
        const skip = new Set(INVALID64[table.name] || []);
        const gaps = [];
        let covered = 0;
        let trapped = 0;

        for(let op = 0; op < 256; op++)
        {
            if(table.table[op] === "." || skip.has(op))
            {
                continue;
            }
            const prefixed = PREFIXED[table.name.replace(/ /g, "") + ":" + op.toString(16)];
            // many 0F opcodes are only valid with a mandatory 66/F3/F2 prefix
            const with_prefix = p => TAILS.map(t => [p, ...table.prefix, op, ...t]);
            const candidates = prefixed || [
                ...TAILS.map(t => [...table.prefix, op, ...t]),
                ...with_prefix(0x66), ...with_prefix(0xF3), ...with_prefix(0xF2),
            ];
            let ok = false;
            for(const bytes of candidates)
            {
                let r = machine.run(bytes);
                if(r === -1)
                {
                    // the machine is unusable now
                    machine = await boot();
                    rebuilds++;
                    r = machine.run(bytes);
                }
                if(r !== 6)
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
        report.push({ name: table.name, covered, invalid: 256 - covered - gaps.length, gaps, trapped });
    }

    for(const r of report)
    {
        console.log("table " + r.name + ": covered " + r.covered +
            ", invalid-in-64-bit " + r.invalid +
            ", #UD for every candidate: " + r.gaps.length +
            (r.gaps.length ? "  [" + r.gaps.map(x => x.toString(16)).join(" ") + "]" : ""));
    }

    console.log("machine rebuilds: " + rebuilds);
    process.exit(0);
};

main();
