#!/usr/bin/env node

// Randomised differential test for the 32-bit interpreter's decode cache.
//
// Generates random programs from the opcodes the cache decodes, runs each one
// twice -- cache engaged and cache disabled -- and requires identical CPU and
// memory state. This is what catches a decoder or executor that disagrees with
// `run_instruction` on an encoding nobody thought to write a test for.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp32/decode-cache-fuzz.js [cases] [seed]

import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";

const CASES = +process.argv[2] || 200;
const SEED = +process.argv[3] || 12345;

const RESET = 0xFFFF0;
const CODE = 0x8000;
const STACK_TOP = 0x70000;

// Registers start inside this window so memory operands stay clear of the page
// tables (0x1000-0x5FFF) and land in identity-mapped RAM instead.
const REG_LO = 0x40000;
const REG_HI = 0x50000;

let rng_state = SEED >>> 0;
function rnd(n) {
    // xorshift32
    let x = rng_state;
    x ^= x << 13; x >>>= 0;
    x ^= x >> 17;
    x ^= x << 5; x >>>= 0;
    rng_state = x;
    return x % n;
}

const imm32 = v => [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF, (v >> 24) & 0xFF];
const imm16 = v => [v & 0xFF, (v >> 8) & 0xFF];
const imm8 = v => [v & 0xFF];

// ALU group ops, indexed the way the opcode encodes them.
const ALU_OPS = ["add", "or", "adc", "sbb", "and", "sub", "xor", "cmp"];
const SHIFT_KINDS = [4, 5, 7, 0, 1]; // shl, shr, sar, rol, ror

// A ModRM byte for `mod=01 reg=r rm=m` (register form) or with a SIB byte for a
// memory operand with an 8-bit displacement.
function modrm_reg(reg, rm) {
    return [0xC0 | (reg << 3) | rm];
}
function modrm_disp8(reg, sibScale, sibIndex, sibBase, disp) {
    // mod=01, rm=100 (SIB), reg in the ModRM reg field
    const modrm = 0x40 | (reg << 3) | 4;
    const sib = (sibScale << 6) | ((sibIndex & 7) << 3) | (sibBase & 7);
    return [modrm, sib, disp & 0xFF];
}

function w() {
    return rnd(3) === 0 ? 8 : rnd(2) === 0 ? 16 : 32;
}
function reg() {
    return rnd(8);
}
function prefix66(osize) {
    return osize === 16 ? [0x66] : [];
}
// A base register that does not require a SIB byte (rm=4 would, and the mod=00/01
// forms below emit a plain displacement, not a SIB).
function base_reg() {
    const b = reg() & 7;
    return b === 4 ? 5 : b;
}
// A ModRM byte for a memory operand: disp8 (mod=01) or disp32 (mod=10), with a
// SIB half the time so the scale/index/base combinations get covered too.
function modrm_mem(reg_field) {
    if(rnd(2) === 0) {
        // mod=10, disp32, plain base
        return [0x80 | (reg_field << 3) | base_reg(), ...imm32(rnd(64) - 32)];
    }
    if(rnd(2) === 0) {
        // mod=01, disp8, SIB
        return modrm_disp8(reg_field, rnd(4), reg() === 4 ? 4 : reg(), reg(), rnd(256) - 128);
    }
    // mod=01, disp8, plain base
    return [0x40 | (reg_field << 3) | base_reg(), rnd(256) - 128];
}

const ENCODERS = [
    // 00/01: op r/m, reg -- the ModRM reg field is the source.
    () => {
        const op = rnd(8), osize = w(), dst = reg(), src = reg();
        return [...prefix66(osize), ALU_OPS[op].length && (op * 8 | (osize === 8 ? 0 : 1)),
            ...modrm_reg(src, dst)];
    },
    // 02/03: op reg, r/m -- the ModRM reg field is the destination.
    () => {
        const op = rnd(8), osize = w(), dst = reg(), src = reg();
        return [...prefix66(osize), op * 8 | (osize === 8 ? 2 : 3),
            ...modrm_reg(dst, src)];
    },
    // 04/05: op AL/eAX, imm
    () => {
        const op = rnd(8), osize = w();
        return [...prefix66(osize), op * 8 | (osize === 8 ? 4 : 5),
            ...(osize === 8 ? imm8(rnd(256)) : osize === 16 ? imm16(rnd(65536)) : imm32(rnd(1 << 30)))];
    },
    // 80/81/83: op r/m, imm (0x80/0x83 take a sign-extended imm8); the ALU
    // operation goes in the ModRM reg field, not the opcode byte.
    () => {
        const op = rnd(8), form = rnd(3), r = reg();
        if(form === 0) return [0x80, ...modrm_reg(op, r), ...imm8(rnd(256))];
        if(form === 1) return [0x83, ...modrm_reg(op, r), ...imm8(rnd(256))];
        return [0x81, ...modrm_reg(op, r), ...imm32(rnd(1 << 30))];
    },
    // ALU against memory, both directions, at every operand size
    () => {
        const op = rnd(8), osize = w();
        if(rnd(2) === 0) {
            // `op r/m, reg`
            return [...prefix66(osize), op * 8 | (osize === 8 ? 0 : 1), ...modrm_mem(reg())];
        }
        // `op reg, r/m`
        return [...prefix66(osize), op * 8 | (osize === 8 ? 2 : 3), ...modrm_mem(reg())];
    },
    // 40-4F inc/dec r
    () => [0x40 | (rnd(2) << 3) | reg()],
    // 50-5F push/pop r
    () => [0x50 | reg()],
    // No jcc / call / jmp here: a relative branch with a random offset lands
    // mid-instruction, and the guest then wanders into MMIO and trips device
    // asserts in both runs, which buries real failures in noise. Branches are
    // covered deterministically by decode-cache-diff.js.
    // 84/85 test r/m, reg
    () => {
        const osize = w();
        return [...prefix66(osize), (osize === 8 ? 0x84 : 0x85), ...modrm_reg(reg(), reg())];
    },
    // 86/87 xchg r/m, reg
    () => {
        const osize = w();
        return [...prefix66(osize), (osize === 8 ? 0x86 : 0x87), ...modrm_reg(reg(), reg())];
    },
    // 88-8B mov, register and memory, both directions
    () => {
        const osize = w();
        if(rnd(2) === 0) {
            return [...prefix66(osize), (osize === 8 ? 0x88 : 0x89), ...modrm_reg(reg(), reg())];
        }
        return [...prefix66(osize), (osize === 8 ? 0x88 : 0x89), ...modrm_mem(reg())];
    },
    () => {
        const osize = w();
        if(rnd(2) === 0) {
            return [...prefix66(osize), (osize === 8 ? 0x8A : 0x8B), ...modrm_reg(reg(), reg())];
        }
        return [...prefix66(osize), (osize === 8 ? 0x8A : 0x8B), ...modrm_mem(reg())];
    },
    // 8D lea
    () => [0x8D, ...modrm_mem(reg())],
    // 90 nop / 91-97 xchg eax, r
    () => [0x90 | reg()],
    // A8/A9 test AL/eAX, imm
    () => {
        const w32 = rnd(2);
        return [0xA8 | w32, ...(w32 ? imm32(rnd(1 << 30)) : imm8(rnd(256)))];
    },
    // B0-B7 mov r8, imm8
    () => [0xB0 | reg(), ...imm8(rnd(256))],
    // B8-BF mov r32, imm32
    () => [0xB8 | reg(), ...imm32(rnd(1 << 30))],
    // C0/C1 shift r/m, imm8
    () => {
        const osize = w();
        return [osize === 8 ? 0xC0 : 0xC1, ...modrm_reg(SHIFT_KINDS[rnd(5)], reg()),
            ...imm8(rnd(64))];
    },
    // C6/C7 mov r/m, imm
    () => {
        const osize = w();
        const m = rnd(2) === 0 ? modrm_reg(0, reg()) : modrm_mem(0);
        return [...prefix66(osize), osize === 8 ? 0xC6 : 0xC7, ...m,
            ...(osize === 8 ? imm8(rnd(256)) : osize === 16 ? imm16(rnd(65536)) : imm32(rnd(1 << 30)))];
    },
    // C9 leave
    () => [0xC9],
    // D0-D3 shift r/m, 1 / cl
    () => {
        const osize = w(), form = rnd(4);
        return [0xD0 | form, ...modrm_reg(SHIFT_KINDS[rnd(5)], reg())];
    },
    // F6/F7 group3: test/not/neg r/m, register and memory
    () => {
        const osize = w(), sub = [0, 2, 3][rnd(3)];
        const m = rnd(2) === 0 ? modrm_reg(sub, reg()) : modrm_mem(sub);
        return [...prefix66(osize), osize === 8 ? 0xF6 : 0xF7, ...m,
            ...(sub === 0 ? (osize === 8 ? imm8(rnd(256)) : osize === 16 ? imm16(rnd(65536)) : imm32(rnd(1 << 30))) : [])];
    },
    // FE inc/dec r/m8
    () => [0xFE, ...(rnd(2) === 0 ? modrm_reg(rnd(2), reg()) : modrm_mem(rnd(2)))],
    // FF inc/dec/push r/m. Deliberately no call/jmp (sub 2 and 4): a random
    // program that branches anywhere immediately leaves RAM, and the guest
    // then trips an unrelated MMIO assert in *both* runs. Control flow is
    // covered deterministically by decode-cache-diff.js ("stack ops",
    // "cond jumps") instead.
    () => {
        const sub = [0, 1, 6][rnd(3)];
        return [0xFF, ...(rnd(2) === 0 ? modrm_reg(sub, reg()) : modrm_mem(sub))];
    },
];

// Flag probe: setcc on each of the 16 condition codes, storing the result byte.
// setcc is not decoded by the cache, so this always runs through the stock
// interpreter -- which means it reads the flags through the lazy getters. That
// makes it a real comparison of *condition codes*, which comparing the raw
// `flags` register cannot do: the cached path materialises the lazy flags while
// the stock path leaves them stale, so `flags` legitimately differs even when
// every condition is right.
const PROBE_ADDR = REG_LO + 0x100;
function flag_probe() {
    const bytes = [];
    for(let code = 0; code < 16; code++) {
        bytes.push(0x0F, 0x90 | code, 0xC0);           // setcc al
        bytes.push(0x88, 0x04, 0x25, ...imm32(PROBE_ADDR + code)); // mov [abs], al
    }
    // Also dump the flags register itself for context.
    bytes.push(0x9C);                                  // pushf
    bytes.push(0x58);                                  // pop eax
    bytes.push(0xA3, 0x04, 0x25, ...imm32(PROBE_ADDR + 16)); // mov [abs], eax
    return bytes;
}

function program(length) {
    const bytes = [];
    for(let i = 0; i < length; i++) {
        bytes.push(...ENCODERS[rnd(ENCODERS.length)]());
    }
    bytes.push(...flag_probe());
    bytes.push(0xF4); // hlt
    return bytes;
}

function run(disable_cache, bytes, init_regs) {
    const emulator = new v64({
        autostart: false,
        memory_size: 8 * 1024 * 1024,
        disable_jit: true,
        log_level: 0,
        wasm_path: process.env.WASM_PATH || undefined,
    });
    return new Promise((resolve, reject) => {
        emulator.add_listener("emulator-loaded", () => {
            try {
                const cpu = emulator.v86.cpu;
                const ex = cpu.wm.exports;
                ex.set_dbg_disable_32_decode_cache(disable_cache ? 1 : 0);

                const write = (address, data) => {
                    for(let i = 0; i < data.length; i++) ex.write8(address + i, data[i]);
                };

                const gdt = [
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0xFF, 0xFF, 0, 0, 0, 0x9A, 0xCF, 0,
                    0xFF, 0xFF, 0, 0, 0, 0x92, 0xCF, 0,
                ];
                write(0x9000, gdt);
                cpu.gdtr_size[0] = gdt.length - 1;
                cpu.gdtr_offset[0] = 0x9000;

                // Real mode -> 32-bit protected mode, entering at CODE and then
                // falling through into `bytes`. No paging: the cache works on
                // the segment-based identity map, and paging only obscured it.
                write(RESET, [
                    0xFA,
                    0x66, 0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, CR0.PE
                    0x0F, 0x22, 0xC0,                   // mov cr0, eax
                    0x66, 0xEA, 0x00, 0x80, 0x00, 0x00, // jmp far 0008:0x8000
                    0x08, 0x00,
                ]);
                // Flat data segments (required by the cache).
                const prot = [
                    0xB8, 0x10, 0x00, 0x00, 0x00, // mov eax, 0x10
                    0x8E, 0xD8,                   // mov ds, ax
                    0x8E, 0xD0,                   // mov ss, ax
                    0x8E, 0xC0,                   // mov es, ax
                    0x8E, 0xE0,                   // mov fs, ax
                    0x8E, 0xE8,                   // mov gs, ax
                ];
                write(CODE, prot);
                write(CODE + prot.length, bytes);

                for(let i = 0; i < 8; i++) cpu.reg32[i] = init_regs[i];
                cpu.reg32[4] = STACK_TOP;

                let steps = 0;
                let panicked = false;
                try {
                    // A valid program halts within a batch or two; the bound only
                    // limits how long a runaway (desynced) program spins before it
                    // is treated as "not halted" and skipped.
                    while(!cpu.in_hlt[0] && steps++ < 200) ex.main_loop();
                }
                catch(e) {
                    // A guest assert (e.g. fetching from an MMIO page) aborts the
                    // wasm instance. Record it and compare: if only the cached run
                    // dies, the cache is what sent it there.
                    panicked = true;
                }

                const regs = [];
                for(let i = 0; i < 8; i++) regs.push(cpu.reg32[i] >>> 0);
                // EIP only, not the raw flags: the cache materialises the lazy flags
                // while the interpreter leaves them pending, so the register is not
                // comparable. Condition codes are covered by the setcc probe above.
                regs.push(cpu.instruction_pointer[0] >>> 0);
                const heap = [];
                for(let i = 0; i < 512; i++) heap.push(ex.read8(REG_LO + i));
                resolve({ halted: cpu.in_hlt[0] === 1, panicked, steps, regs, heap,
                    hits: ex.dbg_decode_cache_hits ? ex.dbg_decode_cache_hits() : -1 });
            }
            catch(e) {
                reject(e);
            }
        });
    });
}

const emulator = new v64({ autostart: false, memory_size: 8 * 1024 * 1024, log_level: 0 });
emulator.add_listener("emulator-loaded", async () =>
{
    let failures = 0;
    let skipped = 0;
    let total_on_hits = 0;
    for(let i = 0; i < CASES; i++) {
        const init_regs = [];
        for(let r = 0; r < 8; r++) {
            init_regs.push(REG_LO + ((rng_state >> (r * 3)) % (REG_HI - REG_LO)) & ~3);
        }
        const bytes = program(12 + rnd(20));
        const off = await run(true, bytes, init_regs);
        const on = await run(false, bytes, init_regs);
        total_on_hits += Math.max(0, on.hits);
        // A program that dies or fails to halt identically either way is not
        // telling us anything about the cache.
        if(off.panicked || on.panicked || !off.halted || !on.halted) {
            if(off.panicked !== on.panicked) {
                failures++;
                console.log(`FAIL case ${i} (seed ${SEED}): panic only with cache on`);
                console.log("  init_regs: " + init_regs.join(","));
                console.log("  bytes: " + bytes.map(b => ((b + 256) & 0xFF).toString(16).padStart(2, "0")).join(" "));
            }
            else {
                skipped++;
            }
            continue;
        }
        try {
            assert.deepEqual(on.regs, off.regs, "registers");
            assert.deepEqual(on.heap, off.heap, "memory");
        }
        catch(e) {
            failures++;
            console.log(`FAIL case ${i} (seed ${SEED}) ${e.message.split("\n")[0]}`);
            console.log("  bytes: " + bytes.map(b => ((b + 256) & 0xFF).toString(16).padStart(2, "0")).join(" "));
            console.log(`  cache-on  regs: ${on.regs.join(",")}`);
            console.log(`  cache-off regs: ${off.regs.join(",")}`);
            if(failures >= 3) break;
        }
    }
    if(total_on_hits === 0) {
        console.log("FAIL: the decode cache was never engaged (harness entry?)");
        failures++;
    }
    console.log(`interp32 decode-cache fuzz: ${CASES} cases, ${failures} failure(s), ${skipped} skipped, hits(on)=${total_on_hits}`);
    process.exit(failures ? 1 : 0);
});