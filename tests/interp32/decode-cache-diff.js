#!/usr/bin/env node

// Differential test for the 32-bit interpreter's decode cache.
//
// Runs the same 32-bit guest programs twice -- once with the cache engaged and
// once with it disabled -- and requires identical CPU and memory state. A
// decode-cache entry that decodes or executes an instruction differently from
// `run_instruction` shows up here as a state mismatch.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/interp32/decode-cache-diff.js`

import assert from "node:assert/strict";
import { v64 } from "../../src/main.js";

const RESET = 0xFFFF0;
const CODE = 0x2000;
const SCRATCH = 0x8000;
const STACK_TOP = 0x30000;

const imm32 = v => [v & 0xFF, v >> 8 & 0xFF, v >> 16 & 0xFF, v >> 24 & 0xFF];
const imm8 = v => [v & 0xFF];

// mov ecx, imm32; loop: <body>; dec ecx; jnz loop; hlt
const loop = (count, body) => [
    0xB9, ...imm32(count),
    ...body,
    0x49,                               // dec ecx
    0x75, (0x100 - (body.length + 3)) & 0xFF, // jnz back to body (dec + jnz = 3 bytes)
    0xF4,
];

// ALU group 0x00-0x3F: `op r/m, reg` and `op reg, r/m`, reg and reg.
const alu_reg_reg = [
    0xB8, ...imm32(0x12345678),         // mov eax, 0x12345678
    0xBB, ...imm32(0x0F0F0F0F),         // mov ebx, 0x0F0F0F0F
    0x31, 0xD8,                         // xor eax, ebx   (01 /r form, reg=ebx)
    0x33, 0xC3,                         // xor ebx, eax   (03 /r form, reg=eax)
    0x29, 0xC8,                         // sub eax, ecx
    0x2B, 0xD8,                         // sub ebx, eax
    0x01, 0xD9,                         // add ecx, ebx
    0x39, 0xC8,                         // cmp eax, ecx
    0xF4,
];

const alu_imm = [
    0xB8, ...imm32(0),                  // mov eax, 0
    0x83, 0xC0, 0xFF,                   // add eax, -1      (imm8 sign-extended)
    0x83, 0xE8, 0x7F,                   // sub eax, 0x7f
    0x81, 0xC0, ...imm32(0x0000FFFF),   // add eax, 0xffff
    0x80, 0xC4, 0x80,                   // add ah, -0x80
    0x05, ...imm32(0x00010000),         // add eax, 0x10000
    0xF4,
];

const test_xchg = [
    0xB8, ...imm32(0xAABBCCDD),
    0xBB, ...imm32(0x11223344),
    0x85, 0xC3,                         // test ebx, eax
    0x87, 0xD8,                         // xchg eax, ebx
    0x84, 0xC4,                         // test ah, al
    0x86, 0xE0,                         // xchg al, ah
    0xF4,
];

// push/pop segment registers (0x06/0x07/0x0E), which share the 0x00-0x3F range
// with the ALU group but are not arithmetic.
const push_pop_seg = [
    0x06,                               // push es
    0x0E,                               // push cs
    0x07,                               // pop es
    0x17,                               // pop ss
    0x06, 0x07,
    0x1E, 0x1F,                         // push/pop ds
    0xF4,
];

const lea_load_store = [
    0xB8, ...imm32(SCRATCH),
    0xB9, ...imm32(0x11223344),
    0x89, 0x08,                         // mov [eax], ecx
    0x8B, 0x11,                         // mov edx, [ecx]
    0x8D, 0x74, 0x08,                   // lea esi, [eax+ecx]
    0x8B, 0x06,                         // mov eax, [esi]
    0xC6, 0x00, 0x5A,                   // mov byte [eax], 0x5a
    0x88, 0x5E, 0x01,                   // mov [esi+1], bl
    0x8B, 0x4D, 0xFC,                   // mov ecx, [ebp-4]
    0x89, 0x45, 0xFC,                   // mov [ebp-4], eax
    0xF4,
];

const shifts = [
    0xB8, ...imm32(0x80000001),
    0xB9, ...imm32(0x0000FFFF),
    0xC1, 0xE0, 0x04,                   // shl eax, 4
    0xD1, 0xF8,                         // sar eax, 1
    0xC1, 0xE9, 0x1F,                   // shr ecx, 31
    0xD3, 0xE0,                         // shl eax, cl
    0xC0, 0xE9, 0x11,                   // shr ecx, 17
    0xF4,
];

const stack_ops = [
    0xB8, ...imm32(0xDEADBEEF),
    0x50,                               // push eax
    0xBB, ...imm32(0x0BADF00D),
    0x53,                               // push ebx
    0x5B,                               // pop ebx
    0x58,                               // pop eax
    0x55,                               // push ebp
    0x5D,                               // pop ebp
    0xE8, 0x00, 0x00, 0x00, 0x00,       // call $+5
    0x5B,                               // pop ebx
    0xC9,                               // leave
    0xF4,
];

const cond_jumps = [
    0xB8, ...imm32(5),
    0x83, 0xF8, 0x05,                   // cmp eax, 5
    0x74, 0x06,                         // je +6
    0xB9, ...imm32(0xBAD0BAD0),         // (skipped)
    0xB9, ...imm32(0x600D5EED),         // taken path
    0x7C, 0x02,                         // jl (not taken)
    0x90,                               // nop
    0xF4,
];

// Straight-line run longer than the per-block instruction cap.
const long_run = loop(200, [
    0x01, 0xD8,                         // add eax, ebx
    0x31, 0xC3,                         // xor ebx, eax
    0xD1, 0xE3,                         // shl ebx, 1
]);

const PROGRAMS = {
    "alu reg/reg": alu_reg_reg,
    "alu imm": alu_imm,
    "test/xchg": test_xchg,
    "push/pop seg": push_pop_seg,
    "lea/load/store": lea_load_store,
    "shifts": shifts,
    "stack ops": stack_ops,
    "cond jumps": cond_jumps,
    "long straight-line run": long_run,
    // Operand-size-16 stack ops (0x66) must push/pop 16 bits, not 32.
    "operand size 16 stack": [
        0xB8, ...imm32(0x12345678),         // mov eax, 0x12345678
        0xBB, ...imm32(0x0000ABCD),         // mov ebx, 0x0000abcd
        0x66, 0x50,                         // push ax
        0x66, 0x53,                         // push bx
        0x66, 0x58,                         // pop ax
        0x66, 0x5B,                         // pop bx
        0xF4,
    ],
    // 16-bit addressing (0x67): the EA uses only the low 16 bits of EBX.
    "address size 16": [
        0xBB, 0x03, 0x80, 0x01, 0x00,       // mov ebx, 0x00018003
        0x67, 0x8B, 0x03,                   // mov eax, [bx]  -> [0x8003]
        0xF4,
    ],
    // 16-bit addressing wraps at 64 KiB: (0xffff + 0x8004) & 0xffff = 0x8003
    "address size 16 wrap": [
        0xBB, 0xFF, 0xFF, 0x01, 0x00,       // mov ebx, 0x0001ffff
        0xBE, 0x04, 0x80, 0x01, 0x00,       // mov esi, 0x00018004
        0x67, 0x8B, 0x00,                   // mov eax, [bx+si] -> [0x8003]
        0xF4,
    ],
};

function run(disable_cache, bytes) {
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

                // Flat GDT, 32-bit, base 0.
                const gdt = [
                    0, 0, 0, 0, 0, 0, 0, 0,
                    0xFF, 0xFF, 0, 0, 0, 0x9A, 0xCF, 0, // 0x08 code, base 0, D=1
                    0xFF, 0xFF, 0, 0, 0, 0x92, 0xCF, 0, // 0x10 data, base 0, D=1
                ];
                write(0x5000, gdt);
                cpu.gdtr_size[0] = gdt.length - 1;
                cpu.gdtr_offset[0] = 0x5000;

                // Real mode -> 32-bit protected mode, then straight into the
                // program. No paging: the cache works on the segment-based
                // identity map, and enabling it only hid the cache behind a
                // mistyped CR3.
                write(RESET, [
                    0xFA,                               // cli
                    0x66, 0xB8, 0x01, 0x00, 0x00, 0x00, // mov eax, CR0.PE
                    0x0F, 0x22, 0xC0,                   // mov cr0, eax
                    0x66, 0xEA, 0x00, 0x20, 0x00, 0x00, // jmp far 0008:0x2000
                    0x08, 0x00,
                ]);
                // Load flat data segments (the cache requires flat segmentation)
                // and fall through into `bytes`.
                const PROT = [
                    0xB8, 0x10, 0x00, 0x00, 0x00, // mov eax, 0x10
                    0x8E, 0xD8,                   // mov ds, ax
                    0x8E, 0xD0,                   // mov ss, ax
                    0x8E, 0xC0,                   // mov es, ax
                ];
                write(CODE, PROT);
                write(CODE + PROT.length, bytes);
                write(CODE + PROT.length + bytes.length, [0xF4]);

                // Stack and a scratch buffer the programs poke at.
                for(let i = 0; i < 256; i++) ex.write8(SCRATCH + i, (i * 7) & 0xFF);
                cpu.reg32[4] = STACK_TOP;
                cpu.reg32[5] = STACK_TOP; // EBP: programs address [ebp-4]

                let steps = 0;
                while(!cpu.in_hlt[0] && steps++ < 200000) ex.main_loop();

                const regs = [];
                for(let i = 0; i < 8; i++) regs.push(cpu.reg32[i] >>> 0);
                // EIP only: the raw `flags` register is not comparable across the
                // two paths, because the cache materialises the interpreter's lazy
                // flags while the interpreter leaves them pending (see the fuzz
                // test's setcc probe for condition-code coverage).
                regs.push(cpu.instruction_pointer[0] >>> 0);
                const scratch = [];
                for(let i = 0; i < 256; i++) scratch.push(ex.read8(SCRATCH + i));
                const stack = [];
                for(let i = 0; i < 64; i++) stack.push(ex.read8(STACK_TOP - 64 + i));

                resolve({
                    halted: cpu.in_hlt[0] === 1,
                    steps,
                    regs,
                    scratch,
                    stack,
                    hits: ex.dbg_decode_cache_hits ? ex.dbg_decode_cache_hits() : -1,
                });
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
    for(const [name, bytes] of Object.entries(PROGRAMS))
    {
        const off = await run(true, bytes);
        const on = await run(false, bytes);

        try
        {
            assert.equal(on.halted, true, "halted");
            assert.equal(on.halted, off.halted, "halted (cache off)");
            assert.ok(on.hits > 0, "cache actually engaged");
            assert.deepEqual(on.regs, off.regs, "registers");
            assert.deepEqual(on.scratch, off.scratch, "scratch memory");
            assert.deepEqual(on.stack, off.stack, "stack");
            console.log(`ok   ${name}`);
        }
        catch(e)
        {
            failures++;
            console.log(`FAIL ${name}: ${e.message.split("\n")[0]}`);
            if(Array.isArray(e.expected))
            {
                console.log(`  regs  cache-on  = ${e.actual.join(",")}`);
                console.log(`  regs  cache-off = ${e.expected.join(",")}`);
            }
            console.log(`  halted on=${on.halted} off=${off.halted} steps on=${on.steps} off=${off.steps}`);
        }
    }
    console.log(`interp32 decode-cache diff: ${failures} failure(s)`);
    process.exit(failures ? 1 : 0);
});