#!/usr/bin/env node

// Roadmap M2: a 64-bit user program runs after a direct boot, and the same
// program keeps working once its hot loop has been JIT-compiled.
//
// The kernel enables SYSCALL, points STAR/LSTAR at a write handler and SYSRETs
// to ring 3. The user program adds one to a counter until it has counted 600
// (hot enough for the JIT) and then asks the kernel to print the digits. It
// has no port access. The test reads COM1.
//
// Requires a debug wasm build: `make build/v64-debug.wasm`
// Run with: `node tests/e2e/user.js`

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { v64 } from "../../src/main.js";

const LOOPS = 600;
const MESSAGE = LOOPS + "\n";
const ENTRY = 0x100000 + 0x200;
const USER_STACK = 0x9E000;

const code = [];
const emit = (...bytes) => code.push(...bytes);
const imm32 = value => [value & 0xFF, value >> 8 & 0xFF, value >> 16 & 0xFF, value >> 24 & 0xFF];
const imm64 = value => [...imm32(value), 0, 0, 0, 0];

function wrmsr(msr, value)
{
    emit(0xB9, ...imm32(Number(msr)));
    emit(0xB8, ...imm32(Number(value & 0xFFFF_FFFFn)));
    emit(0xBA, ...imm32(Number((value >> 32n) & 0xFFFF_FFFFn)));
    emit(0x0F, 0x30);
}

// STAR: kernel CS base 0x10, user CS base 0x30. LSTAR is patched below.
wrmsr(0xC0000080n, 1n);
wrmsr(0xC0000081n, (0x30n << 48n) | (0x10n << 32n));
const lstarAt = code.length;
wrmsr(0xC0000082n, 0n);
wrmsr(0xC0000084n, 0n);

const userRipAt = code.length;
emit(0x48, 0xB9, ...imm64(0));                  // mov rcx, user (patched)
emit(0x49, 0xC7, 0xC3, 0x02, 0x00, 0x00, 0x00); // mov r11, 2
emit(0x48, 0xBC, ...imm64(USER_STACK));         // mov rsp, user stack
emit(0x0F, 0x07);                               // sysret
emit(0xF4);

// SYSCALL handler: the byte is in AL. Write it to COM1 and return.
const handler = ENTRY + code.length;
code.splice(lstarAt + 6, 4, ...imm32(handler));
emit(0xBA, 0xFD, 0x03, 0x00, 0x00); // mov edx, 0x3FD
emit(0x86, 0xC3);                   // xchg al, bl
emit(0xEC, 0xA8, 0x20, 0x74, 0xFB); // in al, dx; test al, 0x20; jz wait
emit(0xBA, 0xF8, 0x03, 0x00, 0x00); // mov edx, 0x3F8
emit(0x88, 0xD8);                   // mov al, bl
emit(0xEE);                         // out dx, al
emit(0x0F, 0x07);                   // sysret

const user = ENTRY + code.length;
code.splice(userRipAt + 2, 8, ...imm64(user));

const userCode = [];
const userEmit = (...bytes) => userCode.push(...bytes);
userEmit(0x31, 0xC0);                  // xor eax, eax
userEmit(0xB9, ...imm32(LOOPS));       // mov ecx, loops
const loopAt = userCode.length;
userEmit(0x83, 0xC0, 0x01);            // add eax, 1
userEmit(0xFF, 0xC9);                  // dec ecx
userEmit(0x75, (loopAt - (userCode.length + 2)) & 0xFF); // jnz loop, from the next instruction
for(const byte of Buffer.from(MESSAGE))
{
    userEmit(0xB0, byte);              // mov al, digit
    userEmit(0x0F, 0x05);              // syscall
}
userEmit(0xF4);
code.push(...userCode);

function bzImage(body)
{
    const setupSects = 4;
    const protStart = (setupSects + 1) * 512;
    const image = new Uint8Array((protStart + 0x200 + body.length + 511) & ~511);
    image[0x1F1] = setupSects;
    image[0x1FE] = 0x55;
    image[0x1FF] = 0xAA;
    image[0x201] = 0x40;
    image.set([0x48, 0x64, 0x72, 0x53], 0x202);
    image[0x206] = 0x0C;
    image[0x207] = 0x02;
    image[0x238] = 0xFF;
    image.set(body, protStart + 0x200);
    return image;
}

const imagePath = path.join(os.tmpdir(), "v64-e2e-user.img");
fs.writeFileSync(imagePath, bzImage(code));

const serial = [];
const emulator = new v64({
    autostart: false,
    memory_size: 8 * 1024 * 1024,
    log_level: 0,
    bzimage: { url: imagePath, async: false },
    cmdline: "",
    direct_boot: true,
    wasm_path: process.env.WASM_PATH || undefined,
});

emulator.add_listener("serial0-output-byte", byte => serial.push(byte));

emulator.add_listener("emulator-loaded", () =>
{
    const cpu = emulator.v86.cpu;
    let steps = 0;
    while(!cpu.in_hlt[0] && steps++ < 100000)
    {
        cpu.wm.exports.main_loop();
    }

    const text = Buffer.from(serial).toString("utf8");
    assert.equal(cpu.in_hlt[0], 1, "user program halted");
    assert.equal(text, MESSAGE, "the user program's writes arrived on COM1");
    assert.ok(cpu.wm.exports.jit64_compiled_count() > 0, "the hot loop was compiled");

    console.log("e2e user: test passed");
    console.log(text.trimEnd());
    process.exit(0);
});
