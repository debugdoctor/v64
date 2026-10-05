#!/usr/bin/env node

import fs from "node:fs";
import fse from "node:fs/promises";
import path from "node:path";
import assert from "node:assert/strict";
import util from "node:util";
import url from "node:url";
import { execFile } from "node:child_process";


import Rand from "./rand.js";

const __dirname = url.fileURLToPath(new URL(".", import.meta.url));

// arithmetic tests
const NUMBER_ARITH_TESTS = 100;

const MAX_PARALLEL_PROCS = +process.env.MAX_PARALLEL_PROCS || 32;

const FLAGS_IGNORE = 0xFFFF3200;
const CF = 1 << 0;
const PF = 1 << 2;
const AF = 1 << 4;
const ZF = 1 << 6;
const SF = 1 << 7;
const OF = 1 << 11;

const BUILD_DIR = __dirname + "/build/";
const LOG_VERBOSE = false;

const exec_file = util.promisify(execFile);

const header = fs.readFileSync(path.join(__dirname, "header.inc"));
const footer = fs.readFileSync(path.join(__dirname, "footer.inc"));

main();

async function main()
{
    try
    {
        fs.mkdirSync(BUILD_DIR);
    }
    catch(e)
    {
        if(e.code !== "EEXIST")
        {
            throw e;
        }
    }

    const tests = create_tests().reverse();

    const workers = [];
    for(let i = 0; i < MAX_PARALLEL_PROCS; i++)
    {
        workers.push(worker(make_test, tests));
    }

    await Promise.all(workers);
}

async function worker(f, work)
{
    while(work.length)
    {
        await f(work.pop());
    }
}

async function make_test(test)
{
    LOG_VERBOSE && console.log("Start", test.name || test.file);
    let asm_file;
    let img_file;
    let tmp_file;

    assert((test.asm && test.name) || test.file);
    if(test.asm)
    {
        asm_file = BUILD_DIR + test.name + ".asm";
        img_file = BUILD_DIR + test.name + ".img";
        tmp_file = "/tmp/" + test.name + ".o";

        let old_code = undefined;

        try
        {
            old_code = await fse.readFile(asm_file, { encoding: "ascii" });
        }
        catch(e)
        {
        }

        if(old_code === test.asm)
        {
            LOG_VERBOSE && console.log("Skip", test.name || test.file);
            return;
        }

        await fse.writeFile(asm_file, test.asm);
    }
    else
    {
        asm_file = path.join(__dirname, test.file);
        img_file = BUILD_DIR + test.file.replace(/\.asm$/, ".img");
        tmp_file = "/tmp/" + test.file + ".o";

        try
        {
            if((await fse.stat(asm_file)).mtime < (await fse.stat(img_file)).mtime)
            {
                return;
            }
        }
        catch(e)
        {
            if(e.code !== "ENOENT") throw e;
        }
    }

    const options = {
        cwd: __dirname,
    };

    LOG_VERBOSE && console.log("nasm", ["-w+error", "-felf32", "-o", tmp_file, asm_file].join(" "));
    await exec_file("nasm", ["-w+error", "-felf32", "-o", tmp_file, asm_file], options);
    LOG_VERBOSE && console.log("ld", ["-g", tmp_file, "-m", "elf_i386", "--section-start=.bss=0x100000", "--section-start=.text=0x80000", "--section-start=.multiboot=0x20000", "-o", img_file].join(" "));
    await exec_file("ld", ["-g", tmp_file, "-m", "elf_i386", "--section-start=.bss=0x100000", "--section-start=.text=0x80000", "--section-start=.multiboot=0x20000", "-o", img_file], options);
    await fse.unlink(tmp_file);

    console.log(test.name || test.file);
}

function create_tests()
{
    const tests = [];

    const asm_files = fs.readdirSync(__dirname).filter(f => f.endsWith(".asm"));
    tests.push.apply(tests, asm_files.map(file => ({ file })));

    for(let i = 0; i < NUMBER_ARITH_TESTS; i++)
    {
        tests.push(create_arith_test(i));
    }

    return tests;
}

function rand_reg_but_not_esp(rng)
{
    let r = rng.int32() & 7;
    return r === 4 ? rand_reg_but_not_esp(rng) : r;
}

function interesting_immediate(rng)
{
    if(rng.int32() & 1)
    {
        return rng.int32();
    }
    else
    {
        return rng.int32() << (rng.int32() & 31) >> (rng.int32() & 31);
    }
}

function create_arith_test(i)
{
    const rng = new Rand(916237867 ^ i);

    const registers_by_size = {
        8: ["al", "ah", "cl", "ch", "dl", "dh", "bl", "bh"],
        16: ["ax", "cx", "dx", "bx", "sp", "bp", "si", "di"],
        32: ["eax", "ecx", "edx", "ebx", "esp", "ebp", "esi", "edi"],
    };
    const mask_by_size = {
        8: 0xFF,
        16: 0xFFFF,
        32: -1,
    };
    const word_by_size = {
        8: "byte",
        16: "word",
        32: "dword",
    };
    const two_operand_instructions = ["add", "sub", "adc", "sbb", "and", "or", "xor", "cmp", "test"];
    const one_operand_instructions = [
        "inc", "dec", "neg",
        "mul", //"idiv", "div", // technically also eax:edx, but are implied by assembler
        "imul", // handled specifically below to also generate 2-/3-operand form
    ];
    const shift_instructions = ["shl", "shr", "sar", "rol", "ror", "rcl", "rcr"];
    // TODO: cmpxchg, xadd, bsf, bsr, shrd/shld, popcnt, bt*
    const instructions = [two_operand_instructions, one_operand_instructions, shift_instructions].flat();
    const conditions = [
        // suffix flag
        ["o", OF],
        ["c", CF],
        ["z", ZF],
        ["p", PF],
        ["s", SF],
        ["be", CF | ZF],
        ["l", SF | OF],
        ["le", SF | OF | ZF],
    ];

    let c = [];
    let address = 0x100000;

    for(let reg of registers_by_size[32])
    {
        if(reg !== "esp")
        {
            c.push(`mov ${reg}, ${interesting_immediate(rng)}`);
        }
    }

    let undefined_flags = 0;

    for(let i = 0; i < 2000; i++)
    {
        const ins = instructions[rng.uint32() % instructions.length];
        const size = [8, 16, 32][rng.uint32() % 3];
        const size_word = word_by_size[size];
        const dst_is_mem = rng.int32() & 1;
        const dst = dst_is_mem ?
            `${size_word} [${nasm_hex(address)}]` :
            registers_by_size[size][rand_reg_but_not_esp(rng)];
        let src_is_mem = false;
        if(ins === "imul" && (rng.int32() & 1)) // other encodings handled in one_operand_instructions
        {
            // dst must be reg, no 8-bit
            const size_imul = [16, 32][rng.int32() & 1];
            const dst_imul = registers_by_size[size_imul][rand_reg_but_not_esp(rng)];
            const src1 = dst_is_mem ?
                `${word_by_size[size_imul]} [${nasm_hex(address)}]` :
                registers_by_size[size_imul][rand_reg_but_not_esp(rng)];
            if(rng.int32() & 1)
            {
                c.push(`${ins} ${dst_imul}, ${src1}`);
            }
            else
            {
                const src2 = nasm_hex(interesting_immediate(rng) & mask_by_size[size_imul]);
                c.push(`${ins} ${dst_imul}, ${src1}, ${src2}`);
            }
        }
        else if(one_operand_instructions.includes(ins))
        {
            c.push(`${ins} ${dst}`);
        }
        else if(two_operand_instructions.includes(ins))
        {
            src_is_mem = !dst_is_mem && (rng.int32() & 1);
            const src = src_is_mem ?
                `${size_word} [${nasm_hex(address)}]` :
                (rng.int32() & 1) ?
                registers_by_size[size][rand_reg_but_not_esp(rng)] :
                nasm_hex(interesting_immediate(rng) & mask_by_size[size]);
            c.push(`${ins} ${dst}, ${src}`);
        }
        else if(shift_instructions.includes(ins))
        {
            if(rng.int32() & 1)
            {
                // unknown CL
                undefined_flags |= AF | OF;
                c.push(`${ins} ${dst}, cl`);
            }
            else
            {
                const shift = interesting_immediate(rng) & 0xFF;
                // TODO: shift mod {8,9,16,17,32,33} depending on bitsize/rotate/with-carry, shifts can clear undefined_flags if shift is not zero
                undefined_flags |= shift === 1 ? AF : AF | OF;
                if(rng.int32() & 1)
                {
                    // known CL
                    c.push(`mov cl, ${nasm_hex(shift)}`);
                    c.push(`${ins} ${dst}, cl`);
                }
                else
                {
                    // immediate
                    c.push(`${ins} ${dst}, ${nasm_hex(shift)}`);
                }
            }
        }

        if(dst_is_mem || src_is_mem)
        {
            if(rng.int32() & 1)
            {
                address += size / 8;
                // initialise next word
                c.push(`mov dword [${nasm_hex(address)}], ${nasm_hex(interesting_immediate(rng) & 0xFF)}`);
            }
        }

        if(ins === "imul" || ins === "mul" || ins === "idiv" || ins === "div")
        {
            undefined_flags = SF | ZF | AF | PF;
        }
        else if(!shift_instructions.includes(ins))
        {
            // adc/sbb/inc/dec read CF, but CF is never undefined
            undefined_flags = 0;
        }

        if(rng.int32() & 1)
        {
            // setcc
            const cond = random_pick(conditions.filter(([_, flag]) => 0 === (flag & undefined_flags)).map(([suffix]) => suffix), rng);
            assert(cond);
            const invert = (rng.int32() & 1) ? "n" : "";
            const ins2 = `set${invert}${cond}`;
            const dst2 = (rng.int32() & 1) ? `byte [${nasm_hex(address++)}]` : registers_by_size[8][rng.int32() & 7];
            c.push(`${ins2} ${dst2}`);
        }
        else if(rng.int32() & 1)
        {
            // cmovcc
            const cond = random_pick(conditions.filter(([_, flag]) => 0 === (flag & undefined_flags)).map(([suffix]) => suffix), rng);
            assert(cond);
            const invert = (rng.int32() & 1) ? "n" : "";
            const ins2 = `cmov${invert}${cond}`;
            const size = (rng.int32() & 1) ? 16 : 32;
            const src2 = registers_by_size[size][rng.int32() & 7];
            const dst2 = registers_by_size[size][rand_reg_but_not_esp(rng)];
            c.push(`${ins2} ${dst2}, ${src2}`);
        }
        else if(rng.int32() & 1)
        {
            c.push("pushf");
            c.push("and dword [esp], ~" + nasm_hex(FLAGS_IGNORE | undefined_flags));
            c.push(`pop ${registers_by_size[32][rand_reg_but_not_esp(rng)]}`);
        }
        else
        {
            // intentionally left blank
        }

        // TODO:
        // cmovcc
        // other random instructions (mov, etc.)
    }

    c.push("pushf");
    c.push("and dword [esp], ~" + nasm_hex(FLAGS_IGNORE | undefined_flags));
    c.push("popf");

    assert(address < 0x102000);

    const name = `arith_${i}`;
    const asm = header + c.join("\n") + "\n" + footer;

    return { name, asm };
}

function nasm_hex(x)
{
    return `0${(x >>> 0).toString(16).toUpperCase()}h`;
}

function random_pick(xs, rng)
{
    return xs[rng.uint32() % xs.length];
}
