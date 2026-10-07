// Check CPUID advertisements and test wiring in both directions.
//
// Tiers
// -----
//   direct   a reference-vector test exercises the feature's instructions.
//            The file must exist and a runner must invoke it.
//   indirect the feature is executed by every guest boot, so a wrong
//            implementation breaks the e2e/CD boot tests immediately.
//   gap      acknowledged as untested: printed, and a failure under --strict.
//
// Run with: `node tests/cpuid-coverage.js`
//        or: `node tests/cpuid-coverage.js --strict`

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = fileURLToPath(new URL("..", import.meta.url));
const STRICT = process.argv.includes("--strict");

// Tests that must exist for the indirect tier. tests/e2e/boot*.js and the CD
// boot tests in tests/full/run.js boot a real kernel, which exercises the whole
// baseline instruction set.
const BOOT = "tests/e2e/linux.js";
const CD_BOOT = "tests/full/run.js";

const DIRECT = {
    // The reference-vector suites added for this gate.
    pclmulqdq: "tests/jit64/differential.js",
    aes: "tests/jit64/differential.js",
    xsave: "tests/jit64/xsave.js",
    osxsave: "tests/jit64/xsave.js",
    avx: "tests/kat/vex256.c",
    avx2: "tests/kat/vex256.c",
    bmi1: "tests/jit64/bmi1.js",
    bmi2: "tests/jit64/bmi1.js",
    rdrand: "tests/jit64/bmi1.js",
    rdseed: "tests/jit64/bmi1.js",
    hypervisor: "tests/kat/vmware.c",
    adx: "tests/jit64/bmi1.js",
    cmov: "tests/jit64/differential.js",
    lm: "tests/jit64/differential.js",
    pge: "tests/jit-paging/run.js",
    apic: "tests/e2e/apic.js",
};

const INDIRECT = {
    fpu: BOOT, pse: BOOT, tsc: BOOT, msr: BOOT, pae: BOOT, cx8: BOOT, sep: BOOT,
    mmx: BOOT, fxsr: BOOT, sse: BOOT, sse2: BOOT, sse3: BOOT, ssse3: BOOT,
    fma: BOOT, sse41: BOOT, sse42: BOOT, mmxext: BOOT, mmxext2: BOOT,
    syscall: BOOT,
    // ERMS promises that REP MOVSB is fast, not that it does anything new. The
    // architecturally visible requirement is that it still copies correctly, so
    // the test that matters is the one for the semantics a "fast" implementation
    // is allowed to get wrong: direction, overlap and page crossings, against a
    // byte-by-byte reference. tests/interp64/rep.js is exactly that.
    erms: "tests/interp64/rep.js",
};

const GAPS = {
};

// Existence is not enough: a suite that is never run passes review and tests
// nothing. Each named file has to be invoked by something that actually runs in
// CI, so check the runner lists it as well as the file being present.
const KAT_MAKEFILE = path.join(ROOT, "tests/kat/Makefile");
function isWiredUp(file)
{
    const base = path.basename(file);
    if(base.endsWith(".c"))
    {
        // A tests/kat test is run by being listed in TESTS or SELFTESTS.
        const mk = fs.readFileSync(KAT_MAKEFILE, "utf8");
        const wanted = base.replace(/\.c$/, "");
        const listed = [...mk.matchAll(/^(?:TESTS|SELFTESTS)\s*=\s*(.*)$/gm)]
            .flatMap(m => m[1].trim().split(/\s+/));
        return listed.includes(wanted);
    }
    const mk = fs.readFileSync(path.join(ROOT, "Makefile"), "utf8");
    // Either the file is named in a target body, or it is the test a KAT-style
    // suite is built from (tests/kat/*.expected are checked by tests/kat).
    return new RegExp("(^|\\s|/)" + base.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).test(mk);
}

// leaf -> register -> bits, exactly as instr_0FA2() sets them.
const ADVERTISED = [
    { leaf: 1, reg: "edx", bits: [0, 3, 4, 5, 6, 8, 9, 11, 13, 15, 23, 24, 25, 26],
        names: { 0: "fpu", 3: "pse", 4: "tsc", 5: "msr", 6: "pae", 8: "cx8", 9: "apic",
                 11: "sep", 13: "pge", 15: "cmov", 23: "mmx", 24: "fxsr", 25: "sse", 26: "sse2" } },
    { leaf: 1, reg: "ecx", bits: [0, 1, 9, 12, 19, 20, 22, 23, 25, 26, 27, 28, 30, 31],
        names: { 0: "sse3", 1: "pclmulqdq", 9: "ssse3", 12: "fma", 19: "sse41", 20: "sse42",
                 22: "mmxext", 23: "mmxext2", 25: "aes", 26: "xsave", 27: "osxsave",
                 28: "avx", 30: "rdrand", 31: "hypervisor" } },
    { leaf: 7, reg: "ebx", bits: [3, 5, 8, 9, 18, 19],
        names: { 3: "bmi1", 5: "avx2", 8: "bmi2", 9: "erms", 18: "rdseed", 19: "adx" } },
    // Keyed by decimal so it matches the cross-check, which finds the leaf by
    // the literal in the source rather than by its hex spelling.
    { leaf: 2147483649, reg: "edx", bits: [11, 29],
        names: { 11: "syscall", 29: "lm" } },
];

const problems = [];
const gapReport = [];
let direct = 0, indirect = 0, gaps = 0;
// What instr_0FA2() really sets, as opposed to what the ADVERTISED table
// claims. Both directions of the cross-check below read this.
const inSource = new Set();

for(const group of ADVERTISED)
{
    for(const bit of group.bits)
    {
        const name = group.names[bit] || ("bit" + bit);
        const where = "leaf 0x" + group.leaf.toString(16) + " " + group.reg.toUpperCase() +
            " bit " + bit + " (" + name + ")";

        if(DIRECT[name])
        {
            const file = DIRECT[name];
            if(!fs.existsSync(path.join(ROOT, file)))
            {
                problems.push(where + " names a missing direct test: " + file);
            }
            else if(!isWiredUp(file))
            {
                // A test file that exists but is never invoked is worse than no
                // test at all: it shows up green in review and checks nothing.
                problems.push(where + " names " + file + ", which no runner invokes");
            }
            else { direct++; }
        }
        else if(INDIRECT[name])
        {
            if(!fs.existsSync(path.join(ROOT, INDIRECT[name])))
            {
                problems.push(where + " names a missing indirect test: " + INDIRECT[name]);
            }
            else { indirect++; }
        }
        else if(GAPS[name])
        {
            gapReport.push(where + ": " + GAPS[name]);
            gaps++;
        }
        else
        {
            problems.push(where + " is advertised but appears in none of the tiers");
        }
    }
}

// Cross-check the table against the CPUID implementation, so a bit that
// instr_0FA2() starts setting cannot slip in without a coverage decision.
const cpuidPath = path.join(ROOT, "src/rust/cpu/interp/instructions_0f.rs");
const cpuidSrc = fs.readFileSync(cpuidPath, "utf8");
const fnStart = cpuidSrc.indexOf("pub unsafe fn instr_0FA2()");
if(fnStart < 0)
{
    problems.push("cannot find instr_0FA2 (the CPUID implementation) to cross-check against");
}
else
{
    const body = cpuidSrc.slice(fnStart, fnStart + 14000);
    const bitsIn = expr =>
    {
        const out = new Set();
        for(const m of expr.matchAll(/1 << (\d+)/g)) out.add(+m[1]);
        return out;
    };
    const declared = new Set();
    for(const g of ADVERTISED)
    {
        for(const b of g.bits) declared.add(g.leaf + ":" + g.reg + ":" + b);
    }

    // Scope assignments to each leaf rather than matching the vendor string.
    const armOf = leaf =>
    {
        const arms = [...body.matchAll(/^\s{8}(0x[0-9a-fA-F]+|\d+) => \{/gm)];
        for(let i = 0; i < arms.length; i++)
        {
            const here = arms[i];
            if(parseInt(here[1], 0) !== leaf) continue;
            const next = arms[i + 1];
            return body.slice(here.index, next ? next.index : body.length);
        }
        return null;
    };

    // Both plain and conditional (`|=`) assignments count: the hypervisor bit is
    // set as `ecx |= 1 << 31` behind a config check, and a scanner that only
    // understood `=` would report it as never advertised.
    const assignments = [
        { leaf: 1, reg: "ecx", re: /ecx = ([^;]+);/ },
        { leaf: 1, reg: "edx", re: /edx = 1 \|([\s\S]*?);/ },
        { leaf: 7, reg: "ebx", re: /ebx = ([^;]+);/ },
        { leaf: 0x80000001, reg: "edx", re: /edx = ([^;]+);/ },
    ];
    const note = (leaf, reg, b) =>
    {
        inSource.add(leaf + ":" + reg + ":" + b);
        if(!declared.has(leaf + ":" + reg + ":" + b))
        {
            problems.push("instr_0FA2 sets leaf 0x" + leaf.toString(16) + " " +
                reg.toUpperCase() + " bit " + b + " but this table does not cover it");
        }
    };
    for(const a of assignments)
    {
        const arm = armOf(a.leaf);
        if(arm === null)
        {
            problems.push("could not find the match arm for leaf 0x" + a.leaf.toString(16) +
                " in instr_0FA2");
            continue;
        }
        const m = arm.match(a.re);
        if(!m)
        {
            problems.push("could not parse leaf 0x" + a.leaf.toString(16) + " " +
                a.reg.toUpperCase() + " from instr_0FA2");
            continue;
        }
        const found = bitsIn(m[1]);
        if(found.size === 0 && a.leaf !== 1)
        {
            problems.push("leaf 0x" + a.leaf.toString(16) + " " + a.reg.toUpperCase() +
                " assignment in instr_0FA2 sets no feature bits at all, which is " +
                "almost certainly a parse failure rather than a real leaf");
            continue;
        }
        for(const b of found) note(a.leaf, a.reg, b);

        // Conditional feature bits in the same arm.
        for(const c of arm.matchAll(new RegExp(a.reg + " \\|= ([^;]+);", "g")))
        {
            for(const b of bitsIn(c[1])) note(a.leaf, a.reg, b);
        }
    }

    // `edx |= 1 << 9` for the APIC is conditional, so it is not part of the
    // assignment above; it still has to be accounted for.
    const leaf1 = armOf(1) || "";
    const apic = leaf1.match(/edx \|= 1 << (\d+);/);
    if(apic)
    {
        inSource.add("1:edx:" + apic[1]);
        if(!declared.has("1:edx:" + apic[1]))
        {
            problems.push("instr_0FA2 conditionally sets leaf 1 EDX bit " + apic[1] +
                " (apic) but this table does not cover it");
        }
    }
}

// Directly covered features must remain advertised.
for(const group of ADVERTISED)
{
    for(const bit of group.bits)
    {
        const name = group.names[bit];
        if(!name || !DIRECT[name]) continue;
        if(!inSource.has(group.leaf + ":" + group.reg + ":" + bit))
        {
            problems.push("leaf 0x" + group.leaf.toString(16) + " " +
                group.reg.toUpperCase() + " bit " + bit + " (" + name + ") is covered by " +
                DIRECT[name] + " but is no longer advertised");
        }
    }
}

if(gapReport.length)
{
    console.log("acknowledged gaps (not failing unless --strict):");
    for(const g of gapReport) console.log("  " + g);
    console.log();
}

if(problems.length)
{
    console.log("cpuid coverage gate: " + problems.length + " problem(s)\n");
    for(const p of problems) console.log("  " + p);
    process.exit(1);
}

if(STRICT && gaps)
{
    console.log("cpuid coverage gate: --strict and " + gaps + " acknowledged gap(s) remain");
    process.exit(1);
}

console.log("cpuid coverage gate: " + direct + " direct, " + indirect + " indirect, " +
    gaps + " acknowledged gap(s); every advertised bit is accounted for");
process.exit(0);
