#!/usr/bin/env node

// Injects the CPU (JIT/interpreter) option keys from config/cpu_config.json
// into the "GENERATED config keys" block of the two hand-written consumers:
//
//   src/rust/config.rs   the wasm switchboard and its key constants
//   src/config.ts        the matching key table and helpers
//
// The keys have to agree between the wasm switchboard and the JS callers; when
// both were hand-written, editing one and forgetting the other was a silent
// misconfiguration. `make check-cpu-config` re-runs this and fails on a diff.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const SOURCE = path.join(ROOT, "config/cpu_config.json");
const OUT = {
    rust: path.join(ROOT, "src/rust/config.rs"),
    js: path.join(ROOT, "src/config.ts"),
};

const BEGIN = "// BEGIN GENERATED config keys (config/cpu_config.json) -- do not edit";
const END = "// END GENERATED config keys";
const HEADER = "// @generated from config/cpu_config.json by tools/gen_cpu_config.js -- do not edit";

const config = JSON.parse(fs.readFileSync(SOURCE, "utf8"));

// A key may only mean one thing, and the numbering belongs to the source file.
const seen = new Map();
for(const group of config.groups)
{
    for(const [name, value] of Object.entries(group.keys))
    {
        if(seen.has(name))
        {
            throw new Error("duplicate key " + name);
        }
        if(!Number.isInteger(value) || value < 0 || value > 0xFFFFFFFF)
        {
            throw new Error(name + ": value must be a u32");
        }
        seen.set(name, value);
    }
}

function generate_rust()
{
    const lines = [BEGIN, HEADER];
    config.groups.forEach((group, i) =>
    {
        if(i > 0)
        {
            lines.push("");
        }
        lines.push("// " + group.name);
        for(const [name, value] of Object.entries(group.keys))
        {
            lines.push("pub const " + name + ": u32 = " + value + ";");
        }
    });
    lines.push(END);
    return lines.join("\n");
}

function generate_js()
{
    const lines = [BEGIN, HEADER, "export const CPU_CONFIG = {"];
    config.groups.forEach((group, i) =>
    {
        if(i > 0)
        {
            lines.push("");
        }
        lines.push("    // " + group.name);
        for(const [name, value] of Object.entries(group.keys))
        {
            lines.push("    \"" + name + "\": " + value + ",");
        }
    });
    lines.push("");
    lines.push("};");
    lines.push(END);
    return lines.join("\n");
}

// Replace the block between the BEGIN and END markers, leaving the rest of the
// hand-written file untouched.
function splice(file, block)
{
    const prev = fs.readFileSync(file, "utf8");
    const begin = prev.indexOf(BEGIN);
    const end = prev.indexOf(END);

    if(begin < 0 || end < 0 || end < begin)
    {
        throw new Error("missing generated block markers in " + path.relative(ROOT, file));
    }

    return prev.slice(0, begin) + block + prev.slice(end + END.length);
}

// --check verifies the checked-in blocks are current instead of writing them.
const check_only = process.argv.includes("--check");
let stale = 0;

for(const [kind, file] of Object.entries(OUT))
{
    if(!fs.existsSync(file))
    {
        throw new Error("missing file " + path.relative(ROOT, file));
    }

    const next = splice(file, kind === "rust" ? generate_rust() : generate_js());

    if(next === fs.readFileSync(file, "utf8"))
    {
        console.log("unchanged " + path.relative(ROOT, file));
        continue;
    }

    if(check_only)
    {
        stale++;
        console.error("stale " + path.relative(ROOT, file));
        continue;
    }

    fs.writeFileSync(file, next);
    console.log("wrote " + path.relative(ROOT, file));
}

if(stale)
{
    console.error("run `make cpu-config` after editing " + path.relative(ROOT, SOURCE));
    process.exit(1);
}
