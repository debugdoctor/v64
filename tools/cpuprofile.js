#!/usr/bin/env node

// Summarise a V8 .cpuprofile by self time (wasm helpers, interpreter, compiled
// blocks, GC, JS) to see whether a boot is CPU-bound in emulation.
//
// Usage:
//   node --cpu-prof --cpu-prof-dir=/tmp/v64-prof tests/e2e/alpine-perf.js
//   node tools/cpuprofile.js /tmp/v64-prof/*.cpuprofile

import fs from "node:fs";

const file = process.argv[2];
if(!file)
{
    console.error("usage: node tools/cpuprofile.js <file.cpuprofile>");
    process.exit(1);
}

const profile = JSON.parse(fs.readFileSync(file, "utf8"));
const nodes = new Map(profile.nodes.map(n => [n.id, n]));

// Self time per node from the sampling timestamps.
const self = new Map();
const samples = profile.samples || [];
const deltas = profile.timeDeltas || [];
for(let i = 0; i < samples.length; i++)
{
    const id = samples[i];
    const dt = deltas[i] || 0;
    self.set(id, (self.get(id) || 0) + dt);
}

const rows = [];
for(const [id, us] of self)
{
    const n = nodes.get(id);
    if(!n) continue;
    const cf = n.callFrame;
    let name = cf.functionName || "(anonymous)";
    if(!name && cf.url && cf.url.startsWith("wasm:"))
    {
        // Wasm frames have no JS name; keep the url to distinguish modules.
        name = cf.url;
    }
    rows.push({ name, url: cf.url || "", us });
}
rows.sort((a, b) => b.us - a.us);

const total = rows.reduce((s, r) => s + r.us, 0) || 1;
console.log("total sampled: " + (total / 1000).toFixed(1) + " ms");
console.log("self ms   %     function");
for(const r of rows.slice(0, 40))
{
    console.log(
        (r.us / 1000).toFixed(1).padStart(8) + "  " +
        (100 * r.us / total).toFixed(2).padStart(6) + "  " +
        r.name + (r.url ? "  [" + r.url + "]" : ""),
    );
}
