// Emit JavaScript from a TypeScript file by stripping its type annotations.
// The development server uses this so the browser can keep requesting
// `./module.js` while the file on disk is `module.ts` (see tools/serve.py).
//
// Uses Node's built-in transform, so there is no extra dependency. This is for
// development only; the release build runs tsc and Closure.
import { readFileSync } from "node:fs";
import { stripTypeScriptTypes } from "node:module";

const file = process.argv[2];

if(!file)
{
    console.error("usage: transpile.mjs <file.ts>");
    process.exit(2);
}

try
{
    const source = readFileSync(file, "utf8");
    process.stdout.write(stripTypeScriptTypes(source, { mode: "strip", sourceUrl: file }));
}
catch(error)
{
    console.error(String(error && error.message || error));
    process.exit(1);
}
