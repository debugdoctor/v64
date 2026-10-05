// Node ESM resolver hook for the incremental TypeScript migration.
//
// Node runs `.ts` files directly (type stripping), but unlike tsc it does not
// rewrite a `.js` specifier to a `.ts` file on disk. While some modules are
// still `.js` and others are already `.ts`, a not-yet-migrated module that
// imports `./const.js` must keep working, so map that specifier to the `.ts`
// file when the `.js` one is gone.
//
// The Makefile exports NODE_OPTIONS="--import ./tools/ts-register.mjs".
import { existsSync } from "node:fs";
import { registerHooks } from "node:module";
import { fileURLToPath } from "node:url";

registerHooks({
    resolve(specifier, context, nextResolve)
    {
        if(specifier.endsWith(".js") && context.parentURL)
        {
            const url = new URL(specifier, context.parentURL);
            const ts = new URL(url.href.slice(0, -3) + ".ts");

            if(!existsSync(fileURLToPath(url)) && existsSync(fileURLToPath(ts)))
            {
                return { url: ts.href, shortCircuit: true };
            }
        }

        return nextResolve(specifier, context);
    },
});
