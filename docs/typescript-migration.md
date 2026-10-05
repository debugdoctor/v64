# TypeScript migration

All runtime JavaScript under `src/` and `lib/` is TypeScript now (54 `.ts`
files; there is no `.js` left there). `config/externs.js` stays JavaScript on
purpose: it is a Closure Compiler externs file, consumed through `--externs`,
not a runtime module.

The release build still uses Closure Compiler ADVANCED, with the same output
size as before the migration (`build/v64_all.js` gzip 95,016 vs 95,002).

## Pipeline

```
src/**/*.ts  --tsc--> build/ts/src/**/*.js  --Closure ADVANCED--> build/v64_all.js
```

`build/ts` is the single input tree for Closure; `config/externs.js` is passed
to Closure separately. `make browser` and the other JS targets depend on
`build/ts/.stamp`, which runs `tsc`.

## Two rules that make it work

1. **Keep the Closure JSDoc.** `tsc` erases TS type annotations, but it passes
   JSDoc comments through verbatim, and Closure reads them for
   `--use_types_for_optimization`. A migrated function therefore carries both a
   TS signature and its Closure JSDoc:

   ```ts
   /**
    * @param {*} cond
    * @param {string=} msg
    */
   export function dbg_assert(cond: unknown, msg?: string): void {}
   ```

   Dropping the JSDoc causes `JSC_WRONG_ARGUMENT_COUNT` and similar errors in
   unrelated files.

2. **`verbatimModuleSyntax: true`.** Closure resolves types across files (e.g.
   `@param {CPU}` in `lib/9p.ts` referring to `src/cpu.ts`). TypeScript would
   normally elide an import that is only referenced from JSDoc, which makes
   Closure report `Unknown type CPU`. `verbatimModuleSyntax` keeps the import
   statements exactly as written.

## Running the sources directly

During development the browser and the Node tests load the `.ts` sources, so
the `.js` specifiers written in the imports have to resolve to `.ts` files:

- Node: `tools/ts-register.mjs` registers a resolver hook. The Makefile exports
  it through `NODE_OPTIONS` for the test targets.
- Browser: `tools/serve.py` strips types on the fly for a `.js` request whose
  file is `.ts`.

## Commands

```sh
pnpm typecheck     # tsc --noEmit
make browser       # tsc + Closure ADVANCED
make eslint
```

## Follow-ups

- `tsconfig.json` runs with `strict` and `noImplicitAny` off; the migration used
  `any`/casts where the original code was dynamically typed. Tighten these per
  module over time.
- eslint still lints only `.js` (the flat config has no TS parser), so `.ts`
  files are currently covered by `tsc` rather than eslint.
- The generated key table lives in `src/config.ts`
  (`tools/gen_cpu_config.js`).
