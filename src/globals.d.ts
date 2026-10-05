// Ambient declarations for globals provided at build time by Closure's
// `--define` (see CLOSURE_FLAGS in the Makefile) and for browser/worker
// globals that the DOM lib does not declare.

/** Replaced with `true`/`false` by Closure; gates debug-only code. */
declare var DEBUG: boolean;

// AudioWorklet global scope (src/browser/speaker.ts).
declare var AudioWorkletProcessor: any;
declare var registerProcessor: any;
declare var sampleRate: number;

// Non-standard / prefixed browser globals.
declare var webkitAudioContext: any;

// Available in workers only; the code guards its use.
declare function importScripts(...urls: string[]): void;

// Debug handles the browser build attaches to `window`.
interface Window
{
    emulator: any;
    cpu: any;
    h: any;
    dump_file: any;
    WabtModule: any;
    cs: any;
}

interface Navigator
{
    keyboard?: any;
}
