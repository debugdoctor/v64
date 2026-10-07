// Load a freestanding ELF and read its output buffer at guest address 0x500000.
//
// Usage: node run_kat.mjs bmi_adx.elf [out.txt]
//   WASM_PATH=../../build/v64.wasm   emulator build to exercise
//   JIT=0                            force the interpreter (default: as built)
import fs from "node:fs";
import { set_cpu_config, JIT_DISABLE_64 } from "../../src/config.js";

const PT_LOAD = 1;

// Must match KAT_OUTPUT_ADDR / KAT_OUTPUT_SIZE / KAT_PROGRESS in kat.h.
const KAT_OUTPUT_SIZE = 65536;
// Matches KAT_BARE_METAL_MAGIC in kat.h. It tells the guest it is running
// bare-metal so it uses the fixed base address instead of mmap()ing, which only
// works under a real kernel. Linux never leaves this in rdx.
const KAT_BARE_METAL_MAGIC = 0x4b41543234364d47n;
const KAT_BARE_METAL_BASE = 0x500000;
// The guest stores the index of the case it is about to run here. There is no
// IDT in this harness, so an unsupported instruction turns into a triple fault
// and a halt with rip pointing at wherever the CPU happened to be -- useless for
// telling *which* case failed. Reading this back after a halt names it.
const KAT_PROGRESS = KAT_BARE_METAL_BASE + 0x10800;

function load_elf(buf) {
  if (buf.readUInt32BE(0) !== 0x7f454c46) throw new Error("not an ELF");
  if (buf[4] !== 2 || buf[5] !== 1) throw new Error("want ELF64 little endian");
  const phoff = Number(buf.readBigUInt64LE(0x20));
  const entry = Number(buf.readBigUInt64LE(0x18));
  const loads = [];
  for (let i = 0; i < buf.readUInt16LE(0x38); i++) {
    const p = phoff + i * 56;
    if (buf.readUInt32LE(p) !== PT_LOAD) continue;
    loads.push({
      offset: Number(buf.readBigUInt64LE(p + 8)),
      vaddr: Number(buf.readBigUInt64LE(p + 0x10)),
      filesz: Number(buf.readBigUInt64LE(p + 0x20)),
      memsz: Number(buf.readBigUInt64LE(p + 0x28)),
    });
  }
  if (!loads.length) throw new Error("no PT_LOAD segments");
  return { entry, loads };
}

export function run_kat(elf_path, { v64, max_steps = 40_000_000 } = {}) {
  if (!v64) throw new Error("pass the v64 constructor in");
  const buf = fs.readFileSync(elf_path);
  const { entry, loads } = load_elf(buf);

  const PML4 = 0x10000, PDPT = 0x11000, PD = 0x12000;
  // Keep page tables separate from the ELF image and staging memory.
  const PT_BASE = 0x1000000;
  const STACK_TOP = 0x400000;

  // Reject overlaps with harness memory.
  const reserved = [
    ["output buffer", KAT_BARE_METAL_BASE, KAT_OUTPUT_SIZE + 0x20000],
    ["stack", 0x3F0000, 0x20000],
  ];
  for (const l of loads) {
    for (const [what, base, size] of reserved) {
      if (l.vaddr < base + size && l.vaddr + l.memsz > base) {
        throw new Error(`image overlaps the ${what} at 0x${base.toString(16)}`);
      }
    }
    if (l.vaddr < PT_BASE + 0x200000 && l.vaddr + l.memsz > PT_BASE) {
      throw new Error("image overlaps the page tables");
    }
  }

  return new Promise((resolve, reject) => {
    const em = new v64({
      autostart: false,
      memory_size: 64 * 1024 * 1024,
      disable_jit: 0,
      log_level: 0,
      wasm_path: process.env.WASM_PATH || undefined,
    });
    em.add_listener("emulator-loaded", () => {
      try {
        const cpu = em.v86.cpu;
        const ex = cpu.wm.exports;
        const w64 = (a, v) => {
          ex.write32(a, Number(v & 0xffffffffn));
          ex.write32(a + 4, Number((v >> 32n) & 0xffffffffn));
        };

        // Identity map the low 2 GiB. One page table covers only 2 MiB, and since
        // paging is on the very first instruction fetch would fault if the image
        // (linked at 0x200000 by lld) were not covered.
        w64(PML4, BigInt(PDPT) | 3n);
        w64(PDPT, BigInt(PD) | 3n);
        for (let i = 0; i < 512; i++) {
          const pt = PT_BASE + i * 0x1000;
          w64(PD + i * 8, BigInt(pt) | 3n);
          for (let j = 0; j < 512; j++) {
            w64(pt + j * 8, BigInt(i * 0x200000 + j * 0x1000) | 3n);
          }
        }

        for (const l of loads) {
          for (let i = 0; i < l.filesz; i++) {
            ex.write8(l.vaddr + i, buf[l.offset + i]);
          }
          for (let i = l.filesz; i < l.memsz; i++) ex.write8(l.vaddr + i, 0);
        }

        // The wasm heap can grow during execution and detach a cached typed
        // array, so build a fresh view rather than holding one across main_loop.
        const u32 = () => new Uint32Array(ex.memory.buffer);
        const set64 = (i, v) => {
          const a = u32();
          a[16 + i] = Number(v & 0xffffffffn);
          a[32 + i] = Number((v >> 32n) & 0xffffffffn);
        };
        // Register indices as the emulator numbers them: 2 = rdx, 4 = rsp.
        for (let i = 0; i < 16; i++) set64(i, 0n);
        set64(4, BigInt(STACK_TOP));
        set64(15, 0x300000n);
        // rdx carries the bare-metal marker and rsi the buffer address; see
        // kat_setup() in kat.h. Linux never puts the magic there, so a KAT run
        // under a real kernel uses its own .bss buffer instead.
        set64(2, KAT_BARE_METAL_MAGIC);
        set64(6, BigInt(KAT_BARE_METAL_BASE));   // 6 = rsi

        if (process.env.JIT !== undefined) {
          set_cpu_config(ex, "JIT_DISABLE",
                          process.env.JIT === "0" ? JIT_DISABLE_64 : 0);
        }
        cpu.flags[0] = 0x2;
        cpu.instruction_pointer[0] = entry;
        cpu.in_hlt[0] = 0;
        ex.jit64_clear_cache();
        ex.enter_long_mode(PML4);

        // Run without hardware timers: this harness has no interrupt handlers.
        let steps = 0;
        while (!cpu.in_hlt[0]) {
          ex.run_exact_instructions(4096);
          steps += 4096;
          if (steps > max_steps) {
            // Report where it is stuck: a bare-metal run that reaches the mmap
            // syscall is the usual cause, and rip says so immediately.
            throw new Error(
              `step limit exceeded; rip=0x${(cpu.instruction_pointer[0] >>> 0).toString(16)}`);
          }
        }

        let text = "";
        for (let i = 0; i < KAT_OUTPUT_SIZE; i++) {
          const c = ex.read8(KAT_BARE_METAL_BASE + i);
          if (!c) break;                  // output is NUL terminated
          text += String.fromCharCode(c);
        }
        // Success is the sentinel line the test emits last, not the value of
        // rip. There is no usable "rip == 0" convention: the emulator halts on
        // the HLT and leaves rip pointing just past it, so a clean run ends at
        // some address inside .text like any other.
        const complete = text.endsWith("== done ==\n");
        let progress = 0;
        for (let i = 0; i < 4; i++) progress += ex.read8(KAT_PROGRESS + i) << (8 * i);

        resolve({
          text,
          progress,
          complete,
          halted: cpu.in_hlt[0] === 1,
          rip: cpu.instruction_pointer[0] >>> 0,
          cr2: cpu.cr[2] >>> 0,
          steps,
        });
      } catch (e) {
        reject(e);
      }
    });
  });
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const { v64 } = await import("../../src/main.js");
  const r = await run_kat(process.argv[2], { v64 });
  // Write rather than print: stdout is block buffered when redirected to a file,
  // and process.exit would discard the tail.
  fs.writeFileSync(process.argv[3] || "/dev/stdout", r.text);
  if (!r.complete) {
    process.stderr.write(
      `\n[kat] ${process.argv[2]} INCOMPLETE: halted=${r.halted} ` +
      `steps=${r.steps} rip=0x${r.rip.toString(16)} cr2=0x${r.cr2.toString(16)} ` +
      `case=${r.progress} ` +
      "(expected the trailing \"== done ==\" marker)");
    process.exitCode = 1;
  } else {
    process.stderr.write(`\n[kat] ${process.argv[2]} steps=${r.steps} complete`);
  }
}
