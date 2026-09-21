# v64 Roadmap: from v86 to x86-64

This document describes the staged plan for extending v86 with x86-64 (long mode) support, with the goal of booting 64-bit Linux in the browser.

## Principles

1. **Maximize reuse of the device layer.** v86's `src/bus.js` is the boundary between the CPU and the devices. The new 64-bit CPU implements the same bus interface so the peripherals can stay as they are.
2. **Correctness before performance.** Get the system booting with an interpreter first, then add the JIT.
3. **Skip the pre-long-mode legacy modes.** Prefer Linux's **x86-64 64-bit boot protocol** to enter long mode directly and jump into the kernel, avoiding investment in 16-bit real mode / 32-bit protected mode.

## What we reuse vs. rewrite

v86 is layered: `src/bus.js` routes memory accesses and port I/O to the individual devices. As long as the new 64-bit CPU implements the **same bus interface**, the device layer can stay largely untouched.

| Part | Location | Plan for 64-bit |
| --- | --- | --- |
| Devices (VGA/IDE/PS2/PCI/virtio/UART…) | `src/*.js` | Reuse |
| Soft-float | `src/rust/softfloat.rs` | Reuse |
| CPU interpreter | `src/cpu.js` | Rewrite (64-bit registers, REX, long mode) |
| Paging | `src/rust/paging.rs`, `page.rs` | Rewrite (2-level → 4-level PML4) |
| JIT / codegen | `src/rust/jit.rs`, `wasmgen/` | Rewrite (emitted WASM assumes 32-bit addressing) |
| Interrupt/timer devices | device layer | Add APIC / IO-APIC / HPET / ACPI (required by 64-bit Linux) |

## Key work items

### 1. CPU state (`src/rust/regs.rs`, `src/cpu.js`)

- Widen the general-purpose registers to 64-bit; add `r8–r15`
- Widen `RIP` to 64-bit; keep `RFLAGS` semantics
- Add the relevant MSRs: `EFER` (LME/LMA/SCE/NXE), `FS_BASE` / `GS_BASE` / `KERNEL_GS_BASE`, `STAR` / `LSTAR` / `SFMASK`
- Control-register semantics: `CR0.PG`, `CR4.PAE`, `CR4.LA57` (optional)

### 2. Instruction decoding and execution

- **REX prefixes** (`0x40–0x4F`): `src/rust/prefix.rs`
- 64-bit operand sizes, `MOVSXD`, `SYSCALL` / `SYSRET`, `SWAPGS`, etc.
- `src/rust/jit_instructions.rs` and the `gen/` instruction tables need to be extended for 64-bit

### 3. Paging (`src/rust/paging.rs`, `page.rs`)

- 2-level (32-bit) → **4-level page tables (PML4 → PDPT → PD → PT)**
- 2MB / 1GB huge pages, `NX` bit
- In long mode the segment registers are basically flat; segment semantics are simplified

### 4. Exceptions and interrupts

- IDT gate semantics and `IST` stack switching in long mode
- 64-bit stack behavior on exception entry/return

### 5. JIT / codegen (`src/rust/jit.rs`, `src/rust/wasmgen/`)

- The current JIT emits WASM that assumes 32-bit addressing and 32-bit registers
- A 64-bit path needs new codegen; the 64-bit path can fall back to the interpreter at first

### 6. Device gaps (required by 64-bit Linux)

| Device | Note |
| --- | --- |
| Local APIC / IO-APIC | v86 has only *partial* APIC; 64-bit Linux SMP/interrupts depend on it |
| HPET / APIC timer | Kernel timekeeping |
| ACPI | Processor/interrupt descriptions; current support is oriented toward 32-bit |
| virtio-blk / 9p | Root filesystem (existing virtio implementations can be reused) |

## Phases

### Phase 0 — Runnable baseline
- [ ] Build v86 with `make` and boot a 32-bit guest (e.g. Buildroot/Alpine)
- [ ] Understand the CPU ↔ device boundary in `src/bus.js`
- [ ] Deliverable: a runnable playground (xterm frontend + snapshot)

### Phase 1 — 64-bit interpreter
- [ ] Widen CPU state to 64-bit
- [ ] Support entering long mode (GDT + `CR0.PG` / `CR4.PAE` / `EFER.LME`)
- [ ] 4-level page tables
- [ ] Implement the Linux x86-64 64-bit boot-protocol load path
- [ ] Goal: `vmlinux` (ELF) + initramfs → **BusyBox shell**
- [ ] Deliverable: boots in 64-bit, but slow (pure interpreter)

### Phase 2 — JIT
- [ ] Extend `wasmgen/` for 64-bit registers/addressing
- [ ] Switch hot paths from the interpreter to the JIT
- [ ] Deliverable: usable performance

### Phase 3 — Devices and frontend
- [ ] APIC / IO-APIC / HPET / ACPI
- [ ] Disk / 9p root filesystem
- [ ] Snapshots (IndexedDB persistence)
- [ ] xterm.js playground

## Milestone criteria

| Milestone | Pass condition |
| --- | --- |
| M1 | A hand-written 64-bit freestanding ELF prints to the serial port |
| M2 | A statically linked 64-bit user-space program runs |
| M3 | Linux kernel + initramfs reaches a BusyBox shell |
| M4 | Full system with disk/9p + snapshots |

## References

- v86 architecture docs: `docs/how-it-works.md`
- Linux x86-64 boot protocol: `Documentation/x86/boot.rst`
- Intel SDM Volume 3 (long mode, paging, exceptions)
