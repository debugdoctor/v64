# v64

v64 runs a full x86 PC in the browser. It emulates an x86 CPU, RAM and the common PC hardware, and translates guest machine code to WebAssembly at runtime for speed.

v64 is a fork of [v86](https://github.com/copy/v86) and keeps its hardware support, its JavaScript API and its license. Upstream documentation lives in [`UPSTREAM_README.md`](./UPSTREAM_README.md).

- License: [BSD-2-Clause](./LICENSE)
- 中文说明: [`README.zh-CN.md`](./README.zh-CN.md)

## Features

Emulated hardware:

- An x86-64 CPU (long mode) with SSE2/SSE3
- An x87 FPU, using Berkeley SoftFloat for precise 80-bit floats
- A VGA/SVGA card with Bochs VBE extensions
- An IDE disk controller and a built-in ISO 9660 CD-ROM
- A floppy controller and an 8042 PS/2 keyboard/mouse controller
- An 8254 PIT, an 8259 PIC, partial APIC support and a CMOS RTC
- A PCI bus and an NE2000 network card
- virtio filesystem, network and balloon devices
- A SoundBlaster 16 sound card and a Hayes-compatible modem

v64 targets 64-bit Linux. The 32-bit support inherited from v86 is still in the tree but has not been re-verified here, so many images listed in [`UPSTREAM_README.md`](./UPSTREAM_README.md) may not work.

> **Alpine is the only image verified so far.**

## Requirements

- `make`
- Rust, with the `wasm32-unknown-unknown` target
- A Rust-compatible version of `clang`
- Node.js (recent; upstream is tested with v24.x)
- `java` (for Closure Compiler; not needed for the debug build)
- For tests: `nasm`, `gdb`, `qemu-system`, `gcc`, `libc-i386`, `rustfmt`

```sh
rustup target add wasm32-unknown-unknown
```

A complete Debian / WSL setup is described in [`tools/docker/test-image/Dockerfile`](./tools/docker/test-image/Dockerfile).

## Build

```sh
# Debug build (output: debug.html, no java needed)
make

# Optimized build (output: index.html)
make all
```

The first build generates `src/rust/gen/*.rs`, compiles the Rust crate for the wasm target, and bundles the JavaScript.

## Run

ROM and disk images are loaded over XHR, so the files must be served over HTTP (`file://` will not work):

```sh
make run
```

Then open the printed URL, e.g. `http://localhost:8000/`.

### Docker

```sh
docker build -f tools/docker/exec/Dockerfile -t v64:alpine .
docker run -it -p 8000:8000 v64:alpine
```

### Dev container

Open the repository in a Dev Container–capable IDE (VS Code, Codespaces, IntelliJ IDEA, …) and run the “Fetch images” task.

## Embed

The JavaScript API is the same as v86:

```javascript
var emulator = new V86({
    screen_container: document.getElementById("screen_container"),
    bios: { url: "./bios/seabios.bin" },
    vga_bios: { url: "./bios/vgabios.bin" },
    cdrom: { url: "./images/linux.iso" },
    autostart: true,
});
```

More examples are in [`examples/`](./examples) (basic, serial terminal, save/restore, networking, …). TypeScript definitions are in [`v86.d.ts`](./v86.d.ts). For bundler setups (Vite/React/Next/Webpack) there is an official npm package, `v86`.

## Tests

Test images are not distributed with the repository:

```sh
mkdir -p images && curl --compressed --output-dir images/ --remote-name-all \
  https://i.copy.sh/{linux.iso,linux3.iso,linux4.iso,buildroot-bzimage68.bin,TinyCore-11.0.iso,oberon.img,msdos.img,openbsd-floppy.img,kolibri.img,windows101.img,os8.img,freedos722.img,mobius-fd-release5.img,msdos622.img}

make tests
```

## Repository layout

```
src/            Emulator core (JS) + Rust JIT (src/rust/)
gen/            Instruction-table generators (Node scripts -> src/rust/gen/*.rs)
bios/           SeaBIOS / VGA BIOS binaries (bring your own)
tools/docker/   Guest image build scripts (including alpine/)
examples/       Embedding examples
docs/           Documentation (how-it-works, filesystem, networking, ...)
tests/          Integration and unit tests
lib/            Bundled third-party libraries (softfloat, zstd)
```

## License

BSD-2-Clause, see [`LICENSE`](./LICENSE). This is a permissive license: closed-source use is allowed and downstream is not required to open source.

Third-party components are listed in [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md).

## Credits

- [v86](https://github.com/copy/v86) — the upstream project this fork is based on
- [QEMU](https://wiki.qemu.org/) — CPU test cases and reference implementation
- [Berkeley SoftFloat](http://www.jhauser.us/arithmetic/SoftFloat.html) — precise 80-bit floating point
- [zstd](https://github.com/facebook/zstd) — state image compression
- [jor1k](https://github.com/s-macke/jor1k) — 9p, filesystem and UART drivers

## Contributing

See [`UPSTREAM_README.md`](./UPSTREAM_README.md) for upstream conventions. Note that **upstream v86 does not accept pull requests or issues written entirely or partially by generative AI tools**; if you plan to contribute back upstream, follow their rules.
