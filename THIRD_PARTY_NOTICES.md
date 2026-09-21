# Third-Party Notices

v64 is a fork of [v86](https://github.com/copy/v86) and ships the third-party components bundled in the v86 repository. Their origins and licenses are listed below.

## v86

- Source: https://github.com/copy/v86
- License: BSD-2-Clause
- Copyright: `Copyright (c) 2012, The v86 contributors`
- Full license text: [`LICENSE`](./LICENSE)

Modifications and additions made in this fork are copyright of the v64 contributors.

## Third-party components bundled in v86

The following components are distributed with the repository under their own licenses:

| Component | Location | License |
| --- | --- | --- |
| Berkeley SoftFloat | `lib/softfloat/softfloat.c` | See file header (BSD-style) |
| zstd (decompression) | `lib/zstd/zstddeclib.c` | BSD-3-Clause / GPL-2.0 dual |
| kvm-unit-tests | `tests/kvm-unit-tests/` | See LICENSE in that directory |
| QEMU test cases | `tests/qemu/` | GPL-2.0 |
| floppy (portions ported from QEMU) | `src/floppy.js` | MIT, see [`LICENSE.MIT`](./LICENSE.MIT) |

> Note: portions of v86 are ported from QEMU (the MIT cases are covered by `LICENSE.MIT`). If code from other projects is added later, add a corresponding entry here.

## Documentation and design references (no code copied)

- [ktock/qemu-wasm](https://github.com/ktock/qemu-wasm)
- [xarantolus/ax](https://github.com/xarantolus/ax) (MIT)
- [r3bb1t/rusty_box](https://github.com/r3bb1t/rusty_box) (LGPL-2.1)
- QEMU

Used as references for principles and architecture only; no code was copied.
