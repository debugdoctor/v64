# Guest images

Large guest images are not committed (see `.gitignore`). Drop them here; the
examples and the end-to-end tests look in this directory by default.

| File | Used by | What it is |
| --- | --- | --- |
| `vmlinuz-virt` | `tests/e2e/linux.js`, `tests/e2e/initramfs.js` | A 64-bit `bzImage` |
| `initramfs-virt` | `tests/e2e/linux.js` | An initramfs |
| `modloop-virt` | the Alpine ISO boot | Kernel modules (squashfs) |
| `busybox` | `tests/e2e/initramfs.js` | A static x86-64 busybox |
| `alpine-virt-3.24.2-x86_64.iso` | `examples/alpine-iso.html` | Alpine live ISO (BIOS / El Torito) |
| `alpine-minirootfs-3.24.2-x86_64.tar.gz` | optional rootfs | Alpine mini rootfs |

`LINUX64_IMAGE`, `LINUX64_INITRD` and `LINUX64_BUSYBOX` override the paths.

## Download

```sh
make test-images       # vmlinuz-virt, initramfs-virt, modloop-virt, busybox, the Alpine ISO
make alpine-example    # only what examples/alpine-iso.html needs (release wasm + xterm + the ISO)
```

Or by hand (the versions the Makefile pins, Alpine 3.24.2):

```sh
base=https://dl-cdn.alpinelinux.org/alpine/v3.24/releases/x86_64

curl -fL -o images/vmlinuz-virt        "$base/netboot/vmlinuz-virt"
curl -fL -o images/initramfs-virt      "$base/netboot/initramfs-virt"
curl -fL -o images/modloop-virt        "$base/netboot/modloop-virt"
curl -fL -o images/alpine-virt-3.24.2-x86_64.iso        "$base/alpine-virt-3.24.2-x86_64.iso"
curl -fL -o images/alpine-virt-3.24.2-x86_64.iso.sha256 "$base/alpine-virt-3.24.2-x86_64.iso.sha256"
curl -fL -o images/alpine-minirootfs-3.24.2-x86_64.tar.gz "$base/alpine-minirootfs-3.24.2-x86_64.tar.gz"

# a static x86-64 busybox (musl build from busybox.net)
curl -fL -o images/busybox https://busybox.net/downloads/binaries/1.35.0-x86_64-linux-musl/busybox
chmod +x images/busybox
```

Verify the ISO against the published checksum:

```sh
cd images && shasum -a 256 -c alpine-virt-3.24.2-x86_64.iso.sha256
```

## Build your own

Buildroot produces a kernel, busybox and an initramfs in one go:

```sh
git clone https://git.busybox.net/buildroot
cd buildroot && make qemu_x86_64_defconfig && make -j"$(nproc)"
# output/images/bzImage, output/images/rootfs.cpio
```

The boot loader parses the Linux setup header, so the kernel must be a
`bzImage` (or a `vmlinuz`), not a raw `vmlinux` ELF.
