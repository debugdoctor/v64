# Guest images

Large guest images are not committed (see `.gitignore`). Drop them here; the
examples and the end-to-end tests look in this directory by default.

| File | Used by | What it is |
| --- | --- | --- |
| `vmlinuz-virt` | `tests/e2e/linux.js`, `tests/e2e/initramfs.js` | A 64-bit `bzImage` |
| `initramfs-virt` | `tests/e2e/linux.js` | An initramfs |
| `modloop-virt` | the Alpine ISO boot | Kernel modules (squashfs) |
| `busybox` | `tests/e2e/initramfs.js` | A static x86-64 busybox |
| `alpine-virt-3.24.2-x86_64.iso` | `tests/e2e/alpine-perf.js` | Alpine live ISO (BIOS / El Torito) |
| `alpine-minirootfs-3.24.2-x86_64.tar.gz` | optional rootfs | Alpine mini rootfs |
| `TinyCorePure64-17.1.iso` | `tests/full/run.js` (Tiny Core Pure64 17 CD) | Tiny Core Linux Pure64 live ISO (x86-64) |
| `openwrt-24.10.5-x86-64-squashfs.img` | `tests/full/run.js` (OpenWrt 24.10) | OpenWrt x86-64 combined image, p2 resized to 64 MB |

`LINUX64_IMAGE`, `LINUX64_INITRD` and `LINUX64_BUSYBOX` override the paths.

## Download

```sh
make test-images       # vmlinuz-virt, initramfs-virt, modloop-virt, busybox, the Alpine ISO
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

# Tiny Core Linux Pure64 (x86-64 live ISO)
curl -fL -o images/TinyCorePure64-17.1.iso \
    http://tinycorelinux.net/17.x/x86_64/release/TinyCorePure64-17.1.iso

# OpenWrt x86-64. The upstream combined image is written for a 128 MB disk
# (121 MB of file); resize_openwrt.py rewrites partition 2 to 64 MB and
# truncates the file, leaving the kernel, GRUB and squashfs untouched.
curl -fL -o images/openwrt-24.10.5-x86-64-generic-squashfs-combined.img.gz \
    https://downloads.openwrt.org/releases/24.10.5/targets/x86/64/openwrt-24.10.5-x86-64-generic-squashfs-combined.img.gz
python3 tools/resize_openwrt.py \
    images/openwrt-24.10.5-x86-64-generic-squashfs-combined.img.gz \
    images/openwrt-24.10.5-x86-64-squashfs.img
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
