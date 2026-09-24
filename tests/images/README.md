# Test images

Large guest images are not committed (see `.gitignore`). Drop them here; the
end-to-end tests look in this directory by default.

| File | Used by | What it is |
| --- | --- | --- |
| `vmlinuz-virt` | `tests/e2e/linux.js`, `tests/e2e/initramfs.js` | A 64-bit `bzImage` |
| `busybox` | `tests/e2e/initramfs.js` | A static x86-64 busybox |
| `initramfs-virt` / `rootfs.cpio` | `tests/e2e/linux.js` | An initramfs |

`LINUX64_IMAGE`, `LINUX64_INITRD` and `LINUX64_BUSYBOX` override the paths.

## Download

Alpine's `vmlinuz-virt` is a real 64-bit `bzImage`:

```sh
curl -fL -o tests/images/vmlinuz-virt \
  https://dl-cdn.alpinelinux.org/alpine/latest-stable/releases/x86_64/netboot/vmlinuz-virt
```

A static x86-64 busybox (musl build from busybox.net):

```sh
curl -fL -o tests/images/busybox \
  https://busybox.net/downloads/binaries/1.35.0-x86_64-linux-musl/busybox
chmod +x tests/images/busybox
```

Verify both:

```sh
file tests/images/vmlinuz-virt   # Linux kernel x86 boot executable bzImage
file tests/images/busybox        # ELF 64-bit, x86-64, statically linked
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
