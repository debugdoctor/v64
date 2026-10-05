#!/usr/bin/env python3
"""Shrink an OpenWrt x86-64 combined image so it is cheap to serve locally.

The upstream `generic-squashfs-combined.img.gz` is written for a 128 MB disk.
Partition 2 ends at ~120 MB, but the squashfs it holds only uses the first
~6 MB; OpenWrt formats an ext4 overlay in the remaining free space at first
boot. Booting the 121 MB file works, but it is larger than the guest needs.

This rewrites the MBR so partition 2 ends at the requested size, then
truncates the file. The kernel, GRUB and the squashfs are in the low part of
the image and are untouched, and the overlay gets a matching partition, so
the result boots and still initialises its overlay.

Usage: resize_openwrt.py <combined.img.gz> <out.img> [size-mb]
"""
import gzip
import struct
import sys

SECTOR = 512
DEFAULT_MB = 64
# The upstream file carries bytes past the gzip stream, which `gunzip` reports
# as trailing garbage. gzip.open only reads through the compressed member, so
# the extra bytes are ignored here.


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)

    src, dst = sys.argv[1], sys.argv[2]
    total_mb = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_MB
    end_lba = total_mb * 1024 * 1024 // SECTOR

    with gzip.open(src, "rb") as f:
        data = bytearray(f.read(end_lba * SECTOR))

    mbr = data[:SECTOR]
    if mbr[510:512] != b"\x55\xaa":
        sys.exit("not an MBR image")

    for i in range(4):
        entry = 446 + i * 16
        start = struct.unpack_from("<I", data, entry + 8)[0]
        sectors = struct.unpack_from("<I", data, entry + 12)[0]
        if not sectors:
            continue
        if entry == 446 + 16:  # partition 2 is the rootfs + overlay
            # Refuse to cut into the squashfs living at the start of p2.
            offset = start * SECTOR
            if data[offset:offset + 4] != b"hsqs":
                sys.exit("partition 2 is not squashfs")
            squashfs_size = struct.unpack_from("<Q", data, offset + 40)[0]
            if end_lba * SECTOR < offset + squashfs_size:
                sys.exit("requested size is smaller than the rootfs")
            struct.pack_into("<I", data, entry + 12, end_lba - start)
            # Keep the CHS end in range for tools that still read it.
            data[entry + 5], data[entry + 6], data[entry + 7] = 0xFE, 0xFF, 0xFF

    with open(dst, "wb") as f:
        f.write(data)

    print("wrote %s (%d MB)" % (dst, total_mb))


if __name__ == "__main__":
    main()
