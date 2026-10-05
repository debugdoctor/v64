#!/bin/sh
# Cross-compiles the relay for every supported platform and writes a manifest
# the web UI reads to offer downloads. The manifest is what ties the page to
# the actual build: if you change the platform list here, the page follows.
#
#   ./tools/relay/dist.sh [output-dir]     (default: build/relay)
set -e

cd "$(dirname "$0")"

OUT="${1:-../../build/relay}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

# sha256sum on Linux, shasum on macOS.
if command -v sha256sum >/dev/null 2>&1; then
    hash() { sha256sum "$1" | cut -d' ' -f1; }
else
    hash() { shasum -a 256 "$1" | cut -d' ' -f1; }
fi

version="$(git -C ../.. describe --tags --always 2>/dev/null || echo unknown)"
entries=""

for os in linux darwin windows; do
    for arch in amd64 arm64; do
        ext=""
        [ "$os" = windows ] && ext=".exe"
        name="relay-$os-$arch$ext"
        GOOS="$os" GOARCH="$arch" go build -trimpath -ldflags "-s -w" -o "$OUT/$name" .
        size="$(wc -c < "$OUT/$name" | tr -d ' ')"
        sum="$(hash "$OUT/$name")"
        printf '%-22s %8s bytes  %s\n' "$name" "$size" "$sum" >&2
        [ -n "$entries" ] && entries="$entries,"
        entries="$entries
        [\"$name\", \"$os\", \"$arch\", $size, \"$sum\"]"
    done
done

cat > "$OUT/manifest.json" <<EOF
{
    "version": "$version",
    "fields": ["name", "os", "arch", "size", "sha256"],
    "files": [$entries
    ]
}
EOF

echo "wrote $OUT/manifest.json" >&2
