#!/bin/sh
# Builds the relay, starts it on a loopback port, runs both test layers and
# stops it again.
#
#   e2e-test.mjs    the framing, driven by a hand-written client
#   client-test.mjs the real v64 WISP client reaching a listener through it
#
# The client test has to run from the repository root: it loads
# build/v64-debug.wasm through a path relative to the working directory.
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

BIN="${TMPDIR:-/tmp}/v64-relay-test"
(cd "$HERE" && go build -o "$BIN" .)

"$BIN" -listen 127.0.0.1:8081 &
relay_pid=$!
trap 'kill "$relay_pid" 2>/dev/null' EXIT

sleep 1
(cd "$HERE" && RELAY=ws://127.0.0.1:8081/ node e2e-test.mjs)
(cd "$ROOT" && WISP_RELAY_URL=wisp://127.0.0.1:8081/ node tools/relay/client-test.mjs)
