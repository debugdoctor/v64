# wisp relay

A local [WISP](https://github.com/MercuryWorkshop/wisp-protocol) server for
v64's `wisp://` network backend, so a guest can reach the network without
depending on a public relay.

## Why WISP and not a TAP proxy

The `wsproxy` backends bridge raw ethernet frames to a TAP device, so the
server has to run DHCP, DNS and NAT, and creating a TAP needs root and
platform-specific code (`/dev/net/tun`, `utun`, `wintun`).

WISP carries only TCP/UDP payloads. The guest-facing network stack -- DHCP,
DNS, ARP, ICMP -- lives in the **client** (`src/browser/fake_network.js`), and
DNS goes out over DoH. This program only dials the address the client asks for
and pumps bytes. That makes it unprivileged, portable and small.

## Run

```sh
make relay                      # listens on 127.0.0.1:8080
```

Then point the emulator at it:

```
http://localhost:8000/?profile=alpine&relay_url=wisp://127.0.0.1:8080/
```

`wisp://` is plaintext (`ws://`); use `wisps://` for TLS.

## Build for every platform

```sh
make relay-dist                 # build/relay/relay-{linux,darwin,windows}-{amd64,arm64}
```

No module downloads are involved: the WebSocket implementation is in the
standard library only, so a cross-compile works offline.

## Notes

- **The server must speak first.** It sends a `CONTINUE` for stream 0 right
  after the handshake. The v64 client starts every stream with zero credit and
  buffers everything, including its own `CONNECT`, until one arrives. Without
  it nothing is ever transmitted. See `src/browser/wisp_network.js`.
- **Bind to localhost by default.** This is an open forwarder; exposing it to a
  network lets anyone dial through it. Add authentication and a host/port
  allowlist before doing that.
- Only TCP streams are implemented. The client also speaks UDP (`stream_type`
  2); it is unused by the guest NIC path here.
- ICMP is answered client-side, so `ping` works without reaching this process.
