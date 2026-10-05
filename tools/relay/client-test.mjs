#!/usr/bin/env node
// Integration test: the real v64 WISP client talking to this relay.
//
// Starts a TCP listener, injects a SYN into the emulated NIC's adapter path and
// expects the relay to dial the listener. That covers what the protocol test
// cannot: that wisp_network.js and this server agree on the framing.
//
//   WISP_RELAY_URL=wisp://127.0.0.1:8080/ node client-test.mjs
//
// tests/relay/test.sh starts the relay and runs this.

import net from "node:net";
import { v64 } from "../../src/main.js";

const RELAY = process.env.WISP_RELAY_URL || "wisp://127.0.0.1:8080/";

function checksum(bytes)
{
    let sum = 0;
    for(let i = 0; i < bytes.length; i += 2)
    {
        sum += (bytes[i] << 8) | (bytes[i + 1] || 0);
        sum = (sum & 0xFFFF) + (sum >>> 16);
    }
    return (~sum) & 0xFFFF;
}

// Minimal IPv4 + TCP SYN, the shape fake_network's state machine expects.
function syn_frame(router_mac, vm_mac, vm_ip, dest_ip, dest_port)
{
    const tcp = new Uint8Array(20);
    tcp[0] = 0x9C; tcp[1] = 0x40;               // source port 40000
    tcp[2] = dest_port >> 8; tcp[3] = dest_port & 0xFF;
    tcp[4] = 0; tcp[5] = 0; tcp[6] = 0x03; tcp[7] = 0xE8;   // seq 1000
    tcp[12] = 0x50;                             // data offset 5
    tcp[13] = 0x02;                             // SYN
    tcp[14] = 0x20; tcp[15] = 0x00;             // window

    const ip = new Uint8Array(20);
    ip[0] = 0x45;
    const total = 20 + tcp.length;
    ip[2] = total >> 8; ip[3] = total & 0xFF;
    ip[8] = 64;
    ip[9] = 6;                                  // TCP
    ip.set(vm_ip, 12);
    ip.set(dest_ip, 16);
    const sum = checksum(ip);
    ip[10] = sum >> 8; ip[11] = sum & 0xFF;

    return new Uint8Array([...router_mac, ...vm_mac, 0x08, 0x00, ...ip, ...tcp]);
}

const server = net.createServer();
await new Promise(resolve => server.listen(0, "127.0.0.1", resolve));
const port = server.address().port;

let accepted = false;
server.on("connection", () => { accepted = true; });

const emulator = new v64({
    autostart: false,
    memory_size: 16 * 1024 * 1024,
    log_level: 0,
    network_relay_url: RELAY,
});

emulator.add_listener("emulator-loaded", async () =>
{
    const adapter = emulator.network_adapter;
    if(!adapter)
    {
        console.error("FAIL: adapter was not created for " + RELAY);
        process.exit(1);
    }

    // The relay must send its opening CONTINUE first: until it does, every
    // stream has zero credit and the CONNECT would sit in the client's buffer.
    const socket = adapter.wispws;
    const deadline = Date.now() + 5000;
    while(socket.readyState !== WebSocket.OPEN ||
        !adapter.connections[0] || adapter.connections[0].congestion <= 0)
    {
        if(Date.now() > deadline)
        {
            console.error("FAIL: relay never granted stream 0 credit");
            process.exit(1);
        }
        await new Promise(resolve => setTimeout(resolve, 50));
    }
    console.log("connected, stream 0 credit = " + adapter.connections[0].congestion);

    adapter.send(syn_frame(
        Array.from(adapter.router_mac || [0x52, 0x54, 0, 1, 2, 3]),
        [0x00, 0x22, 0x15, 0x08, 0xD0, 0x2B],
        Array.from(adapter.vm_ip || [192, 168, 86, 100]),
        [127, 0, 0, 1],
        port,
    ));
    console.log("sent SYN for 127.0.0.1:" + port);

    while(!accepted && Date.now() < deadline)
    {
        await new Promise(resolve => setTimeout(resolve, 50));
    }

    console.log(accepted ? "RESULT: PASS - relay dialled the listener" : "RESULT: FAIL - no dial");
    process.exit(accepted ? 0 : 1);
});
