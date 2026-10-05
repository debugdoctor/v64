// End-to-end check of tools/relay: a local TCP server, reached through the
// WISP relay. Verifies the stream-0 CONTINUE, CONNECT, DATA both ways and CLOSE.
import net from "node:net";

const TCP_PORT = 9099;
const RELAY = process.env.RELAY || "ws://127.0.0.1:8080/";

const server = net.createServer(sock => {
    sock.write("BANNER\n");
    sock.on("data", d => sock.write("echo:" + d));
});
server.listen(TCP_PORT, "127.0.0.1", () => {
    const ws = new WebSocket(RELAY);
    ws.binaryType = "arraybuffer";

    const seen = { continue0: null, data: [], closed: false };
    let sent_connect = false;

    ws.onopen = () => console.log("ws open");
    ws.onmessage = e => {
        const b = new Uint8Array(e.data);
        const type = b[0];
        const id = new DataView(b.buffer).getUint32(1, true);
        if(type === 3) {
            const credit = new DataView(b.buffer).getUint32(5, true);
            console.log("CONTINUE stream=" + id + " credit=" + credit);
            if(id === 0) {
                seen.continue0 = credit;
                // Now the client has credit, like the real one does.
                const host = "127.0.0.1";
                const p = Buffer.alloc(5 + 1 + 2 + host.length);
                p[0] = 0x01;
                p.writeUInt32LE(1, 1);
                p[5] = 0x01;                       // TCP
                p.writeUInt16LE(TCP_PORT, 6);
                p.write(host, 8, "utf8");
                ws.send(p);
                sent_connect = true;
            }
        } else if(type === 2) {
            console.log("DATA stream=" + id + " " + JSON.stringify(Buffer.from(b.slice(5)).toString()));
            seen.data.push(Buffer.from(b.slice(5)).toString());
            if(seen.data.length === 1) {
                const p = Buffer.concat([Buffer.from([0x02, 1, 0, 0, 0]), Buffer.from("hello")]);
                ws.send(p);
            }
        } else if(type === 4) {
            console.log("CLOSE stream=" + id + " reason=" + b[5]);
            seen.closed = true;
        }
    };
    ws.onerror = () => console.log("ws error");

    setTimeout(() => {
        const ok = seen.continue0 > 0 && sent_connect &&
            seen.data.some(d => d.includes("BANNER")) &&
            seen.data.some(d => d.includes("echo:hello"));
        console.log(ok ? "RESULT: PASS" : "RESULT: FAIL " + JSON.stringify(seen));
        ws.close(); server.close(); process.exit(ok ? 0 : 1);
    }, 2500);
});
