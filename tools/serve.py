#!/usr/bin/env python3
"""Static file server with a CORS proxy, for local UI work.

Two jobs:

* Serve the tree. `python3 -m http.server` sends no Cache-Control header, so
  browsers fall back to heuristic caching and keep serving a stale stylesheet
  after an edit; this marks everything no-store.
* Answer `/proxy?url=<url>` by fetching the target server-side and adding the
  CORS headers. The browser cannot fetch a cross-origin host that does not send
  Access-Control-Allow-Origin, which is what `relay_url=fetch` needs.

Point the emulator at it with:

    relay_url=fetch:http://localhost:8000/proxy?url=

The listener is bound to 127.0.0.1 only: this is an open forwarder, so it must
not be reachable from outside the machine.
"""
import functools
import http.server
import os
import socketserver
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
PROXY_PATH = "/proxy"
FORWARDED_HEADERS = ("Content-Type", "Accept", "User-Agent", "Authorization")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSPILE = os.path.join(ROOT, "tools", "transpile.mjs")


def is_proxy(path):
    return urllib.parse.urlparse(path).path == PROXY_PATH


def transpile_typescript(path):
    """Return type-stripped JavaScript for a `.ts` file, or None on failure."""
    try:
        result = subprocess.run(
            ["node", TRANSPILE, path],
            cwd=ROOT, capture_output=True, timeout=30,
        )
    except Exception as error:  # noqa: BLE001 - report anything to the page
        sys.stderr.write("transpile: " + str(error) + "\n")
        return None

    if result.returncode != 0:
        sys.stderr.write(result.stderr.decode(errors="replace"))
        return None

    return result.stdout


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        if is_proxy(self.path):
            super().end_headers()
            return

        # Everything is no-store. A rebuild changes the wasm and the bundle
        # while their URLs stay the same, so any caching policy that outlives a
        # rebuild serves a stale emulator.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "GET /health" in (fmt % args):
            return
        super().log_message(fmt, *args)

    def cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "*")
        self.send_header("Access-Control-Allow-Headers", "*")

    def send_bytes(self, status, content_type, data):
        self.send_response(status)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.cors_headers()
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def proxy(self):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        target = (query.get("url") or [""])[0]

        if not target.startswith(("http://", "https://")):
            self.send_bytes(400, "text/plain", b"missing or invalid ?url=")
            return

        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        request = urllib.request.Request(target, data=body, method=self.command)
        for name in FORWARDED_HEADERS:
            value = self.headers.get(name)
            if value:
                request.add_header(name, value)

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                self.send_bytes(response.status, response.headers.get("Content-Type"), response.read())
        except urllib.error.HTTPError as error:
            self.send_bytes(error.code, error.headers.get("Content-Type"), error.read())
        except Exception as error:  # noqa: BLE001 - report anything to the page
            self.send_bytes(502, "text/plain", ("cors proxy: " + str(error)).encode())

    def do_OPTIONS(self):
        if is_proxy(self.path):
            self.send_response(204)
            self.cors_headers()
            self.end_headers()
            return
        super().do_OPTIONS()

    def serve_typescript(self):
        """Serve `<name>.js` from a `<name>.ts` sibling during the TS migration.

        The browser keeps requesting `./module.js` (that is what tsc emits), but
        the source may already be `module.ts`; strip its types on the fly.
        """
        path = self.translate_path(self.path)

        if path.endswith(".js") and not os.path.isfile(path):
            ts = path[:-3] + ".ts"
            if os.path.isfile(ts):
                data = transpile_typescript(ts)
                if data is None:
                    self.send_bytes(500, "text/plain", b"TypeScript transpile failed")
                else:
                    self.send_bytes(200, "text/javascript", data)
                return True

        return False

    def do_GET(self):
        if is_proxy(self.path):
            self.proxy()
            return
        if self.serve_typescript():
            return
        super().do_GET()

    def do_HEAD(self):
        if is_proxy(self.path):
            self.proxy()
            return
        if self.serve_typescript():
            return
        super().do_HEAD()

    def do_POST(self):
        if is_proxy(self.path):
            self.proxy()
            return
        super().do_POST()

    def do_PUT(self):
        if is_proxy(self.path):
            self.proxy()
            return
        super().do_PUT()

    def do_DELETE(self):
        if is_proxy(self.path):
            self.proxy()
            return
        super().do_DELETE()


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    handler = functools.partial(Handler, directory=".")
    with Server(("127.0.0.1", PORT), handler) as httpd:
        print(f"serving . on http://localhost:{PORT} (all no-store, /proxy?url=)", flush=True)
        httpd.serve_forever()
