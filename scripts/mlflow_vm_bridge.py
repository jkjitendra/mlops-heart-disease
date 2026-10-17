"""Private HTTP bridge from Minikube to the local MLflow server."""

import os
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BRIDGE_HOST = os.environ.get("MLFLOW_BRIDGE_HOST", "192.168.49.1")
BRIDGE_PORT = 5051
MLFLOW_HOST = "127.0.0.1"
MLFLOW_PORT = 5050


class MLflowBridge(BaseHTTPRequestHandler):

    def forward(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else None

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {
                "host", "connection", "proxy-connection",
                "transfer-encoding"
            }
        }

        headers["Host"] = f"{MLFLOW_HOST}:{MLFLOW_PORT}"

        connection = HTTPConnection(
            MLFLOW_HOST, MLFLOW_PORT, timeout=90
        )

        try:
            connection.request(
                self.command,
                self.path,
                body=body,
                headers=headers,
            )

            response = connection.getresponse()
            payload = response.read()

            self.send_response(response.status)

            for key, value in response.getheaders():
                if key.lower() not in {
                    "content-length", "connection",
                    "transfer-encoding"
                }:
                    self.send_header(key, value)

            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()

            if self.command != "HEAD":
                self.wfile.write(payload)

        except Exception as exc:
            self.send_error(502, str(exc))
        finally:
            connection.close()

    do_GET = forward
    do_POST = forward
    do_PUT = forward
    do_PATCH = forward
    do_DELETE = forward
    do_HEAD = forward


server = ThreadingHTTPServer(
    (BRIDGE_HOST, BRIDGE_PORT), MLflowBridge
)

print(
    f"MLflow bridge listening on {BRIDGE_HOST}:{BRIDGE_PORT}",
    flush=True,
)

server.serve_forever()
