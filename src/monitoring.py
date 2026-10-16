"""Lightweight in-memory monitoring for prediction requests."""

from collections import deque
from math import ceil
from threading import Lock
from time import perf_counter

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse


class RequestMetrics:
    """Store aggregate API metrics without patient information."""

    def __init__(self):
        self._lock = Lock()
        self._latencies = deque(maxlen=1000)
        self._total = 0
        self._success = 0
        self._client_errors = 0
        self._server_errors = 0
        self._total_latency_ms = 0.0

    def record(self, status_code: int, latency_ms: float):
        """Record one completed prediction request."""

        with self._lock:
            self._total += 1
            self._total_latency_ms += latency_ms
            self._latencies.append(latency_ms)

            if 200 <= status_code < 300:
                self._success += 1
            elif 400 <= status_code < 500:
                self._client_errors += 1
            elif status_code >= 500:
                self._server_errors += 1

    def snapshot(self):
        """Return counters and latency statistics."""

        with self._lock:
            samples = sorted(self._latencies)

            p95 = (
                samples[ceil(0.95 * len(samples)) - 1]
                if samples else 0.0
            )

            average = (
                self._total_latency_ms / self._total
                if self._total else 0.0
            )

            return {
                "prediction_requests_total": self._total,
                "successful_requests": self._success,
                "client_errors_4xx": self._client_errors,
                "server_errors_5xx": self._server_errors,
                "average_latency_ms": round(average, 2),
                "p95_latency_ms": round(p95, 2),
                "latency_sample_count": len(samples),
            }


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Heart Disease API Monitor</title>
  <style>
    body {
      font-family: system-ui, sans-serif;
      background: #f5f7fb;
      color: #172338;
      padding: 30px;
      max-width: 950px;
      margin: auto;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
      gap: 16px;
      margin-top: 24px;
    }
    .card {
      background: white;
      border-radius: 12px;
      padding: 22px;
      box-shadow: 0 2px 8px #00000012;
    }
    .value {
      display: block;
      font-size: 30px;
      font-weight: bold;
      margin-top: 10px;
    }
    .note { color: #596579; }
  </style>
</head>
<body>
  <h1>Heart Disease Prediction API</h1>
  <p class="note">Local Kubernetes monitoring dashboard</p>
  <p id="status">Loading metrics...</p>

  <div class="grid">
    <div class="card">Prediction requests
      <span class="value" id="total">-</span></div>
    <div class="card">Successful requests
      <span class="value" id="success">-</span></div>
    <div class="card">Client errors (4xx)
      <span class="value" id="client-errors">-</span></div>
    <div class="card">Server errors (5xx)
      <span class="value" id="server-errors">-</span></div>
    <div class="card">Average latency
      <span class="value" id="average">-</span></div>
    <div class="card">p95 latency
      <span class="value" id="p95">-</span></div>
  </div>

  <p class="note">
    Updates every 5 seconds. Metrics are stored in memory and
    reset when the API process restarts. No patient data is stored.
  </p>

  <script>
    async function refreshMetrics() {
      try {
        const response = await fetch("/metrics", {
          cache: "no-store"
        });
        if (!response.ok) throw new Error("Metrics unavailable");

        const m = await response.json();

        document.getElementById("total").textContent =
          m.prediction_requests_total;
        document.getElementById("success").textContent =
          m.successful_requests;
        document.getElementById("client-errors").textContent =
          m.client_errors_4xx;
        document.getElementById("server-errors").textContent =
          m.server_errors_5xx;
        document.getElementById("average").textContent =
          m.average_latency_ms + " ms";
        document.getElementById("p95").textContent =
          m.p95_latency_ms + " ms";
        document.getElementById("status").textContent =
          "Monitoring active";
      } catch (error) {
        document.getElementById("status").textContent =
          "Unable to retrieve metrics";
      }
    }

    refreshMetrics();
    setInterval(refreshMetrics, 5000);
  </script>
</body>
</html>
"""


def install_monitoring(app: FastAPI):
    """Register request metrics and a monitoring dashboard."""

    @app.middleware("http")
    async def track_predictions(request: Request, call_next):
        if request.url.path != "/predict":
            return await call_next(request)

        start = perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            metrics = getattr(
                request.app.state, "request_metrics", None
            )
            if metrics is not None:
                metrics.record(
                    status_code,
                    (perf_counter() - start) * 1000,
                )

    @app.get("/metrics")
    def metrics_endpoint(request: Request):
        metrics = getattr(
            request.app.state, "request_metrics", None
        )
        if metrics is None:
            raise HTTPException(
                status_code=503,
                detail="metrics unavailable",
            )
        return metrics.snapshot()

    @app.get("/monitor", response_class=HTMLResponse)
    def monitor_dashboard():
        return HTMLResponse(
            DASHBOARD_HTML,
            headers={"Cache-Control": "no-store"},
        )
