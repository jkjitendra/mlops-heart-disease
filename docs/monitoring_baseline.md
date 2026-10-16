# Kubernetes API Monitoring Baseline

## Environment

- Application: Heart Disease Prediction FastAPI
- Platform: Local Kubernetes (Colima + K3s)
- Access: Kubernetes LoadBalancer via 127.0.0.1:8088
- Model: MLflow champion, version 1
- Deployment replicas: 1

## Experiment 1: Prediction Latency

Date: 16 October 2026

Method:
- 5 warm-up requests
- 200 sequential prediction requests
- Same sample patient input
- Measured through the Kubernetes LoadBalancer

| Metric | Result |
|---|---:|
| Requests | 200 |
| Successful requests | 200 |
| Average | 27.93 ms |
| p50 | 22.75 ms |
| p95 | 51.32 ms |
| p99 | 56.14 ms |
| Minimum | 16.74 ms |
| Maximum | 91.73 ms |

Prediction consistency: PASS

## Experiment 2: API Reliability

Date: 16 October 2026

Method:
- 100 sequential valid prediction requests
- 10 sequential invalid requests (missing age)
- Measured through the Kubernetes LoadBalancer

| Metric | Result |
|---|---:|
| Valid requests | 100 |
| Valid HTTP 200 responses | 100 |
| Correct predictions | 100 |
| Invalid requests | 10 |
| Expected HTTP 422 responses | 10 |
| Transport/parsing errors | 0 |
| Unexpected outcomes | 0 |

Reliability test: PASS

## Kubernetes Resource Snapshot After Reliability Test

| Metric | Result |
|---|---:|
| CPU usage | 62m |
| Memory usage | 369 MiB |
| Pod ready | 1/1 |
| New restarts during test | 0 |

## Limitations

- Measurements were taken on a local MacBook Air M1.
- Tests were sequential, not concurrent.
- The LoadBalancer was accessed locally.
- Results do not establish a production SLA.
- This is an educational ML deployment, not a clinical system.


## Experiment 3: MLflow Availability and Recovery

Date: 16 October 2026

### Scenario A: Existing Pod During MLflow Outage

Procedure:
- Confirmed the Kubernetes API was ready.
- Temporarily stopped the MLflow server.
- Verified MLflow was unreachable (HTTP status 000).
- Called the existing API's readiness and prediction endpoints.
- Restored MLflow.

Observed results:
- Existing API remained ready with model version 1.
- Prediction succeeded: class 0, probability 0.21225669206996303.
- No additional application Pod restarts were observed.
- MLflow was restored successfully (HTTP 200).

Conclusion:
The already-running API continued inference using its in-memory
model while MLflow was temporarily unavailable.

### Scenario B: New Pod During MLflow Outage

Procedure:
- Temporarily stopped MLflow.
- Created an isolated test Pod with an HTTP readiness probe.
- Observed its status and logs after approximately 42 seconds.
- Checked the existing application's readiness.
- Restored MLflow and deleted the temporary Pod.

Observed results:
- Temporary Pod: Running, 0/1 Ready.
- Startup logs: Waiting for application startup.
- Readiness probe failed repeatedly with connection refused.
- Existing application remained ready throughout the experiment.
- MLflow restored successfully (HTTP 200).
- Temporary test Pod was removed.

Conclusion:
The new Pod did not become ready during the 42-second observation
while MLflow was unavailable. Model retrieval is a likely
startup dependency, but permanent startup failure was not established.

### Reliability Implications

- Loaded models can remain available during a registry outage.
- New Pod startup can be delayed by registry unavailability.
- Kubernetes readiness probes prevent unready instances from
  being treated as ready for Service traffic.
- These observations are specific to the tested local environment.
