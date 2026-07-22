---
title: Runbook — Demand Forecaster Drift Alert
source_type: runbook
service: demand-forecaster
severity: high
owner: platform-oncall
---

# Runbook — Demand Forecaster Drift Alert

This runbook covers the `DriftHigh` alert raised by the monitoring DAG when the
hourly demand forecaster's input distribution diverges from its training baseline.

## Symptoms

- `DriftHigh` firing in the monitoring dashboard.
- Population Stability Index (PSI) above `0.2` on one or more input features.
- Forecast-vs-actual error (MAPE) trending upward over the last few hours.

## Likely causes

- A holiday, weather event, or incident shifting real-world demand.
- An upstream schema or units change in the ingestion pipeline.
- Stale features — the feature materialization job failed or lagged.

## First steps

1. Open the monitoring dashboard and confirm which features breached PSI.
2. Check the feature materialization job ran on schedule and did not error.
3. Compare the current input distribution against the training baseline.
4. If features are stale or wrong, fix ingestion first — do not retrain on bad data.

## Resolution

- Genuine demand shift: trigger the weekly training DAG to refresh the model, and
  verify the new candidate clears the promotion gate before it serves traffic.
- Data-quality issue: repair the upstream source, re-materialize features, and
  confirm PSI returns below `0.2`.

## Escalation

If the promotion gate keeps rejecting candidates, or forecast error stays elevated
after retraining, page the platform on-call and open an incident.
