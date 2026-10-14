---
title: Runbook — Model Promotion Gate
source_type: runbook
service: model-registry
severity: medium
owner: platform-oncall
---

# Runbook — Model Promotion Gate

The promotion gate blocks a newly trained model from serving traffic until it clears
automated checks. This runbook covers what to do when the gate keeps rejecting candidates.

## What the gate checks

- The candidate's backtest error (MAPE) is no worse than the current production model
  by more than 2%.
- No feature the model depends on has breached its drift threshold (PSI below 0.2).
- The model card is present and its metadata validates.

## When the gate keeps rejecting candidates

1. Read the gate's rejection reason in the training DAG logs — it names the failing check.
2. If backtest error regressed, the training data or features likely changed; inspect the
   recent ingestion and feature-materialization runs.
3. If a drift check failed, resolve the drift first (see the drift-alert runbook) before
   retraining — do not retrain on bad data.
4. If the model card failed validation, regenerate it and re-run the gate.

## Escalation

If candidates keep failing after the underlying data issue is fixed, page the platform
on-call and open an incident. Do not manually override the gate.
