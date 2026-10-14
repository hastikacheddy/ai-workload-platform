---
title: Postmortem — Serving Latency Spike
source_type: postmortem
service: serving-gateway
severity: high
---

# Postmortem — Serving Latency Spike

## Summary

On 2026-09-14 the serving gateway's p99 latency rose from 80ms to 1.2s for 40 minutes,
affecting all real-time inference callers.

## Root cause

A deploy enabled request-level logging synchronously in the hot path. The log sink
backpressured under load and blocked serving worker threads waiting to write.

## Resolution

Rolled back the deploy, then re-introduced logging as an asynchronous, buffered writer
off the request path.

## Action items

- Make request logging asynchronous by default.
- Add a p99 latency alert at 300ms wired to automatic canary rollback.
- Load-test any change that touches the request hot path before it reaches production.
