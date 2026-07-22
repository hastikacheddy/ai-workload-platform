# AI Platform

**Cloud-native infrastructure for running production ML & LLM workloads on Kubernetes.**

<p>
<a href="https://github.com/hastikacheddy/ai-workload-platform/actions/workflows/mlops_pipeline.yaml"><img alt="CI" src="https://github.com/hastikacheddy/ai-workload-platform/actions/workflows/mlops_pipeline.yaml/badge.svg"></a>
<img alt="Python" src="https://img.shields.io/badge/python-3.11-blue">
<img alt="Tests" src="https://img.shields.io/badge/tests-161%20passing-brightgreen">
<img alt="Kubernetes" src="https://img.shields.io/badge/Kubernetes-native-blue">
<img alt="Serving" src="https://img.shields.io/badge/serving-KServe%20%7C%20vLLM-orange">
<img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

A framework-neutral platform that lets any team **register, deploy, and serve any AI
workload** — a CPU-bound gradient-boosted model or a GPU-served LLM — through one API,
one scheduler, and one observability stack. It is the internal equivalent of Vertex AI,
SageMaker, or Databricks Model Serving, built as a **working, tested system** rather than
a set of diagrams.

> **Status, stated up front.** The control plane, serving backends, canary routing,
> inference benchmarks, distributed trainer, FinOps, and LLMOps **run and are covered by
> 161 automated tests**. GPU node pools and A100s are **specified as infrastructure-as-code
> and Kubernetes manifests**, not provisioned. Nothing here pretends to own hardware it
> does not — see [What is real vs. designed](#what-is-real-vs-designed).

---

## Contents

- [What it is](#what-it-is) · [Architecture](#architecture) · [Capabilities](#capabilities)
- [The declarative workload interface](#the-declarative-workload-interface) · [Quickstart](#quickstart)
- [Reference workload](#reference-workload-nyc-taxi-demand-forecasting) · [Design & decision records](#design--decision-records)
- [What is real vs. designed](#what-is-real-vs-designed) · [Repository layout](#repository-layout) · [Tech stack](#tech-stack)

---

## What it is

MLOps ships *one* model well. A platform lets *many* teams ship *many* models without a
platform engineer in the loop. This repository is that platform layer:

| Concern | What the platform provides |
|---|---|
| **Self-service** | Register a model and deploy it via a REST API **or** a declarative `workload.yaml` |
| **Framework-neutral** | One control plane serves LightGBM, scikit-learn/ONNX, and vLLM/Transformers LLMs |
| **Compute-aware** | Placement onto **CPU vs GPU pools is derived from the workload**, not requested by the caller |
| **Safe rollout** | Canary traffic splitting with **automatic fallback** to the stable version |
| **Observable & costed** | Prometheus metrics plus per-model / per-request **cost attribution** (FinOps) |
| **GenAI-ready** | Prompt registry, guardrails, RAG, evaluation, and vLLM serving |

The design principle throughout: **the caller never branches on model type.** A team asks
for `{"model": "risk-copilot"}`; the platform resolves the version, selects the pool,
routes traffic, executes the correct backend, records metrics, and falls back on failure.
Adding a new serving engine (Triton, TGI) is a ~40-line backend, not an API change.

---

## Architecture

```
                    Developers / internal services
                                │
                                ▼
              ┌───────────────────────────────────┐
              │        AI Platform API (gateway)   │   src/platform/gateway.py
              │  /v1/models   /v1/workloads        │
              │  /v1/deployments   /v1/inference   │
              │  /v1/costs   /v1/metrics           │
              └───────┬───────────────────┬────────┘
             control  │                   │  data plane
             plane    ▼                   ▼
        ┌────────────────────┐   ┌────────────────────────┐
        │  Model Registry    │   │  Deployment Manager     │   src/platform/
        │  framework-neutral │   │  pool placement · canary│
        │  index + aliases   │   │  · fallback · scaling   │
        └─────────┬──────────┘   └───────────┬────────────┘
                  │                           │ build_backend()
                  │              ┌────────────┴───────────┐
                  │              ▼                        ▼
                  │      LightGBM (CPU pool)        vLLM / LLM (GPU pool)
                  │              └──── KServe · autoscale · canary ────┘
                  ▼
        MLflow registry              Kubernetes + Terraform (AKS)
        (source of truth)     ───────────────────────────────────────
                                  Observability · Security · FinOps
                               Prometheus · Grafana · OPA · Trivy · cosign
```

Three planes, cleanly separated: a **control plane** (what should run), a **data plane**
(serving the request), and a **management plane** (health and cost). The full narrative is
in [`docs/platform/ARCHITECTURE.md`](docs/platform/ARCHITECTURE.md).

---

## Capabilities

| Area | What is built | Where |
|---|---|---|
| **Platform API** | Framework-neutral control plane: register / deploy / infer / cost, plus a declarative `workload.yaml` | [`src/platform/`](src/platform/) |
| **Kubernetes-native serving** | KServe `InferenceService` for CPU (LightGBM) and GPU (vLLM), with canary and concurrency autoscaling | [`kubernetes/serving/`](kubernetes/serving/) |
| **LLM serving** | vLLM backend over the OpenAI-compatible API, scheduled on the GPU pool | [`backends.py`](src/platform/backends.py) · [ADR-002](docs/adr/002-why-vllm.md) |
| **GPU infrastructure** | Taints/tolerations, node affinity, GPU `ResourceQuota`, and priority classes (serving preempts training) | [`gpu-*.yaml`](kubernetes/serving/) · [GPU_DESIGN](docs/platform/GPU_DESIGN.md) |
| **Inference optimization** | Benchmark harness: p50/p95/p99, throughput, GPU utilization, **and TTFT + tokens/sec** for streaming LLMs | [`benchmarks/`](benchmarks/) |
| **Distributed training** | Data-parallel trainer with checkpoint/recovery, plus PyTorch DDP and Ray Train equivalents | [`src/training/distributed/`](src/training/distributed/) |
| **Reliability & multi-tenancy** | Failure catalogue, degradation ladders, canary→stable fallback, per-tenant GPU quotas | [RELIABILITY](docs/platform/RELIABILITY.md) |
| **FinOps** | Cost per request, cost per model, idle-GPU detection, monthly burn | [`finops.py`](src/platform/finops.py) · [COST_MODEL](docs/platform/COST_MODEL.md) |
| **LLMOps** | Prompt registry, guardrails (PII + injection), embeddings, vector store, RAG, evaluation gate | [`src/llmops/`](src/llmops/) |
| **Cloud architecture** | Azure AKS + GPU node pool + ACR + Blob + Key Vault + Monitor, as Terraform (`terraform validate` passes) | [`infra/azure/`](infra/azure/) · [ADR-001](docs/adr/001-why-kubernetes.md) |

---

## The declarative workload interface

The gateway's REST API is imperative; the platform also accepts a **declarative workload
spec** — the same shape describes a CPU model and a GPU LLM, and the platform reconciles
each onto the right pool. This is the "deploy any AI workload" contract.

```yaml
# A CPU ML-model workload                     # A GPU LLM workload — identical shape
name: taxi-forecast                           name: ops-copilot-llm
type: ml-model                                type: llm
runtime: lightgbm                             runtime: vllm
artifact_uri: "models:/TaxiDemand@champion"   artifact_uri: "http://vllm.mlops.svc:8000/v1"
resources: { cpu: "4", memory: 8Gi }          resources: { gpu: 1, gpu_type: nvidia-a100 }
scaling:   { min: 2, max: 10 }                scaling:   { min: 1, max: 4 }
```

```bash
curl -X POST localhost:8090/v1/workloads --data-binary @src/platform/schemas/taxi-forecast.yaml
```

One interface, opposite compute profiles — which is what makes this a *platform* rather
than a *service*.

---

## Quickstart

```bash
pip install -r requirements.txt && pip install -e .

# 1. Run the platform API (control plane + inference data plane)
uvicorn src.platform.gateway:app --port 8090
#    then POST to /v1/models · /v1/workloads · /v1/deployments · /v1/inference · /v1/costs

# 2. Inference-optimization benchmark (zero-dependency mock: vLLM vs vanilla HF)
python -m benchmarks.run_benchmark --demo           # latency + throughput
python -m benchmarks.run_benchmark --stream-demo     # TTFT + tokens/sec

# 3. Distributed training (dependency-free: shard -> all-reduce -> checkpoint -> recover)
python -m src.training.distributed.data_parallel_demo

# 4. LLM ops copilot, grounded in the reference workload's own data
python -m src.llmops.ops_copilot

# 5. Validate the Azure infrastructure
cd infra/azure && terraform init -backend=false && terraform validate

# 6. Tests (161; dependency-free where a backend or GPU would otherwise be required)
pytest tests/
```

---

## Reference workload: NYC taxi demand forecasting

The platform is demonstrated on a **real, production-grade forecasting model**, not a toy —
this is what keeps it honest. The workload forecasts NYC yellow-taxi demand (daily and
hourly) with a **leakage-free** LightGBM forecaster and calibrated 99% Value-at-Risk bands:

| Model | Test window | MAPE | 99% interval coverage |
|---|---|---:|---:|
| Daily | 23 days | **5.4%** | 95.7% |
| Hourly | 24 days (576 h) | **8.1%** | 100% |

It ships with a champion–challenger promotion gate, a closed drift-retraining loop, a
hardened FastAPI serving path, and an Airflow orchestration layer. The platform's **Ops
Copilot** answers operator questions ("why did demand spike yesterday?") grounded in this
workload's own demand data. Modelling detail lives in the
[`src/forecasting/`](src/forecasting/), [`src/monitoring/`](src/monitoring/), and
[`src/serving/`](src/serving/) packages.

---

## Design & decision records

The design documents match the code, decision for decision:

| Document | Subject |
|---|---|
| [ARCHITECTURE](docs/platform/ARCHITECTURE.md) | The three planes, request lifecycle, compute placement |
| [GPU_DESIGN](docs/platform/GPU_DESIGN.md) | Pools, taints, priority/preemption, fractional GPU, autoscaling |
| [RELIABILITY](docs/platform/RELIABILITY.md) | Failure catalogue, degradation ladders, blast-radius isolation |
| [SCALING](docs/platform/SCALING.md) | CPU vs GPU scaling signals, bottleneck order, capacity planning |
| [COST_MODEL](docs/platform/COST_MODEL.md) | cost/request, cost/model, idle-GPU detection, auto scale-down |
| [DISASTER_RECOVERY](docs/platform/DISASTER_RECOVERY.md) | RPO/RTO targets, backup/restore, scenario runbooks |
| [SECURITY](docs/platform/SECURITY.md) | Multi-tenancy isolation, OWASP-LLM threats, secrets |
| [ADRs 001–004](docs/adr/) | Why Kubernetes · why vLLM · model-serving choice · platform API abstraction |

---

## What is real vs. designed

| Component | Status |
|---|---|
| Platform API (models / workloads / deployments / inference / costs / metrics) | **Runs, 161 tests** |
| LightGBM backend (wraps the real forecasting model) | **Real** |
| Canary split, fallback, auto-degrade | **Real, tested** |
| vLLM backend | **Real when pointed at a served model** (`VLLM_BASE_URL`); honest `unavailable` otherwise |
| Inference benchmark (latency / throughput / TTFT / tokens-sec) | **Runs** (mock + real OpenAI-compatible path) |
| Distributed data-parallel trainer | **Runs, tested** (all-reduce ≡ single-worker to 1e-10) |
| FinOps cost attribution | **Runs, tested** |
| LLMOps (prompt registry, guardrails, RAG, eval, copilot) | **Runs, tested** |
| Deployment Manager control loop | **In-process simulation**; production actuation is the KServe manifests |
| GPU node pools / A100s | **Designed as IaC + manifests**, not provisioned |

The engineering that *is* here — the abstraction boundary, routing/fallback, the pool
model, the benchmark methodology, the all-reduce correctness — is exactly what transfers to
a cluster that owns the accelerators.

---

## Repository layout

| Path | What |
|---|---|
| [`src/platform/`](src/platform/) | Control plane — registry, deployments, backends, gateway, workloads, FinOps |
| [`src/llmops/`](src/llmops/) | Prompt registry, guardrails, embeddings, vector store, RAG, evaluation, ops copilot |
| [`src/training/distributed/`](src/training/distributed/) | Data-parallel trainer + PyTorch DDP + Ray Train |
| [`benchmarks/`](benchmarks/) | Inference-optimization harness (latency, throughput, TTFT, tokens/sec) |
| [`kubernetes/serving/`](kubernetes/serving/) | KServe (CPU + vLLM GPU) + GPU scheduling (taints, quotas, priorities) |
| [`infra/azure/`](infra/azure/) | Azure AKS + GPU pool + ACR + Blob + Key Vault Terraform |
| [`docs/platform/`](docs/platform/), [`docs/adr/`](docs/adr/) | Design documents and architecture decision records |
| [`src/forecasting/`](src/forecasting/), [`src/serving/`](src/serving/), [`src/monitoring/`](src/monitoring/) | The reference forecasting workload (model, serving, drift loop) |
| [`dags/`](dags/), [`src/pipelines/`](src/pipelines/) | Airflow orchestration for the reference workload |

---

## Tech stack

`Python 3.11 · FastAPI · Kubernetes · KServe · vLLM · Ray Train · PyTorch DDP · Terraform
(Azure AKS) · MLflow · Prometheus · Grafana · OPA · Trivy · cosign · LightGBM · Feast ·
Airflow · DVC · Bandit · Semgrep`

## License

MIT — see [LICENSE](LICENSE).
