# AI 平台

[English](README.md) | 简体中文

**在 Kubernetes 上运行生产级机器学习与大语言模型（LLM）工作负载的云原生基础设施。**

<p>
<a href="https://github.com/hastikacheddy/ai-workload-platform/actions/workflows/mlops_pipeline.yaml"><img alt="CI" src="https://github.com/hastikacheddy/ai-workload-platform/actions/workflows/mlops_pipeline.yaml/badge.svg"></a>
<img alt="Python" src="https://img.shields.io/badge/python-3.11-blue">
<img alt="Tests" src="https://img.shields.io/badge/tests-161%20passing-brightgreen">
<img alt="Kubernetes" src="https://img.shields.io/badge/Kubernetes-native-blue">
<img alt="Serving" src="https://img.shields.io/badge/serving-KServe%20%7C%20vLLM-orange">
<img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

一个与框架无关的平台，让任何团队都能通过**同一个 API、同一个调度器、同一套可观测性体系**，**注册、部署并运行任意 AI 工作负载**——无论是 CPU 上运行的梯度提升模型，还是 GPU 上部署的大语言模型。它相当于企业内部版的 Vertex AI、SageMaker 或 Databricks Model Serving，并且是一个**真正可运行、经过测试的系统**，而不只是一堆架构图。

> **先说明现状。** 控制平面、推理后端、金丝雀路由、推理基准测试、分布式训练、FinOps 和 LLMOps **均可运行，并由 161 个自动化测试覆盖**。GPU 节点池和 A100 **以基础设施即代码（IaC）和 Kubernetes 清单的形式定义**，并未实际开通。这里不会假装拥有并不存在的硬件——详见[哪些是真实运行的，哪些是设计](#哪些是真实运行的哪些是设计)。

---

## 目录

- [这是什么](#这是什么) · [架构](#架构) · [功能](#功能)
- [声明式工作负载接口](#声明式工作负载接口) · [快速开始](#快速开始)
- [参考工作负载](#参考工作负载纽约出租车需求预测) · [设计文档与决策记录](#设计文档与决策记录)
- [哪些是真实运行的，哪些是设计](#哪些是真实运行的哪些是设计) · [仓库结构](#仓库结构) · [技术栈](#技术栈)

---

## 这是什么

MLOps 能把*一个*模型交付好；平台则让*许多*团队在不需要平台工程师介入的情况下交付*许多*模型。本仓库就是这一平台层：

| 关注点 | 平台提供的能力 |
|---|---|
| **自助服务** | 通过 REST API **或**声明式 `workload.yaml` 注册并部署模型 |
| **框架无关** | 同一个控制平面支持 LightGBM、scikit-learn/ONNX 以及 vLLM/Transformers 大语言模型 |
| **感知算力** | **CPU 池还是 GPU 池由工作负载本身推导决定**，而不是由调用方指定 |
| **安全发布** | 金丝雀流量切分，并可**自动回退**到稳定版本 |
| **可观测、可计费** | Prometheus 指标，以及按模型 / 按请求的**成本归因**（FinOps） |
| **面向生成式 AI** | 提示词注册表、安全护栏、RAG、评估，以及 vLLM 推理服务 |

贯穿始终的设计原则：**调用方永远不需要根据模型类型写分支逻辑。** 团队只需请求 `{"model": "risk-copilot"}`，平台就会解析版本、选择资源池、路由流量、调用正确的后端、记录指标，并在失败时自动回退。接入新的推理引擎（Triton、TGI）只需大约 40 行的后端代码，而不需要修改 API。

---

## 架构

```
                    开发者 / 内部服务
                                │
                                ▼
              ┌───────────────────────────────────┐
              │      AI 平台 API（网关）           │   src/platform/gateway.py
              │  /v1/models   /v1/workloads        │
              │  /v1/deployments   /v1/inference   │
              │  /v1/costs   /v1/metrics           │
              └───────┬───────────────────┬────────┘
              控制平面 │                   │ 数据平面
                      ▼                   ▼
        ┌────────────────────┐   ┌────────────────────────┐
        │  模型注册表         │   │  部署管理器             │   src/platform/
        │  框架无关的索引     │   │  资源池分配 · 金丝雀    │
        │  + 别名             │   │  · 回退 · 扩缩容        │
        └─────────┬──────────┘   └───────────┬────────────┘
                  │                           │ build_backend()
                  │              ┌────────────┴───────────┐
                  │              ▼                        ▼
                  │      LightGBM（CPU 池）         vLLM / LLM（GPU 池）
                  │              └──── KServe · 自动扩缩容 · 金丝雀 ────┘
                  ▼
        MLflow 注册表               Kubernetes + Terraform（AKS）
        （唯一事实来源）      ───────────────────────────────────────
                                  可观测性 · 安全 · FinOps
                               Prometheus · Grafana · OPA · Trivy · cosign
```

三个平面清晰分离：**控制平面**（应该运行什么）、**数据平面**（处理请求）和**管理平面**（健康状况与成本）。完整说明见 [`docs/platform/ARCHITECTURE.md`](docs/platform/ARCHITECTURE.md)。

---

## 功能

| 领域 | 已构建的内容 | 位置 |
|---|---|---|
| **平台 API** | 框架无关的控制平面：注册 / 部署 / 推理 / 成本，外加声明式 `workload.yaml` | [`src/platform/`](src/platform/) |
| **Kubernetes 原生推理** | 用于 CPU（LightGBM）和 GPU（vLLM）的 KServe `InferenceService`，支持金丝雀发布和基于并发的自动扩缩容 | [`kubernetes/serving/`](kubernetes/serving/) |
| **LLM 推理** | 基于 OpenAI 兼容 API 的 vLLM 后端，调度到 GPU 池 | [`backends.py`](src/platform/backends.py) · [ADR-002](docs/adr/002-why-vllm.md) |
| **GPU 基础设施** | 污点/容忍、节点亲和性、GPU `ResourceQuota` 和优先级类（推理可抢占训练） | [`gpu-*.yaml`](kubernetes/serving/) · [GPU_DESIGN](docs/platform/GPU_DESIGN.md) |
| **推理优化** | 基准测试框架：p50/p95/p99、吞吐量、GPU 利用率，**以及流式 LLM 的首 token 时延（TTFT）和每秒 token 数** | [`benchmarks/`](benchmarks/) |
| **分布式训练** | 支持检查点与故障恢复的数据并行训练器，并提供 PyTorch DDP 和 Ray Train 等价实现 | [`src/training/distributed/`](src/training/distributed/) |
| **可靠性与多租户** | 故障目录、降级阶梯、金丝雀→稳定版回退、按租户的 GPU 配额 | [RELIABILITY](docs/platform/RELIABILITY.md) |
| **FinOps** | 每请求成本、每模型成本、闲置 GPU 检测、月度消耗 | [`finops.py`](src/platform/finops.py) · [COST_MODEL](docs/platform/COST_MODEL.md) |
| **LLMOps** | 提示词注册表、安全护栏（PII + 提示注入）、向量嵌入、向量库、RAG、评估门禁 | [`src/llmops/`](src/llmops/) |
| **智能体 RAG 助手** | 混合检索 RAG 服务（BM25 + kNN，RRF 融合）配合 LangGraph 智能体——基于平台运维手册和 ADR 给出有据可查、附引用的回答；已端到端实测验证 | [`platform-copilot/`](platform-copilot/) |
| **云架构** | 以 Terraform 定义的 Azure AKS + GPU 节点池 + ACR + Blob + Key Vault + Monitor（`terraform validate` 通过） | [`infra/azure/`](infra/azure/) · [ADR-001](docs/adr/001-why-kubernetes.md) |

---

## 声明式工作负载接口

网关的 REST API 是命令式的；平台同时也接受**声明式的工作负载描述**——同一种格式既可以描述 CPU 模型，也可以描述 GPU 大模型，平台会把它们分别调度到合适的资源池。这就是“部署任意 AI 工作负载”的约定。

```yaml
# CPU 机器学习模型工作负载                     # GPU 大语言模型工作负载——格式完全相同
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

同一个接口，截然相反的算力需求——这正是它成为*平台*而不只是一个*服务*的原因。

---

## 快速开始

```bash
pip install -r requirements.txt && pip install -e .

# 1. 启动平台 API（控制平面 + 推理数据平面）
uvicorn src.platform.gateway:app --port 8090
#    然后向 /v1/models · /v1/workloads · /v1/deployments · /v1/inference · /v1/costs 发送 POST 请求

# 2. 推理优化基准测试（零依赖模拟：vLLM 对比原生 HF）
python -m benchmarks.run_benchmark --demo           # 延迟 + 吞吐量
python -m benchmarks.run_benchmark --stream-demo     # TTFT + 每秒 token 数

# 3. 分布式训练（无依赖：分片 -> all-reduce -> 检查点 -> 恢复）
python -m src.training.distributed.data_parallel_demo

# 4. LLM 运维助手，基于参考工作负载自身的数据作答
python -m src.llmops.ops_copilot

# 5. 校验 Azure 基础设施
cd infra/azure && terraform init -backend=false && terraform validate

# 6. 测试（161 个；在原本需要后端或 GPU 的地方均已做到无依赖）
pytest tests/
```

---

## 参考工作负载：纽约出租车需求预测

平台以一个**真实的生产级预测模型**作为演示，而不是玩具示例——这让整个平台保持“言行一致”。该工作负载使用**无数据泄漏**的 LightGBM 预测器和经过校准的 99% 风险价值（VaR）区间，预测纽约黄色出租车的每日与每小时需求：

| 模型 | 测试窗口 | MAPE | 99% 区间覆盖率 |
|---|---|---:|---:|
| 按天 | 23 天 | **5.4%** | 95.7% |
| 按小时 | 24 天（576 小时） | **8.1%** | 100% |

它配备了冠军-挑战者晋升门禁、闭环漂移重训练、加固的 FastAPI 推理路径以及 Airflow 编排层。平台的**运维助手（Ops Copilot）**会基于该工作负载自身的需求数据回答运维人员的问题（例如“昨天的需求为什么会激增？”）。建模细节位于 [`src/forecasting/`](src/forecasting/)、[`src/monitoring/`](src/monitoring/) 和 [`src/serving/`](src/serving/)。

---

## 设计文档与决策记录

设计文档与代码一一对应，每一项决策都有据可查：

| 文档 | 主题 |
|---|---|
| [ARCHITECTURE](docs/platform/ARCHITECTURE.md) | 三个平面、请求生命周期、算力分配 |
| [GPU_DESIGN](docs/platform/GPU_DESIGN.md) | 资源池、污点、优先级/抢占、GPU 切分、自动扩缩容 |
| [RELIABILITY](docs/platform/RELIABILITY.md) | 故障目录、降级阶梯、故障影响范围隔离 |
| [SCALING](docs/platform/SCALING.md) | CPU 与 GPU 的扩缩容信号、瓶颈顺序、容量规划 |
| [COST_MODEL](docs/platform/COST_MODEL.md) | 每请求成本、每模型成本、闲置 GPU 检测、自动缩容 |
| [DISASTER_RECOVERY](docs/platform/DISASTER_RECOVERY.md) | RPO/RTO 目标、备份/恢复、场景化运维手册 |
| [SECURITY](docs/platform/SECURITY.md) | 多租户隔离、OWASP LLM 威胁、密钥管理 |
| [ADR 001–004](docs/adr/) | 为什么选 Kubernetes · 为什么选 vLLM · 推理服务选型 · 平台 API 抽象 |

---

## 哪些是真实运行的，哪些是设计

| 组件 | 状态 |
|---|---|
| 平台 API（models / workloads / deployments / inference / costs / metrics） | **可运行，161 个测试** |
| LightGBM 后端（封装真实的预测模型） | **真实** |
| 金丝雀切分、回退、自动降级 | **真实，已测试** |
| vLLM 后端 | **指向已部署模型时为真实运行**（`VLLM_BASE_URL`）；否则如实返回 `unavailable` |
| 推理基准测试（延迟 / 吞吐量 / TTFT / 每秒 token 数） | **可运行**（模拟路径 + 真实 OpenAI 兼容路径） |
| 分布式数据并行训练器 | **可运行，已测试**（all-reduce 与单机结果误差小于 1e-10） |
| FinOps 成本归因 | **可运行，已测试** |
| LLMOps（提示词注册表、护栏、RAG、评估、助手） | **可运行，已测试** |
| 部署管理器控制循环 | **进程内模拟**；生产环境的实际执行由 KServe 清单完成 |
| GPU 节点池 / A100 | **以 IaC + 清单形式设计**，未实际开通 |

这里*真正*实现的工程部分——抽象边界、路由/回退、资源池模型、基准测试方法、all-reduce 正确性——正是可以直接迁移到拥有加速器的集群上的那部分。

---

## 仓库结构

| 路径 | 内容 |
|---|---|
| [`src/platform/`](src/platform/) | 控制平面——注册表、部署、后端、网关、工作负载、FinOps |
| [`src/llmops/`](src/llmops/) | 提示词注册表、护栏、向量嵌入、向量库、RAG、评估、运维助手 |
| [`src/training/distributed/`](src/training/distributed/) | 数据并行训练器 + PyTorch DDP + Ray Train |
| [`benchmarks/`](benchmarks/) | 推理优化测试框架（延迟、吞吐量、TTFT、每秒 token 数） |
| [`kubernetes/serving/`](kubernetes/serving/) | KServe（CPU + vLLM GPU）+ GPU 调度（污点、配额、优先级） |
| [`infra/azure/`](infra/azure/) | Azure AKS + GPU 池 + ACR + Blob + Key Vault 的 Terraform 定义 |
| [`docs/platform/`](docs/platform/)、[`docs/adr/`](docs/adr/) | 设计文档与架构决策记录 |
| [`src/forecasting/`](src/forecasting/)、[`src/serving/`](src/serving/)、[`src/monitoring/`](src/monitoring/) | 参考预测工作负载（模型、推理服务、漂移闭环） |
| [`dags/`](dags/)、[`src/pipelines/`](src/pipelines/) | 参考工作负载的 Airflow 编排 |

---

## 技术栈

`Python 3.11 · FastAPI · Kubernetes · KServe · vLLM · Ray Train · PyTorch DDP · Terraform
(Azure AKS) · MLflow · Prometheus · Grafana · OPA · Trivy · cosign · LightGBM · Feast ·
Airflow · DVC · Bandit · Semgrep`

## 许可证

MIT，详见 [LICENSE](LICENSE)。
