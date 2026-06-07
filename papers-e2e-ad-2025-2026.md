# 2025–2026 端到端自动驾驶论文汇编

> 收集时间：2026-06-06 | 来源：arXiv
> 整理：Orca 🐋

---

## 一、VLA (Vision-Language-Action) 范式

| 论文 | 时间 | 作者机构 | 核心贡献 |
|------|------|----------|----------|
| **Counterfactual VLA** | 2025.12 | NVIDIA / MIT / Stanford | 自反思式 VLA，通过反事实推理增强可解释性 |
| **ColaVLA** | 2025.12 | 港中文 / MMLab | 认知潜推理 + 分层并行轨迹规划 |
| **Spatial-aware VLM** | 2025.12→2026.05 | - | 将空间感知注入 VLM，解决 VLM 的 3D 盲区 |
| **SpaceDrive** | 2025.12→2026.05 | 德国高校 (Geiger) | 增强 VLM 的空间理解能力做端到端 |

**小结：** VLA 在 25–26 年快速升温。核心痛点——VLM 缺乏 3D 空间感知——正在被逐步解决。ColaVLA 的 cognitive latent reasoning 思路值得深挖。

---

## 二、World Model / 潜空间推理

| 论文 | 时间 | 核心思路 |
|------|------|----------|
| **FutureX** | 2025.12 | Latent Chain-of-Thought World Model，在潜空间做多步推演 |
| **Latent CoT World Modeling** | 2025.12 | Shuhan Tan、Boris Ivanovic (NVIDIA) 等 — 潜空间 CoT |
| **WorldRFT** | 2025.12 | Latent World Model + RL Fine-Tuning，用 RL 优化潜空间规划 |
| **PLAN-S** | 2026.06 | Latent Style Dynamics，融合驾驶风格到 world model |
| **Discrete-WAM** | 2026.06 | 统一离散化的 Vision-Action Token，做 World-Policy 联合学习 |
| **CLEAR** | 2026.06 | Cognition and Latent Evaluation for Adaptive Routing |
| **D³-MoE** | 2026.06 | 双解耦扩散 MoE，实现风格可控的端到端驾驶 |

**小结：** World model 是 25–26 年最热的细分方向。核心趋势是从 latent CoT → RL fine-tuning → 离散 token 统一，逐步走向可推理、可控的 world model 范式。

---

## 三、感知 → 规划 联合

| 论文 | 时间 | 核心贡献 |
|------|------|----------|
| **Bridging Predictive Uncertainty and Safe Action** | 2026.06 | Sample-Conditioned 可微规划，将预测不确定性显式桥接到规划 |
| **Rethinking Spatio-Temporal Alignment for E2E 3D Perception** | 2025.12 | 重新审视端到端感知中的时空对齐，非 attention 方案 |
| **End-to-End 3D Spatiotemporal Perception + V2X** | 2025.12 | 多视角协同感知 + V2X 多模态融合 |
| **From Human Intention to Action Prediction** | 2025.12 | Intention-driven 端到端，从人类意图推断动作 |
| **Navigation-Guided Sparse Scene Representation** | 2025.03 | 导航引导的稀疏场景表示 |

---

## 四、安全 / 极端场景 / 评测

| 论文 | 时间 | 核心贡献 |
|------|------|----------|
| **CRASH** | 2026.03 | Cognitive Reasoning Agent 分析安全风险根因 |
| **Driving in Corner Case** | 2025.12→2026.04 | 真实世界对抗闭环评测平台 |
| **Argus** | 2025.11 | 面向端到端系统的弹性安全保证框架 |
| **Challenger** | 2025.05 | 低成本对抗场景视频生成 |
| **EvoDrive** | 2026.06 | 基于 LLM Agent 的 Pareto 进化式安全场景生成 |

**小结：** 安全评测方向正在从 "数据驱动" 转向 "对抗生成 + LLM agent 驱动"，EvoDrive 的 Pareto 进化思路很实用。

---

## 五、数据集 / 基础设施

| 论文 | 时间 | 核心贡献 |
|------|------|----------|
| **StandardE2E** | 2026.06 | 统一端到端自动驾驶数据集格式框架 |
| **MapAgent** | 2026.06 | 工业级城市场景车道级地图生成 Agent |
| **KITScenes** | 2026.06 | 多模态数据集 (KITTI 新扩展) |

---

## 值得重点关注的论文

1. **ColaVLA** — 认知潜推理 + 分层规划，落地可行性高
2. **WorldRFT** — World Model + RL Fine-Tuning，思路直接且有效
3. **Discrete-WAM** — 离散 token 统一 world 和 policy，架构创新
4. **Counterfactual VLA** — 可解释性方案，对功能安全评审有价值
5. **EvoDrive** — 安全场景生成的 LLM Agent 方案，工程实用

---

> 注：搜索工具日志显示 2026.06 新论文密集发布，部分论文刚上线不久。
> 需要我 deep dive 哪篇，或者针对某一方向做技术细节拆解，随时说。
