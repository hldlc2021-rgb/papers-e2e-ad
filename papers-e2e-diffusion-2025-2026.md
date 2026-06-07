# 端到端自动驾驶 + 扩散模型 论文汇编 (2025–2026)

> 收集时间：2026-06-06 | 来源：arXiv  
> 排除：VLA / World Model 路线，纯端到端 + 扩散模型  
> 整理：Orca 🐋

---

## 核心论文

### 1. DiffusionDriveV2 — RL-Constrained Truncated Diffusion
- **作者：** Jialv Zou, Shaoyu Chen, Bencheng Liao, Xinggang Wang et al. (华科 / 地平线)
- **时间：** 2025.12
- **核心贡献：** 在 DiffusionDrive 基础上引入 **RL 约束的截断扩散建模**。原版 DiffusionDrive 用预定义 anchor 划分动作空间解决模式坍塌，但模仿学习导致保守。V2 用 RL 在线优化截断扩散过程，生成更激进且合理的轨迹。
- **看点：** diffusion + RL 结合的工程范本，适合落地场景
- **arXiv ID：** 待查

### 2. FeaXDrive — Feasibility-aware Trajectory-Centric Diffusion Planning
- **作者：** Baoyun Wang, Zhuoren Li, Ran Yu et al.
- **时间：** 2026.04 (arXiv: **2604.12656**)
- **核心贡献：** 提出 **轨迹中心**（trajectory-centric）的扩散规划，而非常见的噪声中心（noise-centric）公式化。三大组件：
  - 自适应曲率约束训练（几何/运动学可行性）
  - 可行驶区域引导的逆扩散采样
  - 基于 GRPO 的可行后训练（feasibility-aware GRPO）
- **看点：** 直接针对轨迹可行性的痛点的解决方案，工程性极强
- **Benchmark：** NAVSIM 达到强闭环性能

### 3. TrajDiff — Trajectory-oriented BEV Conditioned Diffusion
- **作者：** Xingtai Gui, Jianbo Zhao, Wencheng Han et al.
- **时间：** 2025.11 (arXiv: **2512.00723**)
- **核心贡献：** 完全无感知标注的端到端生成方法。用 **Gaussian BEV heatmap** 隐式编码驾驶模态，设计 TrajBEV encoder + TB-DiT（Trajectory-oriented BEV Diffusion Transformer），只用原始传感器 + 未来轨迹训练。
- **看点：** 去掉感知标注，直接 BEV → 轨迹扩散，数据扩展性好
- **Benchmark：** NAVSIM 87.5 PDMS（SOTA among annotation-free），数据扩展后 88.5 PDMS

### 4. Mimir — Hierarchical Goal-Driven Diffusion with Uncertainty
- **作者：** Zebin Xing, Yupeng Zheng, Qichao Zhang et al.
- **时间：** 2025.12 (arXiv: **2512.07130**)
- **核心贡献：** 分层双系统框架，用 **Laplace 分布** 估计目标点不确定性，引入 **多速率引导机制** 提前预测扩展目标点以加速推理。
- **看点：** 不确定性显式建模 + 推理加速（1.6x 提升），实用导向
- **Benchmark：** Navhard/Navtest 上 EPDMS 提升 20%
- **代码：** https://github.com/ZebinX/Mimir-Uncertainty-Driving

### 5. DiffE2E — Hybrid Action Diffusion and Supervised Policy
- **作者：** Rui Zhao, Yuze Fan, Ziguo Chen et al.
- **时间：** 2025.05
- **核心贡献：** 重新审视端到端系统中的动作空间建模，提出混合范式——**扩散模型生成多模态动作分布 + 监督策略选择最终动作**，兼顾多样性（扩散）和可控性（监督）。
- **看点：** 较早探索 diffusion + E2E 的工作，思路简洁

### 6. Multimodal Action Diffusion for Robust E2E
- **作者：** Jorge Daniel Rodríguez-Vidal, Antonio M. López Peña et al. (UAB, Spain)
- **时间：** 2026.06
- **核心贡献：** 多模态动作扩散提升端到端系统的鲁棒性
- **看点：** 最新发表（2026.06），待细读

### 7. D³-MoE — Dual Disentangled Diffusion Mixture-of-Experts
- **作者：** Renju Feng, Pan Zhou, Duanfeng Chu et al.
- **时间：** 2026.06
- **核心贡献：** 双解耦扩散 MoE，解决多司机演示中的 "风格平均化" 问题
- **看点：** 风格可控 + 扩散 + MoE，思路独特

---

## 关键趋势总结

| 趋势 | 代表工作 | 含义 |
|------|----------|------|
| **扩散 + RL 闭环优化** | DiffusionDriveV2, FeaXDrive | 扩散做多模态提议, RL 做安全约束/可行后训练 |
| **去感知标注** | TrajDiff | BEV latent → 轨迹直接扩散, 低成本数据扩展 |
| **可行性显式建模** | FeaXDrive | 从噪声中心转向轨迹中心的扩散公式 |
| **不确定性感知** | Mimir | 扩散本身就是不确定性建模工具, 内禀优势 |
| **风格/动作解耦** | D³-MoE, DiffE2E | 扩散在此方面天然适配 |

---

**建议优先阅读：** FeaXDrive → DiffusionDriveV2 → TrajDiff → Mimir  

> 需要我 deep dive 哪篇的具体方法/代码/实验结果，随时说。
