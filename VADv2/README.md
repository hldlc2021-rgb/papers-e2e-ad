# VADv2: End-to-End Vectorized Autonomous Driving via Probabilistic Planning

> **VADv2** — 从确定性回归改成概率分布采样，解决驾驶行为的不确定性
>
> - 作者：Bo Jiang, Shaoyu Chen, Hao Gao, Bencheng Liao, Qian Zhang, Wenyu Liu, Xinggang Wang
> - 机构：华中科技大学 (HUST) / 地平线 (Horizon Robotics)
> - 发表：ICLR 2026
> - 论文：[arXiv:2402.13243](https://arxiv.org/abs/2402.13243)
> - 代码：[hustvl/VAD](https://github.com/hustvl/VAD)
> - 阅读时间：2026-06-10 🐋

---

## 一、核心问题

现有端到端规划方法（VAD、UniAD 等）采用 **确定性范式**——直接回归出一条轨迹或控制信号。但驾驶行为天然有不确定性（跟车时继续跟还是变道超车？对向来车时让还是超？），确定性回归遇到非凸解空间时，会产出"中间态"的折中轨迹，导致安全隐患。

> **"The variance of human driving behavior causes the ambiguity of the regression target."**

**VADv2 的答案：** 把规划建模成场景条件化的随机过程 `p(action|observation)`，学的是动作空间上的 **概率分布**，然后采样，而不是直接回归一个值。

---

## 二、整体架构

```
多视图图像序列 (6 views)
     ↓
 BEV Encoder (BEVFormer)
     ↓
 Scene Tokens ──────────────────────────────┐
  ├─ Map tokens (车道线/边界/人行道)          │
  ├─ Agent tokens (其他参与者运动预测)         │
  ├─ Traffic element tokens (红绿灯/停止牌)    │  ← 新增
  └─ Image tokens (前视图密集特征)             │  ← 新增
     ↓                                       │
 E_navi (导航目标点) ─────┐                   │
 E_state (自车速度/状态) ──┤ ──→ ego_feats ──┘
                          │       ↓
 Planning Vocabulary ──→ E(action) ──→ Transformer Decoder
 (4096条预设轨迹)    (sin/cos编码)   (action token 做 Query,
                                     scene tokens 做 K,V)
                                           ↓
                          4个分类分支，各输出 [N] 分数
                        ├─ collision score  → 会不会撞
                        ├─ boundary score   → 会不会压线
                        ├─ centerline score → 跟车道对齐度
                        └─ expert score     → 跟专家驾驶匹配度
                                           ↓
                              选最高分的 → PID → 控制信号
```

---

## 三、Scene Encoder（继承 VAD + 增强）

VADv2 的 scene encoder 基本沿用 VAD 的架构（BEV + map tokens + agent tokens），但做了增强：

| 组件 | 来源 | 说明 |
|---|---|---|
| Map tokens | MapTRv2 | 车道中心线、车道分割线、路边界、人行横道 |
| Agent tokens | VAD | 位置、朝向、尺寸、速度、未来轨迹，K 条多模态 |
| **Traffic element tokens** | **新增** | 前视图 → MLP → 红绿灯颜色/触发状态 + 停止牌重叠 |
| **Image tokens** | **新增** | 前视图密集特征，补充实例级 token 丢失的细节 |
| E_navi | MLP | 导航目标点编码，替代 VAD 里复杂的 Goal Generation |
| E_state | MLP | 自车速度、加速度等状态编码 |

导航注入方式从 VAD 的 cross-attention Goal Generation **简化成 embedding 加法**：

```python
# VAD: nav_embed → cross-attn(map) → cross-attn(agent) → goal_feature
# VADv2: 直接 MLP 编码 → 加进 ego_feats
```

---

## 四、Probabilistic Planning — 核心创新

### 4.1 Planning Vocabulary 构建（离线）

vocabulary **不是学出来的**，是用贪心最远点采样从真实驾驶数据中预计算得到：

```
Algorithm: 贪心最远点采样（按终点距离）

输入：所有驾驶轨迹集合 S，目标词汇表大小 N
输出：词汇表 V

1. 从 S 中随机选一条轨迹 → V
2. 重复直到 |V| = N:
   a. 对 S 中每条轨迹 𝒂，计算：
      dist(𝒂, V) = min_{v ∈ V} || endpoint(𝒂) - endpoint(v) ||
                   只算终点(最后一个 waypoint)的距离
   b. 选 dist 最大的那条 → 加入 V
```

**为什么只算 endpoint？** 终点决定了轨迹的大方向（左转/右转/直行/多快），中间路径可以靠 PID 平滑。如果算全轨迹距离，同方向但路径微弯的轨迹会大量重复采样，不同方向的反而覆盖不够。

**代码确认：** 训练时 N=256，推理时 N=4096。

```python
# 从 .npy 文件加载，不参与训练
self.plan_anchors = np.load('./plan_anchors_endpoint_242.npy')
# shape: [N, fut_ts=6, 2] = [4096, 6, (x,y)]
```

### 4.2 E(action) — 无可学习参数的 Sin/Cos 编码

一条 action 轨迹由 T 个 waypoint 组成：

```
𝒂 = (x₁, y₁, x₂, y₂, ..., x_T, y_T)
```

先把每个坐标值用 Γ 函数做高维位置编码（类似 NeRF 和 Transformer positional encoding）：

```
γ(pos, j) = (cos(pos / 10000^{2πj/L}), sin(pos / 10000^{2πj/L}))
                 ↓ 不同频率
          低频 → 粗粒度位置  高频 → 细粒度精度
```

再把所有 waypoint 的编码拼接：

```python
E(𝒂) = [Γ(x₁), Γ(y₁), Γ(x₂), Γ(y₂), ..., Γ(x_T), Γ(y_T)]
```

**无可学习参数**——纯固定函数。学习发生在后续的 Transformer Decoder 和 MLP 中。

```python
# 一次性编码所有轨迹
_tmp = pos2posemb2d(
    self.used_plan_anchors.reshape(1, N * fut_ts, -1),
    num_pos_feats=embed_dims // 2  # 128
).reshape(N, fut_ts, -1)

ego_query = _tmp.unsqueeze(0).repeat(batch, 1, 1, 1).reshape(batch, N, -1)
ego_query = self.ego_query_pre_branch(ego_query)  # MLP: 1536→256
```

### 4.3 特征交互 — 级联 Decoder

```
E(action)  作为 Query  ← 256 条轨迹的 token
E(scene)   作为 Key / Value

步骤：
1. ego_pv_decoder    : action ↔ 前视图图像特征
2. ego_agent_decoder : action ↔ agent tokens（有车挡住没？）
3. ego_map_decoder   : action ↔ map tokens（在车道内吗？）
```

交互完后拼接成最终特征：

```python
ego_feats = (
    ego_agent_feat +    # 场景中其他参与者的信息
    ego_map_feat +      # 车道/路边界信息
    ego_cf_feat +       # 前视图密集图像特征
    ego_status_feat +   # 自车状态（速度等）
    1. * ego_wp_feat +  # 导航目标点
    0. * ego_cmdid_feat # 导航指令（权重0，弃用）
)
```

### 4.4 回归分支被显式禁用

```python
outputs_ego_trajs = self.plan_reg_branch(ego_feats)  # 确实存在
outputs_ego_trajs = outputs_ego_trajs * 0.            # 乘以 0 清掉
                   + self.used_plan_anchors[None]      # 替换成 vocabulary 轨迹

loss_plan_reg = (0. * ego_fut_preds).sum()             # loss 也置 0
```

**模型只学打分，不学生成轨迹。** 最终输出就是 vocabulary 里选出来的那条。

---

## 五、四种打分的详细计算方式 ⭐

这是本节的重点——每条预设轨迹会得到 4 个分数，每个分数对应一个独立的分支。

### 5.1 碰撞分数 — `plan_cls_col_branch`

**结构：** 一个 `Linear(256 → 1)`，输出 sigmoid 后的 0~1 碰撞概率。

**标签生成（`get_plan_col_target`）：**

借助 `PlanningMetric` 类进行几何碰撞检测：

```
对 vocabulary 中的每条轨迹 i:

  1. 获取 GT agent 的包围盒数据（位置/朝向/尺寸）
  2. 把 agent 的包围盒画到 BEV 占据图上（200×200 栅格，0.5m/pixel）：
     - vehicle 类 → segmentation 图
     - human 类 → pedestrian 图
  3. 占据图 = segmentation ∪ pedestrian

  4. 在自车未来的每个时间步：
     a. 把自车矩形轮廓（宽 1.85m × 长 4.084m）放在轨迹对应位置上
     b. 将轮廓转换成 BEV 栅格坐标
     c. 检查轮廓覆盖的栅格是否与占据图重叠

  5. 只要有一个时间步重叠 → label = 1（有碰撞风险）
     全部不重叠 → label = 0（安全）
```

**损失：** `F.binary_cross_entropy(pred, label)`，平均权重。

### 5.2 越界分数 — `plan_cls_bd_branch`

**结构：** 一个 `Linear(256 → 1)`，输出 sigmoid 后的 0~1 越界概率。

**标签生成（`get_plan_bd_target`）：**

```
对每条轨迹 i:

  1. 获取车道边界线（lane boundary class 的 map elements）
  2. 对轨迹每个路段（waypoint i 到 waypoint i+1）：
     a. 找到最近的边界线段
     b. 判断轨迹路段和边界线段是否相交（几何线段相交检测）
     c. 加入三个偏移量的冗余检测：
        - 左偏移 (-0.9m, +2.4m)
        - 右偏移 (+0.9m, +2.4m)
        - 前偏移 (0, +2.4m)
        任意一个偏移导致相交 → 判为越界

  3. 整条轨迹中只要有一个路段越界 → label = 1
```

**损失：** `F.binary_cross_entropy(pred, label)`，平均权重。

### 5.3 车道一致性分数 — `plan_cls_cl_branch`

**结构：** 一个 `Linear(256 → 1)`，输出连续值（经 clamp 到 [-1, 1]）。

**标签生成（`get_plan_cl_target`）：**

```
对每条轨迹 i:

  1. 获取车道中心线（lane centerline class 的 map elements）
     先用 linear 插值把每条中心线从 20 个点加密到 1000 个点

  2. 对轨迹每个时间步 t，找到最近的中心线点和对应线段：
     a. 算轨迹点与所有插值后的中心线点的距离 → 取最近点
     b. 找到最近点所在的线段（前后两个插值点）
     c. 算该中心线线段的方向向量

  3. 算轨迹路段的方向向量 vs 中心线线段方向向量的 cosine similarity:
     cos_sim = F.cosine_similarity(ego_vector, centerline_vector)

  4. 最终 label 的计算（有多个尝试版本，当前使用）:
     label = 1 - 轨迹到中心线的平均距离
     label = clamp(label, min=-1.)
     
     即：离中心线越近 → label 越接近 1
         离中心线越远 → label 越接近 0 或负数
```

**损失：** 带权重的分类/回归 loss。逻辑上是想让模型学会"哪条轨迹跟车道对齐得最好"。

### 5.4 专家匹配分数 — `plan_cls_expert_branch`

**结构：** 一个 2 层 MLP：`Linear(256→256) → LayerNorm → ReLU → Linear(256→1)`，输出 sigmoid。

**标签生成（`get_plan_expert_target`）：**

这是四个分支中**最复杂**的，只有 1 个正样本，其余 255 个都是负样本：

```
1. 计算所有轨迹与 GT 轨迹的 L2 距离：
   traj_dis[i] = Σ_t ||轨迹_i 在时间 t 的 waypoint - GT waypoint||

2. 默认全部打 label=1（负样本），权重 = clamp(traj_dis, 0, 100) × 100
   即：离 GT 越远的轨迹 → 权重越大 → 训练时更受惩罚

3. 如果该轨迹同时有碰撞或越界问题（col=1 或 bd=1）：
   权重再提高到 100

4. 找距离 GT 最近的那条轨迹：
   pos_idx = traj_dis.argmin()
   → 打 label=0（正样本），权重设为 100

5. 选中次数平衡机制：
   traj_selected_cnt[pos_idx] += 1
   scaling = traj_selected_cnt.sum() / traj_selected_cnt[pos_idx] / N
   scaling = clamp(scaling, 0.5, 2.0)
   权重 = 100 × scaling
   
   目的：防止总选中同一条轨迹（某条轨迹被频繁选为正样本后，权重提升，下次模型被迫选别的）
```

**损失：** 带 sample weight 的 binary cross-entropy。

### 5.5 分数组合方式

训练时四个分支各自独立计算 loss 反向传播。推理时可以选择不同的组合策略：

- **最简策略：** 选 `expert_score` 最高的轨迹
- **安全优先：** 在 `col_score < 阈值 && bd_score < 阈值` 的轨迹里选 `expert_score` 最高的
- **综合评分：** 多分数加权组合

论文中 CARLA 闭环默认使用最高分 + PID 控制，不做额外后处理。

---

## 六、损失函数

### 6.1 总 Loss

```
ℒ = ℒ_cls_col + ℒ_cls_bd + ℒ_cls_cl + ℒ_cls_expert + ℒ_token
```

所有回归类的 loss 权重为 0（显式禁用）：

```python
loss_plan_reg = dict(type='L1Loss', loss_weight=0.)
loss_plan_bound = dict(type='PlanMapBoundLoss', loss_weight=0.)
loss_plan_agent_dis = dict(type='PlanAgentDisLoss', loss_weight=0.)
loss_plan_map_theta = dict(type='PlanMapThetaLoss', loss_weight=0.)
```

### 6.2 Scene Token Loss（沿袭 VAD）

| 组件 | Loss | 来源 |
|---|---|---|
| Map tokens | L1 + Focal (分类) | MapTRv2 |
| Agent tokens | L1 (属性回归) + Focal (分类) + L1 (轨迹) + Focal (多模态分类) | VAD |
| Traffic element tokens | Focal (红绿灯状态/触发 + 停止牌触发) | **新增** |

---

## 七、推理流程

```python
if not self.training:
    self.plan_fut_mode = self.plan_fut_mode_testing  # 训练 256 → 推理 4096

# kinodynamic 滤波：筛掉物理不可行的轨迹
kinodynamic_mask = (轨迹首步位移 - 自车按当前速度可行驶距离).abs() < 阈值
used_index = torch.multinomial(kinodynamic_mask.float(), self.plan_fut_mode)
```

1. 4096 条轨迹全部通过 `pos2posemb2d` 编码成 token
2. 过三级 Transformer Decoder 与 scene tokens 交互
3. 四个分支各输出 4096 个分数
4. 选分数最高的轨迹
5. PID 控制器将 waypoints 转为 steer/throttle/brake

**不做轨迹修正**——vocabulary 里的轨迹直接来自真实驾驶数据，天然满足运动学约束。

真实部署场景可选加：top-K 候选 + rule-based 过滤 + optimization-based 后优化精修。

---

## 八、VAD → VADv2 改进一览

| 维度 | VAD (ICCV 2023) | VADv2 (ICLR 2026) |
|---|---|---|
| **规划范式** | 确定性回归 | 概率分布 + 采样 |
| **动作空间** | 连续回归出 waypoints | 离散 vocabulary (4096 条预设轨迹) |
| **Loss 核心** | L1 回归 + collision 约束 | Cross-entropy + conflict + 4 分类 loss |
| **交通信号** | ❌ 无 | ✅ 前视检测红绿灯/停止牌 |
| **导航注入** | Goal Generation (cross-attn) | E_navi 直接加进 ego_feats |
| **前视图应用** | 无 | 密集特征补充 + 交通信号检测 |
| **Rule-based wrapper** | 部分场景需要 | 不需要 |
| **E(action) 编码** | N/A | Sin/cos 位置编码（固定无参数） |
| **CARLA Town05 DS** | 30.3 | **85.1** |
| **RC** | 75.2 | **98.4** |
| **IS** | — | **0.87** |
| **nuScenes** | ✅ SOTA (open-loop) | 未重点评估（聚焦 closed-loop） |

---

## 九、代码实现细节

- `plan_anchors_endpoint_242.npy`：预计算词汇表，不参与训练，仅加载
- 训练时 kinodynamic 滤波保证选出的轨迹物理可行
- GT best-match 轨迹硬编码保底加入 `used_index[-1]`，保证正样本一定在 vocabulary 中
- `traj_selected_cnt`：记录每条轨迹被选为正样本的频次，平衡选中率防止偏向
- 代码中有大量被注释掉的实验版本 (v1~v11)，说明 expert 分支权重调参经历多轮迭代

---

## 十、四维评分

| 维度 | 分数 | 分析 |
|---|---|---|
| **结果指标** | **9/10** | CARLA Town05 DS 85.1 大幅超过现有方法（第二名 76.1），Bench2Drive 也排第一。仅用 camera 达到这个水平很强。减一分因为 nuScenes open-loop 没有系统评估。 |
| **推理速度** | **7/10** | 4096 条轨迹全部编码+decoder 开销不小。代码里没有给具体的 FPS 数字。不过 vocabulary 方法比扩散系列（Diffusion Planner/Drive）省掉迭代去噪的过程，应该是更快的。 |
| **可落地程度** | **8/10** | 预设 vocabulary + 纯分类，工程上简洁明了，不需要 rule-based wrapper 兜底。PID 控制足够轻量。减分点：4096 条轨迹的 cross-attention 对算力有要求，且 vocabulary 依赖驾驶数据的分布覆盖。 |
| **创新价值** | **8/10** | 把规划从回归问题彻底变成分类/打分问题，思路很清晰。受 LLM 启发做概率建模的类比也很直观。但不是完全从零到一——离散化动作空间在 motion prediction 领域（Trajeglish, MotionLM）已有先例。减分在工程创新 > 理论创新。 |

**综合评分：8.0/10** — 实用性很强、效果提升显著的端到端工作，概率建模的方向是对的，值得关注其后续。
