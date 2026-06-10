# VAD: Vectorized Scene Representation for Efficient Autonomous Driving

> **VAD** — 端到端自动驾驶的向量化范式，将驾驶场景建模为纯向量化表示
>
> - 作者：Bo Jiang, Shaoyu Chen, Qing Xu, Bencheng Liao, Jiajie Chen, Helong Zhou, Qian Zhang, Wenyu Liu, Chang Huang, Xinggang Wang
> - 机构：华中科技大学 (HUST) / 地平线 (Horizon Robotics)
> - 发表：ICCV 2023
> - 论文：[arXiv:2303.12077](https://arxiv.org/abs/2303.12077)
> - 代码：[hustvl/VAD](https://github.com/hustvl/VAD)
> - 阅读时间：2026-06-09 🐋

---

## 一、核心问题

端到端自动驾驶中，**场景表示（Scene Representation）** 是连接感知、预测、规划的枢纽。此前主流方法（如 ST-P3、UniAD）使用 **密集栅格化（dense rasterized）** 表示：

- Agent 用 **占用栅格（occupancy map）** 表示
- 地图用 **语义分割图（semantic map）** 表示

这带来两个根本问题：

1. **计算开销大** — 栅格化表示需要 CNN 密集计算，且分辨率影响推理速度与场景细节的 trade-off
2. **丢失实例级结构信息** — 栅格化是像素级的，无法区分不同实例的边界和个体运动意图，导致规划约束不够明确

**VAD 的核心洞察：** 能否将整个驾驶场景——包括动/静态元素——全部用一个 **向量化（vectorized）**、**实例级（instance-level）** 的表示来建模？

---

## 二、算法深度拆解

VAD 是一个 **纯端到端** 的向量化自动驾驶范式，核心思想是 **Vectorized Scene Representation + Vectorized Planning**。

### 0. 全貌：输入→输出

```
输入：6 路环视图像 + HD 地图 + 导航指令
输出：自车轨迹 (未来 3s, 以 waypoints 形式)
中间表示：纯向量化场景 (agent vectors + map vectors)
设计原则：没有像素级表示，全程是 queries + vectors
```

### 2.1 特征提取与 BEV 编码

Backbone（VoVNet-99 等）提取多视图图像特征，通过 **BEVFormer 风格 Transformer** 做视角转换：

- 预定义一组 **BEV queries**（grid-based），通过 cross-attention 在 multi-view 图像特征上采样
- 得到 **BEV 特征图**（通常 200×200 分辨率）

⚠️ VAD **不直接对 BEV 特征图做规划**（这是过去的方法），而是把 BEV 特征作为桥梁去 decode 出向量化元素。

---

### 2.2 向量化场景表示 — 核心创新

这一步分两条线并行，都通过 **Transformer Decoder + learnable queries** 实现：

#### Agent Queries（智能体检测+预测）

```python
# 伪代码逻辑
agent_queries = nn.Embedding(num_queries=900, dim=256)  # 可学习

# Transformer Decoder 迭代 refine
for layer in decoder_layers:
    agent_queries = cross_attention(agent_queries, BEV_features)
    agent_queries = self_attention(agent_queries)  # agent 间交互

# 从 query 解码出:
bbox_3d = MLP(agent_queries)       # → (x, y, z, w, l, h, yaw)
velocity = MLP(agent_queries)       # → (vx, vy)
trajectories = MLP(agent_queries)   # → K 条轨迹 × T 个 waypoint (K=4, T=6)
```

每个 agent query 解码出：3D 检测框、速度向量 (vx, vy)、**多模态轨迹预测**（K 条未来轨迹，每条带置信度分数）。这些预测轨迹用来做规划阶段的碰撞约束。

#### Map Queries（地图向量化）

```python
map_queries = nn.Embedding(num_queries=2000, dim=256)

for layer in decoder_layers:
    map_queries = cross_attention(map_queries, BEV_features)

polyline_points = MLP(map_queries)  # → N 个控制点 × 2 (x, y)
semantic_class = MLP(map_queries)   # → 车道线、路沿、人行道等
```

同一根车道线由多个连续 query 通过 **attention-based grouping** 聚合成一条 polyline。

> ⚠️ **关键区别：VAD 没有显式的 ego_query。**
> 自车（ego）**不**在 query 体系里。自车状态（位置、朝向、速度）来自 CAN 信号，直接被喂入 trajectory head。
> - 不需要 query 是因为自车是已知量，无需从图像里 decode
> - BEV 网格以自车为中心，(0,0) 即自车位置，在特征空间里隐式存在
> - 这个设计被后续 SparseDrive 改进，它引入了显式 ego_query

| Query 类型 | 数量 | 作用 |
|-----------|:----:|------|
| agent queries | ~900 | 检测+预测其他车辆/行人 |
| map queries | ~2000 | 解码车道线、路沿等元素 |
| ego query | ❌ 无 | 自车状态从 CAN 输入 |

---

### 2.3 Goal Generation（目标生成）

**要解决的问题：** 同样的场景，导航说"直行"和"右转"，自车轨迹完全不同。而 agent/map vectors 只告诉你"哪里有路"、"哪里有车"，**不告诉你"你要去哪"**。Goal Generation 的作用就是把导航意图注入规划器。

VAD 里 Goal Generation **不直接输出目标坐标**，而是输出一个**条件特征向量 goal_feature**，让轨迹生成器知道自己该往哪个方向去：

```python
# 导航指令编码
nav_embed = embedding("右转")  # token 化后查 embedding 表

# goal_query 融合导航 + 场景信息
goal_query = nav_embed + positional_encoding

# cross-attention 找目标方向的车道
goal_query = cross_attention(query=goal_query, key/value=map_vectors)
# cross-attention 看目标路径上有没有车挡着
goal_query = cross_attention(query=goal_query, key/value=agent_vectors)

goal_feature = goal_query  # 输出 1×256 向量
```

**为什么需要它？** 没有 goal_feature，trajectory head 同时产出直行/右转/左转的候选轨迹，scoring network 难以对比；有了它，所有候选轨迹都偏向"右转"方向，只需在"都右转"的候选里选最优。**它本质是一个条件先验（conditional prior），解决轨迹歧义性。**

> **一句话：Goal Generation = 把导航信号转化成规划器能理解的条件特征，告诉轨迹生成器"往哪个方向走"。**

---

### 2.4 Constraint Generation（约束生成）— ⭐ 核心创新

**本质：把向量化场景元素（agent vectors + map vectors）转化成可微的代价函数，用来约束自车轨迹。**

VAD 的约束模块不像 MPC 那样解优化问题，而是用 learning 方式来**模拟**约束行为。

#### 第一步：生成约束特征

将 agent 和 map 向量编码成两个约束特征向量，与 goal_feature 拼接后输入 trajectory head：

```python
# 碰撞约束特征 — 从 agent vectors 编码
B_occ = constraint_decoder(
    query=[goal_feature, ego_state],   # 条件：目标方向 + 自车状态
    key/value=agent_vectors             # 关注：哪些 agent 挡路
)

# 交通规则约束特征 — 从 map vectors 编码
B_lane = constraint_decoder(
    query=[goal_feature, ego_state],
    key/value=map_vectors
)

# 拼接作为 trajectory head 的条件
traj_head_input = [goal_feature, B_occ, B_lane, ego_state]
```

#### 第二步：生成带约束的候选轨迹

```python
hidden = MLP(traj_head_input)  # 融合约束信息

candidates = []
for _ in range(6):             # 生成 6 条候选轨迹
    traj = []
    state = ego_state
    for t in range(T):         # 逐时间步预测 waypoint
        delta = GRU(hidden, state)
        waypoint = state + delta
        traj.append(waypoint)
        state = waypoint
    candidates.append(traj)
```

由于输入中包含 B_occ（碰撞约束特征），模型在逐时间步生成轨迹时会**隐式地绕开**那些关注到的 agent。

#### 第三步：显式约束代价评分 ⭐

生成候选轨迹后，VAD 用**显式可微的代价函数**来评选最优轨迹。这些代价函数在训练时指导梯度更新，在推理时用于选优：

```python
def score_trajectory(traj, agent_vectors, map_vectors, goal_feature):
    """
    traj: (T, 2) — 自车候选轨迹 waypoints, T=6, dt=0.5s
    """

    # ============ 1. 碰撞代价 (Collision Cost) ============
    collision_cost = 0
    for k in range(N_agents):
        for m in range(K):  # K=4 多模态预测
            pred_traj = agent[k].pred_trajs[m]  # (T, 2)
            conf = agent[k].confidences[m]      # 置信度权重
            distances = [L2(traj[t], pred_traj[t]) for t in range(T)]
            # 距离 < 安全阈值 ε 则惩罚，越近惩罚越大
            penalty = sum(max(ε - d, 0) for d in distances)
            collision_cost += conf * penalty

    # ============ 2. 车道保持代价 (Lane Cost) ============
    lane_cost = 0
    for t, waypoint in enumerate(traj):
        dist_to_lanes = [L2_to_polyline(waypoint, lane) for lane in lane_polylines]
        nearest_dist = min(dist_to_lanes)
        lane_width = 3.5  # m
        lane_cost += max(nearest_dist - lane_width / 2, 0)

    # ============ 3. 路沿/边界代价 (Boundary Cost) ============
    boundary_cost = 0
    for waypoint in traj:
        for boundary in road_edge_polylines:
            dist = L2_to_polyline(waypoint, boundary)
            if dist < 0.5:  # 太靠近路沿
                boundary_cost += (0.5 - dist) * 10

    # ============ 4. 方向代价 (Direction Cost) ============
    goal_dir = decode_direction(goal_feature)
    direction_cost = 1 - cos_sim(traj[-1] - traj[0], goal_dir)

    # ============ 5. 舒适性代价 (Comfort Cost) ============
    # traj = [(x₀,y₀), (x₁,y₁), ..., (x₅,y₅)], dt=0.5s
    # 速度 = 一阶差分
    v = [(traj[t] - traj[t-1]) / dt for t in range(1, T)]
    # 加速度 = 二阶差分
    a = [(v[t] - v[t-1]) / dt for t in range(1, len(v))]
    # Jerk(加加速度) = 三阶差分
    j = [(a[t] - a[t-1]) / dt for t in range(1, len(a))]

    accel_cost = sum(a_t.x**2 + a_t.y**2 for a_t in a)  # 惩罚急加减速
    jerk_cost  = sum(j_t.x**2 + j_t.y**2 for j_t in j)  # 惩罚顿挫感
    comfort_cost = accel_cost + jerk_cost

    # ============ 综合评分 ============
    total_cost = (λ_coll * collision_cost +
                  λ_lane * lane_cost +
                  λ_bound * boundary_cost +
                  λ_dir * direction_cost +
                  λ_comfort * comfort_cost)

    return -total_cost  # 分数 = 负代价，越高越好

# 选最优
best_traj = candidates[argmax([score_trajectory(t, ...) for t in candidates])]
```

**各代价的作用：** 去掉碰撞约束 → Collision Rate 上升~2x；去掉车道约束 → 轨迹压线/出车道；去掉方向约束 → 轨迹方向漂移；全部去掉 → 退化为纯模仿学习，性能接近 UniAD。

#### ⚠️ 约束代价 vs 模仿学习 loss 的关键区别

| 类型 | 何时生效 | 作用 |
|------|---------|------|
| **L1 waypoint loss** | 训练时 | 让轨迹逼近人类驾驶数据（模仿学习） |
| **约束代价** | **训练 + 推理时** | 训练时指导梯度方向，推理时用于从候选轨迹中选优 |

**关键：VAD 在推理时有一个显式的、基于安全规则的选优过程，而不是纯开盲盒。** 这与 UniAD 只在训练时用 collision loss 有本质区别——VAD 的推理阶段也有规则介入。

---

### 2.5 损失函数（训练）

VAD 是多任务联合训练：

```
L_total = λ_det·L_det + λ_map·L_map + λ_pred·L_pred + λ_plan·L_plan
```

- **L_det**：检测 loss = focal loss（分类）+ L1 loss（3D框回归）
- **L_map**：地图 loss = focal loss（语义分类）+ L1 loss（polyline 控制点回归）
- **L_pred**：预测 loss = L1 waypoint loss + 模式概率交叉熵
- **L_plan**：规划 loss = **L1 waypoint loss**（模仿人类轨迹）+ **collision loss**（碰撞惩罚）

规划部分由模仿学习和碰撞惩罚互补——前者让轨迹贴近人类驾驶习惯，后者让轨迹远离周围 agent。

---

### 2.6 推理流程（汇总）

```
输入 6 张环视图像 → BEV 特征
                         ├→ agent queries → 3D 检测 + 速度 + 轨迹预测
                         └→ map queries  → 车道线/路沿 polyline
                                                  ↓
        导航 → goal_generation ──→ goal_feature ──┐
        agents/maps → constraint_generation → B_occ, B_lane ─┤
                                                              ↓
                                          trajectory_head → 6 候选轨迹
                                                  ↓
                                          显式代价评分 (推理时选优)
                                                  ↓
                                            最优轨迹 (waypoints)
```

---

### 2.7 时序处理 — 模型最大短板

VAD 本质是**单帧模型**。每一帧完全独立处理：

```
第 t 帧:   6 张图像 → BEV → queries → 轨迹输出
第 t+1 帧: 6 张图像 → BEV → queries → 轨迹输出
                         ↑ 每帧独立，无跨帧信息传递
```

| 常见时序机制 | VAD 有没有？ | 对比 UniAD |
|-------------|:----------:|-----------|
| 历史 BEV 特征拼接/融合 | ❌ | 无 |
| 时序 self-attention (跨帧) | ❌ | 无 |
| 检测结果 tracking/关联 | ❌ | UniAD 有 track query 帧间维持 |
| 帧间 query 传递 | ❌ | UniAD 的 query hidden state 跨帧传递 |
| 光流/时序一致性 loss | ❌ | 无 |

**唯一涉及时序的地方：**
1. **轨迹预测输出 waypoints 序列** — 这是**未来**方向的时序，不涉及历史帧
2. **L1 waypoint loss 跨时间步计算** — 但也在当前帧内

**带来的问题：**
- **检测 flicker**：遮挡/光照变化导致同一 agent 帧间 detection 不稳定 → 规划轨迹一跳一跳
- **缺乏轨迹平滑**：帧间输出的轨迹在重叠时间段没有衔接约束
- **运动信息丢失**：单帧难准确推断速度，UniAD 的 track query 帧间关联能提供更稳定的运动估计

**后续改进：**
- **VADv2**：加入时序 BEV 特征对齐 + 记忆队列（缓存 3 帧 BEV → warp → concat）
- **SparseDrive**：引入 ego_query 的时序传播，query hidden state 从上一帧传递给下一帧

---

## 三、实验与结果

### 3.1 数据集

- **nuScenes** — 端到端规划评测的标准 benchmark
- 评测指标：
  - **L2 Displacement Error** (m) — 轨迹 L2 误差
  - **Collision Rate** (%) — 碰撞率
  - 按不同时长（1s、2s、3s）细分

### 3.2 主要结果

| 方法 | L2 (1s)↓ | L2 (2s)↓ | L2 (3s)↓ | Collision (avg)↓ | 推理速度 |
|------|---------|---------|---------|-----------------|---------|
| **ST-P3** (CVPR'22) | 1.15 | 2.40 | 5.38 | 1.93% | - |
| **UniAD** (CVPR'23) | 0.62 | 1.40 | 3.56 | 1.37% | 1x (baseline) |
| **VAD-Base** | **0.54** | **1.14** | **2.73** | **0.97%** | **2.5x** |
| **VAD-Tiny** | 0.67 | 1.36 | 3.01 | 1.19% | **9.3x** |

**关键结论：**

- VAD-Base 相比 UniAD，Collision Rate **降低 29.0%**，速度提升 **2.5x**
- VAD-Tiny 以略低的精度换取了 **9.3x 的速度提升**，碰撞率仅略高于 UniAD
- 所有指标全面超越当时 SOTA，尤其是 **安全性（碰撞率）** 的显著改善

### 3.3 消融实验亮点

- **向量化约束 vs 无约束**：去掉显式规划约束后碰撞率显著上升，验证了 vectorized constraint 的有效性
- **向量化 vs 栅格化**：同等计算量下，向量化的速度和精度都优于栅格化
- **不同 backbone**：在 ResNet50、ResNet101、VoVNet 等 backbone 上均有提升
- **约束消融**：去掉碰撞约束 → Collision Rate 上升~2x；去掉车道约束 → 轨迹压线/出车道；去掉方向约束 → 轨迹方向漂移；全部去掉 → 退化为纯模仿学习，性能接近 UniAD

---

## 四、核心贡献

1. **向量化场景表示范式** — 抛弃栅格化，用纯向量化的 instance-level 表示建模整个驾驶场景
2. **显式实例级规划约束** — 利用 agent/map 向量生成可微的碰撞、交通规则约束，提升规划安全性
3. **极高推理效率** — Base 比 SOTA 快 2.5x，Tiny 快 9.3x，为实时部署提供可能
4. **端到端联合训练** — 感知→预测→规划 全程可微

---

## 五、局限与后续工作

### VAD 的局限

- **缺乏交互推理**：没有显式建模 agent 之间的交互行为
- **⏱️ 无时序处理**：VAD 本质是单帧模型——每帧独立处理，没有跨帧信息传递。导致检测 flicker、轨迹跳跃、运动估计不准
- **无 ego_query**：自车信息靠 CAN 输入而不是 query，对定位误差鲁棒性差（SparseDrive 改善了这一点）
- **约束宽度固定**：碰撞安全阈值 ε 固定超参，不能根据车速动态调整
- **仅做开环评测**：nuScenes 开环评估不能完全反映闭环性能
- **非因果结构**：规划可以"看到"未来帧检测结果，存在信息泄露（开环下是优势，但闭环会出问题）

### 后续工作

VAD 直接催生了大量后续工作，部分由其作者后续团队开发：

- **VADv2** — 改进的向量化范式，加入时序建模
- **SparseDrive** — 用纯稀疏表示替代 BEV 特征 + 引入 ego_query
- **Planning-oriented 系列** — UniAD → VAD → SparseDrive 的发展脉络

---

## 六、个人感悟

VAD 给我最大的启发不是它在精度上的提升，而是 **"表示范式转换"** 带来的系统性优势：

1. **计算效率** — 向量化消除了栅格化的 O(N²) 计算瓶颈，将复杂度从像素级降到实例级
2. **语义明确性** — 向量化的实例天然包含语义边界，使规划约束变得清晰可控
3. **可扩展性** — 增加或修改某种约束只需要操作对应的向量，不需要改整个 pipeline

VAD 证明了在端到端自动驾驶中，**表示（Representation）比架构（Architecture）更重要**。这是一个被后续 SparseDrive 等作品反复验证的洞见。

---

## 四维评分

| 维度 | 评分 | 说明 |
|------|:----:|------|
| **结果指标** | **8.0** | nuScenes 上全面超越 SOTA，Collision Rate 降低 29%；但仅开环评测 |
| **推理速度** | **9.0** | Base 2.5x 加速，Tiny 9.3x 加速；向量化范式本身就是效率导向的设计 |
| **可落地程度** | **7.5** | 架构简洁清晰，代码开源；但缺少时序建模和闭环验证，工程落地仍需进一步开发 |
| **创新价值** | **9.0** | 向量化范式转换是端到端自动驾驶的重要里程碑，对后续作品影响深远 |

### 综合评分：**8.4 / 10**

> 端到端自动驾驶向量化范式的开创之作。虽然在精度提升上是 incremental 的，但在表示范式上的创新是 foundational 的。ICCV 2023 代表作之一。
