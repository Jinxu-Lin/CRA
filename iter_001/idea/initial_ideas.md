### 核心方法

1. **Task-to-Task Influence 矩阵**：定义 $M_{ij} = \mathbb{E}_{z \in \mathcal{D}_i, z' \in \mathcal{D}_j} [\mathcal{I}(z, z')]$，量化任务 $i$ 的训练数据对任务 $j$ 性能的平均影响。
   - 正值 → 互助（正迁移）
   - 负值 → 互害（负迁移）
   - 近零 → 独立

2. **负迁移机制诊断**：对识别出的负迁移任务对，进一步分析其机制：
   - **Representation conflict**：相似 state 但不同最优 action（e.g., 同一个物体在不同任务中需要不同操作）
   - **Optimization conflict**：梯度方向系统性矛盾
   - **Action distribution mismatch**：训练分布的 action space 不兼容

3. **Influence-guided data mixing**：基于 influence 矩阵优化数据配比——对互助任务对增加共训权重，对互害任务对降低或分离。与 Re-Mix (DRO) 和均匀混合做正面对比。