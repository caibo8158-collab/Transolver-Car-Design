# 归一化改进说明：global vs separate

> 日期：2026-07-31  
> 相关提交基准：`8212ee0`（改动前工作区）  
> 目标：修复「体积点压力占位 0 污染压力统计」的问题，并提供可对比实验入口。

---

## 1. 背景与动机

### 1.1 原始数据与标签构造

每个样本由两类 VTK 组成：

| 文件 | 内容 |
|------|------|
| `quadpress_smpl.vtk` | 车身表面点 + 压力标量 |
| `hexvelo_smpl.vtk` | 体积流场点 + 速度向量 |

预处理后每个点：

- 输入 `x`：`[x, y, z, sdf, nx, ny, nz]`（7 维）
- 标签 `y`：`[vx, vy, vz, p]`（4 维）
- `surf`：是否为表面点

体积点**没有压力真值**，代码里把体积压力填为 `0`，训练时压力损失也**只在表面点**上计算。

### 1.2 原归一化的问题（global）

原实现对 **全体点**（表面 + 体积）统一做：

```text
x' = (x - mean_in) / (std_in + 1e-8)
y' = (y - mean_out) / (std_out + 1e-8)
```

体积点上大量压力 `0` 会进入 `mean_out[-1]` / `std_out[-1]`，导致：

1. 压力均值被拉向 0（表面真实均值约 -40 量级，全局均值约 -4）
2. 压力标准差被压小（表面约 47，全局约 20）
3. 表面压力在网络里的中心与尺度被扭曲，更难学准

全量训练集上实测（fold0 训练折）：

| 通道 | global mean | global std |
|------|-------------|------------|
| vx, vy, vz, p | `[0.0026, -0.021, 15.60, -4.17]` | `[1.28, 1.20, 7.79, 20.14]` |

分开统计后表面压力：`mean ≈ -41`，`std ≈ 47`（体积压力占位通道 std 置 1，避免除零）。

### 1.3 与损失权重的关系

训练目标为：

```text
total_loss = loss_velo + weight * loss_press
```

默认 `weight=0.5`，压力项本身权重更小。  
归一化偏差与较小的压力权重是**两件独立的事**，都会让压力相对速度更难优化；本次改动只针对归一化。

---

## 2. 本次改动内容

### 2.1 新增两种归一化模式

通过参数 `--norm_mode` 选择：

| 模式 | 含义 |
|------|------|
| `global`（默认） | 与原版一致：全体点统一 mean/std |
| `separate` | **表面点 / 体积点各自**统计 mean/std，并分别标准化 |

`separate` 下 `coef_norm` 结构：

```python
{
  'mode': 'separate',
  'surf': (mean_in, std_in, mean_out, std_out),  # 仅表面点
  'vol':  (mean_in, std_in, mean_out, std_out),  # 仅体积点
}
```

常数通道（如表面 `sdf=0`、体积压力占位 `0`）的 std 过小时自动置为 `1.0`，避免除零。

### 2.2 涉及文件

| 文件 | 改动 |
|------|------|
| `dataset/dataset.py` | 归一化工具函数；`get_datalist(..., norm_mode=...)`；反归一化 `denormalize_y` |
| `dataset/load_dataset.py` | 读取 `args.norm_mode` 并传入 `get_datalist` |
| `main.py` | 增加 `--norm_mode`；保存路径带模式后缀 |
| `main_evaluation.py` | 支持 separate 反归一化与路径兼容 |
| `train.py` | 验证时额外记录**反归一化后**相对 L2（`val_l2_press` / `val_l2_velo`） |
| `compare_norm.py` | **新增**：自动跑 global 与 separate 并对比 |

### 2.3 为什么用反归一化 L2 对比

不同归一化下，标准化空间的 `val_loss` **不可直接比**（尺度不同）。  
因此对比指标采用物理量纲下的相对 L2：

```text
L2_press = ||p_pred - p_gt|| / ||p_gt||     （仅表面）
L2_velo  = ||u_pred - u_gt|| / ||u_gt||     （仅体积）
```

---

## 3. 使用方法

### 3.1 单独训练（指定归一化）

```bash
# 原版全局归一化
python main.py --norm_mode global --nb_epochs 200 --weight 0.5 --preprocessed 1

# 表面/体积分开归一化
python main.py --norm_mode separate --nb_epochs 200 --weight 0.5 --preprocessed 1
```

权重保存目录示例：

```text
metrics/Transolver/{fold_id}/{nb_epochs}_{weight}_{norm_mode}/
```

评估：

```bash
python main_evaluation.py --norm_mode separate --nb_epochs 200 --weight 0.5
```

（若旧实验目录无 `_global` / `_separate` 后缀，评估脚本会回退到旧路径。）

### 3.2 对比实验（推荐）

各跑 40 epoch，每 5 epoch 验证一次：

```bash
python compare_norm.py --nb_epochs 40 --val_iter 5 --lr 0.001 --weight 0.5 --preprocessed 1 --fold_id 0
```

输出目录：

```text
metrics/norm_compare/Transolver/{fold_id}/ep{epochs}_w{weight}_lr{lr}/
  ├── global/
  ├── separate/
  ├── comparison_summary.json
  └── norm_compare_l2.png
```

主要看 `comparison_summary.json` 与曲线图中的：

- `final_val_l2_press` / `best_val_l2_press`
- `final_val_l2_velo` / `best_val_l2_velo`

---

## 4. 预期影响（定性）

| 项目 | 预期 |
|------|------|
| 表面压力相对 L2 | separate 有望下降（尺度与中心更合理） |
| 体积速度相对 L2 | 变化通常小于压力（速度本就以体积点为主） |
| 标准化 `val_loss` | **不要**跨模式直接对比 |
| 阻力系数 Cd | 依赖表面压力/速度，压力变好时 Cd 也可能改善（需完整评估确认） |

最终结论以 `compare_norm.py` 跑完后的 `comparison_summary.json` 为准。

---

## 5. 未改动的部分（后续可做）

本次**没有**改这些，避免一次改太多变量：

1. 损失中压力权重 `weight`（仍默认 0.5）
2. 点特征本身（仍为 7 维：坐标 + SDF + 法向）
3. SDF 仍是最近邻无符号距离
4. 预处理 npy 缓存格式（只改加载后的归一化，无需重导 VTK）

若 separate 验证有效，可再单独扫 `weight`（如 0.5 / 1.0 / 2.0）。

---

## 6. 实现要点速查

```text
训练集:
  global   → 全体点统计 mean/std → 全体点标准化
  separate → surf / vol 分别统计 → 按 surf mask 分别标准化

验证集:
  复用训练集 coef_norm，规则与上相同

反归一化:
  denormalize_y(y, surf, coef_norm)
  → 评估 Cd、相对 L2、保存物理场时使用
```

辅助函数（均在 `dataset/dataset.py`）：

- `apply_coef_norm`：标准化
- `denormalize_y`：反标准化
- `is_separate_coef`：判断 coef 结构
- `coef_norm_to_jsonable`：写日志

---

## 7. 简要结论

原结果压力误差偏大，至少有两条线索：

1. **全局归一化**被体积压力占位 0 污染（本次已提供 `separate` 修复路径）  
2. **损失权重**默认 `weight=0.5`，压力项更弱（尚未改）

本次交付：可切换的归一化实现 + 反归一化 L2 日志 + `compare_norm.py` 对比脚本。
