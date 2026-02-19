# dl4to-multiscale 科学研究使用说明

本文档用于指导 `dl4to-multiscale` 在科研场景下的标准使用方式，目标是让实验结果具备可复现性、可比较性和可审计性。

## 1. 适用范围

本说明适用于以下研究任务：

- 3D 拓扑优化与多尺度材料设计
- TPMS 微结构参数化与均匀化建模
- 基于可微物理的优化与神经网络耦合
- 宏观设计到微观重建的端到端验证

## 2. 研究流程总览

建议遵循以下标准流程：

1. 固定环境并记录版本信息
2. 运行功能测试，确认代码基线正确
3. 预计算均匀化查找表并持久化
4. 配置优化目标并运行多尺度优化
5. 重建高分辨率结构并做后处理
6. 多随机种子重复实验并汇总统计
7. 输出可复核的图表、日志和配置

## 3. 环境与版本固定

### 3.1 建议环境

- Python `>=3.9`（推荐 `3.11`）
- PyTorch CUDA 版本（如使用 GPU）
- Linux / WSL2 优先

### 3.2 环境记录命令

每组实验开始前保存以下信息：

```bash
python --version
pip freeze > artifacts/pip_freeze.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())" > artifacts/torch_env.txt
git rev-parse HEAD > artifacts/git_commit.txt
```

如果使用 conda，可额外导出：

```bash
conda env export > artifacts/conda_env.yml
```

## 4. 基线正确性验证

在正式实验前，必须运行：

```bash
python test_phases.py
```

预期输出包含：

```text
ALL PHASES PASSED!
```

若基线测试失败，不建议继续进行参数搜索或论文数据生成。

## 5. 可复现性要求

### 5.1 随机种子

建议在每次实验启动时固定随机性：

```python
import os
import random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

set_seed(42)
```

### 5.2 多次重复

每组配置至少运行 `3-5` 个种子，报告：

- 均值 `mean`
- 标准差 `std`
- 最优值 `best`

禁止仅报告单次最优结果作为主结论。

## 6. 参考实验脚手架

下面示例展示从查找表到优化、再到重建的最小研究流程。

```python
from dl4to.homogenization import HomogenizationLookupTable
from dl4to.multiscale_solver import MultiscaleSIMP
from dl4to.multiscale_criteria import GradedPropertyConstraint, LocalVolumeConstraint
from dl4to.criteria import Compliance, VolumeConstraint
from dl4to.reconstruction import TPMSReconstructor

# 1) 查找表
table = HomogenizationLookupTable(
    tpms_types=["gyroid", "schwarz_p"],
    n_samples=19,
    resolution=16,
    E=1.0,
    nu=0.3,
)
table.precompute(verbose=False)
table.save("artifacts/lookup_table.pt")

# 2) 目标函数
criterion = (
    Compliance()
    + 0.1 * VolumeConstraint(max_volume_fraction=0.3)
    + 0.05 * GradedPropertyConstraint(max_gradient=0.3)
    + 0.05 * LocalVolumeConstraint(max_local_vf=0.5, neighborhood_size=3)
)

# 3) 多尺度优化
solver = MultiscaleSIMP(
    criterion=criterion,
    homogenization_table=table,
    n_iterations=100,
    lr=3e-2,
    temperature_init=1.0,
    temperature_decay=0.99,
    return_intermediate_solutions=False,
)

# solution = solver([problem])[0]  # problem 为 dl4to.problem.Problem

# 4) 高分辨率重建
# reconstructor = TPMSReconstructor(upscale_factor=8)
# hr_density = reconstructor.reconstruct(problem, solution.tpms_params)
```

## 7. 建议记录的核心指标

建议至少记录如下字段：

- 目标函数值（`loss`）
- 顺应度（`compliance`）
- 体积分数（`volume_fraction`）
- 体积分数约束惩罚
- 梯度约束惩罚
- 每轮耗时与总耗时
- GPU/CPU 型号与显存

对于本项目，`MultiscaleSIMPIterator` 会在 `solution.logs` 中给出：

- `losses`
- `volumes`
- `durations`
- `relative_max_sigma_vm`（若可用）

## 8. 建议的实验目录结构

```text
experiments/
  exp_001/
    config.yaml
    git_commit.txt
    pip_freeze.txt
    torch_env.txt
    train_logs.json
    final_solution.pt
    reconstruction.pt
    figures/
  exp_002/
    ...
```

## 9. 消融实验建议

建议至少覆盖以下消融维度：

- TPMS 类型集合（单类型 vs 多类型）
- 查找表采样密度（`n_samples`）
- 单胞分辨率（`resolution`）
- 温度退火策略（`temperature_decay`）
- 是否使用空间滤波（`use_filter`）
- 局部体积分数与分级约束权重

并报告每个消融的：

- 结构性能（如 compliance）
- 可制造性（平滑性、二值性、局部体积分数）
- 计算开销（时间、显存）

## 10. 论文报告建议

论文或技术报告中建议明确说明：

- 问题定义：边界条件、载荷、设计域
- 材料参数：`E`, `nu`, `sigma_ys`
- 网格与分辨率：宏观网格、重建倍率
- 优化设置：学习率、迭代次数、初始体积分数
- 重复次数与统计方式：种子数量、均值和标准差

## 11. 常见风险与排查

- `torch.cuda.is_available() == False`：检查驱动、PyTorch CUDA 版本、运行环境权限
- 结果波动大：检查是否固定随机种子及 cudnn 配置
- 内存不足：降低 `resolution`、`upscale_factor` 或批量规模
- 收敛慢：调节学习率、退火速率、约束权重

## 12. 合规与学术诚信

- 保留所有原始实验日志和中间文件，确保可追溯
- 不得只选择性报告有利结果
- 若使用开源数据或代码，请按其许可证与引用要求执行

## 13. 引用

如用于学术发表，建议引用原始 DL4TO 和相关工作（见 `README.md` 的参考文献）。
