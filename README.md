# dl4to-multiscale

`dl4to-multiscale` 是基于 [DL4TO](https://github.com/dl4to/dl4to) 的多尺度拓扑优化扩展版本。  
项目在原始单尺度 SIMP 框架上，加入了 TPMS 微结构参数化、数值均匀化、各向异性求解和高分辨率重建能力，支持端到端可微优化流程。

科研使用说明见：[`SCIENTIFIC_RESEARCH_GUIDE.md`](SCIENTIFIC_RESEARCH_GUIDE.md)

## 1. 项目简介

该项目面向 3D 结构拓扑优化与深度学习结合场景，核心技术栈为 PyTorch。  
与原版 DL4TO 相比，当前仓库增加了微观单胞到宏观结构的多尺度映射能力，适合以下任务：

- 基于体积分数和 TPMS 类型混合的材料分布设计
- 通过均匀化查找表构造空间变系数各向异性刚度场
- 在宏观网格上求解多尺度弹性 PDE
- 对优化结果进行高分辨率 TPMS 重建

## 2. 核心能力

- 可微 TPMS 几何生成（支持多种隐式曲面）
- 周期边界条件下的数值均匀化与查找表插值
- 各向异性刚度场驱动的 FDM 求解器
- 可训练的多尺度表示器与 SIMP 优化流程
- 多尺度约束准则（分级平滑、局部体积分数）
- 宏观到微观的高分辨率体素重建

## 3. 新增模块

| 模块 | 文件 | 说明 |
|---|---|---|
| TPMS 几何 | `dl4to/tpms.py` | TPMS 函数注册、体积分数控制、可微体素化 |
| 数值均匀化 | `dl4to/homogenization.py` | 周期 FDM、`C_eff` 计算、查找表构建与插值 |
| 多尺度 PDE | `dl4to/multiscale_pde.py` | 支持空间变系数各向异性本构的求解器 |
| 多尺度表示器 | `dl4to/multiscale_representer.py` | `vf + type` 参数化并输出刚度场 |
| 多尺度优化器 | `dl4to/multiscale_solver.py` | `MultiscaleSIMPIterator` 与 `MultiscaleSIMP` |
| 多尺度准则 | `dl4to/multiscale_criteria.py` | 分级约束与局部体积分数约束 |
| 重建模块 | `dl4to/reconstruction.py` | TPMS 高分辨率重建与缓存加速 |

## 4. 环境要求

- Python `>=3.9`（推荐 `3.10/3.11`）
- Linux / WSL2 / macOS（优先 Linux 或 WSL2）
- 可选 GPU：NVIDIA 驱动 + CUDA 运行环境（PyTorch CUDA 版）

## 5. 安装

### 5.1 新建环境并安装

```bash
conda create -n dl4to python=3.11 -y
conda activate dl4to
pip install -U pip setuptools wheel
pip install -e .
```

### 5.2 已有 PyTorch 环境（推荐开发机）

如果你已经手动安装了匹配 CUDA 的 PyTorch，为避免被重新解析依赖覆盖：

```bash
conda activate <your_env>
pip install -e . --no-build-isolation --no-deps
```

## 6. GPU 可用性检查

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print('cuda:', torch.cuda.is_available()); print('count:', torch.cuda.device_count())"
```

若 `cuda=False`，优先检查：

- 是否在支持 GPU 的终端/容器中运行
- 驱动是否正常（`nvidia-smi` 是否可用）
- 当前 Python 环境中的 PyTorch 是否为 CUDA 版本

## 7. 快速开始

```python
from dl4to.homogenization import HomogenizationLookupTable
from dl4to.multiscale_solver import MultiscaleSIMP
from dl4to.criteria import Compliance, VolumeConstraint
from dl4to.reconstruction import TPMSReconstructor

# 1) 预计算均匀化查找表
table = HomogenizationLookupTable(
    tpms_types=["gyroid", "schwarz_p"],
    n_samples=19,
    resolution=16,
    E=1.0,
    nu=0.3,
)
table.precompute()

# 2) 定义优化目标
criterion = Compliance() + 0.1 * VolumeConstraint(max_volume_fraction=0.3)

# 3) 创建多尺度求解器并优化
solver = MultiscaleSIMP(
    criterion=criterion,
    homogenization_table=table,
    n_iterations=100,
    lr=3e-2,
)

# solution = solver([problem])[0]  # problem 为 dl4to.Problem 实例

# 4) 高分辨率重建
# reconstructor = TPMSReconstructor(upscale_factor=8)
# hr_density = reconstructor.reconstruct(problem, solution.tpms_params)
```

## 8. 测试

运行多尺度阶段测试：

```bash
python test_phases.py
```

预期输出包含：

```text
ALL PHASES PASSED!
```

## 9. 项目结构

```text
dl4to/
  tpms.py
  homogenization.py
  multiscale_pde.py
  multiscale_representer.py
  multiscale_solver.py
  multiscale_criteria.py
  reconstruction.py
test_phases.py
notebooks/
docs/
```

## 10. 与原版 DL4TO 的关系

- 保留原版 DL4TO 的核心数据结构与求解流程
- 以新增模块方式扩展多尺度功能，尽量保持向后兼容
- 可继续使用原有 `criteria`、`problem`、`solution` 等接口组合实验

## 11. 参考文献

1. Dittmer, Soren, et al. "SELTO: Sample-Efficient Learned Topology Optimization." arXiv:2209.05098 (2023).  
2. Erzmann, David, et al. "DL4TO: A Deep Learning Library for Sample-Efficient Topology Optimization." https://doi.org/10.1007/978-3-031-38271-0_54 (2023).  
3. Dittmer, Soren, et al. "SELTO Dataset." https://doi.org/10.5281/zenodo.7781392 (2023).  
4. Erzmann, David. "Equivariant Deep Learning for 3D Topology Optimization." https://doi.org/10.26092/elib/3439 (2024).

## 12. 许可证

本项目采用 Apache-2.0 许可证，见 `LICENSE`。
