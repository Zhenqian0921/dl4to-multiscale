# dl4to-multiscale 项目说明

## 项目概述

基于 dl4to（PyTorch 3D拓扑优化库）扩展的**多尺度TPMS点阵结构优化框架**。在原有单尺度 SIMP 密度法基础上，新增了 TPMS 微结构参数化、数值均匀化、各向异性求解和高分辨率重建能力，实现从微观到宏观的端到端可微优化。

## 新增模块（7个文件）

### Phase 1: `dl4to/tpms.py` — TPMS 几何模块
- 6种TPMS隐式曲面函数：`Gyroid`, `SchwarzP`, `SchwarzD`, `Neovius`, `FischerKoch`, `IWP`
- `TPMS_REGISTRY`：名称→类的注册表
- `tpms_density_field()`：基于 sigmoid 的可微体素化
- `find_threshold_for_vf()`：二分搜索给定体积分数对应的阈值
- 所有操作基于 PyTorch，支持 autograd 梯度回传

### Phase 2: `dl4to/homogenization.py` — 数值均匀化模块
- `PeriodicFDMSolver`：周期边界条件 FDM 求解器（使用 `torch.roll` 实现周期性）
- `NumericalHomogenizer`：计算单胞等效刚度张量 C_eff (6×6 Voigt)
  - 对6个正则宏观应变模式求解单胞弹性问题
  - 全密度单胞误差 < 1e-8（相对于解析解）
- `HomogenizationLookupTable`：预计算查找表
  - 支持可微线性插值：`C(vf) = C[i] + t * (C[i+1] - C[i])`
  - `batch_interpolate_9x9()` 返回 (T, 9, 9, nx, ny, nz) 用于 FDM 求解器
  - 支持 `save()`/`load()` 磁盘持久化

### Phase 3: `dl4to/multiscale_pde.py` — 各向异性 FDM 求解器
- `MultiscaleFDM(PDESolver)`：核心改造
  - `_apply_C_eff(epsilon, C_field)`：逐体素各向异性本构 `σ_i = C_ij(x) · ε_j(x)`
  - `_A(u, C_field)`：系统算子 `y = J^T · C_eff · J · u`
  - 复用现有 `_J`, `_J_adj`, `FDMDerivatives`, `FDMAssembly`, `AutogradLinearSolver`
  - `set_stiffness_field(C_field)` 接受 (9, 9, nx, ny, nz) 刚度场

### Phase 4: `dl4to/multiscale_representer.py` — 多尺度参数化器
- `MultiscaleRepresenter(nn.Module)`：每个体素的设计变量
  - `vf_logit` → sigmoid → 体积分数 [vf_min, vf_max]
  - `type_logits` → softmax → TPMS 类型权重
  - `forward()` 返回 `(C_field, vf)`
  - 支持空间平滑滤波（`RadialDensityFilter`）
  - 温度退火控制类型选择锐度

### Phase 5a: `dl4to/multiscale_solver.py` — 多尺度优化求解器
- `MultiscaleSIMPIterator`：单次迭代（类比 `SIMPIterator`）
  - 前向：representer → C_field → MultiscaleFDM.solve → criterion → backward
  - 支持温度退火
- `MultiscaleSIMP(TopoSolver)`：完整优化循环（类比 `SIMP`）
  - 自动创建 representer、pde_solver、iterator
  - 支持中间解记录

### Phase 5b: `dl4to/multiscale_criteria.py` — 多尺度准则
- `GradedPropertyConstraint(UnsupervisedCriterion)`：惩罚相邻体素 TPMS 参数突变
- `LocalVolumeConstraint(UnsupervisedCriterion)`：局部区域体积分数约束

### Phase 6: `dl4to/reconstruction.py` — 高分辨率重建
- `TPMSReconstructor`：宏观参数 → 微观体素
  - 每个宏观体素扩展为 `upscale_factor^3` 个微观体素
  - 根据主导 TPMS 类型和体积分数生成单胞
  - 支持高斯平滑处理单胞边界过渡
  - 缓存机制避免重复计算

## 数据流

```
MultiscaleRepresenter
  ├─ vf_logit → sigmoid → vf        (体积分数)
  ├─ type_logits → softmax → weights (类型权重)
  ├─ HomogenizationLookupTable.interpolate(weights, vf)
  │     → C_field (9, 9, nx, ny, nz)
  v
MultiscaleFDM.solve_pde(C_field, problem)
  ├─ _J(u) → ε (应变)
  ├─ _apply_C_eff(ε, C_field) → σ (应力)
  ├─ AutogradLinearSolver (伴随法反向传播)
  v
u, σ, σ_vm
  v
Criterion (Compliance + VolumeConstraint + GradedPropertyConstraint)
  v
loss.backward() → 梯度: loss → u → C_field → vf_logit, type_logits
```

## 使用示例

```python
from dl4to.tpms import TPMS_REGISTRY
from dl4to.homogenization import HomogenizationLookupTable
from dl4to.multiscale_solver import MultiscaleSIMP
from dl4to.multiscale_criteria import GradedPropertyConstraint
from dl4to.reconstruction import TPMSReconstructor
from dl4to.criteria import Compliance, VolumeConstraint

# 1. 预计算均匀化查找表
table = HomogenizationLookupTable(
    tpms_types=['gyroid', 'schwarz_p'],
    n_samples=19, resolution=16, E=1.0, nu=0.3
)
table.precompute()
table.save('lookup_table.pt')

# 2. 定义优化目标
criterion = Compliance(alpha=1e-9) + 0.1 * VolumeConstraint(max_volume_fraction=0.3)

# 3. 运行多尺度优化
solver = MultiscaleSIMP(
    criterion=criterion,
    homogenization_table=table,
    n_iterations=100,
    lr=3e-2,
)
solution = solver([problem])  # problem 是 dl4to.Problem 实例

# 4. 高分辨率重建
reconstructor = TPMSReconstructor(upscale_factor=8)
hr_density = reconstructor.reconstruct(problem, solution.tpms_params)
```

## 对现有代码的修改

仅修改了 `dl4to/plotting.py` 第17行，将 `pv.set_jupyter_backend('pythreejs')` 包裹在 try/except 中以兼容新版 pyvista。**所有其他现有文件未做修改**，完全向后兼容。

## 测试

运行 `python test_phases.py` 验证全部6个阶段：
- Phase 1: TPMS 几何函数、体积分数控制、autograd 梯度
- Phase 2: 全密度均匀化精度 (~1e-9)、C_eff 对称性、单调性
- Phase 3: 各向同性退化验证（与标准 FDM 一致）
- Phase 4: 输出形状、梯度非零、类型权重归一化
- Phase 5: 约束准则正确性
- Phase 6: 重建形状、固体区域密度、二值化

## 依赖

在原有 dl4to 依赖基础上无新增依赖（均使用 PyTorch、SciPy、NumPy）。
