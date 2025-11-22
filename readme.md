---
# 🚁 VRP Drone Delivery Simulation (VRPTW + Battery Constraints)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

这是一个基于 Python 的无人机配送路径规划仿真系统。该项目旨在解决**带时间窗和电池约束的车辆路径问题 (VRPTW + Battery Constraints)**。

系统集成了数据生成、精确求解算法 (Gurobi)、启发式算法 (Clarke-Wright)、实验管理以及丰富的可视化（静态图表 + 动态 GIF 仿真）功能。

## ✨ 主要功能

*   **多算法对比**：
    *   **Gurobi (Exact)**: 用于小规模问题的精确求解，提供最优解基准。
    *   **Clarke-Wright (Heuristic)**: 经典的节约算法，用于大规模问题的快速求解。
*   **复杂约束处理**：
    *   ✅ **容量约束 (Capacity)**: 无人机最大载重限制。
    *   ✅ **时间窗约束 (Time Windows)**: 客户服务必须在指定时间段内进行。
    *   ✅ **电池约束 (Battery/Range)**: 无人机单次飞行的最大距离限制。
*   **自动化实验平台**：
    *   自动运行小规模对比实验 (Small Scale) 和大规模性能测试 (Large Scale)。
    *   自动生成成本对比表格、运行时间分析图。
*   **高级可视化**：
    *   **静态图表**: 路线图、任务甘特图 (Task Timeline)、成本分析图。
    *   **动态仿真**: 生成 GIF 动画，实时展示无人机飞行轨迹、电池消耗和客户服务状态。

## 📂 项目结构

```text
VRP_drone_delivery/
├── 📄 main.py                  # 程序主入口
├── 📄 config.py                # 全局配置参数（随机种子、电池限制、规模等）
├── 📄 data_generator.py        # VRPTW 随机实例生成器
├── 📄 experiment_manager.py    # 实验流程控制器
├── 📄 visualization_manager.py # 可视化流程控制器
│
├── 🧠 Algorithms (Solvers)
│   ├── solver_gurobi.py        # Gurobi 精确求解器 (MILP 模型)
│   └── solver_cw.py            # Clarke-Wright 节约算法求解器
│
├── 🎨 Visualization & Simulation
│   ├── sim.py                  # 算法对比动画生成逻辑
│   ├── animation_utils.py      # 单算法动画生成工具
│   ├── plotting_utils.py       # 静态绘图工具 (Matplotlib)
│   └── matplotlib_config.py    # 绘图样式与字体配置
│
└── 📁 cline_docs/              # 项目文档与开发记忆库
```

## 🚀 快速开始

### 1. 环境准备

确保安装了 Python 3.8+。推荐使用虚拟环境。

安装依赖库：

```bash
pip install numpy matplotlib
```

> **注意**: 如需运行精确求解器，需要安装 `gurobipy` 并拥有有效的 Gurobi License。如果没有安装 Gurobi，程序会自动降级，仅运行 Clarke-Wright 算法。

```bash
# 可选：安装 Gurobi
pip install gurobipy
```

### 2. 运行仿真

直接运行主程序即可启动实验全流程：

```bash
python main.py
```

程序将依次执行：

1. 小规模对比实验 (N=5, 8, 10...)
2. 大规模性能测试 (N=20, 50...)
3. 生成所有图表和 GIF 动画到 `figs/` 目录。

## ⚙️ 配置说明 (Config)

所有实验参数均在 `config.py` 中集中管理，你可以修改该文件来调整实验设置：

```python
# config.py 示例
class ExperimentConfig:
    # 随机种子，保证实验可复现
    random_seed: int = 202511233
  
    # 实验规模设置
    small_n_list: List[int] = [5, 8, 10, 15]
    large_n_list: List[int] = [20, 50, 100]
  
    # 指定为哪些 N 值生成详细的 GIF 动画
    visualization_n_list: List[int] = [10, 15]
  
    # 约束参数
    vehicle_capacity: float = 25.0
    small_battery_limit: float = 150.0  # 小规模电池限制
    large_battery_limit: float = 300.0  # 大规模电池限制
  
    # 动画参数
    animation_fps: int = 20
```

## 📊 输出示例

运行完成后，`figs/` 目录下将生成以下类型的可视化文件：

### 1. 算法对比动画 (GIF)

展示不同算法下无人机的实时调度、路径规划及电池剩余情况。
*(生成文件: `algorithm_comparison_N10.gif`)*

### 2. 路线对比图

直观对比 Gurobi 最优解与 CW 启发式解的路径差异。
*(生成文件: `routes_N10_comparison.png`)*

### 3. 任务时间线 (Gantt Chart)

展示每辆无人机的任务执行顺序及时间窗满足情况。
*(生成文件: `schedule_N10_timeline.png`)*

### 4. 性能分析

* **运行时间扩展性**: `runtime_scalability.png`
* **成本与 Gap 分析**: `cost_gap_smallN.png`
* **大规模求解质量**: `large_scale_comparison.png`

## 🧠 算法细节

### Gurobi Solver (Exact)

* **模型**: 混合整数线性规划 (MILP)。
* **决策变量**: $x_{ijk}$ (车辆 $k$ 是否从 $i$ 到 $j$)。
* **约束**: 流平衡、时间窗 (Big-M method)、子回路消除、电池容量限制。
* **目标**: 最小化总行驶距离。

### Clarke-Wright Solver (Heuristic)

* **原理**: 基于节约值 (Savings) $S_{ij} = d_{i0} + d_{0j} - d_{ij}$ 进行路径合并。
* **增强**: 在合并过程中加入了严格的 `check_validity` 函数，实时检测合并后的路径是否满足容量、电池和时间窗约束。
* **特点**: 速度快，适合大规模问题，但在紧致约束下可能无法找到可行解。

## 📝 License

No License.
---
