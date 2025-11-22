"""
绘图工具 - 静态图表生成函数
"""
from matplotlib_config import setup_matplotlib
setup_matplotlib()
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple
from data_generator import VRPInstance





def plot_runtime_scalability(
    small_results: List[Dict],
    large_results: List[Dict],
    output_file: str
):
    """绘制运行时间可扩展性图"""
    # CW数据
    cw_n = []
    cw_time = []
    for r in small_results + large_results:
        if r.get('cw_time'):
            cw_n.append(int(r['N']))
            cw_time.append(float(r['cw_time']))
    
    # Gurobi数据
    gurobi_n = []
    gurobi_time = []
    for r in small_results:
        if r.get('gurobi_time'):
            gurobi_n.append(int(r['N']))
            gurobi_time.append(float(r['gurobi_time']))
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if cw_n:
        ax.plot(cw_n, cw_time, 'o-', label='Clarke-Wright', linewidth=2)
    if gurobi_n:
        ax.plot(gurobi_n, gurobi_time, 's-', label='Gurobi (small N)', linewidth=2)
    
    ax.set_xlabel("Number of Customers (N)", fontsize=12)
    ax.set_ylabel("Runtime (seconds, log scale)", fontsize=12)
    ax.set_yscale('log')
    ax.set_title("Algorithm Scalability Comparison", fontsize=14, fontweight='bold')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.legend(fontsize=11)
    
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def plot_cost_and_gap(small_results: List[Dict], output_file: str):
    """绘制成本和Gap对比图"""
    n_values = []
    gurobi_costs = []
    cw_costs = []
    gap_values = []
    
    for r in small_results:
        n = int(r['N'])
        if r.get('cw_cost'):
            n_values.append(n)
            cw_costs.append(float(r['cw_cost']))
            gurobi_costs.append(
                float(r['gurobi_cost']) if r.get('gurobi_cost') else float('nan')
            )
        if r.get('gap_pct'):
            gap_values.append((n, float(r['gap_pct'])))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # 成本对比
    if n_values:
        ax1.plot(n_values, cw_costs, 'o-', label='Clarke-Wright', linewidth=2)
        ax1.plot(n_values, gurobi_costs, 's-', label='Gurobi', linewidth=2)
        ax1.set_ylabel("Total Cost", fontsize=12)
        ax1.set_title("Solution Cost Comparison (Small Scale)", fontsize=14, fontweight='bold')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(fontsize=11)
    
    # Gap趋势
    if gap_values:
        gap_n, gap_pct = zip(*gap_values)
        ax2.plot(gap_n, gap_pct, '^-', color='orange', label='Gap (%)', linewidth=2)
        ax2.set_xlabel("Number of Customers (N)", fontsize=12)
        ax2.set_ylabel("Gap (%)", fontsize=12)
        ax2.set_title("Solution Gap vs N", fontsize=14, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(fontsize=11)
    
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

def _plot_routes_for_solution(
    ax,
    coordinates: np.ndarray,
    routes: List[List[int]],
    title: str,
) -> None:
    """
    在给定的 Matplotlib 子图上绘制一组车辆路径。

    图内标签全部使用英文，避免中文字体问题。
    """
    if not routes:
        ax.set_title(f"{title}\n(no feasible route)")
        ax.axis("off")
        return

    # 绘制所有节点
    x_all = coordinates[:, 0]
    y_all = coordinates[:, 1]
    ax.scatter(x_all, y_all, s=20, label="Customers")

    # 仓库高亮
    depot_x = coordinates[0, 0]
    depot_y = coordinates[0, 1]
    ax.scatter([depot_x], [depot_y], s=60, marker="*", label="Depot")

    for idx, route in enumerate(routes):
        xs = [coordinates[node, 0] for node in route]
        ys = [coordinates[node, 1] for node in route]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"Vehicle {idx}")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(fontsize=8)


def plot_routes_comparison(snapshot: Dict, output_file: str):
    """绘制路线对比图"""
    
    instance = snapshot['instance']
    data = instance.get_data()
    coordinates = data['coordinates']
    gurobi_routes = snapshot.get('gurobi_routes', [])
    cw_routes = snapshot.get('cw_routes', [])
    
    if gurobi_routes:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        _plot_routes_for_solution(ax1, coordinates, gurobi_routes, "Gurobi Routes (N=10)")
        _plot_routes_for_solution(ax2, coordinates, cw_routes, "Clarke-Wright Routes (N=10)")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        _plot_routes_for_solution(ax, coordinates, cw_routes, "Clarke-Wright Routes (N=10)")
    
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


def _build_schedule_for_routes(
    instance: VRPInstance,
    routes: List[List[int]],
) -> List[List[Tuple[int, float, float]]]:
    """
    根据给定路径构造任务时间线（到达/服务时间段），用于任务甘特图。

    返回每辆车的任务列表：
    - 每个元素为 (node_index, start_time, end_time)
    """
    data = instance.get_data()
    distance_matrix: np.ndarray = data["distance_matrix"]  # type: ignore[assignment]
    time_windows: List[Tuple[float, float]] = data["time_windows"]  # type: ignore[assignment]
    service_time: float = float(data["service_time"])

    schedules: List[List[Tuple[int, float, float]]] = []

    for route in routes:
        if len(route) < 2:
            schedules.append([])
            continue

        depot = route[0]
        depot_early, _ = time_windows[depot]
        current_time = float(depot_early)

        vehicle_schedule: List[Tuple[int, float, float]] = []

        for idx in range(1, len(route)):
            prev_node = route[idx - 1]
            node = route[idx]

            travel_time = float(distance_matrix[prev_node, node])
            current_time += travel_time

            early, _ = time_windows[node]
            if current_time < early:
                current_time = float(early)

            start_service = current_time
            end_service = start_service
            # 只有客户才有服务时间，仓库视为瞬时
            if node != depot:
                end_service = start_service + service_time

            vehicle_schedule.append((node, start_service, end_service))
            current_time = end_service

        schedules.append(vehicle_schedule)

    return schedules

def plot_task_timeline(snapshot: Dict, output_file: str):
    """绘制任务时间线"""
    
    instance = snapshot['instance']
    cw_routes = snapshot.get('cw_routes', [])
    
    if not cw_routes:
        return
    
    schedules = _build_schedule_for_routes(instance, cw_routes)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for v_idx, vehicle_schedule in enumerate(schedules):
        for node, start, end in vehicle_schedule:
            if node == 0:
                continue
            duration = max(end - start, 1e-6)
            ax.barh(v_idx, duration, left=start, height=0.5, alpha=0.8)
            ax.text(
                start + duration/2, v_idx,
                str(node),
                ha='center', va='center',
                fontsize=9, fontweight='bold'
            )
    
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Vehicle Index", fontsize=12)
    ax.set_title("Task Timeline (Clarke-Wright, N=10)", fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    fig.savefig(output_file, dpi=150)
    plt.close(fig)

def plot_large_scale_comparison(
    large_results: List[Dict],
    output_file: str
):
    """
    绘制大规模测试的成本和时间对比图
    
    Parameters
    ----------
    large_results : List[Dict]
        大规模实验结果列表
    output_file : str
        输出文件路径
    """
    if not large_results:
        print("[提示] 无大规模实验数据")
        return
    
    n_values = []
    costs = []
    times = []
    
    for r in large_results:
        n_values.append(int(r['N']))
        costs.append(float(r['cw_cost']))
        times.append(float(r['cw_time']))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 成本曲线
    ax1.plot(n_values, costs, 'o-', linewidth=2.5, markersize=8, 
             color='#FF6B6B', label='Clarke-Wright')
    ax1.set_ylabel("Total Cost", fontsize=12, fontweight='bold')
    ax1.set_title("Scalability Analysis: Solution Quality vs Problem Size", 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=11)
    
    # 为每个点添加数值标签
    for i, (n, cost) in enumerate(zip(n_values, costs)):
        ax1.text(n, cost, f'{cost:.1f}', 
                ha='center', va='bottom', fontsize=9)
    
    # 时间曲线（对数坐标）
    ax2.plot(n_values, times, 's-', linewidth=2.5, markersize=8,
             color='#4ECDC4', label='Clarke-Wright')
    ax2.set_xlabel("Number of Customers (N)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Runtime (seconds)", fontsize=12, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_title("Scalability Analysis: Computational Time vs Problem Size", 
                  fontsize=14, fontweight='bold')
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend(fontsize=11)
    
    # 为每个点添加数值标签
    for i, (n, t) in enumerate(zip(n_values, times)):
        ax2.text(n, t, f'{t:.4f}s', 
                ha='center', va='bottom', fontsize=9)
    
    # 设置 x 轴刻度
    ax2.set_xticks(n_values)
    
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
