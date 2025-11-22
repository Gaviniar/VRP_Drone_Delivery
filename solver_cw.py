import time
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  
    from data_generator import VRPInstance


def _compute_route_distance(route: List[int], distance_matrix: np.ndarray) -> float:
    """
    计算一条路径的总行驶距离（仅基于距离矩阵，不含服务时间）。

    :param route: 节点序列，例如 [0, 3, 5, 0]。
    :param distance_matrix: 距离矩阵。
    """
    dist = 0.0
    for i in range(len(route) - 1):
        dist += float(distance_matrix[route[i], route[i + 1]])
    return dist


def _check_time_windows_and_battery(
    route: List[int],
    distance_matrix: np.ndarray,
    time_windows: List[Tuple[float, float]],
    service_time: float,
    battery_limit: float,
) -> bool:
    """
    对给定路径进行时间窗和电池约束检查。

    时间窗检查采用前向仿真：
    - 初始时间设为仓库时间窗的起点；
    - 车辆沿路径依次移动，累加行驶时间；
    - 若早于时间窗下界则等待至下界；
    - 若超过时间窗上界则直接判为不可行；
    - 除仓库外，服务完成需要额外 service_time。

    电池约束：
    - 路径总行驶距离不得超过 battery_limit。
    """
    if len(route) < 2:
        return True

    total_distance = _compute_route_distance(route, distance_matrix)
    if total_distance > battery_limit + 1e-6:
        return False

    # 初始时间设为仓库时间窗起点
    depot = route[0]
    depot_early, depot_late = time_windows[depot]
    current_time = float(depot_early)

    for idx in range(1, len(route)):
        prev_node = route[idx - 1]
        node = route[idx]

        travel_time = float(distance_matrix[prev_node, node])
        current_time += travel_time

        early, late = time_windows[node]

        # 提前到达则等待
        if current_time < early:
            current_time = float(early)

        # 超过时间窗上界则不可行
        if current_time > late + 1e-6:
            return False

        # 对非仓库节点追加服务时间
        if node != depot:
            current_time += service_time

    # 回到仓库时也要满足仓库时间窗上界
    if current_time > depot_late + 1e-6:
        return False

    return True


def _check_capacity(
    route: List[int],
    demands: List[int],
    vehicle_capacity: Optional[float],
) -> bool:
    """
    检查路径的容量约束。

    若 vehicle_capacity 为 None，则认为不启用容量约束。
    """
    if vehicle_capacity is None:
        return True

    total_demand = 0.0
    for node in route:
        total_demand += float(demands[node])

    return total_demand <= float(vehicle_capacity) + 1e-6


def _check_validity(
    route: List[int],
    demands: List[int],
    distance_matrix: np.ndarray,
    time_windows: List[Tuple[float, float]],
    service_time: float,
    vehicle_capacity: Optional[float],
    battery_limit: float,
) -> bool:
    """
    合并路径后的综合可行性检查。

    同时检查：
    - 容量约束；
    - 电池约束；
    - 时间窗约束。
    """
    if not _check_capacity(route, demands, vehicle_capacity):
        return False

    if not _check_time_windows_and_battery(
        route=route,
        distance_matrix=distance_matrix,
        time_windows=time_windows,
        service_time=service_time,
        battery_limit=battery_limit,
    ):
        return False

    return True


def solve_cw(instance, vehicle_capacity: float, battery_limit: float):
    """
    使用 Clarke-Wright Savings Algorithm 求解 VRPTW + 电池约束近似解。

    参数说明
    ----------
    instance : VRPInstance
        来自 data_generator.VRPInstance 的实例。
    vehicle_capacity : float
        单辆车的最大载重，用于容量约束。
    battery_limit : float
        单辆车在一次任务中的最大飞行距离（电池约束）。

    返回
    ----------
    (obj_value, runtime, routes)

    - obj_value : float
        近似解的总行驶距离。若构造失败则为 None。
    - runtime : float
        算法运行时间（秒）。
    - routes : List[List[int]]
        路径列表，每条路径为节点序列，例如 [0, 3, 5, 0]。
    """
    if vehicle_capacity <= 0:
        raise ValueError("vehicle_capacity 必须为正数。")
    if battery_limit <= 0:
        raise ValueError("battery_limit 必须为正数。")

    data: Dict[str, object] = instance.get_data()
    num_nodes: int = int(data["num_nodes"])
    depot: int = int(data["depot_index"])
    distance_matrix: np.ndarray = data["distance_matrix"]
    time_windows: List[Tuple[float, float]] = data["time_windows"]
    service_time: float = float(data["service_time"])
    demands: List[int] = data["demands"]  # type: ignore[assignment]

    nodes = list(range(num_nodes))
    customers = [i for i in nodes if i != depot]

    start_time = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. 初始化：每个客户一条独立路径 [0, i, 0]
    # ------------------------------------------------------------------
    routes: List[List[int]] = []
    for i in customers:
        route = [depot, i, depot]
        routes.append(route)

    # 检查初始路径是否满足约束（如果连单客户回路都不可行，则提示用户调整参数）
    for route in routes:
        if not _check_validity(
            route=route,
            demands=demands,
            distance_matrix=distance_matrix,
            time_windows=time_windows,
            service_time=service_time,
            vehicle_capacity=vehicle_capacity,
            battery_limit=battery_limit,
        ):
            end_time = time.perf_counter()
            runtime = end_time - start_time
            # 返回 None 表示给定参数下 CW 算法无法构造可行解
            return None, runtime, []

    # 维护一个从客户节点到其所在路径索引的映射
    node_to_route: Dict[int, int] = {}
    for idx, route in enumerate(routes):
        for node in route:
            if node != depot:
                node_to_route[node] = idx

    # ------------------------------------------------------------------
    # 2. 计算节约值 S_ij = d_i0 + d_0j - d_ij
    # ------------------------------------------------------------------
    savings: List[Tuple[float, int, int]] = []
    for i in customers:
        for j in customers:
            if i >= j:
                continue
            s_ij = (
                float(distance_matrix[i, depot])
                + float(distance_matrix[depot, j])
                - float(distance_matrix[i, j])
            )
            savings.append((s_ij, i, j))

    # 按节约值从大到小排序
    savings.sort(key=lambda x: x[0], reverse=True)

    # ------------------------------------------------------------------
    # 3. 遍历节约值列表，尝试合并路径
    # ------------------------------------------------------------------
    for _, i, j in savings:
        # 如果 i 或 j 当前已不在任何路径中（理论上不应发生），跳过
        if i not in node_to_route or j not in node_to_route:
            continue

        route_i_idx = node_to_route[i]
        route_j_idx = node_to_route[j]

        # 同一路径无法再合并
        if route_i_idx == route_j_idx:
            continue

        route_i = routes[route_i_idx]
        route_j = routes[route_j_idx]

        # 为了得到 0 - ... - i - j - ... - 0 的结构：
        # 要求 i 是其路径中 depot 前的最后一个客户，j 是其路径中 depot 后的第一个客户
        if len(route_i) < 3 or len(route_j) < 3:
            continue

        if route_i[-2] != i:
            continue
        if route_j[1] != j:
            continue

        # 构造合并后的新路径：
        # route_i: [0, ..., i, 0]
        # route_j: [0, j, ..., 0]
        # new_route: [0, ..., i, j, ..., 0]
        new_route = route_i[:-1] + route_j[1:]

        if not _check_validity(
            route=new_route,
            demands=demands,
            distance_matrix=distance_matrix,
            time_windows=time_windows,
            service_time=service_time,
            vehicle_capacity=vehicle_capacity,
            battery_limit=battery_limit,
        ):
            # 合并后不可行则跳过
            continue

        # 合并成功：更新 routes 和 node_to_route
        new_routes: List[List[int]] = []
        for idx, r in enumerate(routes):
            if idx not in (route_i_idx, route_j_idx):
                new_routes.append(r)
        new_routes.append(new_route)
        routes = new_routes

        # 重新构建映射
        node_to_route.clear()
        for idx, route in enumerate(routes):
            for node in route:
                if node != depot:
                    node_to_route[node] = idx

    end_time = time.perf_counter()
    runtime = end_time - start_time

    # 计算最终解的总行驶距离
    total_cost = 0.0
    for route in routes:
        total_cost += _compute_route_distance(route, distance_matrix)

    return float(total_cost), float(runtime), routes


if __name__ == "__main__":
    # 简单自测入口：用于快速验证算法是否能返回合理解。
    from data_generator import VRPInstance

    instance = VRPInstance(num_nodes=10, seed=123)
    obj, rt, rts = solve_cw(
        instance=instance,
        vehicle_capacity=25.0,
        battery_limit=200.0,
    )
    print("Clarke-Wright 求解结果：")
    print("  目标值:", obj)
    print("  运行时间 (s):", rt)
    print("  路径数:", len(rts))
    for idx, r in enumerate(rts):
        print(f"  车辆 {idx}: {r}")
