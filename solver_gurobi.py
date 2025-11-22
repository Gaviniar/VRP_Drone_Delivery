import time
from typing import Dict, List, Tuple

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except Exception:  # pragma: no cover - 环境中未安装 Gurobi 时走降级分支
    GUROBI_AVAILABLE = False


def solve_gurobi(instance, num_vehicles: int, battery_limit: float):
    """
    使用 Gurobi 求解 VRPTW + 电池约束模型。

    参数说明
    ----------
    instance : VRPInstance
        来自 data_generator.VRPInstance 的实例。
    num_vehicles : int
        车辆（无人机）数量上限。
    battery_limit : float
        单辆车在一次任务中的最大飞行距离（电池约束）。

    返回
    ----------
    (obj_value, runtime, routes)

    - obj_value : float
        模型求得的目标值（总行驶距离）。若无可行解则为 None。
    - runtime : float
        Gurobi 求解用时（秒）。
    - routes : List[List[int]]
        每辆车的路径表示为节点序列，例如 [0, 3, 5, 0]。
        若模型无可行解，则为空列表。
    """
    if not GUROBI_AVAILABLE:
        raise RuntimeError(
            "gurobipy 未安装或 Gurobi 不可用，无法运行精确求解器。"
        )

    if num_vehicles <= 0:
        raise ValueError("num_vehicles 必须为正整数。")
    if battery_limit <= 0:
        raise ValueError("battery_limit 必须为正数。")

    data: Dict[str, object] = instance.get_data()
    num_nodes: int = int(data["num_nodes"])
    depot: int = int(data["depot_index"])
    distance_matrix: np.ndarray = data["distance_matrix"]
    time_windows: List[Tuple[float, float]] = data["time_windows"]
    service_time: float = float(data["service_time"])

    nodes = list(range(num_nodes))
    customers = [i for i in nodes if i != depot]
    vehicles = list(range(num_vehicles))

    # -----------------------------
    # 构建 Gurobi 模型
    # -----------------------------
    start_time = time.perf_counter()
    model = gp.Model("VRPTW_with_battery")

    # 决策变量 x[i, j, k]：车辆 k 是否从 i 直接驶向 j
    x = model.addVars(
        nodes,
        nodes,
        vehicles,
        vtype=GRB.BINARY,
        name="x",
    )

    # 到达时间变量 t[i]：某辆车到达节点 i 的服务开始时间
    t = model.addVars(
        nodes,
        vtype=GRB.CONTINUOUS,
        lb=0.0,
        name="t",
    )

    # 目标函数：最小化总行驶距离
    model.setObjective(
        gp.quicksum(
            distance_matrix[i, j] * x[i, j, k]
            for i in nodes
            for j in nodes
            if i != j
            for k in vehicles
        ),
        GRB.MINIMIZE,
    )

    # -----------------------------
    # 约束：每个客户恰好被访问一次
    # -----------------------------
    for j in customers:
        # 所有车辆到达 j 的流量之和为 1
        model.addConstr(
            gp.quicksum(
                x[i, j, k] for i in nodes if i != j for k in vehicles
            )
            == 1,
            name=f"visit_in_{j}",
        )
        # 所有车辆从 j 出发的流量之和为 1
        model.addConstr(
            gp.quicksum(
                x[j, h, k] for h in nodes if h != j for k in vehicles
            )
            == 1,
            name=f"visit_out_{j}",
        )

    # -----------------------------
    # 约束：车辆流守恒 & 车次限制
    # -----------------------------
    for k in vehicles:
        # 每辆车从 depot 出发最多一次
        model.addConstr(
            gp.quicksum(
                x[depot, j, k] for j in nodes if j != depot
            )
            <= 1,
            name=f"depot_depart_{k}",
        )
        # 每辆车回到 depot 最多一次
        model.addConstr(
            gp.quicksum(
                x[i, depot, k] for i in nodes if i != depot
            )
            <= 1,
            name=f"depot_return_{k}",
        )

        # 对每个非仓库节点，车流守恒：
        # ∑_j x[i, j, k] = ∑_j x[j, i, k]
        for i in customers:
            model.addConstr(
                gp.quicksum(
                    x[i, j, k] for j in nodes if j != i
                )
                == gp.quicksum(
                    x[j, i, k] for j in nodes if j != i
                ),
                name=f"flow_cons_{i}_{k}",
            )

    # -----------------------------
    # 约束：时间窗 + 前继/后继时间关系
    # -----------------------------
    # depot 的到达时间设置为其时间窗起点
    depot_tw = time_windows[depot]
    model.addConstr(
        t[depot] == float(depot_tw[0]),
        name="depot_start_time",
    )

    # 节点时间窗约束
    for i in nodes:
        early, late = time_windows[i]
        model.addConstr(t[i] >= float(early), name=f"tw_early_{i}")
        model.addConstr(t[i] <= float(late), name=f"tw_late_{i}")

    # 利用 Big-M 连接 x 和 t
    # t[j] ≥ t[i] + service_time + d_ij - M * (1 - ∑_k x[i, j, k])
    max_travel = float(np.max(distance_matrix))
    max_tw_end = max(float(tw[1]) for tw in time_windows)
    big_m = max_tw_end + max_travel + service_time

    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            # 关键修正：
            # 如果 j 是 depot（仓库），不对边 i -> depot 建立时间前后继约束。
            # 否则 0->i 和 i->0 的 time_link 叠加后，会得到
            #   0 >= 2 * service_time + 2 * distance
            # 这样的矛盾，从而使“从仓库出发又回到仓库”的任何路线都时间上不可行。
            if j == depot:
                continue
            # 对所有车辆求和后的边使用一个联合 Big-M 约束
            model.addConstr(
                t[j]
                >= t[i]
                + service_time
                + float(distance_matrix[i, j])
                - big_m * (
                    1
                    - gp.quicksum(
                        x[i, j, k] for k in vehicles
                    )
                ),
                name=f"time_link_{i}_{j}",
            )

    # -----------------------------
    # 约束：每辆车的电池限制（总行驶距离）
    # -----------------------------
    for k in vehicles:
        model.addConstr(
            gp.quicksum(
                distance_matrix[i, j] * x[i, j, k]
                for i in nodes
                for j in nodes
                if i != j
            )
            <= float(battery_limit),
            name=f"battery_{k}",
        )

    # -----------------------------
    # 求解参数设置
    # -----------------------------
    model.setParam("TimeLimit", 30.0)
    # 对教学/实验场景关闭冗长输出，可按需开启
    model.setParam("OutputFlag", 0)

    model.optimize()
    end_time = time.perf_counter()
    runtime = end_time - start_time

    # -----------------------------
    # 根据求解状态提取解
    # -----------------------------
    status = model.Status
    try:
        status_name = gp.statusToString(status)
    except Exception:  # pragma: no cover - 兼容旧版本 Gurobi 或异常情况
        status_name = str(status)

    # 无论状态码如何，先检查是否存在任何可行解：
    # 包括 Status=TIME_LIMIT 但没有找到可行解（SolCount=0）的情况，
    # 此时变量不允许访问 .X / .ObjVal。
    if model.SolCount == 0:
        print(
            f"[Gurobi] 求解结束但未得到可行解，状态: {status} ({status_name})"
        )
        return None, runtime, []

    # 运行到这里说明至少有 1 个可行解：
    # 若状态不是“已找到最优解”或“时间限制内找到当前最优解”，
    # 则说明只是未证明最优或被中止，此时仍可以安全提取当前最优解。
    if status not in (GRB.OPTIMAL, GRB.TIME_LIMIT):
        print(
            f"[Gurobi] 状态为 {status} ({status_name})，"
            "存在可行解但未证明最优，将使用当前最优解作为输出。"
        )

    obj_value = model.ObjVal

    # 重构每辆车的路径
    routes: List[List[int]] = []
    for k in vehicles:
        # 为车辆 k 构建邻接表
        successor: Dict[int, int] = {}
        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                if x[i, j, k].X > 0.5:
                    successor[i] = j

        if depot not in successor:
            # 该车辆未被使用
            continue

        # 从仓库出发沿后继关系构建路径
        route = [depot]
        current = depot
        visited_safety = 0
        max_hops = num_nodes * 2

        while True:
            if current not in successor:
                break
            nxt = successor.pop(current)
            route.append(nxt)
            visited_safety += 1
            if nxt == depot or visited_safety > max_hops:
                break
            current = nxt

        if route[-1] != depot:
            route.append(depot)

        routes.append(route)

    return float(obj_value), float(runtime), routes


if __name__ == "__main__":
    # 简单自测入口：仅在安装 Gurobi 且环境允许时才会有意义。
    # 注意：原始随机时间窗有可能导致实例本身不可行（VRPTW 常见现象），
    # 这里自测使用“宽松时间窗”，保证更大概率得到可行解，便于你直观验证模型。
    from data_generator import VRPInstance

    if not GUROBI_AVAILABLE:
        print("Gurobi 不可用，跳过 solver_gurobi 自测。")
    else:
        instance = VRPInstance(
            num_nodes=10,
            depot_time_window=(0.0, 10000.0),
            customer_time_window_span=(5000.0, 10000.0),
            seed=123,
        )
        obj, t_used, rts = solve_gurobi(
            instance,
            num_vehicles=3,
            battery_limit=10000.0,
        )
        print("Gurobi 求解结果：")
        print("  目标值:", obj)
        print("  求解时间 (s):", t_used)
        print("  路径:", rts)
        if obj is None or not rts:
            print(
                "  [提示] 即使用宽松时间窗，该随机实例在给定约束下仍不可行；"
                "这在带时间窗的 VRP 中是正常现象，可以通过换 seed 或放宽约束来获得可行解。"
            )
