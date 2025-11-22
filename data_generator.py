import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class VRPData:
    """
    VRP 问题数据封装类。

    所有字段均为只读结构，方便在不同求解器之间传递。
    """

    num_customers: int
    num_nodes: int
    depot_index: int
    coordinates: np.ndarray  # 形状为 (num_nodes, 2)
    demands: List[int]  # 长度为 num_nodes，仓库需求恒为 0
    time_windows: List[Tuple[float, float]]  # 每个节点的时间窗 (early, late)
    service_time: float
    vehicle_capacity: Optional[float]
    distance_matrix: np.ndarray  # 形状为 (num_nodes, num_nodes)


class VRPInstance:
    """
    VRPInstance 用于生成无人机配送 VRPTW + 电池约束的随机测试数据。

    约定：
    - 节点编号 0 为仓库（depot），1..num_customers 为客户。
    - num_nodes = num_customers + 1。
    - 坐标范围为 [0, coord_max] 的正方形区域。
    - 仓库时间窗固定为 [0, depot_tw_end]，客户时间窗在给定范围内随机生成。
    """

    def __init__(
        self,
        num_nodes: int,
        coord_max: float = 100.0,
        demand_range: Tuple[int, int] = (1, 5),
        depot_time_window: Tuple[float, float] = (0.0, 1000.0),
        customer_time_window_span: Tuple[float, float] = (100.0, 400.0),
        service_time: float = 10.0,
        vehicle_capacity: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        """
        :param num_nodes: 客户数量（不含仓库），必须为正整数。
        :param coord_max: 坐标最大值，坐标范围为 [0, coord_max]。
        :param demand_range: 客户需求的整数区间 [low, high]。
        :param depot_time_window: 仓库时间窗 (start, end)。
        :param customer_time_window_span: 客户时间窗长度的随机区间 (min_span, max_span)。
        :param service_time: 每个客户统一的服务时间。
        :param vehicle_capacity: 车辆容量（可选），主要用于启发式算法。
        :param seed: 随机种子，便于实验可复现。
        """
        if num_nodes <= 0:
            raise ValueError("num_nodes 必须为正整数（代表客户数量）。")

        self.num_customers: int = num_nodes
        self.num_nodes: int = num_nodes + 1  # 加上仓库节点
        self.depot_index: int = 0
        self.coord_max: float = coord_max
        self.demand_range: Tuple[int, int] = demand_range
        self.depot_time_window: Tuple[float, float] = depot_time_window
        self.customer_time_window_span: Tuple[float, float] = customer_time_window_span
        self.service_time: float = service_time
        self.vehicle_capacity: Optional[float] = vehicle_capacity
        self.random_state = np.random.RandomState(seed)

        # 内部缓存
        self._coordinates: Optional[np.ndarray] = None
        self._demands: Optional[List[int]] = None
        self._time_windows: Optional[List[Tuple[float, float]]] = None
        self._distance_matrix: Optional[np.ndarray] = None

        # 立即生成数据，避免惰性生成导致的不一致
        self._generate_all()

    # ----------------------------------------------------------------------
    # 公共接口
    # ----------------------------------------------------------------------
    def get_data(self) -> Dict[str, object]:
        """
        以字典形式返回当前实例的全部关键信息。

        返回的字段包括：
        - num_customers, num_nodes, depot_index
        - coordinates, demands, time_windows
        - service_time, vehicle_capacity, distance_matrix
        """
        return VRPData(
            num_customers=self.num_customers,
            num_nodes=self.num_nodes,
            depot_index=self.depot_index,
            coordinates=self._coordinates.copy(),
            demands=list(self._demands),
            time_windows=list(self._time_windows),
            service_time=self.service_time,
            vehicle_capacity=self.vehicle_capacity,
            distance_matrix=self._distance_matrix.copy(),
        ).__dict__

    # ----------------------------------------------------------------------
    # 内部生成逻辑
    # ----------------------------------------------------------------------
    def _generate_all(self) -> None:
        """内部入口：依次生成坐标、需求、时间窗和距离矩阵。"""
        self._coordinates = self._generate_coordinates()
        self._demands = self._generate_demands()
        self._time_windows = self._generate_time_windows()
        self._distance_matrix = self._compute_distance_matrix(self._coordinates)

    def _generate_coordinates(self) -> np.ndarray:
        """
        随机生成所有节点的坐标。

        设计选择：
        - 仓库放在地图中心附近（coord_max / 2, coord_max / 2），
          这样路径更“分散”，更适合展示效果；
        - 客户均匀分布在 [0, coord_max] × [0, coord_max] 区域。
        """
        coord = np.zeros((self.num_nodes, 2), dtype=float)

        # 仓库坐标放在中心
        depot_coord = self.coord_max / 2.0
        coord[self.depot_index] = np.array([depot_coord, depot_coord], dtype=float)

        # 生成客户坐标
        coord[1:] = self.random_state.uniform(
            low=0.0,
            high=self.coord_max,
            size=(self.num_customers, 2),
        )
        return coord

    def _generate_demands(self) -> List[int]:
        """
        生成每个节点的需求量。

        - 仓库需求设置为 0；
        - 客户需求为给定整数区间内的随机值。
        """
        low, high = self.demand_range
        if low <= 0 or high < low:
            raise ValueError("demand_range 必须满足 0 < low <= high。")

        demands: List[int] = [0] * self.num_nodes
        # 为每个客户生成随机需求
        for i in range(1, self.num_nodes):
            demands[i] = int(self.random_state.randint(low, high + 1))
        return demands

    def _generate_time_windows(self) -> List[Tuple[float, float]]:
        """
        生成每个节点的时间窗。

        - 仓库时间窗直接采用 depot_time_window；
        - 客户时间窗通过随机起始时间 + 随机长度生成，
          并截断在仓库时间窗之内，避免完全不可行。
        """
        depot_start, depot_end = self.depot_time_window
        if depot_start < 0 or depot_end <= depot_start:
            raise ValueError("depot_time_window 必须满足 0 <= start < end。")

        min_span, max_span = self.customer_time_window_span
        if min_span <= 0 or max_span < min_span:
            raise ValueError("customer_time_window_span 必须满足 0 < min_span <= max_span。")

        time_windows: List[Tuple[float, float]] = [(0.0, 0.0)] * self.num_nodes

        # 仓库时间窗
        time_windows[self.depot_index] = (float(depot_start), float(depot_end))

        # 客户时间窗
        max_start = depot_end - max_span
        max_start = max(max_start, depot_start)

        for i in range(1, self.num_nodes):
            # 随机选择时间窗长度
            span = float(
                self.random_state.uniform(
                    low=min_span,
                    high=max_span,
                )
            )
            # 随机选择开始时间，保证结束时间不超过仓库结束时间
            if max_start <= depot_start:
                start_earliest = depot_start
            else:
                start_earliest = depot_start
            start_latest = max(depot_end - span, start_earliest)
            start = float(
                self.random_state.uniform(
                    low=start_earliest,
                    high=start_latest + 1e-6,
                )
            )
            end = min(start + span, depot_end)
            time_windows[i] = (start, end)

        return time_windows

    @staticmethod
    def _compute_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
        """
        根据坐标计算欧式距离矩阵。

        :param coordinates: 形状为 (n, 2) 的坐标数组。
        """
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates 必须是形状为 (n, 2) 的二维数组。")

        num_nodes = coordinates.shape[0]
        dist_matrix = np.zeros((num_nodes, num_nodes), dtype=float)

        # 逐对计算欧式距离；虽然可以用向量化方式，但这里保持实现直观清晰
        for i in range(num_nodes):
            xi, yi = coordinates[i]
            for j in range(num_nodes):
                if i == j:
                    dist_matrix[i, j] = 0.0
                else:
                    xj, yj = coordinates[j]
                    dist_matrix[i, j] = math.hypot(xi - xj, yi - yj)

        return dist_matrix

    # ----------------------------------------------------------------------
    # 辅助工具函数
    # ----------------------------------------------------------------------
    def describe(self) -> str:
        """
        返回一个简要描述字符串，便于调试和日志记录。
        """
        return (
            f"VRPInstance(num_customers={self.num_customers}, "
            f"coord_max={self.coord_max}, "
            f"demand_range={self.demand_range}, "
            f"depot_time_window={self.depot_time_window}, "
            f"service_time={self.service_time}, "
            f"vehicle_capacity={self.vehicle_capacity})"
        )


if __name__ == "__main__":
    # 简单自测，用于快速验证生成逻辑是否正常。
    instance = VRPInstance(num_nodes=10, seed=42)
    data = instance.get_data()

    print("=== VRPInstance 简要信息 ===")
    print(instance.describe())
    print(f"节点数量（含仓库）: {data['num_nodes']}")
    print(f"客户数量: {data['num_customers']}")
    print("仓库时间窗:", data["time_windows"][0])
    print("前 5 个坐标:\n", data["coordinates"][:5])
    print("需求列表:", data["demands"])
