"""
实验配置文件 - 集中管理所有可调参数
"""
from dataclasses import dataclass, field
from typing import List, Tuple
from matplotlib_config import setup_matplotlib
setup_matplotlib()

@dataclass
class ExperimentConfig:
    """实验配置类"""
    
    # 随机种子
    random_seed: int = 202511233
    
    # 实验规模
    small_n_list: List[int] = field(default_factory=lambda: [5, 8, 10, 15])
    large_n_list: List[int] = field(default_factory=lambda: [20, 50, 100])
    
    # ⭐ 新增：指定哪些 N 值需要生成详细可视化
    # 可以是单个值 [10] 或多个值 [10, 15]
    visualization_n_list: List[int] = field(default_factory=lambda: [10, 15])
    
    # 车辆与电池参数
    vehicle_capacity: float = 25.0
    small_battery_limit: float = 150.0
    large_battery_limit: float = 300.0
    
    # Gurobi参数
    max_vehicles_factor: int = 1
    gurobi_time_limit: float = 30.0
    
    # 动画参数
    animation_fps: int = 20
    animation_frames: int = 150
    animation_pause_duration: float = 3.0
    
    # 输出目录
    output_dir: str = "figs"
    
    def get_output_path(self, filename: str) -> str:
        """获取输出文件路径"""
        import os
        os.makedirs(self.output_dir, exist_ok=True)
        return os.path.join(self.output_dir, filename)


# 创建全局配置实例
CONFIG = ExperimentConfig()


def update_config(**kwargs):
    """
    更新配置参数
    
    示例:
        update_config(
            random_seed=12345,
            visualization_n_list=[10, 15],  # 同时生成 N=10 和 N=15 的可视化
            animation_pause_duration=5.0
        )
    """
    for key, value in kwargs.items():
        if hasattr(CONFIG, key):
            setattr(CONFIG, key, value)
        else:
            print(f"[警告] 未知配置项: {key}")
