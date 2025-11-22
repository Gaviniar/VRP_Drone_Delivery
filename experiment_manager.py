"""
实验管理器 - 负责组织和执行实验流程
"""
from typing import Dict, List, Optional, Tuple
import time

from data_generator import VRPInstance
from solver_cw import solve_cw
from solver_gurobi import GUROBI_AVAILABLE, solve_gurobi
from config import CONFIG


class ExperimentResult:
    """实验结果封装类"""
    
    def __init__(self):
        self.small_scale_results: List[Dict] = []
        self.large_scale_results: List[Dict] = []
        self.snapshots: Dict[int, Dict] = {}
        self.execution_time: float = 0.0
        
    def add_small_result(self, result: Dict):
        """添加小规模实验结果"""
        self.small_scale_results.append(result)
        
    def add_large_result(self, result: Dict):
        """添加大规模实验结果"""
        self.large_scale_results.append(result)
        
    def add_snapshot(self, n: int, snapshot: Dict):
        """保存指定 N 值的详细数据"""
        self.snapshots[n] = snapshot
        
    def get_summary(self) -> str:
        """生成实验摘要"""
        summary = ["=" * 60]
        summary.append("实验执行摘要".center(60))
        summary.append("=" * 60)
        summary.append(f"总耗时: {self.execution_time:.2f}秒")
        summary.append(f"小规模实验: {len(self.small_scale_results)}组")
        summary.append(f"大规模实验: {len(self.large_scale_results)}组")
        summary.append(f"生成可视化的N值: {list(self.snapshots.keys())}")
        summary.append("=" * 60)
        return "\n".join(summary)


class ExperimentManager:
    """实验管理器 - 统一管理实验流程"""
    
    def __init__(self):
        self.config = CONFIG
        self.result = ExperimentResult()
        
    def run_all_experiments(self) -> ExperimentResult:
        """运行所有实验"""
        print("\n" + "="*60)
        print("开始执行VRP无人机配送实验".center(60))
        print("="*60 + "\n")
        
        start_time = time.time()
        
        # 小规模对比实验
        self._run_small_scale_experiments()
        
        # 大规模性能测试
        self._run_large_scale_experiments()
        
        self.result.execution_time = time.time() - start_time
        
        print("\n" + self.result.get_summary())
        
        return self.result
    
    def _run_small_scale_experiments(self):
        """小规模对比实验"""
        print("=" * 60)
        print("小规模对比实验 (Gurobi vs Clarke-Wright)".center(60))
        print("=" * 60)
        self._print_table_header()
        
        for n in self.config.small_n_list:
            result, snapshot = self._run_single_small_experiment(n)
            self.result.add_small_result(result)
            
            # ⭐ 检查是否需要为该 N 值生成可视化
            if n in self.config.visualization_n_list and snapshot:
                self.result.add_snapshot(n, snapshot)
            
            self._print_result_row(result)
        
        if not GUROBI_AVAILABLE:
            print("\n[提示] 未检测到Gurobi,表中Gurobi列显示为N/A")
    
    def _run_large_scale_experiments(self):
        """大规模性能测试"""
        print("\n" + "=" * 60)
        print("大规模性能测试与可扩展性分析".center(60))  
        print("=" * 60)
        print(f"[算法] Clarke-Wright 启发式")  
        print(f"[规模] N = {self.config.large_n_list}")
        print(f"[电池限制] {self.config.large_battery_limit}")
        print("-" * 60)
        print(f"{'N':>5} | {'成本 (Cost)':>12} | {'时间 (s)':>12}")  
        print("-" * 60)
    
        for n in self.config.large_n_list:
            result = self._run_single_large_experiment(n)
            self.result.add_large_result(result)
        
            print(f"{n:>5} | "
                f"{self._format_float(result['cw_cost']):>12} | "
                f"{self._format_float(result['cw_time'], 4):>12}")

            
    def _run_single_small_experiment(
        self, 
        num_customers: int
    ) -> Tuple[Dict, Dict]:
        """运行单个小规模实验"""
        instance = VRPInstance(
            num_nodes=num_customers,
            vehicle_capacity=self.config.vehicle_capacity,
            seed=self.config.random_seed,
        )
        
        # Clarke-Wright解
        cw_cost, cw_time, cw_routes = solve_cw(
            instance=instance,
            vehicle_capacity=self.config.vehicle_capacity,
            battery_limit=self.config.small_battery_limit,
        )
        
        # Gurobi解
        gurobi_cost = None
        gurobi_time = None
        gurobi_routes = []
        
        if GUROBI_AVAILABLE:
            try:
                num_vehicles = max(
                    1, 
                    self.config.max_vehicles_factor * num_customers
                )
                obj_value, runtime, routes = solve_gurobi(
                    instance=instance,
                    num_vehicles=num_vehicles,
                    battery_limit=self.config.small_battery_limit,
                )
                if obj_value is not None and routes:
                    gurobi_cost = obj_value
                    gurobi_time = runtime
                    gurobi_routes = routes
            except Exception as exc:
                print(f"\n[警告] Gurobi求解N={num_customers}失败: {exc}")
        
        # 计算Gap
        gap_pct = None
        if (gurobi_cost is not None and cw_cost is not None 
            and gurobi_cost > 1e-9):
            gap_pct = (cw_cost - gurobi_cost) / gurobi_cost * 100.0
        
        result = {
            'N': float(num_customers),
            'gurobi_cost': gurobi_cost,
            'cw_cost': cw_cost,
            'gap_pct': gap_pct,
            'gurobi_time': gurobi_time,
            'cw_time': cw_time,
        }
        
        snapshot = {}
        # ⭐ 只在需要可视化时才保存快照
        if num_customers in self.config.visualization_n_list and cw_routes:
            snapshot = {
                'instance': instance,
                'gurobi_routes': gurobi_routes,
                'cw_routes': cw_routes,
                'N': num_customers,  # 记录 N 值
            }
        
        return result, snapshot
    
    def _run_single_large_experiment(self, num_customers: int) -> Dict:
        """运行单个大规模实验"""
        instance = VRPInstance(
            num_nodes=num_customers,
            vehicle_capacity=self.config.vehicle_capacity,
            seed=self.config.random_seed,
        )
        
        cw_cost, cw_time, _ = solve_cw(
            instance=instance,
            vehicle_capacity=self.config.vehicle_capacity,
            battery_limit=self.config.large_battery_limit,
        )
        
        return {
            'N': float(num_customers),
            'cw_cost': cw_cost,
            'cw_time': cw_time,
        }
    
    @staticmethod
    def _print_table_header():
        """打印表头"""
        print(f"{'N':>3} | {'Gurobi成本':>11} | {'CW成本':>8} | "
              f"{'Gap(%)':>8} | {'Gurobi时间(s)':>14} | {'CW时间(s)':>11}")
        print("-" * 70)
    
    @staticmethod
    def _print_result_row(result: Dict):
        """打印结果行"""
        print(f"{int(result['N']):>3} | "
              f"{ExperimentManager._format_float(result['gurobi_cost']):>11} | "
              f"{ExperimentManager._format_float(result['cw_cost']):>8} | "
              f"{ExperimentManager._format_float(result['gap_pct']):>8} | "
              f"{ExperimentManager._format_float(result['gurobi_time'], 4):>14} | "
              f"{ExperimentManager._format_float(result['cw_time'], 4):>11}")
    
    @staticmethod
    def _format_float(
        val: Optional[float], 
        digits: int = 2
    ) -> str:
        """格式化浮点数"""
        if val is None or (isinstance(val, float) and 
                          (val != val)):  # NaN check
            return "N/A"
        return f"{val:.{digits}f}"
