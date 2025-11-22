"""
可视化管理器 - 统一管理所有图表生成
"""

from matplotlib_config import setup_matplotlib
setup_matplotlib()
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, List

from config import CONFIG
from plotting_utils import (
    plot_runtime_scalability,
    plot_cost_and_gap,
    plot_routes_comparison,
    plot_task_timeline,
)
from animation_utils import AnimationGenerator





class VisualizationManager:
    """可视化管理器"""
    
    def __init__(self, experiment_result):
        self.result = experiment_result
        self.config = CONFIG
        
    def generate_all_visualizations(self):
        """生成所有可视化内容"""
        print("\n" + "="*60)
        print("生成可视化图表".center(60))
        print("="*60 + "\n")
        
        # 静态图表（全局）
        self._generate_global_plots()
        
        # 为每个指定的 N 值生成详细可视化
        for n, snapshot in self.result.snapshots.items():
            print(f"\n--- 生成 N={n} 的详细可视化 ---")
            self._generate_n_specific_plots(n, snapshot)
            self._generate_n_specific_animations(n, snapshot)
        
        # 打印生成文件列表
        self._print_output_files()
    
    def _generate_global_plots(self):
        """生成全局对比图表"""
        # 运行时间可扩展性
        output_file = self.config.get_output_path("runtime_scalability.png")
        plot_runtime_scalability(
            self.result.small_scale_results,
            self.result.large_scale_results,
            output_file
        )
        print(f"✓ 已生成: {output_file}")
    
        # 成本与Gap对比
        output_file = self.config.get_output_path("cost_gap_smallN.png")
        plot_cost_and_gap(
            self.result.small_scale_results,
            output_file
        )
        print(f"✓ 已生成: {output_file}")
    
        # 大规模性能对比图
        if self.result.large_scale_results:
            output_file = self.config.get_output_path("large_scale_comparison.png")
            from plotting_utils import plot_large_scale_comparison
            plot_large_scale_comparison(
                self.result.large_scale_results,
                output_file
            )
            print(f"✓ 已生成: {output_file}")

    
    def _generate_n_specific_plots(self, n: int, snapshot: Dict):
        """生成特定 N 值的静态图表"""
        # 路线对比图
        output_file = self.config.get_output_path(f"routes_N{n}_comparison.png")
        plot_routes_comparison(snapshot, output_file)
        print(f"✓ 已生成: {output_file}")
        
        # 任务时间线
        output_file = self.config.get_output_path(f"schedule_N{n}_timeline.png")
        plot_task_timeline(snapshot, output_file)
        print(f"✓ 已生成: {output_file}")
    
    def _generate_n_specific_animations(self, n: int, snapshot: Dict):
        """生成特定 N 值的动画"""
        animator = AnimationGenerator(snapshot, self.config)
        
        # CW单算法动画
        output_file = self.config.get_output_path(f"cw_routes_N{n}_animation.gif")
        animator.generate_cw_animation(output_file)
        print(f"✓ 已生成: {output_file}")
        
        # 多算法对比动画
        output_file = self.config.get_output_path(f"algorithm_comparison_N{n}.gif")
        animator.generate_comparison_animation(output_file)
        print(f"✓ 已生成: {output_file}")
    
    def _print_output_files(self):
        """打印生成的文件列表"""
        print("\n" + "="*60)
        print("已生成文件列表".center(60))
        print("="*60)
    
        # 全局图表
        print("\n[全局对比图表]")
        print(f"  • {self.config.get_output_path('runtime_scalability.png')}")
        print(f"  • {self.config.get_output_path('cost_gap_smallN.png')}")
        
        # ⭐ 新增大规模对比图
        if self.result.large_scale_results:
            print(f"  • {self.config.get_output_path('large_scale_comparison.png')}")
    
        # 各 N 值的详细可视化
        for n in self.result.snapshots.keys():
            print(f"\n[N={n} 详细可视化]")
            print(f"  • {self.config.get_output_path(f'routes_N{n}_comparison.png')}")
            print(f"  • {self.config.get_output_path(f'schedule_N{n}_timeline.png')}")
            print(f"  • {self.config.get_output_path(f'cw_routes_N{n}_animation.gif')}")
            print(f"  • {self.config.get_output_path(f'algorithm_comparison_N{n}.gif')}")
    
        print("="*60)

