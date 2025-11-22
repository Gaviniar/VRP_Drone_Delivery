
from experiment_manager import ExperimentManager
from visualization_manager import VisualizationManager
from config import CONFIG


def main():
    """
    主程序入口
    
    执行流程:
    1. 运行小规模对比实验(Gurobi vs CW)
    2. 运行大规模性能测试(仅CW)
    3. 生成所有可视化图表和动画
    4. 打印实验摘要
    """
    # 打印配置信息
    print_configuration()
    
    # 实验执行
    manager = ExperimentManager()
    result = manager.run_all_experiments()
    
    # 可视化生成
    viz_manager = VisualizationManager(result)
    viz_manager.generate_all_visualizations()
    
    # 完成提示
    print("\n" + "="*60)
    print("实验完成!".center(60))
    print("="*60)
    print("\n所有图表和动画已生成,请查看输出文件。")
    print("GIF动画结束时会暂停数秒,方便查看最终结果。\n")


def print_configuration():
    """打印当前配置"""
    print("\n" + "="*60)
    print("当前实验配置".center(60))
    print("="*60)
    print(f"随机种子: {CONFIG.random_seed}")
    print(f"小规模测试: N = {CONFIG.small_n_list}")
    print(f"大规模测试: N = {CONFIG.large_n_list}")  
    print(f"详细可视化生成: N = {CONFIG.visualization_n_list}")  
    print(f"车辆容量: {CONFIG.vehicle_capacity}")
    print(f"小规模电池限制: {CONFIG.small_battery_limit}")
    print(f"大规模电池限制: {CONFIG.large_battery_limit}")
    print(f"动画帧率: {CONFIG.animation_fps} FPS")
    print(f"动画结束暂停: {CONFIG.animation_pause_duration}秒")
    print("="*60 + "\n")



if __name__ == "__main__":
    main()
