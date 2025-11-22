
from matplotlib_config import setup_matplotlib
setup_matplotlib()
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from typing import Dict, List, Tuple
from data_generator import VRPInstance



def create_algorithm_comparison_animation(
    snapshot: Dict[str, object],
    algorithms: Dict[str, List[List[int]]],
    battery_limit: float,
    output_file: str = "algorithm_comparison.gif"
) -> None:
    """
    生成多算法对比的动态仿真动画（双图并排展示）。
  
    使用矢量图标：
    - 无人机：飞机形状
    - 客户：圆形（颜色变化表示状态）
    - 仓库：五角星
    """
    if not snapshot or not algorithms:
        print("[提示] 数据不足，跳过对比动画生成")
        return
  
    instance: VRPInstance = snapshot["instance"]
    data = instance.get_data()
    coordinates: np.ndarray = data["coordinates"]
    distance_matrix: np.ndarray = data["distance_matrix"]
    time_windows: List[Tuple[float, float]] = data["time_windows"]
    service_time: float = float(data["service_time"])
  
    # 为每个算法构建运动轨迹
    algo_tracks = {}
    for algo_name, routes in algorithms.items():
        if routes:
            algo_tracks[algo_name] = _build_movement_tracks_with_battery(
                instance, routes, distance_matrix, time_windows, service_time
            )
  
    if len(algo_tracks) < 2:
        print("[提示] 至少需要两个算法进行对比")
        return
  
    # 取前两个算法进行对比
    algo_names = list(algo_tracks.keys())[:2]
    algo_1_name = algo_names[0]
    algo_2_name = algo_names[1]
  
    # 计算全局时间范围
    t_min = float('inf')
    t_max = float('-inf')
    for algo_name in algo_names:
        tracks = algo_tracks[algo_name]
        for vehicle_track in tracks:
            if vehicle_track:
                t_min = min(t_min, vehicle_track[0]["start"])
                t_max = max(t_max, vehicle_track[-1]["end"])
  
    if t_max <= t_min:
        print("[提示] 时间范围异常")
        return
  
    # 创建帧序列（包含暂停帧）
    from config import CONFIG
    normal_frames = CONFIG.animation_frames
    pause_frames = int(CONFIG.animation_pause_duration * CONFIG.animation_fps)
    total_frames = normal_frames + pause_frames
    
    times_normal = np.linspace(t_min, t_max, normal_frames)
    times_pause = np.full(pause_frames, t_max)
    times = np.concatenate([times_normal, times_pause])
  
    # 布局：4行2列 (地图 + 电池 + 距离 + 统计表格)
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1, 0.8], hspace=0.45, wspace=0.35)
  
    # 两个主地图
    ax_map_1 = fig.add_subplot(gs[0, 0])
    ax_map_2 = fig.add_subplot(gs[0, 1])
  
    # 电池状态图
    ax_battery_1 = fig.add_subplot(gs[1, 0])
    ax_battery_2 = fig.add_subplot(gs[1, 1])
  
    # 累积距离图
    ax_distance_1 = fig.add_subplot(gs[2, 0])
    ax_distance_2 = fig.add_subplot(gs[2, 1])
  
    # 统计信息表格
    ax_stats_1 = fig.add_subplot(gs[3, 0])
    ax_stats_2 = fig.add_subplot(gs[3, 1])
  
    # 颜色方案
    colors = {
        algo_1_name: '#FF6B6B',
        algo_2_name: '#4ECDC4'
    }
  
    # 为两个算法分别初始化可视化
    algo_viz = {}
    axes_map = {algo_1_name: ax_map_1, algo_2_name: ax_map_2}
    axes_battery = {algo_1_name: ax_battery_1, algo_2_name: ax_battery_2}
    axes_distance = {algo_1_name: ax_distance_1, algo_2_name: ax_distance_2}
    axes_stats = {algo_1_name: ax_stats_1, algo_2_name: ax_stats_2}  # ⭐ 新增
  
    for algo_name in algo_names:
        ax_map = axes_map[algo_name]
        ax_bat = axes_battery[algo_name]
        ax_dist = axes_distance[algo_name]
        ax_stat = axes_stats[algo_name]  # ⭐ 新增
        color = colors[algo_name]
      
        # === 地图初始化 ===
        ax_map.set_xlabel("X Coordinate", fontsize=10)
        ax_map.set_ylabel("Y Coordinate", fontsize=10)
        ax_map.set_title(f"{algo_name} Algorithm", fontsize=12, fontweight='bold', pad=10)
        ax_map.set_aspect("equal", adjustable="box")
        ax_map.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
      
        x_margin = (coordinates[:, 0].max() - coordinates[:, 0].min()) * 0.1
        y_margin = (coordinates[:, 1].max() - coordinates[:, 1].min()) * 0.1
        ax_map.set_xlim(coordinates[:, 0].min() - x_margin, coordinates[:, 0].max() + x_margin)
        ax_map.set_ylim(coordinates[:, 1].min() - y_margin, coordinates[:, 1].max() + y_margin)
      
        # 绘制仓库
        depot_x, depot_y = coordinates[0]
        depot_star = patches.RegularPolygon(
            (depot_x, depot_y), 5, radius=3.5,
            orientation=np.pi/2,
            facecolor='gold', edgecolor='darkred',
            linewidth=2.5, zorder=10
        )
        ax_map.add_patch(depot_star)
        ax_map.text(depot_x, depot_y - 6, 'Depot', 
                   ha='center', va='top', fontsize=9, fontweight='bold')
      
        # 绘制客户节点
        customer_circles = []
        for i in range(1, len(coordinates)):
            x, y = coordinates[i]
            circle = patches.Circle(
                (x, y), radius=2.0,
                facecolor='lightgray', edgecolor='black',
                linewidth=1.5, zorder=5
            )
            ax_map.add_patch(circle)
            customer_circles.append(circle)
          
            ax_map.text(x, y, str(i), ha='center', va='center',
                       fontsize=7, fontweight='bold', color='black', zorder=6)
      
        # 创建无人机和路径
        drone_artists = []
        path_lines = []
      
        for _ in range(len(algo_tracks[algo_name])):
            line, = ax_map.plot([], [], linewidth=2.5, color=color, 
                               alpha=0.7, zorder=3)
            path_lines.append(line)
          
            drone_body = patches.FancyBboxPatch(
                (0, 0), 2, 1.5, boxstyle="round,pad=0.1",
                facecolor=color, edgecolor='black', linewidth=1.5, zorder=8
            )
            ax_map.add_patch(drone_body)
            drone_artists.append(drone_body)
      
        # === 电池状态柱状图 ===
        ax_bat.set_title(f"{algo_name} Battery Status", fontsize=10, fontweight='bold')
        ax_bat.set_ylabel("Battery (%)", fontsize=9)
        ax_bat.set_ylim(0, 110)
        ax_bat.set_xlim(-0.5, len(algo_tracks[algo_name]) + 0.5)
        ax_bat.set_xticks(range(len(algo_tracks[algo_name])))
        ax_bat.set_xticklabels([f"V{i+1}" for i in range(len(algo_tracks[algo_name]))], fontsize=8)
        ax_bat.grid(axis='y', linestyle='--', alpha=0.3)
        ax_bat.axhline(y=20, color='red', linestyle='--', linewidth=1, alpha=0.5)
      
        battery_bars = []
        for i in range(len(algo_tracks[algo_name])):
            bar = ax_bat.bar(i, 100, width=0.6, color=color, 
                            edgecolor='black', linewidth=1.5, alpha=0.8)
            battery_bars.append(bar[0])
      
        # === 累积距离折线图 ===
        ax_dist.set_title(f"{algo_name} Cumulative Distance", fontsize=10, fontweight='bold')
        ax_dist.set_xlabel("Time", fontsize=9)
        ax_dist.set_ylabel("Total Distance", fontsize=9)
        ax_dist.grid(True, linestyle='--', alpha=0.3)
      
        distance_line, = ax_dist.plot([], [], linewidth=2.5, color=color, 
                                      marker='o', markersize=4, alpha=0.8)
        distance_data = {'times': [], 'distances': []}
      
        # 统计表格初始化
        ax_stat.axis('tight')
        ax_stat.axis('off')
        
        # 创建空表格（将在动画中更新）
        table = ax_stat.table(
            cellText=[
                ['Metric', 'Value'],
                ['Total Distance', '0.00'],
                ['Avg Battery', '0.00 / 200.00'],
                ['Battery Usage', '0.0%'],
                ['Status', 'Running']
            ],
            cellLoc='left',
            loc='center',
            colWidths=[0.5, 0.5]
        )
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        # 表头样式
        for i in range(2):
            cell = table[(0, i)]
            cell.set_facecolor(color)
            cell.set_text_props(weight='bold', color='white')
            cell.set_edgecolor('black')
            cell.set_linewidth(1.5)
        
        # 数据行样式
        for i in range(1, 5):
            for j in range(2):
                cell = table[(i, j)]
                cell.set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                cell.set_edgecolor('gray')
                cell.set_linewidth(0.5)
      
        # 保存所有可视化元素
        algo_viz[algo_name] = {
            'ax_map': ax_map,
            'color': color,
            'customer_circles': customer_circles,
            'customer_status': [0] * (len(coordinates) - 1),
            'drone_artists': drone_artists,
            'path_lines': path_lines,
            'battery_bars': battery_bars,
            'battery_levels': [battery_limit] * len(algo_tracks[algo_name]),
            'distance_line': distance_line,
            'distance_data': distance_data,
            'stats_table': table  # ⭐ 保存表格对象
        }
  
    # 全局时间文本
    time_text = fig.text(0.5, 0.98, '', ha='center', va='top', fontsize=14, 
                        fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
  
    # === 动画更新函数 ===
    def update(frame_idx: int):
        t = float(times[frame_idx])
        is_paused = frame_idx >= normal_frames
        
        if is_paused:
            time_text.set_text(f'Time: {t:.1f} [PAUSED - View Results]')
        else:
            time_text.set_text(f'Time: {t:.1f}')
      
        artists = [time_text]
      
        for algo_name in algo_names:
            viz = algo_viz[algo_name]
            tracks = algo_tracks[algo_name]
          
            viz['customer_status'] = [0] * (len(coordinates) - 1)
          
            total_distance = 0.0
            total_battery_used = 0.0
          
            # 更新每辆车的状态
            for v_idx, vehicle_track in enumerate(tracks):
                pos, path_points, battery, dist = _get_vehicle_state_at_time(
                    vehicle_track, t, coordinates[0], battery_limit
                )
              
                total_distance += dist
                total_battery_used += (battery_limit - battery)
              
                # 更新路径
                if path_points:
                    xs = [p[0] for p in path_points]
                    ys = [p[1] for p in path_points]
                    viz['path_lines'][v_idx].set_data(xs, ys)
                    artists.append(viz['path_lines'][v_idx])
                  
                    for p in path_points[-5:]:
                        for c_idx in range(1, len(coordinates)):
                            if np.linalg.norm(p - coordinates[c_idx]) < 0.5:
                                viz['customer_status'][c_idx - 1] = max(
                                    viz['customer_status'][c_idx - 1], 1
                                )
              
                # 更新无人机位置
                drone_body = viz['drone_artists'][v_idx]
                drone_body.set_x(pos[0] - 1)
                drone_body.set_y(pos[1] - 0.75)
                artists.append(drone_body)
              
                # 更新电池
                battery_pct = (battery / battery_limit) * 100
                viz['battery_levels'][v_idx] = battery
                viz['battery_bars'][v_idx].set_height(battery_pct)
                artists.append(viz['battery_bars'][v_idx])
          
            # 更新客户状态颜色
            for c_idx, circle in enumerate(viz['customer_circles']):
                status = viz['customer_status'][c_idx]
                if status == 2:
                    color = '#2ECC71'
                elif status == 1:
                    color = '#F39C12'
                else:
                    color = 'lightgray'
                circle.set_facecolor(color)
                artists.append(circle)
          
            # 更新距离曲线
            viz['distance_data']['times'].append(t)
            viz['distance_data']['distances'].append(total_distance)
            viz['distance_line'].set_data(
                viz['distance_data']['times'],
                viz['distance_data']['distances']
            )
            artists.append(viz['distance_line'])
          
            # 动态调整距离图范围
            if viz['distance_data']['distances']:
                axes_distance[algo_name].set_xlim(t_min, t_max)
                
                max_dist = max(viz['distance_data']['distances'])
                min_dist = min(viz['distance_data']['distances'])
                
                if max_dist - min_dist > 1e-6:
                    axes_distance[algo_name].set_ylim(
                        min_dist * 0.95,
                        max_dist * 1.15
                    )
                elif max_dist > 1e-6:
                    axes_distance[algo_name].set_ylim(0, max_dist * 1.2)
                else:
                    axes_distance[algo_name].set_ylim(0, 10)
            
            # 更新统计表格
            avg_battery_remain = (
                len(tracks) * battery_limit - total_battery_used
            ) / max(len(tracks), 1)
            battery_usage_pct = (total_battery_used / (len(tracks) * battery_limit)) * 100
            
            # 更新表格数据
            table = viz['stats_table']
            table[(1, 1)].get_text().set_text(f'{total_distance:.2f}')
            table[(2, 1)].get_text().set_text(f'{avg_battery_remain:.2f} / {battery_limit:.0f}')
            table[(3, 1)].get_text().set_text(f'{battery_usage_pct:.1f}%')
            table[(4, 1)].get_text().set_text('PAUSED' if is_paused else 'Running')
            
            # 暂停时高亮状态行
            if is_paused:
                table[(4, 0)].set_facecolor('#FFE5B4')
                table[(4, 1)].set_facecolor('#FFE5B4')
                table[(4, 1)].get_text().set_weight('bold')
            else:
                table[(4, 0)].set_facecolor('#f0f0f0')
                table[(4, 1)].set_facecolor('#f0f0f0')
                table[(4, 1)].get_text().set_weight('normal')
            
            # 将表格中的所有cell添加到artists（matplotlib表格更新需要）
            for key, cell in table.get_celld().items():
                artists.append(cell)
      
        return artists
  
    # 生成动画
    try:
        from config import CONFIG
        anim = FuncAnimation(
            fig, update, frames=total_frames,
            interval=1000 // CONFIG.animation_fps,
            blit=True
        )
      
        writer = PillowWriter(fps=CONFIG.animation_fps)
        anim.save(output_file, writer=writer)
        print(f"[信息] 已生成算法对比动画：{output_file}")
      
    except Exception as exc:
        print(f"[警告] 动画生成失败：{exc}")
  
    plt.close(fig)




def _build_movement_tracks_with_battery(
    instance: VRPInstance,
    routes: List[List[int]],
    distance_matrix: np.ndarray,
    time_windows: List[Tuple[float, float]],
    service_time: float
) -> List[List[dict]]:
    """构建包含电池消耗信息的运动轨迹"""
    
    data = instance.get_data()
    coordinates: np.ndarray = data["coordinates"]
    
    all_tracks = []
    
    for route in routes:
        if len(route) < 2:
            all_tracks.append([])
            continue
        
        depot = route[0]
        depot_early, _ = time_windows[depot]
        current_time = float(depot_early)
        
        segments = []
        cumulative_distance = 0.0
        
        for idx in range(1, len(route)):
            prev_node = route[idx - 1]
            node = route[idx]
            
            prev_coord = np.array(coordinates[prev_node], dtype=float)
            node_coord = np.array(coordinates[node], dtype=float)
            
            # 移动阶段
            travel_distance = float(distance_matrix[prev_node, node])
            travel_time = travel_distance
            move_start = current_time
            move_end = current_time + travel_time
            cumulative_distance += travel_distance
            
            segments.append({
                "type": "move",
                "start": move_start,
                "end": move_end,
                "start_coord": prev_coord,
                "end_coord": node_coord,
                "distance": travel_distance,
                "cumulative_distance": cumulative_distance
            })
            current_time = move_end
            
            # 等待阶段
            early, _ = time_windows[node]
            if current_time < early:
                segments.append({
                    "type": "wait",
                    "start": current_time,
                    "end": float(early),
                    "start_coord": node_coord,
                    "end_coord": node_coord,
                    "distance": 0.0,
                    "cumulative_distance": cumulative_distance
                })
                current_time = float(early)
            
            # 服务阶段
            if node != depot:
                segments.append({
                    "type": "service",
                    "start": current_time,
                    "end": current_time + service_time,
                    "start_coord": node_coord,
                    "end_coord": node_coord,
                    "distance": 0.0,
                    "cumulative_distance": cumulative_distance
                })
                current_time += service_time
        
        all_tracks.append(segments)
    
    return all_tracks


def _get_vehicle_state_at_time(
    track: List[dict],
    t: float,
    depot_pos: np.ndarray,
    battery_limit: float
) -> Tuple[np.ndarray, List[np.ndarray], float, float]:
    """获取车辆在时刻t的状态"""
    if not track:
        return depot_pos, [depot_pos], battery_limit, 0.0
    
    path_points = [depot_pos.copy()]
    current_pos = depot_pos.copy()
    cumulative_distance = 0.0
    
    for seg in track:
        s, e = seg["start"], seg["end"]
        p0, p1 = seg["start_coord"], seg["end_coord"]
        seg_dist = seg.get("distance", 0.0)
        
        if t >= e:
            path_points.append(p1.copy())
            current_pos = p1.copy()
            cumulative_distance = seg["cumulative_distance"]
        elif t <= s:
            break
        else:
            ratio = (t - s) / max(e - s, 1e-9)
            pos = p0 + ratio * (p1 - p0)
            path_points.append(pos.copy())
            current_pos = pos.copy()
            
            if seg["type"] == "move":
                cumulative_distance = seg.get("cumulative_distance", 0.0) - seg_dist * (1 - ratio)
            else:
                cumulative_distance = seg.get("cumulative_distance", 0.0)
            break
    
    remaining_battery = battery_limit - cumulative_distance
    
    return current_pos, path_points, remaining_battery, cumulative_distance
