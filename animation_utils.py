"""
动画工具 - 增强版动画生成功能
"""
from matplotlib_config import setup_matplotlib
setup_matplotlib()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import FancyBboxPatch
from typing import Dict, List, Tuple

from data_generator import VRPInstance



class AnimationGenerator:
    """动画生成器 - 带统计信息和暂停功能"""
    
    def __init__(self, snapshot: Dict, config):
        self.snapshot = snapshot
        self.config = config
        self.instance: VRPInstance = snapshot['instance']
        self.data = self.instance.get_data()
        self.n = snapshot.get('N', 10)  # 获取当前 N 值
        
    def generate_cw_animation(self, output_file: str):
        """生成CW算法动画(增强版)"""
        cw_routes = self.snapshot.get('cw_routes', [])
        if not cw_routes:
            print(f"[提示] N={self.n} 无CW路径,跳过CW动画")
            return
        
        self._create_enhanced_animation(
            routes=cw_routes,
            title=f"Clarke-Wright Algorithm (N={self.n})",
            output_file=output_file,
            battery_limit=self.config.small_battery_limit
        )
    
    def generate_comparison_animation(self, output_file: str):
        """生成算法对比动画"""
        gurobi_routes = self.snapshot.get('gurobi_routes', [])
        cw_routes = self.snapshot.get('cw_routes', [])
        
        if not gurobi_routes or not cw_routes:
            print(f"[提示] N={self.n} 需要两种算法的路径才能生成对比动画")
            return
        
        algorithms = {
            'Gurobi': gurobi_routes,
            'Clarke-Wright': cw_routes
        }
        
        from sim import create_algorithm_comparison_animation
        create_algorithm_comparison_animation(
            snapshot=self.snapshot,
            algorithms=algorithms,
            battery_limit=self.config.small_battery_limit,
            output_file=output_file
        )
    
    
    def _create_enhanced_animation(
        self,
        routes: List[List[int]],
        title: str,
        output_file: str,
        battery_limit: float
    ):
        """创建增强版动画 - 带统计信息和结束暂停"""
        coordinates = self.data['coordinates']
        
        # 构建运动轨迹
        tracks = self._build_tracks(routes)
        if not tracks:
            return
        
        # 计算时间范围
        t_min = min(seg['start'] for track in tracks for seg in track)
        t_max = max(seg['end'] for track in tracks for seg in track)
        
        # 创建帧序列 (包含暂停帧)
        normal_frames = self.config.animation_frames
        pause_frames = int(
            self.config.animation_pause_duration * 
            self.config.animation_fps
        )
        total_frames = normal_frames + pause_frames
        
        # 正常播放时间点
        times_normal = np.linspace(t_min, t_max, normal_frames)
        # 暂停阶段保持在最后时刻
        times_pause = np.full(pause_frames, t_max)
        times = np.concatenate([times_normal, times_pause])
        
        # 创建图形
        fig, (ax_map, ax_stats) = plt.subplots(
            2, 1, 
            figsize=(10, 10),
            gridspec_kw={'height_ratios': [3, 1]}
        )
        
        # 地图初始化
        self._init_map_axes(ax_map, coordinates, title)
        
        # 创建车辆和路径对象
        vehicle_artists = []
        path_lines = []
        for _ in tracks:
            # 车辆
            drone = FancyBboxPatch(
                (0, 0), 2, 1.5,
                boxstyle="round,pad=0.1",
                facecolor='#FF6B6B',
                edgecolor='black',
                linewidth=1.5,
                zorder=8
            )
            ax_map.add_patch(drone)
            vehicle_artists.append(drone)
            
            # 路径线
            line, = ax_map.plot(
                [], [], 
                linewidth=2, 
                color='#FF6B6B',
                alpha=0.7
            )
            path_lines.append(line)
        
        # 统计信息面板
        stats_text = self._create_stats_panel(ax_stats)
        
        # 动画更新函数
        def update(frame_idx: int):
            t = times[frame_idx]
            is_paused = frame_idx >= normal_frames
            
            total_distance = 0.0
            total_battery_used = 0.0
            
            artists = []
            
            for v_idx, track in enumerate(tracks):
                pos, path_points, battery_remain, dist = \
                    self._get_vehicle_state(
                        track, t, coordinates[0], battery_limit
                    )
                
                total_distance += dist
                total_battery_used += (battery_limit - battery_remain)
                
                # 更新车辆位置
                vehicle_artists[v_idx].set_x(pos[0] - 1)
                vehicle_artists[v_idx].set_y(pos[1] - 0.75)
                artists.append(vehicle_artists[v_idx])
                
                # 更新路径
                if path_points:
                    xs = [p[0] for p in path_points]
                    ys = [p[1] for p in path_points]
                    path_lines[v_idx].set_data(xs, ys)
                artists.append(path_lines[v_idx])
            
            # 更新统计信息
            avg_battery_remain = (
                len(tracks) * battery_limit - total_battery_used
            ) / max(len(tracks), 1)
            
            stats_info = (
            f"{'='*50}\n"
            f"Time: {t:.1f}s  "
            f"{'[PAUSED - View Results]' if is_paused else ''}\n"
            f"{'-'*50}\n"
            f"Total Distance: {total_distance:.2f}\n"
            f"Avg Battery Remaining: {avg_battery_remain:.2f} / {battery_limit:.2f}\n"
            f"Battery Usage: {(total_battery_used/(len(tracks)*battery_limit)*100):.1f}%\n"
            f"{'='*50}"
            )
            
            stats_text.set_text(stats_info)
            artists.append(stats_text)
            
            return artists
        
        # 生成动画
        try:
            anim = FuncAnimation(
                fig, update,
                frames=total_frames,
                interval=1000 // self.config.animation_fps,
                blit=True
            )
            
            writer = PillowWriter(fps=self.config.animation_fps)
            anim.save(output_file, writer=writer)
            
        except Exception as exc:
            print(f"[警告] 动画生成失败: {exc}")
        
        plt.close(fig)
    
    def _init_map_axes(self, ax, coordinates, title):
        """初始化地图轴"""
        # 绘制所有节点
        ax.scatter(
            coordinates[1:, 0],
            coordinates[1:, 1],
            s=80,
            c='lightgray',
            edgecolors='black',
            linewidths=1.5,
            label='Customers',
            zorder=5
        )
        
        # 仓库
        ax.scatter(
            [coordinates[0, 0]],
            [coordinates[0, 1]],
            s=200,
            marker='*',
            c='gold',
            edgecolors='darkred',
            linewidths=2,
            label='Depot',
            zorder=10
        )
        
        ax.set_xlabel("X Coordinate", fontsize=11)
        ax.set_ylabel("Y Coordinate", fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(fontsize=9)
    
    def _create_stats_panel(self, ax):
        """创建统计信息面板"""
        ax.axis('off')
        text = ax.text(
            0.5, 0.5, '',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=10,
            family='monospace',
            bbox=dict(
                boxstyle='round',
                facecolor='wheat',
                alpha=0.9,
                pad=1.0
            )
        )
        return text
    
    
    
    def _build_tracks(self, routes: List[List[int]]) -> List[List[dict]]:
        
        
        def _build_movement_tracks(
            instance: VRPInstance,
            routes: List[List[int]],
            ) -> List[List[dict]]:
            """
            为每辆车构造连续的运动片段列表，用于动画驱动。

            对每条 route 生成若干 segment：
            - type: "move" / "wait" / "service"
            - start, end: 时间区间
            - start_coord, end_coord: np.ndarray, shape (2,)
            """
            data = instance.get_data()
            coordinates: np.ndarray = data["coordinates"]  # type: ignore[assignment]
            distance_matrix: np.ndarray = data["distance_matrix"]  # type: ignore[assignment]
            time_windows: List[Tuple[float, float]] = data["time_windows"]  # type: ignore[assignment]
            service_time: float = float(data["service_time"])

            all_tracks: List[List[dict]] = []

            for route in routes:
                if len(route) < 2:
                    all_tracks.append([])
                    continue

                depot = route[0]
                depot_early, _ = time_windows[depot]
                current_time = float(depot_early)

                segments: List[dict] = []

                for idx in range(1, len(route)):
                    prev_node = route[idx - 1]
                    node = route[idx]

                    prev_coord = np.array(coordinates[prev_node], dtype=float)
                    node_coord = np.array(coordinates[node], dtype=float)

                    # 1) 移动阶段：prev -> node
                    travel_time = float(distance_matrix[prev_node, node])
                    move_start = current_time
                    move_end = current_time + travel_time
                    segments.append(
                        {
                            "type": "move",
                            "start": move_start,
                            "end": move_end,
                            "start_coord": prev_coord,
                            "end_coord": node_coord,
                        }
                    )
                    current_time = move_end

                    # 2) 等待阶段：如果早于时间窗下界
                    early, _ = time_windows[node]
                    if current_time < early:
                        wait_start = current_time
                        wait_end = float(early)
                        segments.append(
                            {
                                "type": "wait",
                                "start": wait_start,
                                "end": wait_end,
                                "start_coord": node_coord,
                                "end_coord": node_coord,
                            }
                        )
                        current_time = wait_end

                    # 3) 服务阶段：非仓库节点有服务时间
                    if node != depot:
                        service_start = current_time
                        service_end = service_start + service_time
                        segments.append(
                            {
                                "type": "service",
                                "start": service_start,
                                "end": service_end,
                                "start_coord": node_coord,
                                "end_coord": node_coord,
                            }
                        )
                        current_time = service_end

                all_tracks.append(segments)

            return all_tracks
        """构建车辆运动轨迹"""
        # 复用原有逻辑
        return _build_movement_tracks(self.instance, routes)
    
    def _get_vehicle_state(
        self,
        track: List[dict],
        t: float,
        depot_pos: np.ndarray,
        battery_limit: float
    ) -> Tuple[np.ndarray, List[np.ndarray], float, float]:
        """获取车辆状态"""
        # 复用原有逻辑
        if not track:
            return depot_pos, [depot_pos], battery_limit, 0.0
        
        path_points = [depot_pos.copy()]
        current_pos = depot_pos.copy()
        traveled = 0.0
        
        for seg in track:
            s, e = seg['start'], seg['end']
            p0, p1 = seg['start_coord'], seg['end_coord']
            
            if t >= e:
                if seg['type'] == 'move':
                    traveled += float(np.linalg.norm(p1 - p0))
                path_points.append(p1.copy())
                current_pos = p1
            elif t <= s:
                break
            else:
                if seg['type'] == 'move':
                    ratio = (t - s) / max(e - s, 1e-9)
                    pos = p0 + ratio * (p1 - p0)
                    path_points.append(pos.copy())
                    current_pos = pos
                    traveled += float(np.linalg.norm(p1 - p0) * ratio)
                else:
                    path_points.append(p0.copy())
                    current_pos = p0
                break
        
        remaining = max(battery_limit - traveled, 0.0)
        return current_pos, path_points, remaining, traveled
