"""
Matplotlib 全局配置
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings


def setup_matplotlib():
    """配置 matplotlib 以避免字体警告"""
    
    # 禁用特定的字体警告
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    # 方案1: 使用无衬线字体（推荐）
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = [
        'Microsoft YaHei',  # 微软雅黑（Windows）
        'SimHei',           # 黑体（Windows）
        'Arial Unicode MS', # Mac
        'DejaVu Sans',      # Linux
        'sans-serif'
    ]
    
    # 关键：使用 ASCII 减号而不是 Unicode 减号
    mpl.rcParams['axes.unicode_minus'] = False
    
    # 设置默认字体大小
    mpl.rcParams['font.size'] = 10
    
    # 提高图表质量
    mpl.rcParams['figure.dpi'] = 100
    mpl.rcParams['savefig.dpi'] = 150
    
    # 网格样式
    mpl.rcParams['grid.alpha'] = 0.3
    mpl.rcParams['grid.linestyle'] = '--'


# 自动在导入时配置
setup_matplotlib()
