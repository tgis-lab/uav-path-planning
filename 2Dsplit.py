import numpy as np
from plyfile import PlyData
import matplotlib.pyplot as plt

def convert_ply_to_npy(ply_file_path, center_x, center_y, grid_size, obstacle_height, output_file_path):
    # 读取PLY文件
    ply_data = PlyData.read(ply_file_path)

    # 获取顶点坐标数据
    vertices = ply_data['vertex']

    # 获取顶点的x、y、z坐标
    x = vertices['x']
    y = vertices['y']
    z = vertices['z']

    # 定义截取区域的边界
    half_size = grid_size // 2
    min_x = center_x - half_size
    max_x = center_x + half_size
    min_y = center_y - half_size
    max_y = center_y + half_size

    # 筛选在指定范围内且高度大于等于障碍高度的顶点
    valid_indices = np.where((x >= min_x) & (x <= max_x) & (y >= min_y) & (y <= max_y) & (z>= obstacle_height))
    valid_x = x[valid_indices]
    valid_y = y[valid_indices]

    # 计算截取区域的网格坐标
    grid_x = np.floor(valid_x - min_x).astype(int)
    grid_y = np.floor(valid_y - min_y).astype(int)

    # Replace the existing code block:
    grid_x = np.floor(valid_x - min_x).astype(int)
    grid_y = np.floor(valid_y - min_y).astype(int)

    # Add the following code block below:
    valid_indices = np.where((grid_x >= 0) & (grid_x < grid_size) & (grid_y >= 0) & (grid_y < grid_size))
    grid_x = grid_x[valid_indices]
    grid_y = grid_y[valid_indices]

    # 创建二维栅格地图，并将所有位置初始化为0
    grid_map = np.zeros((grid_size, grid_size), dtype=np.int8)
    grid_map[grid_y, grid_x] = 1

    # 保存二维栅格地图为.npy文件
    np.save(output_file_path, grid_map)

def visualize_2d_grid(grid_data, grid_size=5):
    """
    可视化二维栅格地图数据。
    """
    # 计算网格的大小
    grid_height, grid_width = grid_data.shape

    # 计算每个网格单元的大小
    grid_unit_size = 1

    # 设置图形大小
    plt.figure(figsize=(10, 10))

    # 绘制二维栅格地图
    plt.imshow(grid_data, cmap='binary', origin='lower', extent=[0, grid_width * grid_unit_size, 0, grid_height * grid_unit_size])

    # 设置X和Y轴的刻度
    x_ticks = np.arange(0, grid_width * grid_unit_size + 1, 50)
    y_ticks = np.arange(0, grid_height * grid_unit_size + 1, 50)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)

    # 添加标签和标题
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Grid Map at 78m')
    # 设置X和Y轴的刻度范围
    #plt.xlim(0, 2000)
    #plt.ylim(0, 2000)

    # 显示栅格地图
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)  # 添加网格线
    plt.show()

# 示例用法
ply_file_path = 'voxel_grid.ply'
center_x, center_y = 1200,1200
grid_size = 750
obstacle_height = 38
output_file_path = '2dslice_38.npy'
convert_ply_to_npy(ply_file_path, center_x, center_y, grid_size, obstacle_height, output_file_path)
grid_data = np.load(output_file_path)
visualize_2d_grid(grid_data)
