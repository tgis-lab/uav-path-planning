import numpy as np
import heapq
import open3d as o3d
import time

def read_glb_file(filename):
    mesh = o3d.io.read_triangle_mesh(filename)
    vertices = np.asarray(mesh.vertices)
    return vertices

def create_3d_grid(coordinates, cell_size):
    # 计算每个方向上的网格数量
    grid_size = np.ceil((np.max(coordinates, axis=0) - np.min(coordinates, axis=0)) / cell_size).astype(int)

    # 创建网格
    grid = np.zeros(grid_size, dtype=np.int8)

    # 使用GLB文件中的最小坐标作为原点位置
    origin = np.min(coordinates, axis=0)

    # 遍历坐标并标记占据的网格单元
    for x, y, z in coordinates:
        ix, iy, iz = int(np.floor((x - origin[0]) / cell_size[0])), \
                     int(np.floor((y - origin[1]) / cell_size[1])), \
                     int(np.floor((z - origin[2]) / cell_size[2]))
        if 0 <= ix < grid_size[0] and 0 <= iy < grid_size[1] and 0 <= iz < grid_size[2]:
            grid[ix, iy, iz] = 1

    return grid, grid_size

def convert_index_to_coordinate(index, cell_size, origin):
    x = origin[0] + (index[0]+0.5) * cell_size[0]
    y = origin[1] + (index[1]+0.5) * cell_size[1]
    z = origin[2] + (index[2] +0.5)* cell_size[2]
    return x, y, z

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

def a_star_search(graph, start, goal):
    neighbors_offsets = [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    while open_set:
        _, _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]
        for offset in neighbors_offsets:
            neighbor = (current[0] + offset[0], current[1] + offset[1], current[2] + offset[2])
            if 0 <= neighbor[0] < graph.shape[0] and 0 <= neighbor[1] < graph.shape[1] and 0 <= neighbor[2] < graph.shape[2]:
                if graph[neighbor] == 1:
                    continue
                tentative_g_score = g_score[current] + 1
                if tentative_g_score < g_score.get(neighbor, np.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], id(neighbor), neighbor))
    return []

def create_path_point_cloud(path_coordinates, color=[1, 0, 0]):
    # 创建一个空的点云对象
    path_cloud = o3d.geometry.PointCloud()
    # 将路径坐标转换为Open3D中的点云
    path_cloud.points = o3d.utility.Vector3dVector(np.array(path_coordinates))
    # 设置点云颜色
    path_cloud.colors = o3d.utility.Vector3dVector(np.array([color for _ in path_coordinates]))
    return path_cloud


# 使用示例
start_time = time.time()
filename = 'export.glb'
coordinates = read_glb_file(filename)
# 直接确定网格的边长（以米为单位）
cell_size_meters = 2  # 调整网格大小以匹配你的数据
# 创建3D网格
grid, grid_size = create_3d_grid(coordinates, (cell_size_meters, cell_size_meters, cell_size_meters))
# 保存网格为Numpy数组文件
np.save('grid.npy', grid)
# 保存网格大小为Numpy数组文件
np.save('grid_size.npy', grid_size)
start = (300, 300, 16)  # 定义起点
goal = (843,825,38)   # 定义终点，需要根据实际情况调整

path = a_star_search(grid, start, goal)
print("Found path:", path)
# 输出 xyz 方向上的网格数量
print("Grid size (xyz directions):", grid_size)
# 转换路径中的网格索引坐标为原始地理坐标
path_coordinates = [convert_index_to_coordinate(index, (cell_size_meters, cell_size_meters, cell_size_meters), np.min(coordinates, axis=0)) for index in path]
print("Path coordinates:", path_coordinates)

# 检查路径是否为空
if not path:
    print("No path found.")
    # 检查路径是否为空
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time)
#可视化
# 加载原始GLB文件
mesh = o3d.io.read_triangle_mesh(filename)
# 转换路径坐标为点云
path_cloud = create_path_point_cloud(path_coordinates, color=[1, 0, 0]) # 红色路径
# 可视化原始地图和路径
o3d.visualization.draw_geometries([mesh, path_cloud])
