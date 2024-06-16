import numpy as np
import heapq
import open3d as o3d
import time

def parse_coordinates_from_text(file_path):
    """解析文本文件中的坐标点，返回一个字典，其中键是区域名称，值是起始和终止坐标的元组"""
    region_data = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 4):
            region_file = lines[i].strip()
            start_coords = tuple(map(int, lines[i+1].split(':')[1].strip().split(',')))
            end_coords = tuple(map(int, lines[i+2].split(':')[1].strip().split(',')))
            region_data[region_file] = (start_coords, end_coords)
    print(region_data)
    return region_data


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

#要修改起始点和终点，
def main(file_path):
    region_data = parse_coordinates_from_text(file_path)
    regions = list(region_data.keys())
    num_regions = len(regions)
    print(len(regions))
    path_lengths = np.zeros((num_regions+1, num_regions+1), dtype=object)  # 路径长度矩阵
    print(path_lengths.shape)

    path_lengths[1:, 0] = regions
    path_lengths[0, 1:] = regions

    for i in range(len(regions)-1):
        for j in range(i+1,len(regions)):
            region1 = regions[i]
            region2 = regions[j]
            start1, end1 = region_data[region1]
            start2, end2 = region_data[region2]
            # 起始点到终止点
            path1 = a_star_search(grid, start1, end2)
            path_coordinates= [convert_index_to_coordinate(index, (cell_size_meters, cell_size_meters, cell_size_meters), np.min(coordinates, axis=0)) for index in path1]
            print("Path coordinates from", region1, "start to", region2, "end:", path_coordinates)
            print(f"Path length: {len(path1)}")
            print("")

            # 保存路径到文本文件
            save_path_to_file(path_coordinates, f"path_{region1}_start_to_{region2}_end.txt")

            # 保存路径长度到矩阵
            path_lengths[i+1, j+1] = len(path1)

            path2 = a_star_search(grid, start2, end1)
            path_coordinates= [convert_index_to_coordinate(index, (cell_size_meters, cell_size_meters, cell_size_meters), np.min(coordinates, axis=0)) for index in path2]
            print("Path coordinates from", region2, "start to", region1, "end:", path_coordinates)
            print(f"Path length: {len(path2)}")
            print("")

            # 保存路径到文本文件
            save_path_to_file(path_coordinates, f"path_{region2}_start_to_{region1}_end.txt")

            # 保存路径长度到矩阵
            path_lengths[j+1, i+1] = len(path2)


    # 保存路径长度矩阵到文件
    save_matrix_to_file(path_lengths, "path_lengths.txt")

# 保存路径到文本文件
def save_path_to_file(path_coordinates, file_path):
    with open(file_path, "w") as f:
        for coordinates in path_coordinates:
            f.write(f"{coordinates}\n")

# 保存矩阵到文件
def save_matrix_to_file(matrix, file_path):
    with open(file_path, "w") as f:
        for row in matrix:
            f.write('\t'.join(str(item) for item in row))
            f.write('\n')

# 使用示例
file_path = 'SG.txt'
# 使用示例
start_time = time.time()
filename = 'export.glb'
coordinates = read_glb_file(filename)
# 直接确定网格的边长（以米为单位）
cell_size_meters = 2  # 调整网格大小以匹配你的数据
# 创建3D网格
grid, grid_size = create_3d_grid(coordinates, (cell_size_meters, cell_size_meters, cell_size_meters))
# 转换路径中的网格索引坐标为原始地理坐标
# 输出 xyz 方向上的网格数量
end_time = time.time()
execution_time = end_time - start_time
print("Execution time:", execution_time)
main(file_path)
