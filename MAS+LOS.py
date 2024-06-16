import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import label, find_objects
import os


def load_scan_area(file_path):
    """加载.npy文件，其中True表示可扫描空间。"""
    return np.load(file_path)

def load_grid(npy_file_path):
    return np.load(npy_file_path)

def calculate_global_offset(center, size):
    half_size = size // 2
    return (center[0] - half_size, center[1] - half_size)

def load_and_extract_subregion(file_path, center, size):
    """加载.npy文件并从中提取子区域的坐标，并进行全局偏移。"""
    # 加载.npy文件，其中True表示可扫描空间
    scan_area = np.load(file_path)
    # 从整个区域中提取子区域，获取值为True的坐标
    subregion_coords = np.argwhere(scan_area)
    # 反转坐标，将行列索引互换
    subregion_coords_flipped = np.flip(subregion_coords, axis=1)
    # 进行全局偏移
    offset = calculate_global_offset(center, size)
    subregion_coords_offset = subregion_coords_flipped + offset
    print("子区域大小：",len(subregion_coords_offset))
    return subregion_coords_offset

def calculate_scan_paths(scan_area, sweep_width):
    """计算每个子区域的最优弓字形扫描路径，并确保路径的连续性。"""
    labeled_area, num_features = label(scan_area)
    slices = find_objects(labeled_area)

    raw_scan_paths = []
    last_end_point = None

    for idx, slice_obj in enumerate(slices):
        start_row, end_row = slice_obj[0].start, slice_obj[0].stop
        start_col, end_col = slice_obj[1].start, slice_obj[1].stop

        height = end_row - start_row
        width = end_col - start_col
        scan_horizontal = width > height  # 选择扫描方向基于长边

        if scan_horizontal:
            # 水平扫描
            adjusted_start_row = start_row + sweep_width // 2  # 调整起始行
            for row in range(adjusted_start_row, end_row, sweep_width):
                row_path = [(row, col) for col in range(start_col, end_col) if scan_area[row, col] == 1]
                if (row - adjusted_start_row) // sweep_width % 2 != 0:
                    row_path.reverse()

                if row_path:
                    if last_end_point and (last_end_point[0] != row_path[0][0]):
                        # 添加垂直连接
                        step = np.sign(row_path[0][0] - last_end_point[0])
                        vertical_connect = [(r, last_end_point[1]) for r in range(last_end_point[0] + step, row_path[0][0], step) if scan_area[r, last_end_point[1]] == 1]
                        if vertical_connect:
                            raw_scan_paths.append(vertical_connect)

                    raw_scan_paths.append(row_path)
                    last_end_point = row_path[-1]

        else:
            # 垂直扫描
            adjusted_start_col = start_col + sweep_width // 2  # 调整起始列
            for col in range(adjusted_start_col, end_col, sweep_width):
                col_path = [(r, col) for r in range(start_row, end_row) if scan_area[r, col] == 1]
                if (col - adjusted_start_col) // sweep_width % 2 != 0:
                    col_path.reverse()

                if col_path:
                    if last_end_point and (last_end_point[1] != col_path[0][1]):
                        # 添加水平连接
                        step = np.sign(col_path[0][1] - last_end_point[1])
                        horizontal_connect = [(last_end_point[0], c) for c in range(last_end_point[1] + step, col_path[0][1], step) if scan_area[last_end_point[0], c] == 1]
                        if horizontal_connect:
                            raw_scan_paths.append(horizontal_connect)

                    raw_scan_paths.append(col_path)
                    last_end_point = col_path[-1]

    # 去除重复的坐标点
    scan_paths = []
    unique_points_set = set()
    for path in raw_scan_paths:
        unique_path = []
        for point in path:
            if point not in unique_points_set:
                unique_path.append(point)
                unique_points_set.add(point)
        scan_paths.append(unique_path)

    return scan_paths



def process_scan_paths(scan_paths):
    """处理扫描路径，将坐标取反并添加第三维坐标。"""

    global_offset = calculate_global_offset(center, size)
    print(global_offset)
    viewpoints = []
    for path in scan_paths:
        for point in path:
            viewpoint = (point[1] +  global_offset[0], point[0] +  global_offset[1], 38)  # 添加第三维坐标，高度为15
            viewpoints.append(viewpoint)
    print("观察点数量：",len(viewpoints))
    return viewpoints


def create_cone_volume(viewpoint, fov_angle, grid_scale, grid):
    """生成视锥内部的地面点。"""
    height = viewpoint[2] - 16  # Adjust height for ground level
    radius = height * np.tan(np.radians(fov_angle / 2)) / grid_scale
    points = []

    for dx in range(-int(radius), int(radius) + 1):
        for dy in range(-int(radius), int(radius) + 1):
            if dx**2 + dy**2 <= radius**2:
                x = int(viewpoint[0] + dx)
                y = int(viewpoint[1] + dy)
                z = 16  # 地面高度为16
                if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1]:
                    points.append((x, y, z))
    return points

def bresenham_3d(x0, y0, z0, x1, y1, z1):
    """生成3D空间中直线上的点集合。"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    dz = abs(z1 - z0)
    xs = 1 if x1 > x0 else -1
    ys = 1 if y1 > y0 else -1
    zs = 1 if z1 > z0 else -1

    # Driving axis is the axis of greatest delta
    if dx >= dy and dx >= dz:        # x is the driving axis
        p1 = 2*dy - dx
        p2 = 2*dz - dx
        while x0 != x1:
            points.append((x0, y0, z0))
            x0 += xs
            if p1 >= 0:
                y0 += ys
                p1 -= 2*dx
            if p2 >= 0:
                z0 += zs
                p2 -= 2*dx
            p1 += 2*dy
            p2 += 2*dz
    elif dy >= dx and dy >= dz:      # y is the driving axis
        p1 = 2*dx - dy
        p2 = 2*dz - dy
        while y0 != y1:
            points.append((x0, y0, z0))
            y0 += ys
            if p1 >= 0:
                x0 += xs
                p1 -= 2*dy
            if p2 >= 0:
                z0 += zs
                p2 -= 2*dy
            p1 += 2*dx
            p2 += 2*dz
    else:                            # z is the driving axis
        p1 = 2*dy - dz
        p2 = 2*dx - dz
        while z0 != z1:
            points.append((x0, y0, z0))
            z0 += zs
            if p1 >= 0:
                y0 += ys
                p1 -= 2*dz
            if p2 >= 0:
                x0 += xs
                p2 -= 2*dz
            p1 += 2*dy
            p2 += 2*dx

    points.append((x1, y1, z1))  # Ensure the last point is added
    return points

def is_line_of_sight_clear(grid, viewpoint, ground_point, los_cache):
    """检查从视点到地面点的线路是否通畅，使用3D Bresenham算法。"""
    x0, y0, z0 = map(int, viewpoint)
    x1, y1, z1 = map(int, ground_point)
    line_points = bresenham_3d(x0, y0, z0, x1, y1, z1)

    for x, y, z in line_points:
        if (x, y, z) in los_cache:
            if not los_cache[(x, y, z)]:
                return False
        else:
            if 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and 0 <= z < grid.shape[2] and grid[x, y, z] == 1:
                los_cache[(x, y, z)] = False
                return False
            los_cache[(x, y, z)] = True
    return True

def calculate_visibility(grid, viewpoints, fov_angle, grid_scale, subregion_coords_offset):
    """计算所有视点的可视区域，返回被阻挡的点和可见的点。"""
    visible_points = set()
    los_cache = {}
    for viewpoint in viewpoints:
        cone_points = create_cone_volume(viewpoint, fov_angle, grid_scale, grid)
        for point in cone_points:
            if is_line_of_sight_clear(grid, viewpoint, point, los_cache):
                # 只有当点的视线清晰，并且该点在地图上不是障碍物时，才认为它是可见的
                if grid[point[0], point[1], point[2]] != 1:
                    visible_points.add(point)
    print(f"检查完成.")
    return np.array(visible_points)


def is_within_subregion(point, subregion_coords):
    """检查点是否在子区域范围内。"""
    print(point)
    return np.any(np.all(subregion_coords == point, axis=1))




def calculate_invisible_points(grid, visible_points, subregion_coords_offset):
    invisible_points = set()

    # 遍历子区域坐标偏移列表
    for coord in subregion_coords_offset:
        x, y = coord
        # 检查点是否在子区域范围内，并且满足 grid[x, y, 6] == 0 的条件，然后添加到不可见点集合中
        if is_within_subregion((x, y), subregion_coords_offset) and grid[x, y, 16] == 0:
            invisible_points.add((x, y, 16))

    # 从不可见点集合中移除可见点集合中的点
    invisible_points -= visible_points
    print(len(invisible_points))
    return invisible_points

def adjust_sweep_width_for_subregions(directory, initial_sweep_width):
    """遍历每个子区域，并动态调整扫描宽度，直到满足条件为止。返回每个子区域的扫描路径和最终的扫描宽度。"""
    final_sweep_widths = {}
    scan_areas = {}

    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            file_path = os.path.join(directory, filename)
            scan_area = load_scan_area(file_path)
            scan_areas[filename] = scan_area
            sweep_width = initial_sweep_width

            while True:
                scan_paths = calculate_scan_paths(scan_area, sweep_width)
                viewpoints = process_scan_paths(scan_paths)  # 使用空的全局偏移
                subregion_coords_offset = load_and_extract_subregion(file_path, center, size)
                visible_points = calculate_visibility(grid, viewpoints, fov_angle, grid_scale, subregion_coords_offset)
                invisible_points = calculate_invisible_points(grid, visible_points, subregion_coords_offset)
                if len(invisible_points) < 0.001*len(subregion_coords_offset):
                    final_sweep_widths[filename] = sweep_width
                    print("my")
                    # 保存路径到文件中
                    save_path_to_file(subpath_directory, filename, viewpoints)
                    break
                else:
                    adjusted_sweep_width = find_next_divisible_width(sweep_width, initial_sweep_width)
                    print("调整后宽度：", adjusted_sweep_width)
                    if adjusted_sweep_width < 1:
                        final_sweep_widths[filename] = sweep_width
                        break
                    sweep_width = adjusted_sweep_width
            print("最后宽度：", final_sweep_widths)
    return scan_areas, final_sweep_widths


def find_next_divisible_width(sweep_width, initial_sweep_width):
    """找到下一个可被可视域宽度除尽的宽度。"""
    while True:
        sweep_width -= 1
        if initial_sweep_width % sweep_width  == 0:
            return sweep_width

def save_path_to_file(subpath_directory, filename, viewpoints):
    """保存扫描路径到文件中。"""
    os.makedirs(subpath_directory, exist_ok=True)  # 确保目录存在
    subpath_file = os.path.join(subpath_directory, f"{os.path.splitext(filename)[0]}.txt")
    with open(subpath_file, 'w') as file:
        for point in viewpoints:
            file.write(f"{point[0]},{point[1]},{point[2]}\n")


if __name__ == "__main__":
    directory = 'subregion'
    subpath_directory = 'subpaths'
    grid = load_grid('grid.npy')
    center = (1200, 1200)  # 假定的区域中心点，根据需要调整
    initial_sweep_width = 36  # 扫描宽度
    size = 750
    fov_angle = 80  # 视场角度为70度
    grid_scale = 1  # 栅格比例为1
    scan_areas,final_sweep_widths = adjust_sweep_width_for_subregions(directory, initial_sweep_width)






