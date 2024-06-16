import numpy as np
import matplotlib.pyplot as plt
import random
import os

def calculate_drone_coverage_width(fov_angle, altitude, ground_resolution):
    """
    Calculate the ground coverage width for a drone at a specific altitude and field of view angle.
    fov_angle: Field of view angle (degrees)
    altitude: Altitude (units the same as ground_resolution)
    ground_resolution: Ground resolution (units/pixel)
    """
    ground_coverage_width = 2 * (np.tan(np.radians(fov_angle / 2)) * altitude)
    grid_width = ground_coverage_width / ground_resolution
    print(f"Physical coverage width: {ground_coverage_width:.2f}")
    print(f"Coverage width (in grid units): {grid_width:.2f}")
    return int(grid_width)

def calc_connectivity(slice):
    """
    Calculate the connectivity of a slice and return the connected regions.
    0 represents passable areas, 1 represents obstacles.
    """
    connectivity = 0
    last_data = 1  # Initialize as 1 (obstacle)
    open_part = False
    connective_parts = []
    for i, data in enumerate(slice):
        if last_data == 1 and data == 0:
            open_part = True
            start_point = i
        elif last_data == 0 and data == 1 and open_part:
            open_part = False
            connectivity += 1
            end_point = i
            connective_parts.append((start_point, end_point))
        last_data = data
    if open_part:
        connective_parts.append((start_point, len(slice)))
        connectivity += 1
    return connectivity, connective_parts

def get_adjacency_matrix(parts_left, parts_right):
    adjacency_matrix = np.zeros([len(parts_left), len(parts_right)])
    for l, lparts in enumerate(parts_left):
        for r, rparts in enumerate(parts_right):
            # Check if there is an overlap between parts from left and right
            if min(lparts[1], rparts[1]) - max(lparts[0], rparts[0]) > 0:
                adjacency_matrix[l, r] = 1
    return adjacency_matrix

def bcd(erode_img, drone_width, save_path):
    rows, cols = erode_img.shape
    separate_img = np.zeros_like(erode_img, dtype=np.int32)
    current_cell = 1
    last_connectivity_parts = []

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for col in range(drone_width // 2, cols, drone_width):
        start_col = max(col - drone_width // 2, 0)
        end_col = min(col + drone_width // 2 + 1, cols)  # 加1确保包含边界
        center_col = col

        # Get the values of the center column
        current_slice = erode_img[:, center_col]

        connectivity, connective_parts = calc_connectivity(current_slice)

        if col == drone_width // 2 or not last_connectivity_parts:
            current_cells = list(range(current_cell, current_cell + connectivity))
            current_cell += connectivity
        else:
            adj_matrix = get_adjacency_matrix(last_connectivity_parts, connective_parts)
            current_cells = [0] * len(connective_parts)

            for i in range(len(last_connectivity_parts)):
                connected = np.where(adj_matrix[i, :] > 0)[0]
                if len(connected) == 1:
                    current_cells[connected[0]] = last_connectivity_parts[i][2]
                else:
                    for j in connected:
                        current_cells[j] = current_cell
                        current_cell += 1

            for idx, connected in enumerate(current_cells):
                if connected == 0:
                    current_cells[idx] = current_cell
                    current_cell += 1

        # Update the map within the full bandwidth
        for idx, part in enumerate(connective_parts):
            separate_img[part[0]:part[1], start_col:end_col] = current_cells[idx]
            np.save(os.path.join(save_path, f'region_{current_cells[idx]}.npy'), separate_img == current_cells[idx])

        last_connectivity_parts = [(part[0], part[1], current_cells[idx]) for idx, part in enumerate(connective_parts)]

    return separate_img, current_cell - 1

def display_separate_map(separate_map, cells, erode_img):
    # 创建显示图像
    display_img = np.zeros([*separate_map.shape, 3], dtype=np.uint8)

    # 随机生成颜色，为每个子区域分配颜色
    random_colors = np.random.randint(0, 255, [cells + 1, 3])
    random_colors[0] = [255, 255, 255]  # 将障碍物区域设置为白色

    # 遍历每个细胞标识，为其分配颜色
    for cell_id in range(1, cells + 1):
        display_img[separate_map == cell_id] = random_colors[cell_id]

    # 将障碍物标记为白色
    display_img[erode_img == 1] = [255, 0, 0]  # 假设障碍物在erode_img中标记为1

    plt.imshow(display_img, origin='lower')
    plt.title('Subregions with Obstacles')
    plt.show()

if __name__ == '__main__':
    file_path = '2dslice_38.npy'
    erode_img = np.load(file_path)
    fov_angle = 80  # Field of view angle
    ground_height = 34
    height = 78
    altitude = height - ground_height  # Altitude
    ground_resolution = 2  # Ground resolution (units/pixel)
    drone_width = calculate_drone_coverage_width(fov_angle, altitude, ground_resolution)
    save_path = 'subregions'
    separate_img, cells = bcd(erode_img, drone_width, save_path)
    print('Total cells:', cells-2)
    display_separate_map(separate_img, cells, erode_img)  # 添加原始地图信息
    print(f"Drone coverage width: {drone_width} grids")
