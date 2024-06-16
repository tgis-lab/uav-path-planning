import open3d as o3d
import numpy as np

def create_voxel_grid_from_glb(glb_file, voxel_size):
    # 加载 GLB 文件
    mesh = o3d.io.read_triangle_mesh(glb_file)

    # 确定栅格地图的边界框
    min_bound = np.min(mesh.vertices, axis=0)
    max_bound = np.max(mesh.vertices, axis=0)

    # 将边界框形状调整为 (3,)
    min_bound = np.squeeze(min_bound).astype(np.float64)
    max_bound = np.squeeze(max_bound).astype(np.float64)

    # 打印边界值
    print("Min Bound:", min_bound)
    print("Max Bound:", max_bound)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)

    return voxel_grid

# 用法示例
glb_file = "export.glb"
voxel_size = 2  # 体素大小
voxel_grid = create_voxel_grid_from_glb(glb_file, voxel_size)

# 保存栅格地图为 .ply 文件
o3d.io.write_voxel_grid("voxel_grid.ply", voxel_grid)

# 可视化栅格地图
#o3d.visualization.draw_geometries([voxel_grid])
