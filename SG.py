import os

def get_coordinates(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    first_coordinate = None
    last_coordinate = None

    for line in lines:
        # 假设每行包含一个三维坐标，格式为"x,y,z"
        coordinates = line.strip().split(',')
        if len(coordinates) == 3:
            if first_coordinate is None:
                first_coordinate = tuple(map(int, coordinates))
            last_coordinate = tuple(map(int, coordinates))

    return first_coordinate, last_coordinate


def process_files(folder_path, output_file):
    with open(output_file, 'w') as output:
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(folder_path, file_name)
                first_coordinate, last_coordinate = get_coordinates(file_path)

                if first_coordinate is not None and last_coordinate is not None:
                    file_name_without_extension = os.path.splitext(file_name)[0]  # 去除文件名的扩展名
                    output.write(f'{file_name_without_extension}\n')
                    output.write(f'First Coordinate: {", ".join(str(coord) for coord in first_coordinate)}\n')
                    output.write(f'Last Coordinate: {", ".join(str(coord) for coord in last_coordinate)}\n')
                    output.write('\n')

# 指定文件夹路径和输出文件路径
folder_path = 'subpaths'
output_file = 'SG.txt'

process_files(folder_path, output_file)
