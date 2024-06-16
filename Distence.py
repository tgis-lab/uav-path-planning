import os
def replace_zeros_with_valid_lines(directory):
    path_lengths_file = os.path.join("path_lengths（1）.txt")
    with open(path_lengths_file, 'r') as file:
        lines = file.readlines()

    for i in range(1, len(lines)):
        line = lines[i].strip().split('\t')
        for j in range(1, len(line)):
            if line[j] == '0':
                filename = line[0]+'.txt'
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r') as txt_file:
                    valid_lines = [txt_line for txt_line in txt_file if txt_line.strip()]
                    print(len(valid_lines),filename)
                    #line[j] = str(len(valid_lines))

        lines[i] = '\t'.join(line) + '\n'

    with open(path_lengths_file, 'w') as file:
        file.writelines(lines)

# 示例用法
directory_path = "subpaths"
replace_zeros_with_valid_lines(directory_path)
print("已将path_lengths.txt文件中的零值替换为有效行数")
