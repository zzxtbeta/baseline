def modify_dataset(input_filename, output_filename):
    with open(input_filename, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    for line in lines:
        # 用制表符分割每一行的数字
        # values = line.strip().split('\t')
        values = line.strip().split(' ')

        # 将前两列的每个数字加1
        values[0] = str(int(values[0]) -1)
        # values[1] = str(int(values[1]) + 1)

        # 将修改后的行加入列表
        modified_lines.append('\t'.join(values))

    # 将修改后的数据保存到新文件
    with open(output_filename, 'w') as file:
        file.write('\n'.join(modified_lines))

    return modified_lines

# 指定输入文件和输出文件
input_filename = '/home/yrgu/topic/baseline/Baselines/node2vec/emb/otc_embedding.txt'
output_filename = '/home/yrgu/topic/baseline/Baselines/node2vec/emb/otc_embedding.txt'

# 调用函数进行修改
modified_lines = modify_dataset(input_filename, output_filename)