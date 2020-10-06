import os
import math

# 创建新路径
def make_dirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)

# 获取文件的行数
def get_total_lines(file_path):
    if not os.path.exists(file_path):
        return 0
    cmd = 'wc -l %s' % file_path
    return int(os.popen(cmd).read().split()[0])

# 函数split_file_by_row: 按行切分文件
# filepath: 切分的目标文件
# new_filepath: 生成新文件的路径
# row_cnt: 每个文件最多包含几行
# suffix_type: 新文件后缀类型，如两位字母或数字
# return: 切分后的文件列表
def split_file_by_row(filepath, new_filepath, row_cnt, suffix_type='-d'):
    tmp_dir = "/split_file_by_row/"
    make_dirs(new_filepath)
    make_dirs(new_filepath+tmp_dir)
    total_rows = get_total_lines(filepath)
    file_cnt = int(math.ceil(total_rows*1.0/row_cnt))
    command = "split -l %d -a 2 %s %s %s" % (row_cnt, suffix_type, filepath, new_filepath+tmp_dir)
    print(command)
    os.system(command)
    filelist = os.listdir(new_filepath+tmp_dir)
    command = "mv %s/* %s"%(new_filepath+tmp_dir, new_filepath)
    print(command)
    os.system(command)
    command = "rm -r %s"%(new_filepath+tmp_dir)
    print(command)
    os.system(command)
    return [new_filepath+fn for fn in filelist]

lines = get_total_lines("/home/forest/QA/src/TextClassification-Keras/data/qa.csv")
print("file lines: {}".format(str(lines)))
split_file_by_row("/home/forest/QA/src/TextClassification-Keras/data/qa.csv", "/home/forest/QA/src/TextClassification-Keras/data/tmp", 20000, suffix_type='-d')