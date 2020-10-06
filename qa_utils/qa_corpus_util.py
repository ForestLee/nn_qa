import pandas as pd
import numpy as np
import os
import csv
from sklearn.model_selection import train_test_split
import random

'''
input_file:
label   sentence
output_file:
sentence    label
'''
def label_sentence_switch(input_file, output_file):
    with open(input_file, encoding='utf-8') as f:
        with open(output_file, "w", encoding='utf-8') as fw:
            line = f.readline()
            while line:
                if len(line) > 2:
                    try:
                        line = line.replace('\n', '')
                        ds = line.split('  ')
                        if len(ds) < 2:
                            ds = line.split('\t')
                        label = ds[0]
                        sentence = ds[1]
                        fw.write("{}\t{}\n".format(sentence, label))
                    except:
                        print("error")
                line = f.readline()


# 从导入的csv文件中获取处理后的语料，去重，转换为语料-子功能的csv格式
def clean_corpus(input_path, size, output_path):
    data = pd.read_csv(input_path, header=None, iterator=True, sep='\t')
    data = data.get_chunk(size)
    print(int(data.size/2))

    data = np.array(data)
    print('数据总量为：', int(data.size/2))

    data_dict = {}
    for i in data:
        data_dict[i[1]] = i[0]
    print(data_dict.__len__())

    if not os.path.exists(output_path):
        f = open(output_path, 'w')
        f.close()
    else:
        os.remove(output_path)
        f = open(output_path, 'w')
        f.close()
    i = 0
    with open(output_path, 'r+', encoding='utf-8') as f:
        for k, v in data_dict.items():
            if isinstance(k, str) and isinstance(v, str):
                tmp_k = str(k).split()
                tmp_v = str(v).split()
                strip_k = ''.join(tmp_k)
                strip_v = ''.join(tmp_v)
                line = strip_k + '\t' + strip_v + '\n'
                f.write(line)
                i += 1
            else:
                print('k:', k)
                print('v:', v)
                print('语料中存在为空的情况')
    print('语料处理完成,总条数为%d条************************************************************************************************' % i)

# split train and test corpus, according percent
def train_test_split_corpus(origin_path, test_path, train_path, threshold_num):
    with open(origin_path, 'r+', encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = [line[0] for line in reader]
        print('原语料长度为：', lines.__len__())

    templines = set(lines)
    corpus_dict = {}
    for line in templines:
        lens = line.find('\t')
        if lens > 0:
            content = line[0:lens]
            head = line[lens + 1:]
            if corpus_dict.get(head):
                corpus_dict.get(head).append(line)
            else:
                corpus_dict[head] = [line]

    test_lines = []
    train_lines = []
    for k, v in corpus_dict.items():
        arr = np.array(v)
        if threshold_num and arr.size >= threshold_num:
            random.shuffle(arr)
            arr = arr[:threshold_num]
        if arr.size > 10:
            random.shuffle(arr)
            train_set, test_set = train_test_split(arr, test_size=0.1)
            train_lines.extend(train_set)
            test_lines.extend(test_set)
        elif arr.size >= 5:
            random.shuffle(arr)
            train_set, test_set = train_test_split(arr, test_size=0.2)
            train_lines.extend(train_set)
            test_lines.extend(test_set)
        else:
            print('len:', arr.size)
            print('数量较少的子分类语义为：', arr)

    test_corpus_num = 0
    train_corpus_num = 0

    if not os.path.exists(test_path):
        f = open(test_path, 'w')
        f.close()
    else:
        os.remove(test_path)
        f = open(test_path, 'w')
        f.close()

    if not os.path.exists(train_path):
        f = open(train_path, 'w')
        f.close()
    else:
        os.remove(train_path)
        f = open(train_path, 'w')
        f.close()

    with open(test_path, 'r+', encoding='utf-8') as f:
        for line in test_lines:
            f.write(line + '\n')
            test_corpus_num += 1

    with open(train_path, 'r+', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line + '\n')
            train_corpus_num += 1
    print('语料处理完成，总数量为：%d条,其中训练集为：%d条，测试集为：%d******************************************************'
          '**************' % (test_corpus_num+train_corpus_num, train_corpus_num, test_corpus_num))

# split train and test corpus, according num
def train_test_split_corpus_by_num(origin_path, test_path, train_path, threshold_num, extract_num):
    with open(origin_path, 'r+', encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = [line[0] for line in reader]
        print('原语料长度为：', lines.__len__())

    templines = set(lines)
    corpus_dict = {}
    for line in templines:
        lens = line.find('\t')
        if lens > 0:
            content = line[0:lens]
            head = line[lens + 1:]
            tmp_content = str(content).split()
            tmp_head = str(head).split()
            strip_content = ''.join(tmp_content)
            strip_head = ''.join(tmp_head)
            strip_line = strip_content + '\t' + strip_head

            if corpus_dict.get(strip_head):
                corpus_dict.get(strip_head).append(strip_line)
            else:
                corpus_dict[strip_head] = [strip_line]

    test_lines = []
    train_lines = []
    for k, v in corpus_dict.items():
        arr = np.array(v)
        if threshold_num and arr.size >= threshold_num:
            random.shuffle(arr)
            arr = arr[:threshold_num]
        if arr.size >= 5 and extract_num < arr.size:
            random.shuffle(arr)
            train_set, test_set = arr[extract_num:], arr[:extract_num]
            train_lines.extend(train_set)
            test_lines.extend(test_set)
        else:
            print('len:', arr.size)
            print('数量较少的子分类语义为：', arr)

    test_corpus_num = 0
    train_corpus_num = 0

    if not os.path.exists(test_path):
        f = open(test_path, 'w')
        f.close()
    else:
        os.remove(test_path)
        f = open(test_path, 'w')
        f.close()

    if not os.path.exists(train_path):
        f = open(train_path, 'w')
        f.close()
    else:
        os.remove(train_path)
        f = open(train_path, 'w')
        f.close()

    with open(test_path, 'r+', encoding='utf-8') as f:
        for line in test_lines:
            f.write(line + '\n')
            test_corpus_num += 1

    with open(train_path, 'r+', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line + '\n')
            train_corpus_num += 1
    print('语料处理完成，总数量为：%d条,其中训练集为：%d条，测试集为：%d******************************************************'
          '**************' % (test_corpus_num+train_corpus_num, train_corpus_num, test_corpus_num))

# split train and test corpus, according num
# origin_path  origin corpus path
# extract_num  the quantity of every class extract corpus
def train_test_split_corpus_by_label_num(origin_path, extract_num):
    train_test_split_corpus_by_num(origin_path, './data/small_2_test.csv', './data/small_2_train.csv', 500, extract_num)


def fetch_num_per_label(file_path, extract_num):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = [line[0] for line in reader]
        print('原语料长度为：', lines.__len__())

    templines = set(lines)
    corpus_dict = {}
    for line in templines:
        lens = line.find('\t')
        if lens > 0:
            content = line[0:lens]
            head = line[lens + 1:]
            tmp_content = str(content).split()
            tmp_head = str(head).split()
            strip_content = ''.join(tmp_content)
            strip_head = ''.join(tmp_head)
            strip_line = strip_content + '\t' + strip_head

            if corpus_dict.get(strip_head):
                corpus_dict.get(strip_head).append(strip_line)
            else:
                corpus_dict[strip_head] = [strip_line]

    train_lines = []
    test_lines = []
    err_count = 0
    for k, v in corpus_dict.items():
        arr = np.array(v)
        random.shuffle(arr)

        if extract_num < arr.size:
            train_set = arr[extract_num:]
            test_set = arr[:extract_num]
            train_lines.extend(train_set)
            test_lines.extend(test_set)
        else:
            test_lines.extend(arr)
            print('len:', arr.size)
            print('数量较少的子分类语义为：', arr[0])
            err_count += 1
    print("totally {} labels, {} labels less than 2 samples".format(len(corpus_dict), err_count))

    with open(file_path+".train", 'w', encoding='utf-8') as f:
        for line in train_lines:
            f.write(line + '\n')
    with open(file_path+".test", 'w', encoding='utf-8') as f:
        for line in test_lines:
            f.write(line + '\n')


def get_en_corpus(file_path, save_en_path, not_en_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        lines = [line[0] for line in reader]
        print('原语料长度为：', lines.__len__())

    if not os.path.exists(save_en_path):
        f = open(save_en_path, 'w')
        f.close()
    else:
        os.remove(save_en_path)
        f = open(save_en_path, 'w')
        f.close()

    if not os.path.exists(not_en_path):
        f = open(not_en_path, 'w')
        f.close()
    else:
        os.remove(not_en_path)
        f = open(not_en_path, 'w')
        f.close()

    en_line = []
    not_en_line = []
    for line in lines:
        import re
        contain_en = bool(re.search('[a-zA-Z]', line))
        if contain_en:
            en_line.append(line)
        else:
            not_en_line.append(line)

    with open(save_en_path, 'r+', encoding='utf-8') as f:
        for line in en_line:
            f.write(line + '\n')
    with open(not_en_path, 'r+', encoding='utf-8') as f:
        for line in not_en_line:
            f.write(line + '\n')

def get_all_train_except_test(origin_path, test_path, all_train_path):
    with open(origin_path, 'r', encoding='utf-8') as f:
        all_corpus = []
        reader = csv.reader(f)
        for s in reader:
            line = s[0]
            lens = line.find('\t')
            if lens > 0:
                content = line[0:lens]
                head = line[lens + 1:]
                tmp_content = str(content).split()
                tmp_head = str(head).split()
                strip_content = ''.join(tmp_content)
                strip_head = ''.join(tmp_head)
                strip_line = strip_content + '\t' + strip_head
                all_corpus.append(strip_line)
        print('原语料长度为：', all_corpus.__len__())

    with open(test_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        test_corpus = [line[0] for line in reader]
        print('测试集语料长度为：', test_corpus.__len__())

    # train_corpus = list(set(all_corpus).difference(set(test_corpus)))
    # train_corpus = [line for line in all_corpus if line not in test_corpus]
    train_corpus = []
    for line in all_corpus:
        if line not in test_corpus:
            train_corpus.append(line)


    print('训练集语料长度为：', train_corpus.__len__())

    if not os.path.exists(all_train_path):
        f = open(all_train_path, 'w')
        f.close()
    else:
        os.remove(all_train_path)
        f = open(all_train_path, 'w')
        f.close()

    with open(all_train_path, 'r+', encoding='utf-8') as f:
        for line in train_corpus:
            f.write(line + '\n')
    print('语料处理完毕！*******************************************************************************************')






if __name__ == '__main__':
    # clean_corpus('./data/20200829_core_corpus.csv', 180000, './data/qa_core_st_corpus.csv')
    # train_test_split_corpus('./data/qa_st_corpus.csv', './data/test.csv', './data/train.csv', None)
    # train_test_split_corpus('./data/qa_st_corpus.csv', './data/small_test.csv', './data/small_train.csv', 100)
    # train_test_split_corpus_by_num('./data/qa_core_st_corpus.csv', './data/core_2_test.csv', './data/core_2_train.csv', 100, 1)
    # train_test_split_corpus_by_label_num('./data/qa_st_corpus.csv', 2)
    # get_en_corpus('./data/qa_st_corpus.csv', './data/en_corpus.csv', './data/not_en_corpus.csv')
    # get_all_train_except_test('./data/qa_st_corpus.csv', './data/small_500_2_test.csv', './data/small_500_2_other_all_train.csv')


    #clean_corpus('./data/20200829_core_corpus.csv', 180000, './data/qa_core_st_corpus.csv')
    #train_test_split_corpus_by_num('./data/qa_core_st_corpus.csv', './data/core_2_test.csv', './data/core_2_train.csv', 100, 1)

    fetch_num_per_label(os.getcwd()+"/../data/new_all.csv", 1)

    #label_sentence_switch("/home/forest/QA/textcnndata/newdomain_0921.txt", "/home/forest/QA/textcnndata/newdomain_0924.txt")
    #fetch_num_per_label("/home/forest/QA/textcnndata/newdomain_0924.txt", 1)



















