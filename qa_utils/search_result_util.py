import os
import requests


def dic_label_get(input_path):
    with open(input_path, 'r') as f:
        list_for_dic = f.read().splitlines()
        dic_label = {}
        for i, j in enumerate(list_for_dic):
            dic_label[str(i)] = j
    return dic_label


def list_data_get(input_path):
    with open(input_path, 'r') as f:
        list_data = []
        for i in f.read().splitlines():
            list_data.append(i.split('\t'))
    return list_data


def get_result(list_data, dic_label):
    num_error = 0
    list_error = []
    for i in list_data:
        input_str = i[0]
        textmod = {'text': input_str, 'nbest': '3'}
        r = requests.post('http://10.43.10.24:9000/webnlp/usermanual', json=textmod, headers={'Connection': 'close'})
        a = r.json()['data'][0]['index']
        if dic_label[str(a)] != i[1]:
            num_error += 1
            list_error.append([i, dic_label[str(a)]])
    final_error_rate = 100 * (1 - num_error / len(list_data))
    return num_error, final_error_rate, list_error


def write_to_file(num_error, final_error_rate, list_error, out_path):
    with open(out_path, 'w') as ff:
        ff.write(str(num_error) + '\t' + str(final_error_rate) + '%')
        ff.write('\n')
        for i in list_error:
            ff.write(i[0][0] + ',' + i[0][1] + '\t' + i[1])
            ff.write('\n')


def run(path_dic, path_data, out_path):
    dic_label = dic_label_get(path_dic)
    list_data = list_data_get(path_data)
    num_error, final_error_rate, list_error=get_result(list_data, dic_label)
    write_to_file(num_error, final_error_rate, list_error, out_path)
if __name__ == '__main__':
    path_dic=os.getcwd() + "/../save/label_list.txt"
    path_data=os.getcwd() + "/../data/new_all.csv"
    out_path=os.getcwd() + "/../data/output_data_error"
    run(path_dic, path_data, out_path)
