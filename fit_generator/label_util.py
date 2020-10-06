
import numpy as np

def load_label_dict_from_file(file_dict):
    label_dict = {}
    with open(file_dict, "r", encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.replace('\n', '')
            if line != "":
                ds = line.split('\t')
                label_dict[ds[0]] = int(ds[1])
            line = f.readline()

    return label_dict


def generate_label_dict_from_file(file_corpus, file_corpus2=None, file_label_dict=None):
    label_set = set()
    with open(file_corpus, "r", encoding='utf-8') as f:
        line = f.readline()
        while line:
            line = line.replace('\n', '')
            if line != "" and len(line) > 2:
                ds = line.split('\t')
                label = ds[1]
                label_set.add(label)
            line = f.readline()
    if not file_corpus2 is None:
        with open(file_corpus2, "r", encoding='utf-8') as f:
            line = f.readline()
            while line:
                line = line.replace('\n', '')
                if line != "" and len(line) > 2:
                    ds = line.split('\t')
                    label = ds[1]
                    label_set.add(label)
                line = f.readline()

    if file_label_dict is None:
        file_label_dict_used = file_corpus + ".label_dict.txt"
    else:
        file_label_dict_used = file_label_dict
    with open(file_label_dict_used, "w", encoding='utf-8') as fw:
        i = 0
        labels_dict = {}
        for label in label_set:
            fw.write(label + "\t" + str(i) + "\n")
            labels_dict[label] = i
            i += 1
    return labels_dict


def __binary_transform(label_dict, label):
    size = len(label_dict)
    label_array = np.zeros((size))
    label_array[label] = 1
    return label_array



def binary_transform(label_dict, labels):
    bin_array = []
    for label in labels:
        label_array = __binary_transform(label_dict, label)
        bin_array.append(label_array)
    return np.array(bin_array)

if __name__ == "__main__":
    from config import conf
    generate_label_dict_from_file(conf.TRAIN_FILE, conf.TEST_FILE, conf.LABEL_DICT_FILE)
    label_dict = load_label_dict_from_file(conf.LABEL_DICT_FILE)
    # generate_label_dict_from_file("/home/forest/QA/textcnndata/newdomain_0924.txt.train", "/home/forest/QA/textcnndata/newdomain_0924.txt.test", "/home/forest/QA/textcnndata/label_dict.txt")
    # label_dict = load_label_dict_from_file("/home/forest/QA/textcnndata/label_dict.txt")
    for key in label_dict:
        print("{}\t{}".format(key, label_dict[key]))