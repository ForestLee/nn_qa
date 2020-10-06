
def convert(corpus_path, save_path, label_dict_path):
    LABEL_START = "#	{\"label\": \""
    labels = set()
    with open(corpus_path, encoding='utf-8') as f:
        with open(save_path, "w", encoding='utf-8') as fw:
            startLen = len(LABEL_START)
            sentence = ""
            label = ""
            line = f.readline()
            state = "not_start"
            while line:
                if line.startswith(LABEL_START) and state == "not_start":
                    line = line.replace('\n', '')
                    line1 = line[startLen:]
                    ds = line1.split('"')
                    label = ds[0]
                    labels.add(label)
                    sentence = ""
                    state = "start"
                elif line == "\n":
                    if sentence != "":
                        str1 = "{}\t{}\n".format(sentence, label)
                        print(str1)
                        fw.write(str1)
                        sentence = ""
                        label = ""
                    state = "not_start"
                elif state == "start":
                    [char, _] = line.rstrip().split('\t')
                    sentence += char
                line = f.readline()

    label_dict = generate_label_index(labels, label_dict_path)
    return labels, label_dict

def generate_label_index(labels, label_dict_path):
    label_dict = dict()
    i = 0
    with open(label_dict_path, "w", encoding='utf-8') as fw:
        for label in labels:
            label_dict[label] = i
            fw.write(label+","+str(i)+"\n")
            i += 1
    return label_dict

def convert_label_index(corpus_path, corpus_index_path, label_dict):
    with open(corpus_path, "r", encoding='utf-8') as f:
        with open(corpus_index_path, "w", encoding='utf-8') as fw:
            line = f.readline()
            while line:
                line = line.replace('\n', '')
                ds = line.split('\t')
                sentence = ds[0]
                label = ds[1]
                i = label_dict[label]
                fw.write(sentence+"\t"+str(i)+"\n")
                line = f.readline()


#SAVE_PATH="/home/forest/QA/src/git/sentence-encoding-qa/data/small_ner/"
#CORPUS_FILE="train.conllx"

SAVE_PATH="/home/forest/QA/src/git/sentence-encoding-qa/data/ner/"
CORPUS_FILE="new_tagged_cls.conllx"

CORPUS_CONV_FILE="ner_qa.csv"
CORPUS_INDEX_FILE="ner_index_qa.csv"
CORPUS_LABEL_DICT="ner_label_dict.csv"

labels, label_dict = convert(SAVE_PATH+CORPUS_FILE, SAVE_PATH+CORPUS_CONV_FILE, SAVE_PATH+CORPUS_LABEL_DICT)
convert_label_index(SAVE_PATH+CORPUS_CONV_FILE, SAVE_PATH+CORPUS_INDEX_FILE, label_dict)