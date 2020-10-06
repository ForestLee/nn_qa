import numpy as np
from fit_generator.fit_generator_base import FitGeneratorBase
from fit_generator.label_util import binary_transform

class ClassFitGenerator(FitGeneratorBase):
    def generate_array_from_file(self):
        """
        倒车灯怎么工作的	倒车灯定义
        给我打开遮阳窗帘	遮阳窗帘开启
        倒车灯是做什么用	倒车灯定义
        """
        file_corpus = self.file_corpus
        file_corpus_shuffle = self.file_corpus+".shuffle"
        batch_size = self.batch_size
        max_len = self.max_len
        while 1:
            self._shuffle(file_corpus, file_corpus_shuffle)
            f = open(file_corpus_shuffle, encoding='utf-8')
            cnt = 0
            idxs = []
            labels = []
            for line in f:
                line = line.replace("\n", "")
                if line == "":
                    continue
                sentence, sentence_label = line.split("\t")
                idx = self.tokenizer.get_input_indexs(sentence, max_len)
                idxs.append(idx)
                labels.append(self.label_dict[sentence_label])
                cnt += 1
                if cnt == batch_size:
                    cnt = 0
                    idxs=np.array(idxs)
                    labels=np.array(labels)
                    targets = binary_transform(self.label_dict, labels)
                    for k in range(len(labels)):
                        targets[k][labels[k]] = 1
                    yield ([idxs, targets], targets)
                    idxs = []
                    labels = []

            f.close()


    def _convert_label_index(slef, file_corpus, labels_set_input=None):
        labels_set = set()
        with open(file_corpus, "r", encoding='utf-8') as f:
            line = f.readline()
            size = 0
            while line:
                try:
                    line = line.replace('\n', '')
                    _, label = line.split('\t')
                except:
                    print("error")
                labels_set.add(label)
                line = f.readline()
                size += 1

        corpus_index_path = file_corpus + ".label_idx.txt"
        with open(corpus_index_path, "w", encoding='utf-8') as fw:
            i = 0
            labels_dict = {}
            for label in labels_set:
                fw.write(label + "\t" + str(i) + "\n")
                labels_dict[label] = i
                i += 1
        return labels_dict, size, labels_set
