import numpy as np
from fit_generator.fit_generator_base import FitGeneratorBase
from fit_generator.label_util import binary_transform

class PairFitGenerator(FitGeneratorBase):
    def generate_array_from_file(self):
        """
        倒车灯怎么工作的	给我打开遮阳窗帘    0
        倒车灯是做什么用	倒车灯怎么工作的    1
        """
        file_corpus = self.file_corpus
        file_corpus_shuffle = self.file_corpus+".shuffle"
        batch_size = self.batch_size
        max_len = self.max_len
        while 1:
            self._shuffle(file_corpus, file_corpus_shuffle)
            f = open(file_corpus_shuffle, encoding='utf-8')
            cnt = 0
            idxs1 = []
            idxs2 = []
            labels = []
            for line in f:
                line = line.replace("\n", "")
                if line == "":
                    continue
                sentence1, sentence2, sentence_label = line.split("\t")
                idx1 = self.tokenizer.get_input_indexs(sentence1, max_len)
                idx2 = self.tokenizer.get_input_indexs(sentence2, max_len)
                idxs1.append(idx1)
                idxs2.append(idx2)
                labels.append(int(sentence_label))
                cnt += 1
                if cnt == batch_size:
                    cnt = 0
                    idxs1 = np.array(idxs1)
                    idxs2 = np.array(idxs2)
                    targets = np.array(labels)
                    yield ([idxs1, idxs2, targets], targets)
                    idxs1 = []
                    idxs2 = []
                    labels = []

            f.close()

    def read_corpus(self, file_corpus=None, maxlen=64):
        return self._read_common(file_corpus=file_corpus, maxlen=maxlen)


    def _read_common(self, file_corpus=None, maxlen=64):
        """
        倒车灯怎么工作的	给我打开遮阳窗帘    0
        倒车灯是做什么用	倒车灯怎么工作的    1
        """
        sentences1 = []
        sentences2 = []
        labels = []
        with open(file_corpus, 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                line = line.strip("\n")
                if line != "" and len(line) > 2:
                    d = line.split("\t")
                    sentence1 = d[0]
                    sentence2 = d[1]
                    label = int(d[2])
                    sentence1 = self.tokenizer.get_input_indexs(sentence1, maxlen)
                    sentences1.append(sentence1)
                    sentence2 = self.tokenizer.get_input_indexs(sentence2, maxlen)
                    sentences2.append(sentence2)
                    labels.append(label)

                line = f.readline()
        self.counts = len(labels)

        sentences1 = np.array(sentences1)
        sentences2 = np.array(sentences2)
        labels = np.array(labels)
        return sentences1, sentences2, labels