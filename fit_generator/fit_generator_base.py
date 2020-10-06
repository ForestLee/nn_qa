
from corpus.token_util import TokenUtil
import numpy as np
import random

class FitGeneratorBase():
    def __init__(self, file_vocab=None, file_corpus=None, batch_size=0, max_len=0, vector_dim=0, ner_vocab=False, label_dict = None):
        self.file_vocab = file_vocab
        self.file_corpus = file_corpus
        self.batch_size = batch_size
        self.max_len = max_len
        self.vector_dim=vector_dim
        self._init_token(ner_vocab)
        self.line_count = self._get_line_count(file_corpus=file_corpus)
        self.label_dict = label_dict


    def _get_label_index(self, label):
        return self.label_dict[label]

    def _init_token(self, ner_vocab = False):
        self.tokenizer = TokenUtil(self.file_vocab, ner_vocab)

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def _get_line_count(self, file_corpus):
        with open(file_corpus, "r", encoding='utf-8') as f:
            line = f.readline()
            size = 0
            while line:
                line = line.replace('\n', '')
                if line != "":
                    size += 1
                line = f.readline()
            return size

    def get_line_count(self):
        return self.line_count

    def get_label_count(self):
        if self.label_dict is None:
            return 0
        else:
            return len(self.label_dict)

    def generate_array_from_file(self):
        raise NotImplementedError

    def _convert_label_index(self, file_corpus):
        labels_set = set()
        with open(file_corpus, "r", encoding='utf-8') as f:
            line = f.readline()
            size = 0
            while line:
                line = line.replace('\n', '')
                ds = line.split('\t')
                anchor_label = ds[1]
                positive_label = ds[3]
                negative_label = ds[5]
                labels_set.add(anchor_label)
                labels_set.add(positive_label)
                labels_set.add(negative_label)
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

    def read_corpus(self, file_corpus=None, maxlen=64):
        """
        for validation
        @param file_corpus:
        @param maxlen:
        @return:
        """
        sentences, sentence_labels = self._read_common(file_corpus=file_corpus, maxlen=maxlen)
        label_idx_bins = []
        size = len(self.label_dict)
        for label in sentence_labels:
            label_idx_bin = np.zeros((size))
            label_idx_bin[self.label_dict[label]] = 1
            label_idx_bins.append(label_idx_bin)

        return  sentences, np.array(label_idx_bins)

    def read_raw_corpus(self, file_corpus=None, maxlen=64):
        return self._read_common(file_corpus=file_corpus, maxlen=maxlen)

    def _read_common(self, file_corpus=None, maxlen=64):
        """
            corpus file format like below:
            打电话给秀芝。,16
            嗯去深圳松岗。,4
            拨打俏梅电话。,16
        @param file_corpus:
        @param maxlen:
        @return:
        """
        sentences = []
        labels = []
        with open(file_corpus, 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                line = line.strip("\n")
                if line != "":
                    d = line.split("\t")
                    len_d = len(d)
                    sentence = d[0]
                    sentence = self.tokenizer.get_input_indexs(sentence, maxlen)
                    sentences.append(sentence)

                    label = d[len_d - 1]
                    labels.append(label)

                line = f.readline()
        self.counts = len(labels)

        sentences = np.array(sentences)
        labels = np.array(labels)
        return sentences, labels

    def _shuffle(self, file, file_out = None):
        lines = []
        with open(file, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip("\n")
                lines.append(line)

        random.shuffle(lines)
        len1 = len(lines)
        if file_out == None:
            shuffle_file = file + ".shuffle.txt"
        else:
            shuffle_file = file_out
        with open(shuffle_file, 'w', encoding='utf-8') as outfile:
            for line in lines:
                outfile.write(line+"\n")
        print("\tshuffle file done, len={}".format(len1))
        return shuffle_file

    def read_line(self, line_num, file_corpus=None):
        if file_corpus == None:
            file_corpus = self.file_corpus
        with open(file_corpus, 'r', encoding='utf-8') as infile:
            i = 0
            for line in infile:
                if i == line_num:
                    return line
                i += 1

        return ""
