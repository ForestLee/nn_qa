
from corpus.token_util import TokenUtil
import numpy as np
import random


class CorpusBase(object):
    def __init__(self, file_vocab, file_corpus, ner_vocab=False):
        self.file_corpus = file_corpus
        self.file_vocab = file_vocab
        self.tokenizer = TokenUtil(self.file_vocab, ner_vocab)
        #self.labels_dict, self.line_count, self.label_set = self._convert_label_index(file_corpus)
        print("corpus is {}".format(self.file_corpus))

    def get_counts(self):
        return self.counts

    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def read(self, file_corpus=None, maxlen=64):
        raise NotImplementedError

    def read_infer(self, file_corpus=None, maxlen=64):
        raise NotImplementedError

    def read_direct(self, file_corpus=None, maxlen=64):
        raise NotImplementedError


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
        if file_corpus == None:
            file_corpus = self.file_corpus
        sentences = []
        labels = []
        label_dim = set()
        with open(file_corpus, 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                d = line.split("\t")
                sentence = d[0]
                label = d[1].strip("\n")
                sentence = self.tokenizer.get_input_indexs(sentence, maxlen)
                sentences.append(sentence)
                labels.append(label)
                label_dim.add(label)

                line = f.readline()
        self.counts = len(labels)

        sentences = np.array(sentences)
        labels = np.array(labels)
        return sentences, labels, label_dim

    def shuffle(self, file):
        if file == None:
            file = self.file_corpus
        lines = []
        with open(file, 'r', encoding='utf-8') as infile:
            for line in infile:
                lines.append(line)

        random.shuffle(lines)
        shuffle_file = file+".shuffle"
        with open(shuffle_file, 'w', encoding='utf-8') as outfile:
            for line in lines:
                outfile.write(line)
        print("shuffle file done")
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



if __name__ == "__main__":
    def shuffle(file, file_out = None):
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
        print("shuffle file done, len={}".format(len1))
        return shuffle_file
    #shuffle("/home/forest/QA/src/git/sentence-encoding-qa/data/chx_qa_full_3_pair/qa_full_three_pair_2.txt")
    shuffle("/home/forest/QA/src/git/sentence-encoding-qa/data/qa_st_corpus.csv", "/home/forest/QA/src/git/sentence-encoding-qa/data/qa_st_corpus.csv")