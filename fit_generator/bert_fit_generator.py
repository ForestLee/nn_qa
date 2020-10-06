import numpy as np
from fit_generator.classify_fit_generator import ClassFitGenerator
from model.bert4keras.tokenizers import Tokenizer
from model.bert4keras.snippets import sequence_padding
from fit_generator.label_util import binary_transform

dict_path = '/home/forest/BERT/models/chinese_L-12_H-768_A-12/vocab.txt'

class BertFitGenerator(ClassFitGenerator):
    def __init__(self, file_vocab=None, file_corpus=None, batch_size=0, max_len=0, vector_dim=0, ner_vocab=False, label_dict = None):
        super(BertFitGenerator, self).__init__(file_vocab=file_vocab, file_corpus=file_corpus, batch_size=batch_size, max_len=max_len, vector_dim=vector_dim, ner_vocab=ner_vocab, label_dict = label_dict)

    def _init_token(self, ner_vocab = False):
        self.tokenizer = Tokenizer(dict_path, do_lower_case=True)

    def generate_array_from_file(self):
        """
        倒车灯怎么工作的	倒车灯定义
        给我打开遮阳窗帘	遮阳窗帘开启
        倒车灯是做什么用	倒车灯定义
        """
        file_corpus = self.file_corpus
        batch_size = self.batch_size
        max_len = self.max_len
        while 1:
            f = open(file_corpus)
            cnt = 0

            token_ids = []
            segment_ids = []
            labels = []
            for line in f:
                line = line.replace("\n", "")
                sentence, sentence_label, = line.split("\t")
                sentence_token_ids, sentence_segment_ids = self.tokenizer.encode(sentence, maxlen=max_len)
                token_ids.append(sentence_token_ids)
                segment_ids.append(sentence_segment_ids)
                labels.append(self.label_dict[sentence_label])
                cnt += 1
                if cnt == batch_size:
                    cnt = 0
                    token_ids = sequence_padding(token_ids, self.max_len)
                    segment_ids = sequence_padding(segment_ids, self.max_len)

                    labels = np.array(labels)
                    targets = binary_transform(self.label_dict, labels)
                    for k in range(len(labels)):
                        targets[k][labels[k]] = 1
                    #yield ([[token_ids, segment_ids], targets], targets)  #tensorflow 2.0
                    yield ([token_ids, segment_ids, targets], targets)   #tensorflow 1.15
                    token_ids = []
                    segment_ids = []
                    labels = []

            f.close()

    def read_corpus(self, file_corpus=None, maxlen=64):
        """
        for validation
        @param file_corpus:
        @param maxlen:
        @return:
        """
        token_ids, segment_ids, labels = self._read_common(file_corpus=file_corpus, maxlen=maxlen)
        label_idx_bins = []
        size = len(self.label_dict)
        for label in labels:
            label_idx_bin = np.zeros((size))
            label_idx_bin[self.label_dict[label]] = 1
            label_idx_bins.append(label_idx_bin)

        return  [token_ids, segment_ids], np.array(label_idx_bins)

    def read_raw_corpus(self, file_corpus=None, maxlen=64):
        token_ids, segment_ids, labels = self._read_common(file_corpus=file_corpus, maxlen=maxlen)
        return  [token_ids, segment_ids], labels

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
        token_ids = []
        segment_ids = []
        labels = []
        with open(file_corpus, 'r', encoding='utf-8') as f:
            line = f.readline()
            while line:
                line = line.strip("\n")
                if line != "":
                    d = line.split("\t")
                    sentence = d[0]
                    label = d[1]
                    sentence_token_ids, sentence_segment_ids = self.tokenizer.encode(sentence, maxlen=maxlen)
                    token_ids.append(sentence_token_ids)
                    segment_ids.append(sentence_segment_ids)
                    labels.append(label)

                line = f.readline()
        self.counts = len(labels)

        token_ids = sequence_padding(token_ids, maxlen)
        segment_ids = sequence_padding(segment_ids, maxlen)

        #token_ids = np.array(token_ids)
        #segment_ids = np.array(segment_ids)
        labels = np.array(labels)
        return token_ids, segment_ids, labels