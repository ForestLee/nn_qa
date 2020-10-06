
from sklearn.preprocessing import LabelBinarizer
from corpus.corpus_base import CorpusBase

class ClassifyCorpus(CorpusBase):
    def read(self, file_corpus=None, maxlen=64):
        shuffle_file = self.shuffle(file_corpus)
        return self._read_classify(shuffle_file, maxlen)

    def _read_classify(self, file_corpus=None, maxlen=64):
        """
        for classify training
        @param file_corpus:
        @param maxlen:
        @return:
        """
        sentences, sentence_labels, label_dim = self._read_common(file_corpus=file_corpus, maxlen=maxlen)

        self.le = LabelBinarizer().fit(sentence_labels.reshape(-1, 1))
        labels = self.le.transform(sentence_labels)

        return  sentences, labels, len(label_dim)
