

from corpus.classify_corpus import ClassifyCorpus


class CorpusWrapper():
    def __init__(self, type, file_vocab, file_corpus, ner_vocab=False):
        print("corpus file: "+ file_corpus)
        if type == "classify":
            self.corpus = ClassifyCorpus(file_vocab, file_corpus, ner_vocab)
        else:
            print("wrong type: "+str(type))

    def get_counts(self):
        return self.corpus.get_counts()

    def get_vocab_size(self):
        return self.corpus.get_vocab_size()

    def read(self, file_corpus=None, maxlen=64):
        return self.corpus.read(file_corpus, maxlen)

    def read_infer(self, file_corpus=None, maxlen=64):
        return self.corpus.read_infer(file_corpus, maxlen)

    def read_direct(self, file_corpus=None, maxlen=64):
        return self.corpus.read_direct(file_corpus, maxlen)

    def read_line(self, line_num, file_corpus=None):
        return self.corpus.read_line(line_num, file_corpus)