

from fit_generator.classify_fit_generator import ClassFitGenerator
from fit_generator.pair_fit_generator import PairFitGenerator
from fit_generator.bert_fit_generator import BertFitGenerator
from fit_generator.label_util import load_label_dict_from_file

class FitGeneratorWrapper():

    def __init__(self, type = "class", file_vocab=None, file_corpus=None, batch_size=0, max_len=0, vector_dim=0, ner_vocab=False, label_dict_file=None):
        print("fit generator type: {}, corpus: {}".format(type, file_corpus))
        self.file_corpus = file_corpus
        self.batch_size = batch_size
        self.max_len = max_len
        self.vector_dim=vector_dim

        if type == "bert":
            label_dict = load_label_dict_from_file(label_dict_file)
            self.fit = BertFitGenerator(file_vocab, file_corpus, batch_size, max_len, vector_dim, ner_vocab=ner_vocab, label_dict=label_dict)
        elif type == "class":
            label_dict = load_label_dict_from_file(label_dict_file)
            self.fit = ClassFitGenerator(file_vocab, file_corpus, batch_size, max_len, vector_dim, ner_vocab=ner_vocab, label_dict=label_dict)
        elif type == "pair":
            label_dict = None
            self.fit = PairFitGenerator(file_vocab, file_corpus, batch_size, max_len, vector_dim, ner_vocab=ner_vocab, label_dict=label_dict)
        else:
            print("wrong type: "+type)

        self.labels_dict = self.fit.label_dict


    def get_vocab_size(self):
        return self.fit.get_vocab_size()


    def get_line_count(self):
        return self.fit.get_line_count()


    def get_label_count(self):
        return self.fit.get_label_count()


    def generate(self):
        return self.fit.generate_array_from_file()

    def read_corpus(self, file_corpus=None, maxlen=64):
        return self.fit.read_corpus(file_corpus=file_corpus, maxlen=maxlen)

    def read_raw_corpus(self, file_corpus=None, maxlen=64):
        return self.fit.read_raw_corpus(file_corpus=file_corpus, maxlen=maxlen)

    def read_line(self, line_num, file_corpus=None):
        return self.fit.read_line(line_num, file_corpus)