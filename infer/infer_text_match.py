import os
import sys
import numpy as np
from config import conf
from fit_generator.fit_generator_wrapper import FitGeneratorWrapper
from infer import InferTest
from model import ModelWrapper
from infer.best_threshold import evaluate_best_threshold_value

class InferTextMatchTest(InferTest):
    def __init__(self):
        super(InferTextMatchTest, self).__init__(backbone="TEXT_MATCH", type="pair", header="value")

    def _prepare_test_data(self, file_corpus=conf.PAIR_TEST_FILE):
        print('***************************prepare corpus***************************')
        self.fit_gen = FitGeneratorWrapper(type=self.conf.type, file_vocab=self.conf.VOCAB_FILE, file_corpus=self.conf.PAIR_TEST_FILE,
                                           batch_size=self.conf.batch_size, max_len=self.conf.maxlen, vector_dim=self.conf.vector_dim,
                                           ner_vocab=self.conf.pretrain_vocab, label_dict_file=self.conf.LABEL_DICT_FILE)
        self.vocab_size = self.fit_gen.get_vocab_size()
        self.labels_num = self.fit_gen.get_label_count()
        sentences1, sentences2, labels = self.fit_gen.read_raw_corpus(file_corpus=file_corpus)
        return [sentences1, sentences2], labels

    def file_infer_test(self, model_file, values_file=None):
        sentences, labels = self._prepare_test_data()

        print('***************************build model***************************')
        model = ModelWrapper.model(self.conf, train=False, vocab_size=self.vocab_size, labels_num = 1)
        model.summary()
        self._load_model(model_file, model)

        print('***************************infer test***************************')
        # if os.path.exists(values_file):
        #     print("load cache file " + values_file)
        #     values = np.load(values_file)
        # else:
        values = self.do_predict(model, sentences, vector_dim=self.conf.vector_dim, header=self.conf.predict_header, type=self.conf.type)
        print("save cache file "+values_file)
        np.save(values_file, values)

        tpr, fpr, accuracy, best_thresholds = evaluate_best_threshold_value(values, labels, nrof_folds=10)
        tpr = np.mean(tpr)
        fpr = np.mean(fpr)
        accuracy = np.mean(accuracy)
        best_thresholds = np.mean(best_thresholds)
        print(
            "cosine: (正样本的召回率tp/(tp+fn))tpr={} (负样本的错误率fp/(fp+tn))fpr={} acc={} threshold={}".format(tpr, fpr, accuracy,
                                                                                                     best_thresholds))
        return best_thresholds


if __name__ == '__main__':
    conf.backbone = "TEXT_MATCH"

    if len(sys.argv) > 1:
        model_file=sys.argv[1]
    else:
        if conf.attention_enable == False:
            model_file = "{}{}_".format(conf.SAVE_DIR, conf.backbone)
        else:
            model_file = "{}{}_{}_".format(conf.SAVE_DIR, conf.backbone, "ATTENTION")
        model_file = model_file+"2020-09-21-13-28-10"+".h5"

    values_file = os.getcwd() + "/../save/values_npy.npy"

    infer_test = InferTextMatchTest()
    infer_test.file_infer_test(model_file=model_file, values_file=values_file)