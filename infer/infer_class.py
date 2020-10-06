import os
import sys
import datetime
import numpy as np
from config import conf

from infer import InferTest
from model import ModelWrapper
from infer.best_threshold import evaluate_best_threshold_value, evaluate_best_threshold
from fit_generator.fit_generator_wrapper import FitGeneratorWrapper
from distance import calc_two_embeddings_distance

class InferClassTest(InferTest):
    def __init__(self):
        super(InferClassTest, self).__init__(backbone="CNN_CLASS", type="class", header="index")

    def _prepare_test_data(self, file_corpus=conf.TEST_FILE):
        print('***************************prepare corpus***************************')
        self.fit_gen = FitGeneratorWrapper(type=self.conf.type, file_vocab=self.conf.VOCAB_FILE, file_corpus=self.conf.TRAIN_FILE,
                                           batch_size=self.conf.batch_size, max_len=self.conf.maxlen, vector_dim=self.conf.vector_dim,
                                           ner_vocab=self.conf.pretrain_vocab, label_dict_file=self.conf.LABEL_DICT_FILE)
        self.vocab_size = self.fit_gen.get_vocab_size()
        self.labels_num = self.fit_gen.get_label_count()
        self.labels_num = self.fit_gen.get_label_count()

        self.x_test, self.y_test = self.fit_gen.read_corpus(file_corpus=file_corpus)

        labels = []
        for label in self.y_test:
            index = np.argmax(label)
            labels.append(index)

        return self.x_test, np.array(labels)

    def _get_num_2(self, values, first):
        reset_values = []
        for i in range(len(values)):
            if i != first:
                reset_values.append(values[i])
            else:
                reset_values.append(0)
        reset_values = np.array(reset_values)
        value2 = np.max(reset_values)
        index2 = np.argmax(reset_values)
        return value2, index2

    def _class_predict_index(self, model, sentences):
        indexs = np.zeros((sentences.shape[0]), dtype=np.int)
        values = np.zeros((sentences.shape[0]), dtype=np.float)
        indexs2 = np.zeros((sentences.shape[0]), dtype=np.int)
        values2 = np.zeros((sentences.shape[0]), dtype=np.float)
        all_values = []
        max_len = sentences.shape[1]
        i = 0
        start_t = datetime.datetime.now()
        for sentence in sentences:
            sentence = sentence.reshape(1, max_len)
            logits_output = model.predict(sentence)
            squeeze_array = np.squeeze(logits_output)
            all_values.append(squeeze_array)
            values[i] = np.max(squeeze_array)
            indexs[i] = np.argmax(squeeze_array)
            value2, index2 = self._get_num_2(squeeze_array, indexs[i])
            values2[i] = value2
            indexs2[i] = index2
            i += 1
        end_t = datetime.datetime.now()
        print("{} sentences infer time {} seconds".format(sentences.shape[0], (end_t - start_t).seconds))
        return indexs, values, all_values, indexs2, values2


    def file_infer_test(self, model_file, values_file=None):
        sentences, labels = self._prepare_test_data()

        print('***************************build model***************************')
        model = ModelWrapper.model(self.conf, train=False, vocab_size=self.vocab_size, labels_num = self.labels_num)
        model.summary()
        self._load_model(model_file, model)

        print('***************************infer test***************************')
        indexs, values, all_values, indexs2, values2 = self.do_predict(model, sentences, vector_dim=self.conf.vector_dim, header=self.conf.predict_header, type=self.conf.type)

        correct_num = 0
        for i in range(len(indexs)):
            if indexs[i] != labels[i]:
                labels[i] = 0
            else:
                labels[i] = 1
                correct_num += 1

        print("validat set precise {}, error number {}".format(correct_num/len(labels), len(labels)-correct_num))

        tpr, fpr, accuracy, best_thresholds = evaluate_best_threshold_value(values, labels, nrof_folds=10)
        tpr = np.mean(tpr)
        fpr = np.mean(fpr)
        accuracy = np.mean(accuracy)
        best_thresholds = np.mean(best_thresholds)
        print(
            "cosine: (正样本的召回率tp/(tp+fn))tpr={} (负样本的错误率fp/(fp+tn))fpr={} acc={} threshold={}".format(tpr, fpr, accuracy,
                                                                                                     best_thresholds))


    def find_best_threshold_for_second(self, model_file):
        sentences, labels = self._prepare_test_data()

        print('***************************build model***************************')
        model = ModelWrapper.model(self.conf, train=False, vocab_size=self.vocab_size, labels_num = self.labels_num)
        model.summary()
        self._load_model(model_file, model)

        print('***************************infer test***************************')
        indexs, values, all_values, indexs2, values2 = self.do_predict(model, sentences, vector_dim=self.conf.vector_dim, header=self.conf.predict_header, type=self.conf.type)

        ground_truth = labels.copy()
        correct_num = 0
        err_min_second = 1.0      #错误的case中第二个的最小值
        err_max_gap = 0.0         #错误的case中第一个和第二个的最大差距
        correct_max_second = 0.0  #正确的case中第二个的最大值
        correct_min_gap = 1.0     #正确的case中第一个和第二个的最小差距
        for i in range(len(indexs)):
            if indexs[i] != labels[i]:
                labels[i] = 0
                if err_min_second > values2[i]:
                    err_min_second = values2[i]
                if err_max_gap < values[i] - values2[i]:
                    err_max_gap = values[i] - values2[i]
            else:
                labels[i] = 1
                correct_num += 1
                if correct_max_second < values2[i]:
                    correct_max_second = values2[i]
                if correct_min_gap > values[i] - values2[i]:
                    correct_min_gap = values[i] - values2[i]

        print("validat set precise {}, error number {}".format(correct_num/len(labels), len(labels)-correct_num))
        print("err_min_second {}, err_max_gap {}".format(err_min_second, err_max_gap))
        print("correct_max_second {}, correct_min_gap {}".format(correct_max_second, correct_min_gap))

        best_gap, best_gap_rate = self.evaluate_gaps(ground_truth, indexs, values, indexs2, values2, correct_min_gap, err_max_gap)
        print("best_gap {}, best_gap_rate {}".format(best_gap, best_gap_rate))

        best_threshold, best_threshold_rate = self.evaluate_second_thresholds(ground_truth, indexs, values2, err_min_second, correct_max_second)
        print("best_threshold {}, best_threshold_rate {}".format(best_threshold, best_threshold_rate))

    def evaluate_gaps(self, ground_truth, indexs, values, indexs2, values2, gap1, gap2):
        num = int((gap2 - gap1) / 0.005)
        gaps = np.linspace(gap1, gap2+0.005, num=num)
        recall_rate = []
        for gap_idx, gap in enumerate(gaps):
            rate = self.evaluate_gap(ground_truth, indexs, values, values2, gap)
            recall_rate.append(rate)
        best_index = recall_rate.index(max(recall_rate))
        return gaps[best_index], recall_rate[best_index]

    def evaluate_gap(self, ground_truth, indexs, values, values2, gap):
        err_indexs = []
        for i in range(len(ground_truth)):
            if indexs[i] != ground_truth[i]:
                err_indexs.append(i)

        pickup_list = []
        for i in range(len(ground_truth)):
            if values[i] - values2[i] < gap:
                pickup_list.append(i)

        print("gap={} pickup={} err={}".format(gap, pickup_list, err_indexs))
        if set(err_indexs) <= set(pickup_list):
            return len(err_indexs) / len(pickup_list)
        else:
            return 0.0

    def evaluate_second_thresholds(self, ground_truth, indexs, values2, threshold1, threshold2):
        num = int((threshold2 - threshold1) / 0.005)
        thresholds = np.linspace(threshold1, threshold2+0.005, num=num)
        recall_rate = []
        for threshold_idx, threshold in enumerate(thresholds):
            rate = self.evaluate_second_threshold(ground_truth, indexs, values2, threshold)
            recall_rate.append(rate)
        best_index = recall_rate.index(max(recall_rate))
        return thresholds[best_index], recall_rate[best_index]

    def evaluate_second_threshold(self, ground_truth, indexs, values2, threshold):
        err_indexs = []
        for i in range(len(ground_truth)):
            if indexs[i] != ground_truth[i]:
                err_indexs.append(i)

        pickup_list = []
        for i in range(len(ground_truth)):
            if values2[i] > threshold:
                pickup_list.append(i)

        print("gap={} pickup={} err={}".format(threshold, pickup_list, err_indexs))
        if set(err_indexs) <= set(pickup_list):
            return len(err_indexs) / len(pickup_list)
        else:
            return 0.0

    # def get_best_threshold(self, model_file):
    #     sentences, labels = self._prepare_test_data(file_corpus=conf.TRAIN_TEST_FILE)
    #
    #     print('***************************build model***************************')
    #     model = ModelWrapper.model(self.conf, train=False, vocab_size=self.vocab_size, labels_num=self.labels_num)
    #     model.summary()
    #     self._load_model(model_file, model)
    #
    #     print('***************************infer test***************************')
    #     indexs, values, all_values = self.do_predict(model, sentences, vector_dim=self.conf.vector_dim, header=self.conf.predict_header, type=self.conf.type)
    #
    #     print('***************************gen pairs***************************')
    #     embeddings1, embeddings2, issame = self.generate_full_pairs(all_values, labels)
    #     print('***************************evaluate_best_threshold***************************')
    #     tpr, fpr, accuracy, best_thresholds = evaluate_best_threshold(embeddings1, embeddings2, issame, dist_fun=calc_two_embeddings_distance, type="euclidean")
    #     tpr = np.mean(tpr)
    #     fpr = np.mean(fpr)
    #     accuracy = np.mean(accuracy)
    #     best_thresholds = np.mean(best_thresholds)
    #     print("cosine: (正样本的召回率tp/(tp+fn))tpr={} (负样本的错误率fp/(fp+tn))fpr={} acc={} threshold={}".format(tpr, fpr, accuracy, best_thresholds))
    #     return best_thresholds


if __name__ == '__main__':

    if len(sys.argv) > 1:
        model_file = sys.argv[1]
    else:
        model_file = "{}{}_{}.h5".format(conf.SAVE_DIR, "CNN_CLASS", "2020-09-25-07-28-40")
    values_file = os.getcwd() + "/../save/class_npy.npy"

    infer_test = InferClassTest()
    infer_test.file_infer_test(model_file=model_file, values_file=values_file)
    #infer_test.find_best_threshold_for_second(model_file=model_file)