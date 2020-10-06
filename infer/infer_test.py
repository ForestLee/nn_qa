import os
import sys
import datetime
import tensorflow as tf
from model import ModelWrapper
from config import conf
import numpy as np
import time
import pandas as pd
from tensorflow import keras
import tensorflow.keras.backend as K
from fit_generator.fit_generator_wrapper import FitGeneratorWrapper
from infer.best_threshold import evaluate_best_threshold
from distance import calc_two_embeddings_distance, calc_distances_to_mean
from gemutils.graph.serving import SaveTensorflowServingModel
from gemutils.graph.editor import GraphEditor
from qa_utils.unfreeze_util import unfreeze_pb


class InferTest(object):
    def __init__(self, backbone="", type="", header=""):
        print(os.getcwd() + "/../")
        sys.path.append(os.getcwd() + "/../")

        if len(sys.argv) == 2 and sys.argv[1] == "gpu":
            if tf.test.is_gpu_available():
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 0 for V100, 1 for P100
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        self.fit_gen = None
        self.vocab_size = 0
        self.labels_num = 0
        self.conf = conf
        if backbone!="":
            self.conf.backbone = backbone

        if type!="":
            self.conf.type=type

        if header!="":
            self.conf.predict_header=header


    def do_predict(self, model, sentences, vector_dim=0, header="", type=""):
        if header == "vector":
            return self.__predict_vectors(model, sentences, vector_dim, type=type)
        elif header == "index":
            return self._class_predict_index(model, sentences)
        elif header == "value":
            return self.__predict_value(model, sentences)
        else:
            print("wrong header"+header)
            return None

    def __predict_vectors(self, model, sentences, vector_dim, type=""):
        if type == "class":
            if len(sentences) == 1:
                return self.__class_predict_vectors(model, sentences, vector_dim)
            elif len(sentences) > 1:
                return self.__class_batch_predict_vectors(model, sentences)
        elif type == "bert":
            return self.__bert_predict_vectors(model, sentences, vector_dim)

    def __class_predict_vectors(self, model, sentences, vector_dim):
        """
        infer sentences and get vectors
        @param model:
        @param sentences:
        @param vector_dim:
        @return:
        """
        vectors = np.zeros((sentences.shape[0], vector_dim))
        max_len = sentences.shape[1]
        i = 0
        for sentence in sentences:
            sentence = sentence.reshape(1, max_len)
            logits_output = model.predict(sentence, verbose=1)
            vectors[i] = logits_output
            i += 1
        return vectors

    def __class_batch_predict_vectors(self, model, sentences):
        max_len = sentences.shape[1]
        for index in range(max_len):
            sentences[index] = sentences[index].reshape(1, max_len)

        vectors = model.predict_on_batch(sentences)
        return vectors

    def _class_predict_index(self, model, sentences):
        indexs = np.zeros((sentences.shape[0]), dtype=np.int)
        max_len = sentences.shape[1]
        i = 0
        start_t = datetime.datetime.now()
        for sentence in sentences:
            sentence = sentence.reshape(1, max_len)
            logits_output = model.predict(sentence)
            indexs[i] = logits_output
            i += 1
        end_t = datetime.datetime.now()
        print("{} sentences infer time {} seconds".format(sentences.shape[0], (end_t - start_t).seconds))
        return indexs

    def __predict_value(self, model, sentences):
        return self.__class_predict_value(model, sentences)
        # if len(sentences) == 1:
        #     return self.__class_predict_value(model, sentences)
        # elif len(sentences) > 1:
        #     return self.__class_batch_predict_value(model, sentences)

    def __class_predict_value(self, model, sentences):
        sentences1 = sentences[0]
        sentences2 = sentences[1]
        values = np.zeros((sentences1.shape[0]), dtype=np.float)

        dim=sentences1[0].shape[0]
        start_t = datetime.datetime.now()
        for i in range(len(sentences1)):
            x1 = sentences1[i].reshape((-1, dim))
            x2 = sentences2[i].reshape((-1, dim))
            logits_output = model.predict(x=[x1, x2])
            values[i] = logits_output
            #t2 = datetime.datetime.now()
            #print("{}/{} sentences infer time {} seconds".format(i, len(sentences1), (t2 - start_t).seconds))

        end_t = datetime.datetime.now()
        print("{} sentences infer time {} seconds".format(sentences1.shape[0], (end_t - start_t).seconds))
        return values

    def __class_batch_predict_value(self, model, sentences):
        sentences1 = sentences[0]
        sentences2 = sentences[1]

        dim=sentences1[0].shape[0]
        inputs = []
        start_t = datetime.datetime.now()
        for i in range(len(sentences1)):
            x1 = sentences1[i].reshape((-1, dim))
            x2 = sentences2[i].reshape((-1, dim))
            inputs.append([x1, x2])

        values = model.predict_on_batch(np.array(inputs))

        end_t = datetime.datetime.now()
        print("{} sentences infer time {} seconds".format(sentences1.shape[0], (end_t - start_t).seconds))
        return values



    def __bert_predict_vectors(self, model, sentences, vector_dim):
        token_ids = sentences[0]
        segment_ids = sentences[1]
        vectors = np.zeros((token_ids.shape[0], vector_dim))
        for i in range(len(token_ids)):
            input_1 = np.array(token_ids[i]).reshape((1, token_ids.shape[1]))
            input_2 = np.array(segment_ids[i]).reshape((1, token_ids.shape[1]))
            logits_output = model.predict([input_1, input_2])
            vectors[i] = logits_output
        return vectors

    def generate_full_pairs(self, vectors, labels):
        embeddings1 = []
        embeddings2 = []
        issame = []
        n = len(vectors)
        for i in range(n):
            for j in range(n):
                if i != j:
                    embeddings1.append(vectors[i])
                    embeddings2.append(vectors[j])
                    if labels[i] == labels[j]:
                        issame.append(True)
                    else:
                        issame.append(False)
        return np.array(embeddings1), np.array(embeddings2), np.array(issame)


    def __calc_distances_top_k(self, all_distances_arrays, labels, k):
        top_k_distance_label = []
        for row_distances in all_distances_arrays:
            distance_label_dict = {}
            i = 0
            for distance in row_distances:
                distance_label_dict[distance] = labels[i]
                i += 1
            sorted_dict = sorted(distance_label_dict.items(), key=lambda x:x[0],reverse=False)
            top_k_distance_label.append(sorted_dict[:k])

        np_file = os.getcwd() + "/../log/all_distances_arrays.npy"
        np.save(np_file, all_distances_arrays)
        np_file = os.getcwd() + "/../log/labels.npy"
        np.save(np_file, labels)
        np_file = os.getcwd() + "/../log/top_k_distance_label.npy"
        np.save(np_file, top_k_distance_label)
        return top_k_distance_label

    def __top_k_select(self, top_k_distance_label):
        top_closest_labels = []
        top_closest_distances = []
        for distances_labels in top_k_distance_label:
            labels = []
            distances_labels = distances_labels[1:]
            for distance_label in distances_labels:
                labels.append(distance_label[1])
            label_count = pd.Series(labels)
            label_count = label_count.value_counts()
            max_count = label_count[0]
            best_label = label_count.keys()[0]
            for i in range(len(label_count)):
                if max_count < label_count[i]:
                    max_count = label_count[i]
                    best_label = label_count.keys()[i]

            top_closest_labels.append(best_label)
            for distance_label in distances_labels:
                if distance_label[1] == best_label:
                    top_closest_distances.append(distance_label[0])
                    break
        return top_closest_labels, top_closest_distances

    def test_distances_top_k(self):

        np_file=os.getcwd()+"/../log/all_distances_arrays.npy"
        if os.path.exists(np_file):
            all_distances_arrays=np.load(np_file)

        np_file=os.getcwd()+"/../log/labels.npy"
        if os.path.exists(np_file):
            labels=np.load(np_file)

        np_file=os.getcwd()+"/../log/top_k_distance_label.npy"
        if os.path.exists(np_file):
            top_k_distance_label=np.load(np_file)

        log_file = open(self.conf.LOG_DIR + "topk.txt", "w")
        line = 1
        for distances_labels in top_k_distance_label:
            log_file.write("line #"+str(line)+"\n")
            line +=1
            for distance_label in distances_labels:
                log_file.write(str(distance_label)+"\n")

        log_file.close()
        print("done")

    def __calc_correct_rate_by_idxs(self, fit_gen, labels, mean_labels_list, mean_closest_label_idxs, mean_closest_distances, name=""):
        all_closest_labels = []
        for i in mean_closest_label_idxs:
            all_closest_labels.append(mean_labels_list[i])
        all_closest_labels = np.array(all_closest_labels)
        return self.__calc_correct_rate(fit_gen, labels, all_closest_labels, mean_closest_distances, name=name)


    def __calc_correct_rate(self, corpus, labels, all_closest_labels, all_closest_distances, name=""):
        """
        calculate correct infer ration
        @param corpus:
        @param labels:
        @param all_closest_labels:
        @param all_closest_distances:
        """
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S.err.log", time.localtime())
        log_file = open(self.conf.LOG_DIR+name+"_infer_"+timestamp, "w")
        correct = 0
        for i in range(len(all_closest_labels)):
            # print("sentence #{} label[{}], most closed is: {}".format(str(i+1), labels[i], str(all_closest_labels[i])))
            if labels[i] == all_closest_labels[i]:
                correct += 1
            else:
                error_msg = "line #{} label[{}], infer[{}] distance[{}]".format(str(i + 1), labels[i],
                                                                                str(all_closest_labels[i]),
                                                                                all_closest_distances[i])
                line = corpus.read_line(i)
                error_msg = error_msg + '.\tsentence:[' + line.split("\t")[0] + ']\n'
                print(error_msg)
                log_file.write(error_msg)

        precise_str="precise is {}%, total:{}, correct:{}".format(100 * correct / len(labels), str(len(labels)), str(correct))
        print(precise_str)
        log_file.write(precise_str+"\n")

        log_file.close()


    def __get_index_list_per_class(self, labels):
        labels_list = {}.fromkeys(labels).keys()
        class_index_list = []
        label_list = labels.tolist()
        for class_data in labels_list:
            class_index = self.__get_index(label_list, class_data)
            class_index_list.append(class_index)
        return class_index_list, list(labels_list)

    def __get_class_mean(self, vectors, class_index_list, vect_dim):
        """
        calculate mean vector for each class
        @param vectors:
        @param class_index_list:
        @param vect_dim:
        @return:
        """
        vect_mean = []
        for each in class_index_list:
            label_count = len(each)
            a = np.array([0] * vect_dim)
            for i in range(label_count):
                a = a + vectors[each[i]]
            mean = a / label_count
            vect_mean.append(mean)
        return vect_mean



    def __get_index(self, label_list, label, n=None):
        if n is None:
            n = label_list.count(label)
        if n <= label_list.count(label):
            all_index = [key for key, value in enumerate(label_list) if value == label]
            return tuple(all_index)
        else:
            return None

    def __get_train_mean_vec(self, fit_gen, model, model_file, train_file, vector_dim, type=""):
        mean_vec_file = model_file + ".mean.vec.npy"
        mean_vec_txt_file = model_file + ".mean.vec.txt"
        labels_list_file = model_file + ".label_list.npy"
        labels_list_txt_file = model_file + ".label_list.txt"

        # if os.path.exists(mean_vec_file):
        #     print("load mean vector from {}".format(mean_vec_file))
        #     mean_vector=np.load(mean_vec_file)
        #     labels_list=np.load(labels_list_file)
        # else:
        sentences, labels = fit_gen.read_raw_corpus(file_corpus=train_file)
        class_index_list, labels_list = self.__get_index_list_per_class(labels)
        vectors = self.do_predict(model, sentences, vector_dim=self.conf.vector_dim, header=self.conf.predict_header, type=self.conf.type)
        mean_vector = self.__get_class_mean(vectors, class_index_list, vect_dim=vector_dim)
        np.save(mean_vec_file, mean_vector)
        np.savetxt(mean_vec_txt_file, mean_vector)
        np.save(labels_list_file, labels_list)
        np.savetxt(labels_list_txt_file, labels_list, fmt='%s')
        print("save mean vector to {}".format(mean_vec_file))
        return mean_vector, labels_list

    def __load_mean_vec(self, model_file):
        mean_vec_file = model_file + ".mean.vec.npy"
        labels_list_file = model_file + ".label_list.npy"

        if os.path.exists(mean_vec_file):
            print("load mean vector from {}".format(mean_vec_file))
            mean_vector=np.load(mean_vec_file)
            labels_list=np.load(labels_list_file)
            return mean_vector, labels_list
        return None, None

    def save_data_for_projector(self, vectors, labels):
        np.savetxt(os.getcwd() + "/../log/vectors.txt", vectors, delimiter='\t')
        label_sentences = []
        with open(self.conf.TEST_FILE, 'r') as infile:
            i = 0
            for line in infile:
                sent = line.strip("\n").split('\t')[0].strip(" ")
                label_sentences.append([labels[i].strip(" ") + "_" + sent])
            i += 1
            np.savetxt(os.getcwd() + "/../log/labels.txt", label_sentences, fmt="%s", delimiter='\t')

    def _prepare_test_data(self, file_corpus=conf.TEST_FILE):
        print('***************************prepare corpus***************************')
        self.fit_gen = FitGeneratorWrapper(type=self.conf.type, file_vocab=self.conf.VOCAB_FILE, file_corpus=self.conf.TEST_FILE,
                                           batch_size=self.conf.batch_size, max_len=self.conf.maxlen, vector_dim=self.conf.vector_dim,
                                           ner_vocab=self.conf.pretrain_vocab, label_dict_file=self.conf.LABEL_DICT_FILE)
        self.vocab_size = self.fit_gen.get_vocab_size()
        self.labels_num = self.fit_gen.get_label_count()
        return self.fit_gen.read_raw_corpus(file_corpus=file_corpus)

    def _load_model(self, model_file, model):
        print('***************************load model***************************')
        print("load weight from " + model_file)
        model.summary()
        model.load_weights(model_file, by_name=True)


    def file_infer_test(self, model_file, saved_model_file=None):
        sentences, labels = self._prepare_test_data()

        print('***************************build model***************************')
        model = ModelWrapper.model(self.conf, train=False, vocab_size=self.vocab_size, labels_num = 0)

        self._load_model(model_file, model)
        model.summary()

        print('***************************infer test***************************')
        vectors = self.do_predict(model, sentences, vector_dim=self.conf.vector_dim, header=self.conf.predict_header, type=self.conf.type)
        if self.conf.save_data_for_projector == True:
            self.save_data_for_projector(vectors, labels)

        # print("****************calculate distances**************")
        # all_distances_arrays, all_closest_labels, all_closest_distances = calc_distances(vectors, labels)
        # __calc_correct_rate(fit_gen, labels, all_closest_labels, all_closest_distances, name="top1")
        #
        # print("****************calculate top k distances**************")
        # top_k_distance_label = __calc_distances_top_k(all_distances_arrays, labels, k=3)
        # top_closest_labels, top_closest_distances = __top_k_select(top_k_distance_label)
        # __calc_correct_rate(fit_gen, labels, top_closest_labels, top_closest_distances, name="top3")

        if not saved_model_file is None:
            print('***************************save model***************************')
            tf.saved_model.save(model, saved_model_file)

        print("****************calculate distance to mean vector for each class**************")
        #class_index_list, labels_list = __get_index_list_per_class(labels)
        #mean_vector = __get_class_mean(vectors, class_index_list, vect_dim=conf.vector_dim)
        mean_vector, mean_labels_list = self.__get_train_mean_vec(self.fit_gen, model, model_file, self.conf.TRAIN_FILE, self.conf.vector_dim, self.conf.type)
        print("*********************calc_cosine_distances_to_mean start*****************************")
        mean_distances_arrays, mean_closest_label_idxs, mean_closest_distances = calc_distances_to_mean(vectors, mean_vector, type="cosine")   #euclidean or cosine
        print("*********************calc_cosine_distances_to_mean end*****************************")
        self.__calc_correct_rate_by_idxs(self.fit_gen, labels, mean_labels_list, mean_closest_label_idxs, mean_closest_distances, name="mean")
        np.savetxt(os.getcwd() + "/../save/mean_distances_arrays.txt", mean_distances_arrays, fmt='%s')

        print("****************done**************")

    def get_best_threshold(self, model_file):
        sentences, labels = self._prepare_test_data(file_corpus=self.conf.TRAIN_TEST_FILE)

        print('***************************build model***************************')
        model = ModelWrapper.model(self.conf, train=False, vocab_size=self.vocab_size, labels_num = self.labels_num)

        self._load_model(model_file, model)

        print('***************************infer vectors***************************')
        vectors = self.do_predict(model, sentences, vector_dim=self.conf.vector_dim, header=self.conf.predict_header, type=self.conf.type)
        print('***************************gen pairs***************************')
        embeddings1, embeddings2, issame = self.generate_full_pairs(vectors, labels)
        print('***************************evaluate_best_threshold***************************')
        tpr, fpr, accuracy, best_thresholds = evaluate_best_threshold(embeddings1, embeddings2, issame, dist_fun=calc_two_embeddings_distance, type="cosine")
        tpr = np.mean(tpr)
        fpr = np.mean(fpr)
        accuracy = np.mean(accuracy)
        best_thresholds = np.mean(best_thresholds)
        print("cosine: (正样本的召回率tp/(tp+fn))tpr={} (负样本的错误率fp/(fp+tn))fpr={} acc={} threshold={}".format(tpr, fpr, accuracy, best_thresholds))
        return best_thresholds

    def mean_inside_model_generate_test(self, model_file, saved_model_file):
        sentences, labels = self._prepare_test_data()

        print('***************************build model***************************')
        if self.conf.predict_header == "index":
            mean_vector, mean_labels_list = self.__load_mean_vec(model_file)
        elif self.conf.predict_header == "vector":
            mean_vector = None
        model = ModelWrapper.model(self.conf, train=False, vocab_size=self.vocab_size, labels_num = 0, mean_vects=mean_vector)

        self._load_model(model_file, model)

        print('***************************infer test***************************')
        if self.conf.predict_header == "index":
            indexs = self.do_predict(model, sentences, vector_dim=self.conf.vector_dim, header=self.conf.predict_header, type=self.conf.type)

            i = 0
            correct = 0
            for idx in indexs:
                if mean_labels_list[idx] == labels[i]:
                    if self.conf.DEBUG == True:
                        print("OK {} {} label {}, predict {}".format(i, idx, labels[i], mean_labels_list[idx]))
                    correct += 1
                else:
                    if self.conf.DEBUG == True:
                        print("err {} {} label {}, predict {}".format(i, idx, labels[i], mean_labels_list[idx]))
                i += 1

            precise_str = "precise is {}%, total:{}, correct:{}".format(100 * correct / len(labels), str(len(labels)),
                                                                        str(correct))
            print(precise_str)


            print('***************************save index h5***************************')
            model.save(os.getcwd() + "/../save/qa_model_index.h5", include_optimizer=False)
        elif self.conf.predict_header == "vector":
            vectors = self.do_predict(model, sentences, vector_dim=self.conf.vector_dim, header=self.conf.predict_header, type=self.conf.type)

            mean_vector, mean_labels_list = self.__get_train_mean_vec(self.fit_gen, model, model_file, self.conf.TRAIN_FILE,
                                                                 self.conf.vector_dim, self.conf.type)
            print("*********************calc_cosine_distances_to_mean start*****************************")
            mean_distances_arrays, mean_closest_label_idxs, mean_closest_distances = calc_distances_to_mean(vectors,
                                                                                                            mean_vector,
                                                                                                            type="cosine")  # euclidean or cosine
            print("*********************calc_cosine_distances_to_mean end*****************************")
            self.__calc_correct_rate_by_idxs(self.fit_gen, labels, mean_labels_list, mean_closest_label_idxs, mean_closest_distances,
                                        name="mean")

        print('***************************saved index model***************************')
        tf.saved_model.save(model, saved_model_file)

        print('***************************save frozen model***************************')
        K.set_learning_phase(0)

        saved_model_path = os.path.join(saved_model_file+"_index_frozen")
        freozen_pb_path = os.path.join(saved_model_file+"_index_frozen", 'frozen_model.pb')

        SaveTensorflowServingModel(K.get_session(), model.input, model.output, saved_model_path)

        input_node_names = [x.name.replace(':0', '') for x in model.inputs]
        output_node_names = [x.name.replace(':0', '') for x in model.outputs]

        editor = GraphEditor.from_saved_model(
            saved_model_path,
            output_node_names)

        editor = editor.remove_dropout()
        editor = editor.optimize_for_inference(input_node_names, output_node_names, [x.dtype.as_datatype_enum for x in model.inputs])
        editor = editor.save_as(freozen_pb_path)
        print("****************done**************")

    def mean_inside_model_load_test(self, model_file, saved_model_file):
        sentences, labels = self._prepare_test_data()

        print('***************************build model***************************')
        mean_vector, mean_labels_list = self.__load_mean_vec(model_file)

        print('***************************load model***************************')
        model = keras.models.load_model(saved_model_file)

        print('***************************infer test***************************')
        if self.conf.predict_header == "index":
            indexs = self.do_predict(model, sentences, vector_dim=self.conf.vector_dim, header=self.conf.predict_header, type=self.conf.type)

            i = 0
            correct = 0
            for idx in indexs:
                if mean_labels_list[idx] == labels[i]:
                    if self.conf.DEBUG == True:
                        print("OK {} {} label {}, predict {}".format(i, idx, labels[i], mean_labels_list[idx]))
                    correct += 1
                else:
                    if self.conf.DEBUG == True:
                        print("err {} {} label {}, predict {}".format(i, idx, labels[i], mean_labels_list[idx]))
                i += 1

            precise_str = "precise is {}%, total:{}, correct:{}".format(100 * correct / len(labels), str(len(labels)),
                                                                        str(correct))
            print(precise_str)

        elif self.conf.predict_header == "vector":
            vectors = self.do_predict(model, sentences, vector_dim=self.conf.vector_dim, header=self.conf.predict_header, type=self.conf.type)

            mean_vector, mean_labels_list = self.__get_train_mean_vec(self.fit_gen, model, model_file, self.conf.TRAIN_FILE,
                                                                 self.conf.vector_dim, self.conf.type)
            print("*********************calc_cosine_distances_to_mean start*****************************")
            mean_distances_arrays, mean_closest_label_idxs, mean_closest_distances = calc_distances_to_mean(vectors,
                                                                                                            mean_vector,
                                                                                                            type="cosine")  # euclidean or cosine
            print("*********************calc_cosine_distances_to_mean end*****************************")
            self.__calc_correct_rate_by_idxs(self.fit_gen, labels, mean_labels_list, mean_closest_label_idxs, mean_closest_distances,
                                        name="mean")

    def unfreeze_pb(self):
        export_dir_path = '../save/unfreezed_model/1'
        freezed_model_file_path = '../save/qa_model_index_frozen/frozen_model.pb'
        unfreeze_pb(export_dir_path, freezed_model_file_path)

if __name__=='__main__':

    if len(sys.argv) > 1:
        model_file=sys.argv[1]
    else:
        if conf.attention_enable == False:
            model_file = "{}{}_".format(conf.SAVE_DIR, conf.backbone)
        else:
            model_file = "{}{}_{}_".format(conf.SAVE_DIR, conf.backbone, "ATTENTION")
        model_file = model_file+"2020-10-04-16-30-20"+".h5"

    saved_model_file = os.getcwd() + "/../save/qa_model"

    infer_test = InferTest()
    #infer_test.file_infer_test(model_file=model_file, saved_model_file=saved_model_file+"_vector")
    best_thresholds = infer_test.get_best_threshold(model_file)

    # infer_test.mean_inside_model_generate_test(model_file=model_file, saved_model_file=saved_model_file)
    # infer_test.unfreeze_pb()
    #infer_test.mean_inside_model_load_test(model_file=model_file, saved_model_file=saved_model_file)

