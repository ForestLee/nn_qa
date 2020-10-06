import os
import sys

import joblib
from sklearn.mixture import GaussianMixture

print(os.getcwd() + "/../")
sys.path.append(os.getcwd() + "/../")

from model import ModelWrapper
from config import conf
import numpy as np
from fit_generator.fit_generator_wrapper import FitGeneratorWrapper

def __predict_vectors(model, sentences, vector_dim):
    return __class_predict_vectors(model, sentences, vector_dim)


def __class_predict_vectors(model, sentences, vector_dim):
    """
    infer sentences and get vectors
    @param model:
    @param sentences:
    @param vector_dim:
    @return:
    """
    vectors = np.zeros((sentences.shape[0], vector_dim))
    i = 0
    for sentence in sentences:
        sentence = sentence.reshape(1, 64)
        logits_output = model.predict(sentence)
        vectors[i] = logits_output
        i += 1
    return vectors
def gauss_change_data(vectors,labels):
    vectors_list=vectors.tolist()
    labels_list=labels.tolist()
    dic = {}
    dic2 = {}
    for i in range(len(labels_list)):
        dic[str(vectors_list[i])] = labels_list[i]
        dic2[str(vectors_list[i])] = vectors_list[i]
    set1 = set(labels_list)
    labels_list_after_set= list(set1)
    dic_all = {}
    for i in labels_list_after_set:
        list_a = []
        for j in dic:
            if dic[j] == i:
                list_a.append(dic2[j])
        dic_all[i] = list_a
    return dic_all,labels_list_after_set

def file_infer_test(model_file):
    # fit_gen_test = FitGeneratorWrapper(type=conf.type, file_vocab=conf.VOCAB_FILE, file_corpus=conf.TEST_FILE,
    #                                    batch_size=conf.batch_size, max_len=conf.maxlen, vector_dim=conf.vector_dim,
    #                                    pretrain_vocab=conf.pretrain_vocab, label_dict_file=conf.LABEL_DICT_FILE)
    path_data = '../data/new_all.csv.test'

    list_data = []
    with open(path_data, 'r') as f:
        for i in f.read().splitlines():
            list_data.append(i.split('\t'))
    fit_gen_train = FitGeneratorWrapper(type=conf.type, file_vocab=conf.VOCAB_FILE, file_corpus=conf.TRAIN_FILE,
                                        batch_size=conf.batch_size, max_len=conf.maxlen, vector_dim=conf.vector_dim,
                                        ner_vocab=conf.pretrain_vocab, label_dict_file=conf.LABEL_DICT_FILE)
    vocab_size_train = fit_gen_train.get_vocab_size()
    sentences_train, labels_train = fit_gen_train.read_raw_corpus(file_corpus=conf.TRAIN_FILE)
    sentences_test, labels_test = fit_gen_train.read_raw_corpus(file_corpus=conf.TEST_FILE)

    model = ModelWrapper.model(conf, train=False, vocab_size=vocab_size_train, labels_num=0)
    model.load_weights(model_file, by_name=True)
    model.summary()
    vectors_train = __predict_vectors(model, sentences_train, conf.vector_dim)
    vectors_test= __predict_vectors(model, sentences_test, conf.vector_dim)

    dic_all, labels_list_after_set = gauss_change_data(vectors_train, labels_train)
    models = {}
    n_components = 3
    model_dir = "/Users/chenhengxi/PycharmProjects/work2/sentence-encoding-qa/data/model"
    for domain in labels_list_after_set:
        modelx = GaussianMixture(n_components=n_components, covariance_type='diag', reg_covar=0.0001, max_iter=200,
                                verbose=0, verbose_interval=1)
        data = np.array(dic_all[domain])
        modelx.fit(data)
        models[domain] = modelx
        joblib.dump(modelx, "{0}/{1}.joblib".format(model_dir,domain))
    final_dic = {}
    final_num=0
    error=[]
    for i in range(len(vectors_test)):
        print(i)
        accept_scores = {}
        for domain in labels_list_after_set:
            models[domain] = joblib.load("{0}/{1}.joblib".format(model_dir,domain))
            a=np.squeeze(vectors_test[i])
            #vectors_test[i]=a.reshape(-1, 1)
            point_array = models[domain].score_samples(a.reshape(1,conf.vector_dim))
            point = point_array[0]
            accept_scores[str(point)] = domain
        list_to_max = []
        for num in accept_scores:
            list_to_max.append(float(num))
        max_num = max(list_to_max)
        label_final = accept_scores[str(max_num)]
        final_dic[str(vectors_test[i])] = label_final
        if list_data[i][1]!=label_final:
            final_num+=1
            error.append([list_data[i][0],list_data[i][1],label_final])
    print((1-final_num/len(vectors_test)))
    print(error)

if __name__=='__main__':

    model_file = "{}callback_triplet_single_{}_{}.h5".format(conf.SAVE_DIR, conf.backbone, "2020-09-08-11-20-30")       #new corpus

    if len(sys.argv) > 1:
        model_file=sys.argv[1]

    file_infer_test(model_file=model_file)