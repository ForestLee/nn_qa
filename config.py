import os

class Conf:

    DEBUG = False

    SAVE_DIR = os.getcwd() + "/../save/"
    OUT_DIR = os.getcwd() + "/../runs/"
    LOG_DIR = os.getcwd() + "/../log/"

    #VOCAB_FILE = os.getcwd() + "/../data/vocab.txt"
    VOCAB_FILE = os.getcwd() + "/../data/words.txt"
    #VOCAB_FILE = os.getcwd() + "/../data/ner_vocab.txt"     #from NER project
    pretrain_vocab = False
    pretrain_ckpt_path = "/../data/warm_start_ckp/"
    pretrain_ckpt_file = "model.ckpt-1276686"
    pretrain_emb_name = "task_independent/Variable_1"

    TRAIN_TEST_FILE = os.getcwd() + "/../data/new_all.csv"
    TRAIN_FILE = os.getcwd() + "/../data/new_all.csv.train"
    TEST_FILE = os.getcwd() + "/../data/new_all.csv.test"

    # TRAIN_TEST_FILE = "/home/forest/QA/textcnndata/newdomain_0924.txt"
    # TRAIN_FILE = "/home/forest/QA/textcnndata/newdomain_0924.txt.train"
    # TEST_FILE = "/home/forest/QA/textcnndata/newdomain_0924.txt.test"

    PAIR_TRAIN_FILE = os.getcwd() + "/../data/pair_all.csv.train"
    PAIR_TEST_FILE = os.getcwd() + "/../data/pair_all.csv.test"
    PAIR_TRAIN_TEST_FILE = os.getcwd() + "/../data/pair_all.csv.train_test"
    LABEL_DICT_FILE = os.getcwd() + "/../data/label_dict.txt"
    #LABEL_DICT_FILE = "/home/forest/QA/textcnndata/label_dict.txt"

    embedding_dims = 32    #word embedding
    vector_dim = 32   #64       #sentence vector
    epochs = 200
    lr = 0.05
    loss_weights = [1, 0.01] #xentropy vs center/triplet loss weights
    batch_size = 64
    margin = 1   #5.0



    #CNN model
    cnn_num_filters = 128
    cnn_filter_sizes = [3, 4, 5]
    cnn_drop_rate = 0.5   #0.5  #0.3   #0.5
    cnn_dense_l2 = 0.01

    #LSTM
    lstm_unit = 100


    # eayly stopping and reluce_lr
    early_stop_patience = 30   #Early stop patience. Only available when -best_fit=True
    reduce_lr_patience = 3     #Reduce learning rate on plateau patience.  Only available when -best_fit=True
    reduce_lr_factor = 0.75    #Reduce learning rate on plateau factor.  Only available when -best_fit=True

    # keras dense kernel initializer
    dense_initialize_flag = False
    FIT_GENERATE = False
    attention_enable = False
    predict_header = "vector"  # index   or vector
    backbone = "BERT_CLASS"  # "BERT_ARCFACE"  #"CNN_ARCFACE"  # CNN, CNN_CLASS, LSTM, BERT, CNN_ARCFACE, LSTM_ARCFACE, BERT_ARCFACE, BERT_CLASS
    type = "bert"  #"bert"    #"class"  #bert input is different with classify
    maxlen = 64

    save_data_for_projector = False
    GAN_ENABLE = False

    RBF = False
    MAXOUT = False
    maxout_num = 2

conf = Conf()
