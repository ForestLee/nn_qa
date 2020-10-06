import os
import numpy as np
from model.text_cnn_classify import TextCNNClassify
from model.bert_classify import BertClass
from model.text_cnn_archface import TextCNNArchface
from model.lstm_archface import LstmArchface
from model.bert_archface import BertArchface
from model.text_match import TextMatch
from model.esim import ESIM
from qa_utils.get_emb_from_ckpt import get_embedding_from_ckpt

class ModelWrapper():

    @staticmethod
    def model(conf, train=True, vocab_size=0, labels_num = 0, mean_vects=None):
        if train == False:
            dropout = 0
        else:
            dropout = conf.cnn_drop_rate

        init_emb = None
        if conf.pretrain_vocab == True:
            init_emb = get_embedding_from_ckpt(ckpt_path=os.getcwd()+conf.pretrain_ckpt_path, ckpt_file=conf.pretrain_ckpt_file, emb_name=conf.pretrain_emb_name)
            #to use BERT format tokenization, expand "[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"
            expand_char = np.zeros((conf.embedding_dims))
            for _ in ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]:
                init_emb = np.row_stack((init_emb, expand_char))

        if conf.backbone == "CNN_ARCFACE":
            model = TextCNNArchface.model(inputlen=conf.maxlen,
                                          vocabulary=vocab_size,
                                          num_class=labels_num,
                                          vector_dim=conf.vector_dim,
                                          embedding_dim=conf.embedding_dims,
                                          num_filters=conf.cnn_num_filters,
                                          filter_sizes=conf.cnn_filter_sizes,
                                          drop_rate=dropout,
                                          l2=conf.cnn_dense_l2,
                                          train=train,
                                          init_emb_enable=conf.pretrain_vocab,
                                          init_emb=init_emb,
                                          mean_vects = mean_vects,
                                          attention_enable=conf.attention_enable)

        elif conf.backbone == "BERT_CLASS":
            model = BertClass.model(num_class=labels_num, train=train, batch_size=conf.batch_size)

        elif conf.backbone == "BERT_ARCFACE":
            model = BertArchface.model(vector_dim=conf.vector_dim,
                                       num_class=labels_num,
                                       batch_size=conf.batch_size,
                                       drop_rate=dropout,
                                       train=train)

        elif conf.backbone == "LSTM_ARCFACE":
            model = LstmArchface.model(inputlen=conf.maxlen,
                                       vocabulary=vocab_size,
                                       vector_dim=conf.vector_dim,
                                       embedding_dim=conf.embedding_dims,
                                       lstm_unit=conf.lstm_unit,
                                       num_class=labels_num,
                                       drop_rate=dropout,
                                       l2=conf.cnn_dense_l2,
                                       train=train,
                                       init_emb_enable=conf.pretrain_vocab,
                                       init_emb=init_emb,
                                       attention_enable=conf.attention_enable)

        elif conf.backbone == "CNN_CLASS":
            model = TextCNNClassify.model(inputlen=conf.maxlen,
                                          vocabulary=vocab_size,
                                          num_class=labels_num,
                                          vector_dim=conf.vector_dim,
                                          embedding_dim=conf.embedding_dims,
                                          num_filters=conf.cnn_num_filters,
                                          filter_sizes=conf.cnn_filter_sizes,
                                          drop_rate=dropout,
                                          train=train,
                                          init_emb_enable=conf.pretrain_vocab,
                                          init_emb=init_emb,
                                          attention_enable=conf.attention_enable,
                                          rbf = conf.RBF,
                                          maxout = conf.MAXOUT,
                                          maxout_num = conf.maxout_num)

        elif conf.backbone == "TEXT_MATCH":
            model = TextMatch.model(vocabulary=vocab_size, embedding_dim=conf.embedding_dims, inputlen=conf.maxlen, train=train, dropout=dropout)

        elif conf.backbone == "ESIM":
            model = ESIM.model(vocabulary=vocab_size, embedding_dim=conf.embedding_dims, n_classes=labels_num, max_length=conf.maxlen, hidden_units=300,
                               dropout = dropout, train = train)

        else:
            print("wrong backbone:" + conf.backbone)

        return model