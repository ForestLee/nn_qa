#! -*- coding:utf-8 -*-
# 句子对分类任务，LCQMC数据集
# val_acc: 0.887071, test_acc: 0.870320

from model.bert4keras.backend import set_gelu
from model.bert4keras.models import build_transformer_model
from tensorflow.keras.layers import Dropout, Dense, Softmax, BatchNormalization, Reshape, Flatten, Conv2D, MaxPool2D, Concatenate
from tensorflow.keras import Model
import tensorflow as tf

set_gelu('tanh')  # 切换gelu版本

config_path = '/home/forest/BERT/models/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/forest/BERT/models/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/forest/BERT/models/chinese_L-12_H-768_A-12/vocab.txt'


class BertClass(object):
    @staticmethod
    def model(num_class=173, train=False, batch_size=64):
        # 加载预训练模型
        bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            with_pool=False,
            return_keras_model=False,
        )

        filter_sizes = [3, 4, 5]
        bert_output = bert.model.get_layer("Transformer-11-FeedForward-Norm").output
        reshape = Reshape((batch_size, bert.hidden_size, 1))(bert_output)

        conv_0 = Conv2D(128,
                        kernel_size=(filter_sizes[0], bert.hidden_size),
                        padding='valid',
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))(
            reshape)
        conv_1 = Conv2D(128,
                        kernel_size=(filter_sizes[1], bert.hidden_size),
                        padding='valid',
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))(
            reshape)
        conv_2 = Conv2D(128,
                        kernel_size=(filter_sizes[2], bert.hidden_size),
                        padding='valid',
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))(
            reshape)

        maxpool_0 = MaxPool2D(pool_size=(batch_size - filter_sizes[0] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(batch_size - filter_sizes[1] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(batch_size - filter_sizes[2] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])

        flatten = Flatten()(concatenated_tensor)

        if train == True:
            dropout_output = Dropout(rate=0.1)(flatten)
        else:
            dropout_output = flatten

        dense_output = Dense(units=num_class)(dropout_output)

        bn = BatchNormalization()(dense_output)

        logits = Softmax()(bn)

        model = Model(bert.model.input, logits)

        return model
