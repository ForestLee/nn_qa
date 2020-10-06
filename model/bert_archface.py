from model.bert4keras.backend import set_gelu
from model.bert4keras.models import build_transformer_model
from tensorflow.keras.layers import Dropout, Dense, Input, BatchNormalization, Reshape, Flatten, Conv2D, MaxPool2D, Concatenate
from tensorflow.keras import Model
from layers import ArcFace
import tensorflow as tf

set_gelu('tanh')  # 切换gelu版本

config_path = '/home/forest/BERT/models/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/forest/BERT/models/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = '/home/forest/BERT/models/chinese_L-12_H-768_A-12/vocab.txt'


class BertArchface(object):
    @staticmethod
    def model(vector_dim, num_class = 0, drop_rate=0.5, batch_size = 64, l2=0.01, train = False):
        # 加载预训练模型
        bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            with_pool=False,  #"relu",   #tanh
            return_keras_model=False,
        )

        for layer in bert.model.layers:
            if not isinstance(layer, BatchNormalization):
                layer.trainable = False
        #
        # # #reshape_output = Flatten()(bert.model.output)
        # # #reshape_output = Reshape((-1, bert.hidden_size))(bert.model.output)
        # # layer_output = bert.model.get_layer("Transformer-11-FeedForward-Norm").output
        # # tmp_shape1 = layer_output
        # # tmp_shape = tmp_shape1[:, 1:2, :]
        # # reshape_output = tf.squeeze(tmp_shape, axis=1, name="squeeze_out")
        # bert_output = bert.model.get_layer("Transformer-11-FeedForward-Norm").output
        # slice_output = tf.slice(bert_output, [0,1,0], [1,1,768])
        # reshape_output = tf.squeeze(slice_output, axis=1, name="squeeze_out")
        #
        # if drop_rate != 0:
        #     dropout = Dropout(rate=drop_rate)(reshape_output)
        # else:
        #     dropout = reshape_output
        # #logits = Dense(units=vector_dim, kernel_initializer=bert.initializer, activity_regularizer=regularizers.l2())(dropout)
        # dense = Dense(vector_dim, activity_regularizer=regularizers.l2(l2))(dropout)
        #
        # bn = BatchNormalization()(dense)
        #
        # if train == True:
        #     label = Input(shape=(num_class))
        #     output = ArcFace(n_classes=num_class)([bn, label])
        #     #output = ArcFace(n_classes=num_class)([dropout, label])
        #     model = Model([bert.model.input, label], output)
        # else:
        #     #model = Model(bert.model.input, bn)
        #     #model = Model(bert.model.input, dropout)
        #     #layer_output = bert.model.get_layer("Transformer-11-FeedForward-Norm").output
        #     #layer_output2 = bert.model.get_layer("Embedding-Segment").output
        #     #layer_output3 = bert.model.get_layer("Embedding-Dropout").output
        #     model = Model(bert.model.input, bn)
        #
        # return model
        filter_sizes = [3,4,5]
        bert_output = bert.model.get_layer("Transformer-11-FeedForward-Norm").output
        reshape = Reshape((batch_size, bert.hidden_size, 1))(bert_output)

        conv_0 = Conv2D(128,
                        kernel_size=(filter_sizes[0], bert.hidden_size),
                        padding='valid',
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))(reshape)
        conv_1 = Conv2D(128,
                        kernel_size=(filter_sizes[1], bert.hidden_size),
                        padding='valid',
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))(reshape)
        conv_2 = Conv2D(128,
                        kernel_size=(filter_sizes[2], bert.hidden_size),
                        padding='valid',
                        kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))(reshape)

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

        if drop_rate != 0:
            dropout = Dropout(drop_rate)(flatten)
        else:
            dropout = flatten

        #dense = Dense(vector_dim, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None), activity_regularizer=tf.python.keras.regularizers.l2(l2))(dropout)
        dense = Dense(vector_dim,
                      kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))(dropout)

        bn = BatchNormalization()(dense)

        if train == True:
            label = Input(shape=(num_class))
            logits = ArcFace(n_classes=num_class)([bn, label])
            model = Model(inputs=[bert.model.input, label], outputs=logits)
        else:
            #model = Model(bert.model.input, [bert_output, conv_0, conv_1, conv_2, maxpool_0, maxpool_1, maxpool_2, dense, bn])
            model = Model(bert.model.input, bn)

        return model