import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Attention, GlobalMaxPooling2D, Embedding, Concatenate, Reshape, MaxPool2D, Dense, Flatten, Input, Conv2D, Dropout, LayerNormalization
from layers import ArcFace, VectorDistance
from config import conf

class TextCNNArchface(object):
    @staticmethod
    def model(inputlen, vocabulary, num_class, vector_dim,
              embedding_dim=128,
              num_filters=128,
              filter_sizes=[3, 4, 5],
              drop_rate=0.5,
              l2=0.01,
              train=False,
              init_emb_enable=False,
              init_emb=None,
              mean_vects = None,
              attention_enable = False):

        input = Input(shape=(inputlen), name="input")
        if train == True:
            label = Input(shape=(num_class))

        if init_emb_enable==False:
            embeddings_init = 'uniform'
        else:
            embeddings_init = keras.initializers.constant(init_emb)

        embedding = Embedding(input_dim=vocabulary,
                              output_dim=embedding_dim,
                              embeddings_initializer = embeddings_init,
                              input_length=inputlen,
                              trainable=True)(input)

        reshape = Reshape((inputlen, embedding_dim, 1))(embedding)

        conv_0 = Conv2D(num_filters,
                        kernel_size=(filter_sizes[0], embedding_dim),
                        padding='valid',
                        activation='relu')(reshape)
        conv_1 = Conv2D(num_filters,
                        kernel_size=(filter_sizes[1], embedding_dim),
                        padding='valid',
                        activation='relu')(reshape)
        conv_2 = Conv2D(num_filters,
                        kernel_size=(filter_sizes[2], embedding_dim),
                        padding='valid',
                        activation='relu')(reshape)

        maxpool_0 = MaxPool2D(pool_size=(inputlen - filter_sizes[0] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_0)
        maxpool_1 = MaxPool2D(pool_size=(inputlen - filter_sizes[1] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_1)
        maxpool_2 = MaxPool2D(pool_size=(inputlen - filter_sizes[2] + 1, 1),
                              strides=(1, 1),
                              padding='valid')(conv_2)

        concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])

        if attention_enable == True:
            attention_output = Attention()([concatenated_tensor, concatenated_tensor])
            pool_output1 = GlobalMaxPooling2D()(concatenated_tensor)
            pool_output2 = GlobalMaxPooling2D()(attention_output)
            flatten = Concatenate()([pool_output1, pool_output2])
        else:
            flatten = Flatten()(concatenated_tensor)

        if drop_rate != 0:
            dropout = Dropout(drop_rate)(flatten)
        else:
            dropout = flatten

        dense = Dense(vector_dim, activity_regularizer=tf.python.keras.regularizers.l2(l2))(dropout)

        bn = LayerNormalization()(dense)

        if train == True:
            logits = ArcFace(n_classes=num_class)([bn, label])
            model = Model(inputs=[input, label], outputs=logits)
        else:
            if not mean_vects is None:
                label_dim = len(mean_vects)
                mean_vects = keras.initializers.constant(mean_vects)
                output = VectorDistance(label_dim=label_dim, vect_dim=vector_dim, mean_distances_init=mean_vects, name="output")(bn)
                output = tf.identity(output, name='output')
                model = Model(inputs=input, outputs=output)
            else:
                output = tf.identity(bn, name='output')
                model = Model(inputs=input, outputs=output)

        return model

