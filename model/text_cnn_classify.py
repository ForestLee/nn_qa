from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Softmax, Attention, GlobalMaxPooling2D, Embedding, Concatenate, Reshape, MaxPool2D, Dense, Flatten, Input, Conv2D, Maximum
import tensorflow.keras as keras
from layers import RBFSoftmax, MaxoutDense


class TextCNNClassify(object):

    @staticmethod
    def model(inputlen, vocabulary, num_class, vector_dim,
              embedding_dim=128,
              num_filters=128,
              filter_sizes=[3, 4, 5],
              drop_rate=0.5,
              train=False,
              init_emb_enable=False,
              init_emb=None,
              attention_enable = False,
              rbf = False,
              maxout = False,
              maxout_num = 2):

        input = Input(shape=(inputlen))

        if init_emb_enable == False:
            embeddings_init = 'uniform'
        else:
            embeddings_init = keras.initializers.constant(init_emb)

        embedding = Embedding(input_dim=vocabulary,
                              output_dim=embedding_dim,
                              embeddings_initializer=embeddings_init,
                              input_length=inputlen,
                              name="embedding")(input)
        reshape = Reshape((inputlen, embedding_dim, 1))(embedding)

        # if train == True:
        #     reshape = Dropout(drop_rate)(reshape)

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
            concatenated_tensor = Concatenate()([pool_output1, pool_output2])

        flatten = Flatten()(concatenated_tensor)
        # if train == True:
        #     flatten = Dropout(drop_rate)(flatten)

        #dense_out = Dense(units=num_class, kernel_regularizer=keras.regularizers.l2(0.01), activity_regularizer=keras.regularizers.l1(0.01))(flatten)
        if rbf == True:
            dense_out = Dense(units=num_class, activation="tanh")(flatten)
            logits = RBFSoftmax(n_classes=num_class, feature_dim=num_class)(dense_out)
            logits = Softmax()(logits)
        elif maxout == True:
            flatten = MaxoutDense(num_class=num_class, nb_feature=maxout_num)(inputs=flatten)
            logits = Softmax()(flatten)
        else:
            logits = Dense(units=num_class, activation="softmax")(flatten)

        model = Model(input, logits)
        return model

