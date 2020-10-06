

from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras import Model
from tensorflow.keras.layers import Attention, GlobalMaxPooling1D, Concatenate, Embedding, Dropout, Reshape, Dense, Input, BatchNormalization
from tensorflow.keras import regularizers
from layers import ArcFace
import tensorflow.keras as keras

class LstmArchface(object):

    @staticmethod
    def model(inputlen, vocabulary, vector_dim,
              embedding_dim=128,
              lstm_unit=100,
              num_class = 0,
              drop_rate=0.5,
              l2=0.01,
              train = False,
              init_emb_enable=False,
              init_emb=None,
              attention_enable=False):

        input = Input(shape=(inputlen))
        if train == True:
            label = Input(shape=(num_class))

        if init_emb_enable == False:
            embeddings_init = 'uniform'
        else:
            embeddings_init = keras.initializers.constant(init_emb)

        embedding = Embedding(input_dim=vocabulary,
                              output_dim=embedding_dim,
                              embeddings_initializer=embeddings_init,
                              input_length=inputlen)(input)
        reshape = Reshape((inputlen, embedding_dim))(embedding)

        if attention_enable == True:
            lstm = Bidirectional(LSTM(lstm_unit, return_sequences=True))(reshape)
            attention_output = Attention()([lstm, lstm])
            pool_output1 = GlobalMaxPooling1D()(lstm)
            pool_output2 = GlobalMaxPooling1D()(attention_output)
            lstm = Concatenate()([pool_output1, pool_output2])
        else:
            lstm = Bidirectional(LSTM(lstm_unit, return_sequences=False))(reshape)

        if drop_rate != 0:
            dropout = Dropout(drop_rate)(lstm)
        else:
            dropout = lstm
        dense = Dense(vector_dim, activity_regularizer=regularizers.l2(l2))(dropout)

        bn = BatchNormalization()(dense)

        if train == True:
            output = ArcFace(n_classes=num_class)([bn, label])
            model = Model([input, label], output)
        else:
            model = Model(input, bn)

        return model
