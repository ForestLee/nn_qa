"""
Definition of the ESIM model.
"""

from tensorflow.keras.layers import Input, BatchNormalization, SpatialDropout1D
from tensorflow.keras import Model
import tensorflow as tf
from layers.esim_layers import EmbeddingLayer, LocalInferenceLayer, InferenceCompositionLayer, PoolingLayer, MLPLayer, EncodingLayer


class ESIM(object):
    """
    ESIM model for Natural Language Inference (NLI) tasks.
    """

    @staticmethod
    def model(vocabulary=7000, embedding_dim = 128, n_classes=1, max_length=64, hidden_units=300,
                 dropout=0.5, train = True):
        embedding_weights = None

        a = Input(shape=(max_length,), dtype='int32', name='premise')
        b = Input(shape=(max_length,), dtype='int32', name='hypothesis')

        # ---------- Embedding layer ---------- #
        embedding = EmbeddingLayer(vocabulary, embedding_dim,
                                   embedding_weights,
                                   max_length=max_length)

        embedded_a = embedding(a)
        embedded_b = embedding(b)

        batchnorm = True
        if batchnorm == True:
            bn_embedded_a = BatchNormalization(axis=2)(embedded_a)
            bn_embedded_b = BatchNormalization(axis=2)(embedded_b)

            embedded_a = SpatialDropout1D(0.25)(bn_embedded_a)
            embedded_b = SpatialDropout1D(0.25)(bn_embedded_b)

        # ---------- Encoding layer ---------- #
        encoded_a = EncodingLayer(hidden_units,
                                  max_length,
                                  dropout=dropout)(embedded_a)
        encoded_b = EncodingLayer(hidden_units,
                                  max_length,
                                  dropout=dropout)(embedded_b)

        # ---------- Local inference layer ---------- #
        m_a, m_b = LocalInferenceLayer()([encoded_a, encoded_b])

        # ---------- Inference composition layer ---------- #
        composed_a = InferenceCompositionLayer(hidden_units,
                                               max_length,
                                               dropout=dropout)(m_a)
        composed_b = InferenceCompositionLayer(hidden_units,
                                               max_length,
                                               dropout=dropout)(m_b)

        # ---------- Pooling layer ---------- #
        pooled = PoolingLayer()([composed_a, composed_b])

        # ---------- Classification layer ---------- #
        prediction = MLPLayer(hidden_units, n_classes,
                              dropout=dropout)(pooled)

        model = Model(inputs=[a, b], outputs=prediction)
        # model.compile(optimizer=Adam(lr=self.learning_rate),     #learning_rate=0.0004,
        #               loss='categorical_crossentropy', metrics=['accuracy'])

        return model


if __name__ == '__main__':
    print("start")
    model = ESIM.model()
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    adam_optimizer = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-6, clipvalue=5)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer,
                       metrics=['binary_crossentropy', 'accuracy'])
    model.summary()
    print("done")
