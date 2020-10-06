from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, Flatten, Input, Conv2D, Dropout, BatchNormalization, LayerNormalization
import tensorflow.keras as keras
import tensorflow as tf
from layers import MatchingLayer

class TextMatch(object):
    @staticmethod
    def model(vocabulary, embedding_dim=128, inputlen = 64, train=True, dropout=0):
        def _make_inputs():
            input_left = Input(
                name='text_left',
                shape=input_shapes[0]
            )
            input_right = Input(
                name='text_right',
                shape=input_shapes[1]
            )
            return [input_left, input_right]

        def _make_embedding_layer(
                name: str = 'embedding',
                **kwargs
        ):
            return Embedding(input_dim=vocabulary,
                             output_dim=embedding_dim,
                             trainable=train,
                             name=name,
                             **kwargs
                             )

        def _make_output_layer() -> keras.layers.Layer:
            #return Dense(num_label, activation='softmax')
            return Dense(num_label, activation='linear')

        def _conv_block(
                x,
                kernel_count: int,
                kernel_size: int,
                padding: str,
                activation: str
        ):
            output = Conv2D(kernel_count,
                            kernel_size,
                            padding=padding,
                            activation=activation)(x)
            return output

        input_shapes = [(inputlen,), (inputlen,)]
        num_blocks = 2
        kernel_count = [16, 32]
        kernel_size = [[3, 3], [3, 3]]
        activation = 'relu'
        padding = 'same'
        num_label = 1

        input_left, input_right = _make_inputs()

        embedding = _make_embedding_layer()
        embed_left = embedding(input_left)
        embed_right = embedding(input_right)

        # Interaction
        matching_layer = MatchingLayer(matching_type='dot')
        embed_cross = matching_layer([embed_left, embed_right])

        for i in range(num_blocks):
            embed_cross = _conv_block(
                embed_cross,
                kernel_count[i],
                kernel_size[i],
                padding,
                activation
            )

        embed_pool = tf.nn.max_pool(embed_cross,
                                    [1, inputlen - 2, inputlen - 9, 1],  # 21
                                    [1, 1, 1, 1],  # [1, max_len-2, max_len-9, 1],    #6
                                    "VALID",
                                    name="max_pooling")

        embed_flat = Flatten()(embed_pool)
        if dropout != 0:
            dropout_out = Dropout(dropout)(embed_flat)
        else:
            dropout_out = embed_flat

        inputs = [input_left, input_right]
        x_out = _make_output_layer()(dropout_out)
        x_out = BatchNormalization(name="bn_output")(x_out)
        #x_out = LayerNormalization(name="bn_output")(x_out)

        model = Model(inputs=inputs, outputs=x_out)
        return model




if __name__ == '__main__':
    print("start")
    model = TextMatch.model()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    model.summary()
    print("done")

