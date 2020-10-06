import tensorflow as tf
from tensorflow.keras.layers import Layer


class RBFSoftmax(Layer):
    def __init__(self, scale=15, gamma=1, feature_dim=32, n_classes=173, **kwargs):
        super(RBFSoftmax, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.scale = scale
        self.gamma = gamma
        self.feature_dim = feature_dim

    def build(self, input_shape):
        super(RBFSoftmax, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(self.n_classes, self.feature_dim),
                                 initializer='glorot_uniform',
                                 trainable=True
                                 )

    def call(self, inputs):
        #W [1, 173, 173]                   input[1, 1, 173]
        diff = tf.expand_dims(self.W, 0) - tf.expand_dims(inputs, 1)  #[1, 173, 173]
        diff1 = tf.multiply(diff, diff)     #[1, 173, 173]
        metric = tf.reduce_sum(diff1, axis=-1)     #[1, 173]
        kernal_metric = tf.exp(-1.0 * metric / self.gamma)
        train_logits = self.scale * kernal_metric
        return train_logits

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "n_classes": self.n_classes,
            "gamma": self.gamma,
            "scale": self.scale,

        })
        return config


if __name__ == '__main__':


    print("exit")
    # with tf.Session() as sess:
    #     gamma = 1
    #     scale = 4
    #     W = tf.random.normal([173,32], mean=0, stddev=1)
    #     W=tf.expand_dims(W, 0)
    #     inputs = tf.random.normal([1, 32], mean=0, stddev=1)
    #     inputs1= tf.expand_dims(inputs, 1)
    #     diff = inputs1 - W
    #     diff1 = tf.multiply(diff, diff)
    #     metric = tf.reduce_sum(diff1, axis=-1)
    #     kernal_metric = tf.exp(-1.0 * metric / gamma)
    #     train_logits = scale * kernal_metric
    #     print("result:")
    #     print(sess.run(train_logits))
    #     print("done")

    with tf.Session() as sess:
        gamma = 1
        scale = 4
        W = tf.random.normal([173,173], mean=0, stddev=1)
        W=tf.expand_dims(W, 0)    #[1, 173, 32]
        inputs = tf.random.normal([1, 173], mean=0, stddev=1)
        inputs= tf.expand_dims(inputs, -1)    #[1, 173, 1]
        diff = inputs - W
        diff1 = tf.multiply(diff, diff)
        metric = tf.reduce_sum(diff1, axis=-1)
        kernal_metric = tf.exp(-1.0 * metric / gamma)
        train_logits = scale * kernal_metric
        print("result:")
        print(sess.run(train_logits))
        print("done")
