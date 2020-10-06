
# maxout 网络层类的定义
from tensorflow.keras.layers import Dense
import tensorflow as tf

class MaxoutDense(object):

    def __init__(self, num_class, nb_feature=2):
        self.num_class = num_class
        self.nb_feature = nb_feature
        self.denses = []
        for i in range(self.nb_feature):
            dense = Dense(units=self.num_class, name="maxout_dense_{}".format(i))
            self.denses.append(dense)


    def __call__(self, inputs):
        axis = 2
        dense_outs = []
        for i in range(self.nb_feature):
            dense_out = self.denses[i](inputs)
            dense_outs.append(dense_out)
        concatenated_tensor = tf.stack(dense_outs, axis=axis, name="maxout_stack")

        shape = concatenated_tensor.get_shape().as_list()
        if shape[0] is None:
            shape[0] = -1

        num_channels = shape[axis]
        if num_channels % 1:
            raise ValueError('number of features({}) is not '
                             'a multiple of num_class({})'.format(num_channels, 1))
        shape[axis] = 1
        shape += [num_channels // 1]
        outputs = tf.reduce_max(tf.reshape(concatenated_tensor, shape, name="maxout_reshape"), -1, keep_dims=False, name="maxout_reduce_max")
        outputs = tf.squeeze(outputs, axis=2)

        return outputs

# if __name__ == '__main__':
#     import numpy as np
#
#     def max_out(inputs, num_units, axis=None):
#         shape = inputs.get_shape().as_list()
#         if shape[0] is None:
#             shape[0] = -1
#         if axis is None:  # Assume that channel is the last dimension
#             axis = -1
#         num_channels = shape[axis]
#         if num_channels % num_units:
#             raise ValueError('number of features({}) is not '
#                              'a multiple of num_units({})'.format(num_channels, num_units))
#         shape[axis] = num_units
#         shape += [num_channels // num_units]
#         outputs = tf.reduce_max(tf.reshape(inputs, shape), -1, keep_dims=False)
#         return outputs
#
#     def kdense(kdim, num_class, inputs):
#         dense_out1 = Dense(units=num_class)(inputs)
#         dense_out2 = Dense(units=num_class)(inputs)
#         #concatenated_tensor = tf.stack([dense_out1, dense_out2], axis=2)
#         xx=[]
#         xx.append(dense_out1)
#         xx.append(dense_out2)
#         concatenated_tensor = tf.stack(xx, axis=2)
#         return dense_out1, dense_out2, concatenated_tensor
#
#     with tf.Session() as sess:
#         inputs = np.random.uniform(size=(1, 384))
#         #inputs = tf.random_normal([-1, 10], name="g1")
#         #shape = inputs.get_shape().as_list()
#         dense_out1, dense_out2, concatenated_tensor = kdense(2, 173, inputs)
#         maxout = max_out(concatenated_tensor, 1, axis=2)
#         maxout = tf.squeeze(maxout, axis=2)
#         sess.run(tf.global_variables_initializer())
#         out1 = dense_out1.eval()
#         out2 = dense_out2.eval()
#         concat = concatenated_tensor.eval()
#         maxout_value = maxout.eval()
#         print("out1")
#         print(out1)
#         print("out2")
#         print(out2)
#         print("concat")
#         print(concat)
#         print("maxout_value")
#         print(maxout_value)
#         print("done")