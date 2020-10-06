
import tensorflow as tf
from tensorflow.python.keras import initializers
from tensorflow.keras.layers import Layer


class VectorDistance(Layer):
    def __init__(self, label_dim = 173, vect_dim=32, mean_distances_init=None, **kwargs):
        super(VectorDistance, self).__init__(**kwargs)
        self.label_dim = label_dim
        self.vect_dim = vect_dim
        self.type = type
        if mean_distances_init is None:
            self.mean_distances_init = None
        else:
            self.mean_distances_init = initializers.get(mean_distances_init)

    def build(self, input_shape):
        super(VectorDistance, self).build(input_shape[0])
        self.mean_distances = self.add_weight(shape=(self.label_dim, self.vect_dim),
                                              initializer=self.mean_distances_init, dtype=tf.float32,
                                              trainable=False, name="mean_distances")

        self.built = True

    def call(self, inputs):
        index, distances_arrays = self.__calc_distances_to_mean(inputs, self.mean_distances, type="cosine")

        return index


    def compute_output_shape(self, input_shape):

        return 1

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            "label_dim":self.label_dim,
            "vect_dim":self.vect_dim,
            "mean_distances_init":initializers.serialize(self.mean_distances_init)
        })
        return config

    # def __cosine_distance(self, x, y):
    #     x1=tf.reshape(x, (x.shape[1],))
    #     x2=tf.squeeze(x1)
    #     y2=tf.squeeze(y)
    #     x_norm = tf.sqrt(tf.reshape(tf.reduce_sum(tf.square(x2)), (1,)))
    #     y_norm = tf.sqrt(tf.reshape(tf.reduce_sum(tf.square(y2)), (1,)))
    #     m = tf.reduce_sum(tf.multiply(x, y))
    #     cos_sim = m / (x_norm * y_norm)
    #     return 1 - cos_sim

    def __cosine_distance(self, x, y):
        x_norm = tf.sqrt(tf.reshape(tf.reduce_sum(tf.square(x)), (1,)))
        y_norm = tf.sqrt(tf.reshape(tf.reduce_sum(tf.square(y)), (1,)))
        m = tf.reduce_sum(tf.multiply(x, y))
        cos_sim = m / (x_norm * y_norm)
        return 1 - cos_sim

    def __euclidean_distance(self, x, y):
        distance = tf.reduce_sum(tf.square(x - y), axis=-1, keepdims=True)
        return distance

    def __calc_distances_to_mean(self, vectors, mean_vector, type="cosine"):
        vector_dim = vectors.shape[1]
        label_dim = mean_vector.shape[0]
        distances_arrays = []
        for j in range(label_dim):
            mean_v = tf.slice(mean_vector, [j, 0], [1, vector_dim])
            if type=="cosine":
                distance = self.__cosine_distance(x=vectors, y=mean_v)
            elif type=="euclidean":
                distance = self.__euclidean_distance(x=vectors, y=mean_v)
            distances_arrays.append(distance)

        return tf.math.argmin(distances_arrays), distances_arrays

if __name__=='__main__':
    import datetime
    time1 = 0.0
    time_tot = 0.0

    time4 = 0.0
    time5 = 0.0

    def __euclidean_distance(x, y):
        global time4
        time4_start_t = datetime.datetime.now()
        distance = tf.reduce_sum(tf.square(x - y), axis=-1, keepdims=True)
        time4_end_t = datetime.datetime.now()
        time4 += (time4_end_t - time4_start_t).seconds * 1000000 + (time4_end_t - time4_start_t).microseconds
        return distance

    def __calc_euclidean_distances_to_mean(vectors, mean_vector):
        global time1, time_tot, time4
        vector_dim = vectors.shape[1]
        label_dim = mean_vector.shape[0]
        distances_arrays = []
        for j in range(label_dim):
            time1_start_t = datetime.datetime.now()
            mean_v = tf.slice(mean_vector, [j, 0], [1, vector_dim])
            time1_end_t = datetime.datetime.now()
            time1 += (time1_end_t - time1_start_t).seconds * 1000000 + (time1_end_t - time1_start_t).microseconds

            time_start_t = datetime.datetime.now()
            distance = __euclidean_distance(x=vectors, y=mean_v)
            time_end_t = datetime.datetime.now()
            time_tot += (time_end_t - time_start_t).seconds * 1000000 + (time_end_t - time_start_t).microseconds
            distances_arrays.append(distance)

        print("time1 {}".format(time1))
        print("tf.reduce_sum(tf.square(x - y): {}".format(time4))
        print("time_tot {}\n".format(time_tot))
        return tf.math.argmin(distances_arrays), distances_arrays



    def __cosine_distance(x, y):
        global time4, time5
        time4_start_t = datetime.datetime.now()
        x_norm = tf.sqrt(tf.reshape(tf.reduce_sum(tf.square(x)), (1,)))
        #x_norm = tf.keras.backend.sqrt(tf.reshape(tf.reduce_sum(tf.keras.backend.square(x)), (1,)))

        y_norm = tf.sqrt(tf.reshape(tf.reduce_sum(tf.square(y)), (1,)))
        #y_norm = tf.keras.backend.sqrt(tf.reshape(tf.reduce_sum(tf.keras.backend.square(y)), (1,)))

        time4_end_t = datetime.datetime.now()
        time4 += (time4_end_t - time4_start_t).seconds * 1000000 + (time4_end_t - time4_start_t).microseconds

        time5_start_t = datetime.datetime.now()
        m = tf.reduce_sum(tf.multiply(x, y))
        #m = tf.reduce_sum(tf.keras.layers.multiply([x, y]))

        cos_sim = m / (x_norm * y_norm)
        time5_end_t = datetime.datetime.now()
        time5 += (time5_end_t - time5_start_t).seconds * 1000000 + (time5_end_t - time5_start_t).microseconds
        return 1 - cos_sim




    def __calc_cosine_distances_to_mean(vectors, mean_vector):
        global time1, time_tot, time4, time5
        vector_dim = vectors.shape[1]
        label_dim = mean_vector.shape[0]
        distances_arrays = []
        for j in range(label_dim):
            time1_start_t = datetime.datetime.now()
            mean_v = tf.slice(mean_vector, [j, 0], [1, vector_dim])
            time1_end_t = datetime.datetime.now()
            time1 += (time1_end_t - time1_start_t).seconds * 1000000 + (time1_end_t - time1_start_t).microseconds

            time_start_t = datetime.datetime.now()
            distance = __cosine_distance(x=vectors, y=mean_v)
            time_end_t = datetime.datetime.now()
            time_tot += (time_end_t - time_start_t).seconds * 1000000 + (time_end_t - time_start_t).microseconds
            distances_arrays.append(distance)

        print("time1 {}".format(time1))
        print("tf.sqrt tf.reshape tf.reduce_sum tf.square: {} {}%".format(time4, int(100*time4/(time4+time5))))
        print("tf.reduce_sum tf.multiply & m / (x_norm * y_norm): {} {}%".format(time5, int(100*time5/(time4+time5))))
        print("time_tot {}\n".format(time_tot))
        return tf.math.argmin(distances_arrays), distances_arrays

    mean_vector = tf.random_normal([173,32], mean=0.0, stddev=1.0, dtype=tf.float32, seed=1000)
    vectors = tf.random_normal([1,32], mean=0.0, stddev=1.0, dtype=tf.float32, seed=1000)

    start_t = datetime.datetime.now()
    for i in range(100):
        index = __calc_euclidean_distances_to_mean(vectors, mean_vector)
    end_t = datetime.datetime.now()
    print("time {}".format((end_t - start_t).seconds * 1000000 + (end_t - start_t).microseconds))