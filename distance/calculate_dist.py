
import numpy as np
from tensorflow.keras import backend as K
import math


def calc_distances_to_mean(vectors, mean_vector, type="cosine"):  #euclidean or cosine
    all_distances_arrays = []
    all_closest_label_idxs = []
    all_closest_distances = []
    vectors_len = len(vectors)

    for i in range(vectors_len):
        #print("{}/{}".format(i, vectors_len))
        full_distances_array = []
        min = -1
        min_label = ''
        for j in range(len(mean_vector)):
            #print("{}/{}: {}/{}".format(i, vectors_len, j, len(mean_vector)))
            distance = calc_two_vect_distance(type=type, x=vectors[i], y=mean_vector[j])
            full_distances_array.append(distance)
            if min == -1:
                min = distance
                min_label = j
            elif min > distance:
                min = distance
                min_label = j

        all_distances_arrays.append(full_distances_array)
        all_closest_label_idxs.append(min_label)
        all_closest_distances.append(min)
    return all_distances_arrays, all_closest_label_idxs, all_closest_distances


def calc_distances(vectors, labels):
    """
    calculate all distances between 2 vectors
    @param vectors:
    @param labels:
    @return:
            all_distances_arrays:   all distances between every 2 vectors
            all_closest_labels:     each vector has a closest vector, this is closest vector's label
            all_closest_distances:  the closest distance of each vector
    """
    all_distances_arrays = []
    all_closest_labels = []
    all_closest_distances = []
    vectors_len = len(vectors)
    for i in range(vectors_len):
        each_distances_array = []
        min = -1
        min_label = ''
        for j in range(len(vectors)):
            if i!=j:
                distance = calc_two_vect_distance(type="cosine", x=vectors[i], y=vectors[j])
                each_distances_array.append(distance)
                if min == -1:
                    min = distance
                    min_label = labels[j]
                if min > distance:
                    min = distance
                    min_label = labels[j]
            else:
                each_distances_array.append(0)

        all_distances_arrays.append(each_distances_array)
        all_closest_labels.append(min_label)
        all_closest_distances.append(min)

    # np.savetxt(model_file+".distances.csv", full_distances_arrays, delimiter=',')
    return all_distances_arrays, all_closest_labels, all_closest_distances

def calc_two_embeddings_distance(type, embeddings1, embeddings2):
    n = len(embeddings1)
    dists = []
    for i in range(n):
        dist = calc_two_vect_distance(type, embeddings1[i], embeddings2[i])
        dists.append(dist)
    return np.array(dists)

def calc_two_vect_distance(type=None, x=None, y=None):
    if type=="euclidean":
        return __euclidean_distance(x, y)
    elif type=="cosine":
        return __cosine_distance(x, y)

def __euclidean_distance(x, y):
    #distance = K.sum(K.square(x - y), axis=-1, keepdims=True)
    #distance = round(distance.numpy()[0], 3)

    distance = np.sum(np.square(x - y), axis=-1, keepdims=True)
    distance = round(distance[0], 3)
    return distance

def __cosine_distance(x, y):
    len_a = 0.0
    len_b = 0.0
    in_product=0.0
    for i in range(len(x)):
        len_a += math.pow(x[i], 2)
        len_b += math.pow(y[i], 2)
        in_product += x[i]*y[i]
    c=math.sqrt(len_a) * math.sqrt(len_b)
    c = max(c, 1e-07)
    cosine_distance = in_product * 1.0 / c
    distance = 1-cosine_distance
    distance = round(distance, 3)
    return distance

def __calc_distances_to_mean(vectors, labels_list, mean_vector):
    all_distances_arrays = []
    all_closest_labels = []
    all_closest_distances = []
    vectors_len = len(vectors)
    for i in range(vectors_len):
        full_distances_array = []
        min = -1
        min_label = ''
        for j in range(len(mean_vector)):
            distance = calc_two_vect_distance(type="euclidean", x=vectors[i], y=mean_vector[j])
            full_distances_array.append(distance)
            if min == -1:
                min = distance
                min_label = labels_list[j]
            elif min > distance:
                min = distance
                min_label = labels_list[j]

        all_distances_arrays.append(full_distances_array)
        all_closest_labels.append(min_label)
        all_closest_distances.append(min)

    # np.savetxt(model_file+".distances.csv", full_distances_arrays, delimiter=',')
    return all_distances_arrays, all_closest_labels, all_closest_distances