
"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from sklearn.model_selection import KFold
import datetime
from matplotlib import pyplot as plt
plt.switch_backend('agg')
from PIL import Image
import io
import os

def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, dist_fun=None, type=""):
    assert (embeddings1.shape[0] == embeddings2.shape[0])
    assert (embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    if dist_fun is None:
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff), 1)
    else:
        dist = dist_fun(type, embeddings1, embeddings2)

    #t1 = datetime.datetime.now()
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        #t2 = datetime.datetime.now()
        #print("fold_idx {} time {} seconds".format(fold_idx, (t2 - t1).seconds))
                                                            
        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy(threshold, dist[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy(threshold,
                                                                                                 dist[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy(thresholds[best_threshold_index], dist[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds

def calculate_roc_value(thresholds, values, actual_issame, nrof_folds=10):

    nrof_pairs = min(len(actual_issame), values.shape[0])
    nrof_thresholds = len(thresholds)
    k_fold = KFold(n_splits=nrof_folds, shuffle=False)

    tprs = np.zeros((nrof_folds, nrof_thresholds))
    fprs = np.zeros((nrof_folds, nrof_thresholds))
    accuracy = np.zeros((nrof_folds))
    best_thresholds = np.zeros((nrof_folds))
    indices = np.arange(nrof_pairs)

    #t1 = datetime.datetime.now()
    for fold_idx, (train_set, test_set) in enumerate(k_fold.split(indices)):
        #t2 = datetime.datetime.now()
        #print("fold_idx {} time {} seconds".format(fold_idx, (t2 - t1).seconds))

        acc_train = np.zeros((nrof_thresholds))
        for threshold_idx, threshold in enumerate(thresholds):
            _, _, acc_train[threshold_idx] = calculate_accuracy_value(threshold, values[train_set], actual_issame[train_set])
        best_threshold_index = np.argmax(acc_train)
        best_thresholds[fold_idx] = thresholds[best_threshold_index]
        for threshold_idx, threshold in enumerate(thresholds):
            tprs[fold_idx, threshold_idx], fprs[fold_idx, threshold_idx], _ = calculate_accuracy_value(threshold,
                                                                                                 values[test_set],
                                                                                                 actual_issame[
                                                                                                     test_set])
        _, _, accuracy[fold_idx] = calculate_accuracy_value(thresholds[best_threshold_index], values[test_set],
                                                      actual_issame[test_set])

    tpr = np.mean(tprs, 0)
    fpr = np.mean(fprs, 0)
    return tpr, fpr, accuracy, best_thresholds

def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / dist.size
    return tpr, fpr, acc

def calculate_accuracy_value(threshold, values, actual_issame):
    predict_issame = np.greater(values, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    tpr = 0 if (tp + fn == 0) else float(tp) / float(tp + fn)
    fpr = 0 if (fp + tn == 0) else float(fp) / float(fp + tn)
    acc = float(tp + tn) / values.size
    return tpr, fpr, acc

def evaluate_best_threshold(embeddings1, embeddings2, issame, nrof_folds=10, dist_fun=None, type=""):
    """

    @param embeddings1:
    @param embeddings2:
    @param issame:
    @param nrof_folds:
    @param dist_fun:
    @param type: euclidean or cosine
    @return:
    """
    # Calculate evaluation metrics
    if type == "euclidean":
        thresholds = np.arange(0, 6, 0.01)
    elif  type == "cosine":
        thresholds = np.arange(0, 1, 0.005)

    tpr, fpr, accuracy, best_thresholds = calculate_roc(thresholds, embeddings1, embeddings2,
                                                        np.asarray(issame), nrof_folds=nrof_folds, dist_fun=dist_fun,
                                                        type=type)

    buf = gen_plot(fpr, tpr)
    roc_curve = Image.open(buf)
    # print("\ntpr:")
    # print(tpr)
    # print("\nfpr:")
    # print(fpr)
    # print("\naccuracy:")
    # print(accuracy)
    # print("\nbest_thresholds:")
    # print(best_thresholds)
    np.savetxt(os.getcwd()+"/../log/tpr.txt", tpr)
    np.savetxt(os.getcwd() + "/../log/fpr.txt", fpr)

    return tpr, fpr, accuracy, best_thresholds

def evaluate_best_threshold_value(values, issame, nrof_folds=10):
    thresholds = np.arange(0, 1, 0.005)

    tpr, fpr, accuracy, best_thresholds = calculate_roc_value(thresholds, values,
                                                        np.asarray(issame), nrof_folds=nrof_folds)

    np.savetxt(os.getcwd()+"/../log/tpr.txt", tpr)
    np.savetxt(os.getcwd() + "/../log/fpr.txt", fpr)

    return tpr, fpr, accuracy, best_thresholds

