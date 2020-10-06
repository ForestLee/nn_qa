from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
from tensorflow.keras.callbacks import Callback
from sklearn import metrics
import numpy as np

class SaveCallback(Callback):

    def __init__(self, save_path=None, backbone= None, model = None, validation_data=None, timestamp="", save_name=""):
        self.save_path = save_path
        self.backbone = backbone
        self.model = model
        self.save_name = save_name
        if timestamp != "":
            self.timestamp = timestamp
        else:
            self.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        if not validation_data is None:
            self.validation_data = validation_data
            y_test = self.validation_data[1]
            self.val_test = list(map(lambda y:np.argmax(y), [y for y in y_test]))
        else:
            self.validation_data = None



    def __valid(self):
        if not self.validation_data is None:
            x_test = self.validation_data[0]
            y_test = self.validation_data[1]
            y_predict = self.model.predict([x_test, y_test])
            val_predict= list(map(lambda y: np.argmax(y), [y for y in y_predict]))
            precise = metrics.accuracy_score(self.val_test, val_predict)
            f1_score = metrics.f1_score(self.val_test, val_predict, average="weighted")
            return round(precise,3), round(f1_score,3)

    def on_epoch_end(self, epoch, logs=None):
        if not self.validation_data is None:
            precise, f1_score = self.__valid()

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.model.save(self.save_name)
        if not self.validation_data is None:
            print("\nsave to {} precise={} f1={}".format(self.save_name, precise, f1_score))
        else:
            print("\nsave to {}".format(self.save_name))
