
import tensorflow as tf
from config import conf
from train.train_base import TrainBase
from tensorflow.keras import callbacks
from train_callback.callbacks import SaveCallback
from tensorflow.keras.callbacks import TensorBoard

class BertClassTrain(TrainBase):
    def __init__(self, backbone = "BERT_CLASS", type="bert"):
        super(BertClassTrain, self).__init__(backbone=backbone, type=type)

    def do_train(self):
        print("***************************start training***************************")
        save_callback = SaveCallback(save_path=conf.SAVE_DIR, backbone=conf.backbone, model=self.model,
                                     timestamp=self.timestamp,
                                     save_name=self.save_name)  # , validation_data=[x_test, y_test])
        early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True,
                                                      patience=conf.early_stop_patience, verbose=1, mode='auto')
        reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=conf.reduce_lr_factor,
                                                                  patience=conf.reduce_lr_patience, verbose=1,
                                                                  mode='auto', epsilon=0.0001, cooldown=0,
                                                                  min_lr=0.00001)
        tensorboard_callback = TensorBoard(log_dir=conf.OUT_DIR)

        callbacks_list = []
        callbacks_list.append(save_callback)
        callbacks_list.append(early_stop_callback)
        callbacks_list.append(reduce_lr_callback)
        callbacks_list.append(tensorboard_callback)
        if conf.FIT_GENERATE == True:
            self.model.fit(self.fit_gen.generate(),
                           epochs=conf.epochs,
                           steps_per_epoch=self.corpus_size / conf.batch_size,
                           callbacks=callbacks_list,
                           validation_data=([self.x_test[0], self.x_test[1], self.y_test], self.y_test),
                           verbose=1)
        else:
            self.model.fit(x=[self.x_train[0], self.x_train[1], self.y_train],
                           y=self.y_train, batch_size=conf.batch_size,
                           epochs=conf.epochs,
                           callbacks=callbacks_list,
                           validation_data=([self.x_test[0], self.x_test[1], self.y_test], self.y_test),  # validation_split=0.02,
                           verbose=1)

if __name__ == '__main__':
    train = BertClassTrain()
    train.prepare_data()
    save_name = "{}{}_{}.h5".format(conf.SAVE_DIR, "BERT_CLASS", "2020-09-23-23-32-05")
    train.build()
    train.do_train()
    train.post_test()