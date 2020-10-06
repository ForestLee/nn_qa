
import time
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import callbacks

from config import conf
from model.model_wrapper import ModelWrapper
from train_callback.callbacks import SaveCallback
from train.train_base import TrainBase

class BertArcfaceTrain(TrainBase):
    def __init__(self, backbone = "BERT_ARCFACE", type="bert"):
        super(BertArcfaceTrain, self).__init__(backbone=backbone, type=type)

    def build(self):
        print('***************************build model***************************')
        self.model = ModelWrapper.model(conf, train=True, vocab_size=self.vocab_size, labels_num=self.labels_num)

        # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])   #optimizer=Adam()  keras.optimizers.Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
        self.model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
                      metrics=["accuracy"])
        self.model.summary()


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
            # tensorflow 2.0
            # model.fit(x=[x_train, y_train], y=y_train, batch_size=conf.batch_size, epochs=conf.epochs,
            #          callbacks=[TensorBoard(log_dir=conf.OUT_DIR), save_callback], validation_data=([x_test, y_test], y_test), #validation_split=0.02,
            #          verbose=1)
            self.model.fit(x=[self.x_train[0], self.x_train[1], self.y_train],
                           y=self.y_train, batch_size=conf.batch_size,
                           epochs=conf.epochs,
                           callbacks=callbacks_list,
                           validation_data=([self.x_test[0], self.x_test[1], self.y_test], self.y_test),  # validation_split=0.02,
                           verbose=1)

if __name__ == '__main__':
	train = BertArcfaceTrain()
	train.prepare_data()
	train.build()
	train.do_train()
	train.post_test()