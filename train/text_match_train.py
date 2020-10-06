
from fit_generator.fit_generator_wrapper import FitGeneratorWrapper
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import callbacks

from config import conf
from model.model_wrapper import ModelWrapper
from train_callback.callbacks import SaveCallback
from train.train_base import TrainBase
import tensorflow as tf

class TextMatchTrain(TrainBase):
    def __init__(self, backbone = "TEXT_MATCH", type="pair"):
        super(TextMatchTrain, self).__init__(backbone=backbone, type=type)

    def prepare_data(self):
        print('***************************prepare corpus***************************')
        self.fit_gen = FitGeneratorWrapper(type=self.type, file_vocab=conf.VOCAB_FILE, file_corpus=conf.PAIR_TRAIN_FILE,
                                           batch_size=conf.batch_size, max_len=conf.maxlen, vector_dim=conf.vector_dim,
                                           ner_vocab=conf.pretrain_vocab, label_dict_file=conf.LABEL_DICT_FILE)
        self.vocab_size = self.fit_gen.get_vocab_size()
        self.orpus_size = self.fit_gen.get_line_count()

        self.x_test1, self.x_test2, self.y_test = self.fit_gen.read_corpus(file_corpus=conf.PAIR_TEST_FILE)
        if conf.FIT_GENERATE == False:
            self.x_train1, self.x_train2, self.y_train = self.fit_gen.read_corpus(file_corpus=conf.PAIR_TEST_FILE)

    def build(self):
        print('***************************build model***************************')
        self.model = ModelWrapper.model(conf, train=True, vocab_size=self.vocab_size, labels_num=1)

        self.model.compile('adam', 'mse', metrics=['accuracy'])

        self.model.summary()

    def do_train(self):
        print("***************************start training***************************")
        save_callback = SaveCallback(save_path=conf.SAVE_DIR, backbone=conf.backbone, model=self.model,
                                     timestamp=self.timestamp,
                                     save_name=self.save_name)  # , validation_data=[x_test, y_test])
        early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', restore_best_weights=True,
                                                      patience=conf.early_stop_patience, verbose=1, mode='auto')
        reduce_lr_callback = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=conf.reduce_lr_factor,
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
                           validation_data=([self.x_test1, self.x_test2], self.y_test),
                           verbose=1)
        else:
            self.model.fit(x=[self.x_train1, self.x_train2],
                           y=self.y_train,
                           batch_size=conf.batch_size,
                           epochs=conf.epochs,
                           callbacks=callbacks_list,
                           validation_data=([self.x_test1, self.x_test2], self.y_test),
                           verbose=1)

    def post_test(self):
        print("no test")

if __name__ == '__main__':
	train = TextMatchTrain()
	train.prepare_data()
	train.build()
	train.do_train()
	train.post_test()