
import os
import tensorflow as tf
from config import conf
from model.model_wrapper import ModelWrapper
from train.train_base import TrainBase



class ClassTrain(TrainBase):

    def build(self, model_file=None):
        print('***************************build model***************************')
        self.model = ModelWrapper.model(conf, train=True, vocab_size=self.vocab_size, labels_num=self.labels_num)
        adam_optimizer = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-6, clipvalue=5)
        #self.model.compile(loss=[self.focal_loss(gamma=2., alpha=.25)], optimizer=self.optimizer, metrics=self.metrics)
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
        if not model_file is None:
            self.model.load_weights(model_file, by_name=True)
        self.model.summary()


    def post_test(self):
        print("***************************start infer test***************************")
        infer = "cd " + os.getcwd() + "/../infer;python " + os.getcwd() + "/../infer/infer_class.py " + self.save_name
        os.system(infer)



if __name__ == '__main__':
    train = ClassTrain(backbone="CNN_CLASS", gan_enable=conf.GAN_ENABLE)
    train.prepare_data()
    save_name = "{}{}_{}.h5".format(conf.SAVE_DIR, "CNN_CLASS", "2020-09-23-23-32-05")
    train.build()
    train.do_train()
    train.post_test()