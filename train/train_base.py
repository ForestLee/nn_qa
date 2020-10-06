import os
import sys
import time
import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import callbacks
import tensorflow as tf
from model.model_wrapper import ModelWrapper
from config import conf
from fit_generator.fit_generator_wrapper import FitGeneratorWrapper
from train_callback.callbacks import SaveCallback

from model.bert4keras.backend import search_layer


class TrainBase(object):
	def __init__(self, backbone = "CNN_ARCFACE", type="class", loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"], gan_enable=False):
		conf.backbone=backbone
		self.type=type
		if gan_enable == True:
			self.loss= self.gan_loss_with_gradient_penalty
		else:
			self.loss=loss
		self.optimizer=optimizer
		self.metrics=metrics

		print(os.getcwd() + "/../")
		sys.path.append(os.getcwd() + "/../")

		if len(sys.argv) == 2 and sys.argv[1] == "gpu":
			if tf.test.is_gpu_available():
				os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 0 for V100, 1 for P100
		else:
			os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

		self.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
		if conf.attention_enable == False:
			self.save_name = "{}{}_{}.h5".format(conf.SAVE_DIR, conf.backbone, self.timestamp)
		else:
			self.save_name = "{}{}_{}_{}.h5".format(conf.SAVE_DIR, conf.backbone, "ATTENTION", self.timestamp)

	def focal_loss(self, gamma=2., alpha=1 ):
		def focal_loss_fixed(y_true, y_pred):
			pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
			pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
			return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
		return focal_loss_fixed

	def gan_loss_with_gradient_penalty(self, y_true, y_pred, epsilon=2):
		"""带梯度惩罚的loss
        """
		loss = K.mean(K.categorical_crossentropy(y_true, y_pred))
		# 查找Embedding层
		for output in self.model.outputs:
			embedding_layer = search_layer(output, "embedding")
			if embedding_layer is not None:
				break
		if embedding_layer is None:
			raise Exception('Embedding layer not found')

		embeddings = embedding_layer.embeddings
		gp = K.sum(K.gradients(loss, [embeddings])[0].values ** 2)
		return loss + 0.5 * epsilon * gp


	def prepare_data(self):
		print('***************************prepare corpus***************************')
		self.fit_gen = FitGeneratorWrapper(type=self.type, file_vocab=conf.VOCAB_FILE, file_corpus=conf.TRAIN_FILE,
                                           batch_size=conf.batch_size, max_len=conf.maxlen, vector_dim=conf.vector_dim,
                                           ner_vocab=conf.pretrain_vocab, label_dict_file=conf.LABEL_DICT_FILE)
		self.vocab_size = self.fit_gen.get_vocab_size()
		self.corpus_size = self.fit_gen.get_line_count()
		self.labels_num = self.fit_gen.get_label_count()
		self.x_test, self.y_test = self.fit_gen.read_corpus(file_corpus=conf.TEST_FILE)
		if conf.FIT_GENERATE == False:
			self.x_train, self.y_train = self.fit_gen.read_corpus(file_corpus=conf.TRAIN_FILE)


	def build(self):
		print('***************************build model***************************')
		self.model = ModelWrapper.model(conf, train=True, vocab_size=self.vocab_size, labels_num=self.labels_num)
		self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)  # optimizer=Adam()  keras.optimizers.Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
		# model.compile(loss=focal_loss(), optimizer='adam', metrics=["accuracy"])   #optimizer=Adam()  keras.optimizers.Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

		self.model.summary()

	def do_train(self):
		print("***************************start training***************************")
		save_callback = SaveCallback(save_path=conf.SAVE_DIR, backbone=conf.backbone, model=self.model,
									 timestamp=self.timestamp, save_name=self.save_name)  # , validation_data=[x_test, y_test])
		early_stop_callback = callbacks.EarlyStopping(monitor='val_loss', patience=conf.early_stop_patience, verbose=1, mode='auto', restore_best_weights=True)
		reduce_lr_callback = callbacks.ReduceLROnPlateau(monitor='val_acc', factor=conf.reduce_lr_factor, patience=conf.reduce_lr_patience, verbose=1,
														 mode='auto', epsilon=0.0001, cooldown=0, min_lr=0.00001)
		tensorboard_callback = TensorBoard(log_dir=conf.OUT_DIR)

		callbacks_list = []
		callbacks_list.append(save_callback)
		callbacks_list.append(early_stop_callback)
		#callbacks_list.append(reduce_lr_callback)
		callbacks_list.append(tensorboard_callback)

		if conf.FIT_GENERATE == True:
			self.model.fit(self.fit_gen.generate(),
						   epochs=conf.epochs,
						   steps_per_epoch=self.corpus_size / conf.batch_size,
					  	   callbacks=callbacks_list,
					  	   validation_data=([self.x_test, self.y_test], self.y_test), verbose=1)
		else:
			self.model.fit(x=[self.x_train,
							  self.y_train],
						   	  y=self.y_train,
						      batch_size=conf.batch_size,
						      epochs=conf.epochs,
					  		  callbacks=callbacks_list,
					          validation_data=([self.x_test, self.y_test], self.y_test),  # validation_split=0.02,
					          verbose=1)
		print("***************************train done***************************")

	def post_test(self):
		print("***************************start infer test***************************")
		infer = "cd " + os.getcwd() + "/../infer;python " + os.getcwd() + "/../infer/infer_test.py " + self.save_name
		os.system(infer)


	def save(self):
		print("save to {}".format(self.save_name))
		if not os.path.exists(conf.SAVE_DIR):
			os.mkdir(conf.SAVE_DIR)
		self.model.save(self.save_name)

		print("***************************save done***************************")

if __name__ == '__main__':
	train = TrainBase(type="class", loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
	train.prepare_data()
	train.build()
	train.do_train()
	train.post_test()