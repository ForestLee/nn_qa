
import tensorflow as tf
from train.text_match_train import TextMatchTrain
from config import conf
from model.model_wrapper import ModelWrapper


class ESIMTrain(TextMatchTrain):
	def __init__(self, backbone="ESIM", type="pair"):
		super(ESIMTrain, self).__init__(backbone=backbone, type=type)

	def build(self):
		print('***************************build model***************************')
		self.model = ModelWrapper.model(conf, train=True, vocab_size=self.vocab_size, labels_num=1)
		adam_optimizer = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-6, clipvalue=5)
		self.model.compile(loss='binary_crossentropy', optimizer=adam_optimizer,
						   metrics=['binary_crossentropy', 'accuracy'])

		self.model.summary()


if __name__ == '__main__':
	train = ESIMTrain()
	train.prepare_data()
	train.build()
	train.do_train()
	train.post_test()
