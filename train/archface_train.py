
from train.train_base import TrainBase
from config import conf


if __name__ == '__main__':
	train = TrainBase(gan_enable=conf.GAN_ENABLE)
	train.prepare_data()
	train.build()
	train.do_train()
	train.post_test()