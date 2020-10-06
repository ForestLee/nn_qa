
import tensorflow as tf
import numpy as np


def get_embedding_from_ckpt(ckpt_path=None, ckpt_file=None, emb_name=None):
    if ckpt_path != None and ckpt_file != None and emb_name != None:
        emb=__get_tensor_array_from_ckpt(ckpt_path, ckpt_file, emb_name)
        np.save(ckpt_path+"emb.npy", emb)
        np.savetxt(ckpt_path+"emb.txt", emb)
        return emb
    else:
        print("get_embedding_from_ckpt fail")
        return None

def __get_tensor_array_from_ckpt(ckpt_path, ckpt_file, tensor_name):
    checkpoint=tf.train.load_checkpoint(ckpt_path+ckpt_file)
    tensor_array = checkpoint.get_tensor(tensor_name)
    print("from {} load tensor {}, shape {}".format(ckpt_path+ckpt_file, tensor_name, tensor_array.shape))
    return tensor_array

if __name__ == "__main__":
    default_ckpt_path = "/home/forest/QA/src/git/sentence-encoding-qa/data/warm_start_ckp/"
    default_ckpt_file = "model.ckpt-564188"
    default_emb_name = "task_independent/Variable_1"
    get_embedding_from_ckpt(ckpt_path=default_ckpt_path, ckpt_file=default_ckpt_file, emb_name=default_emb_name)
