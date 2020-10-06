import os
import sys
from config import conf
from infer import InferTextMatchTest


class InferESIMTest(InferTextMatchTest):
    def __init__(self):
        super(InferTextMatchTest, self).__init__(backbone="ESIM", type="pair", header="value")



if __name__ == '__main__':
    conf.backbone = "ESIM"

    if len(sys.argv) > 1:
        model_file=sys.argv[1]
    else:
        model_file = "{}{}_{}.h5".format(conf.SAVE_DIR, conf.backbone, "2020-09-20-21-18-22")

    values_file = os.getcwd() + "/../save/esim_values_npy.npy"

    infer_test = InferESIMTest()
    infer_test.file_infer_test(model_file=model_file, values_file=values_file)