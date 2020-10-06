import tensorflow as tf
from tensorflow.python import ops

from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import gfile


def unfreeze_pb(export_dir_path:str, freezed_model_file_path:str):
    input_graph_def = graph_pb2.GraphDef()
    with gfile.Open(freezed_model_file_path, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    with tf.Session() as session:
        tf.import_graph_def(input_graph_def, name="")
        tf.saved_model.simple_save(
            session,
            export_dir_path,
            inputs={"input:0": session.graph.get_tensor_by_name("input:0")},
            outputs={
                "output:0": session.graph.get_tensor_by_name("output:0")
            },
        )


if __name__ == "__main__":
    export_dir = "/home/lyj/workspace/sentence-encoding-qa/save/unfreezed_model"
    freezed_model_file = "/home/lyj/workspace/sentence-encoding-qa/save/qa_model_index_frozen/frozen_model.pb"

    input_graph_def = graph_pb2.GraphDef()
    with gfile.Open(freezed_model_file, "rb") as f:
        data = f.read()
        input_graph_def.ParseFromString(data)

    with tf.Session() as session:
        tf.import_graph_def(input_graph_def, name="")

        tf.saved_model.simple_save(
            session,
            export_dir,
            inputs={"input:0": session.graph.get_tensor_by_name("input:0")},
            outputs={
                "output:0": session.graph.get_tensor_by_name("output:0")
            },
        )
