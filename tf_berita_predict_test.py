import csv

import tensorflow as tf
import numpy as np
from nawanlp.feature.base_filter import BaseFilter
from nawanlp.feature.tokenizer.tokenizer import Tokenizer

from tensorflow.contrib.saved_model.python.saved_model import reader
from tensorflow.python.saved_model import loader, signature_def_utils

from dataset_builder import get_data, preprocessing


def create_flags():
    tf.flags.DEFINE_string('signature_def', "proba", '')
    tf.flags.DEFINE_string("saved_model", "model/export/1541665264", "Directory of SavedModel")
    tf.flags.DEFINE_string('label_file', "data/label.txt", '')
    tf.flags.DEFINE_string("tag", "serve", "SavedModel tag, serve|gpu")

FLAGS = tf.flags.FLAGS


def main(args):

    bf = BaseFilter()
    tokenizer = Tokenizer()

    data_test = get_data("data/test-10.csv")
    data_test = preprocessing(bf, tokenizer, data_test)

    label = [v.strip() for v in tf.gfile.Open(FLAGS.label_file).readlines()]


    tag = FLAGS.tag
    output_key = "scores"

    saved_model = reader.read_saved_model(FLAGS.saved_model)

    meta_graph = None
    for meta_graph_def in saved_model.meta_graphs:
        if FLAGS.tag in meta_graph_def.meta_info_def.tags:
            meta_graph = meta_graph_def
            break

    if meta_graph is None:
        raise ValueError("Cannot find saved_model with tag" + FLAGS.tag)


    # print(meta_graph)

    signature_def = meta_graph.signature_def[FLAGS.signature_def]

    print(signature_def.inputs["inputs"].name)

    output_tensor = signature_def.outputs[output_key].name

    with tf.Session() as sess:
        loader.load(sess, [tag], FLAGS.saved_model)

        example = []
        for data in data_test:
            text = [tf.compat.as_bytes(x) for x in data.get("tokens")]

            record = tf.train.Example()
            record.features.feature["text"].bytes_list.value.extend(text)


            example.append(record.SerializeToString())

            inputs_feed_dict = {
                signature_def.inputs["inputs"].name: [record.SerializeToString()],
            }

            outputs = sess.run(output_tensor,
                               feed_dict=inputs_feed_dict)

            index = np.argmax(outputs)
            print(" ".join(data.get("tokens")))
            print("benar" if label[index] == data.get("label") else "salah", "predict: ", label[index], "harusnya:", data.get("label"))
            print()



if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    create_flags()
    tf.app.run()
