from nawanlp.feature.base_filter import BaseFilter
from nawanlp.feature.tokenizer.tokenizer import Tokenizer

import tensorflow as tf
import csv

LABEL_X = "text"
LABEL_Y = "category"

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def get_data(file_loc):

    with open(file_loc) as f:
        reader = csv.DictReader(f)
        data = [dict(d) for d in reader]
        return data

def preprocessing(bf, tokenizer, data):
    results = []
    for d in data:

        text_clear = bf.clear_all_native_str(d.get(LABEL_X))

        tokens = tokenizer._split_all(text_clear)
        results.append({
            "tokens": tokens,
            "label": bf.clear_all_native_str(d.get(LABEL_Y)),
        })

    return results

def get_vocab(data, field):
    vocab = {}

    for d in data:
        for t in d.get(field):
            if not vocab.get(t):
                vocab[t] = 1
            else:
                vocab[t] += 1

    return vocab

def get_label(data, field):
    vocab = {}

    for d in data:
        t = d.get(field)
        if not vocab.get(t):
            vocab[t] = 1
        else:
            vocab[t] += 1

    return vocab

def save_vocab(vocab):
    with open("data/vocab.txt", "w") as f:
        f.write("<unk>") # unkown word

        for word in vocab:
            f.write("\n")
            f.write(word)

def save_label(labels):

    with open("data/label.txt", "w") as f:
        f.write("\n".join(labels))

def write_tfrecord(filename, data):
    writer = tf.python_io.TFRecordWriter(filename)

    for d in data:
        feature = {
            "text": _bytes_feature([w.encode() for w in d.get("tokens")]),
            "label": _bytes_feature([d.get("label").encode()])
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()

if __name__ == '__main__':
    train_file = "data/train-100.csv"
    dev_file = "data/dev-10.csv"
    test_file = "data/test-10.csv"

    data_train_ori = get_data(train_file)
    data_dev_ori = get_data(dev_file)
    data_test_ori = get_data(test_file)


    bf = BaseFilter()
    tokenizer = Tokenizer()

    data_train_pre = preprocessing(bf, tokenizer, data_train_ori)
    data_dev_pre = preprocessing(bf, tokenizer, data_dev_ori)
    # data_test_pre = preprocessing(bf, tokenizer, data_test_ori)

    write_tfrecord("data/train.tfrecords", data_train_pre)
    write_tfrecord("data/dev.tfrecords", data_dev_pre)
    # write_tfrecord("test.tfrecords", data_test_pre)

    vocab = [word for word in get_vocab(data_train_pre, "tokens")]
    save_vocab(vocab)

    labels = [word for word in get_label(data_train_pre, "label")]
    save_label(labels)
