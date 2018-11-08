import tensorflow as tf
from tensorflow import SparseTensor


def create_flags():
    tf.flags.DEFINE_string('vocab_file', "data/vocab.txt", '')
    tf.flags.DEFINE_string('label_file', "data/label.txt", '')

    tf.flags.DEFINE_string('data_path_train', "data/train.tfrecords", '')
    tf.flags.DEFINE_string('data_path_dev', "data/dev.tfrecords", '')

    tf.flags.DEFINE_string('model_dir', 'model', '')
    tf.flags.DEFINE_string('export_dir', 'model/export', '')
    tf.flags.DEFINE_integer('epochs', 10, '')
    tf.flags.DEFINE_integer('embedding_dims', 100, '')
    tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for training")

FLAGS = tf.flags.FLAGS

def parser(serialized_example):

    feature = {
        'text': tf.VarLenFeature(tf.string),
        'label': tf.VarLenFeature(tf.string),
    }

    features = tf.parse_single_example(serialized_example, features=feature)

    return {"text": features["text"].values}, features["label"].values[0]


def Exports(probs, embedding):
    exports = {
        "proba": tf.estimator.export.ClassificationOutput(scores=probs),
        "embedding": tf.estimator.export.RegressionOutput(value=embedding),
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: \
            tf.estimator.export.ClassificationOutput(scores=probs),
    }
    return exports

def model_fn(features, labels, mode, params):
    # print(features["text"])
    if type(features["text"]) == SparseTensor:
        features["text"] = tf.sparse_tensor_to_dense(features["text"],
                                                     default_value=" ")
    text_lookup_table = tf.contrib.lookup.index_table_from_file(FLAGS.vocab_file, default_value=0)
    text_ids = text_lookup_table.lookup(features["text"])

    text_embedding_w = tf.Variable(tf.random_uniform([params.get("total_vocab"), FLAGS.embedding_dims], -0.1, 0.1))
    text_embedding = tf.reduce_mean(tf.nn.embedding_lookup(text_embedding_w, text_ids), axis=-2, name="text_embedding")
    input_layer = text_embedding

    num_classes = params.get("num_classes")
    logits = tf.contrib.layers.fully_connected(
        inputs=input_layer, num_outputs=num_classes,
        activation_fn=None)

    predictions = tf.argmax(logits, axis=-1)
    probs = tf.nn.softmax(logits)
    loss, train_op = None, None

    metrics = {}
    if mode != tf.estimator.ModeKeys.PREDICT:
        label_lookup_table = tf.contrib.lookup.index_table_from_file(
            FLAGS.label_file, vocab_size=params.get("num_classes"))
        labels = label_lookup_table.lookup(labels)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits))
        opt = tf.train.AdamOptimizer(params["learning_rate"])

        train_op = opt.minimize(loss, global_step=tf.train.get_global_step())
        metrics = {
            "accuracy": tf.metrics.accuracy(labels, predictions)
        }

    exports = {}

    if FLAGS.export_dir:
        exports = Exports(probs, text_embedding)

    return tf.estimator.EstimatorSpec(
        mode, predictions=predictions, loss=loss, train_op=train_op,
        eval_metric_ops=metrics, export_outputs=exports)


def input_fn():

    ds = tf.data.TFRecordDataset([FLAGS.data_path_train])
    ds = ds.map(parser)
    ds = ds.repeat(FLAGS.epochs)
    ds = ds.batch(1)
    return ds

def eval_input():

    ds = tf.data.TFRecordDataset([FLAGS.data_path_dev])
    ds = ds.map(parser)
    ds = ds.batch(1)
    return ds


def main(args):

    label = [v.strip() for v in tf.gfile.Open(FLAGS.label_file).readlines()]
    vocab = [v.strip() for v in tf.gfile.Open(FLAGS.vocab_file).readlines()]
    params = {
        "learning_rate": FLAGS.learning_rate,
        "num_classes": len(label),
        "total_vocab": len(vocab),
    }


    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=params)

    print("STARTING TRAIN")
    estimator.train(input_fn=input_fn)
    print("TRAIN COMPLETE")

    print("EVALUATE")
    result = estimator.evaluate(input_fn=eval_input)
    print(result, "\n")

    if FLAGS.export_dir:
        print("EXPORTING")
        parse_spec = {"text": tf.VarLenFeature(tf.string)}
        serving_input_receiver_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(parse_spec)

        estimator.export_savedmodel(FLAGS.export_dir, serving_input_receiver_fn=serving_input_receiver_fn)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    create_flags()
    tf.app.run()
