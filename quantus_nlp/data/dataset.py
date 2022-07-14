import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import List, Callable
import json

log = logging.getLogger(__name__)


def _configure_ds(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.shuffle(1000).cache().prefetch(tf.data.AUTOTUNE)


def save_dataset(preprocessor: Callable):
    (train, val, test), metadata = tfds.load(
        "ag_news_subset",
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
        batch_size=100,
        split=["train[:80%]", "train[80%:]", "test"],
        try_gcs=True,
    )

    train = train.map(lambda x, y: (preprocessor(x), y))
    train = _configure_ds(train)
    tf.data.experimental.save(
        train, "/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/dataset/train"
    )

    val = val.map(lambda x, y: (preprocessor(x), y))
    val = _configure_ds(val)
    tf.data.experimental.save(
        val,
        "/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/dataset/validation",
    )

    tf.data.experimental.save(
        test, "/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/dataset/test"
    )

    lm = {
        "num_classes": metadata.features["label"].num_classes,
        "class_names": metadata.features["label"].names,
    }

    s = json.dumps(lm)
    tf.io.write_file(
        "/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/dataset/metadata.json",
        s,
    )


def sample_messages(num_datapoints=10) -> List[str]:
    test = tf.data.experimental.load(
        "/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/dataset/test"
    )

    x = test.unbatch().take(num_datapoints).map(lambda i, j: i)
    x = list(x.as_numpy_iterator())
    return [i.decode("utf-8") for i in x]
