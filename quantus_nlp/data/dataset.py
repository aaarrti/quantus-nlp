import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import List, Callable
import json

log = logging.getLogger(__name__)


def _configure_ds(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.shuffle(1000).cache().prefetch(tf.data.AUTOTUNE)


def save_dataset(preprocessor: Callable):
    (train, test), metadata = tfds.load(
        "ag_news_subset",
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
        batch_size=100,
        split=["train", "test"],
        try_gcs=True,
    )

    train = train.map(lambda x, y: (preprocessor(x), y))
    train = _configure_ds(train)
    tf.data.experimental.save(train, '/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/dataset/train')

    test = test.map(lambda x, y: (preprocessor(x), y))
    test = _configure_ds(test)
    tf.data.experimental.save(test, '/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/dataset/test')

    lm = {
        'num_classes': metadata.features["label"].num_classes,
        'class_names': metadata.features["label"].names
    }

    s = json.dumps(lm)
    tf.io.write_file('Users/artemsereda/Documents/PycharmProjects/quantus-nlp/dataset/metadata.json', s)



