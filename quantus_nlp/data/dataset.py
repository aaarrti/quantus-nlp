import logging
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import List
from dataclasses import dataclass


log = logging.getLogger(__name__)


@dataclass
class LabelMetadata:
    num_classes: int
    class_names: List[str]


def _configure_ds(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.cache().prefetch(tf.data.AUTOTUNE)


def dataset() -> (tf.data.Dataset, tf.data.Dataset, LabelMetadata):
    (train, test), metadata = tfds.load(
        "ag_news_subset",
        as_supervised=True,
        with_info=True,
        shuffle_files=True,
        batch_size=100,
        split=["train", "test"],
        try_gcs=True,
    )

    train = _configure_ds(train)
    test = _configure_ds(test)

    lm = LabelMetadata(
        num_classes=metadata.features["label"].num_classes,
        class_names=metadata.features["label"].names,
    )

    return train, test, lm
