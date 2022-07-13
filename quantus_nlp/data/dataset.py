import tensorflow as tf
import tensorflow_datasets as tfds


def map_to_x_and_multi_hot_label():
    pass


def go_emotions_ds() -> (tf.data.Dataset, tf.data.Dataset, tf.data.Dataset):
    ds, metadata = tfds.load(
        "goemotions",
        split=["train", "validation", "test"],
        try_gcs=True,
        with_info=True,
    )
