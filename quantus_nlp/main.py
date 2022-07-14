from __future__ import print_function
from __future__ import with_statement
from __future__ import annotations

import logging
import click

from models import Classifier, fine_tune, pre_process_model
from data import save_dataset
import tensorflow as tf
import json

LOG_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)


def base_path(tpu: bool):
    if tpu:
        return 'gs://quantus-nlp'
    else:
        return '/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/'


def init_tpu():
    print('Connecting to TPU')
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    return strategy


@click.group()
def main():
    pass


@main.command()
def dataset():
    pl = pre_process_model()
    save_dataset(pl)


@main.command()
@click.option("--tpu", default=False, is_flag=True, help='Enable TPU')
def train(tpu):
    device = init_tpu() if tpu else tf.distribute.OneDeviceStrategy('cpu')

    with device.scope():
        p = base_path(tpu)

        train = tf.data.experimental.load(f'{p}/dataset/train')
        val = tf.data.experimental.load(f'{p}/dataset/test')
        metadata = tf.io.read_file(f'{p}/dataset/metadata.json').numpy()
        metadata = json.loads(metadata)

        nn = Classifier(metadata['num_classes'])
        fine_tune(model=nn, train_ds=train, val_ds=val)


@main.command()
def xai():
    pass


if __name__ == "__main__":
    tf.config.set_soft_device_placement(True)
    main()
