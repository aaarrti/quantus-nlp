from __future__ import print_function
from __future__ import with_statement
from __future__ import annotations

import logging
import click

from bert import Classifier, fine_tune, pre_process_model
from data import save_dataset
import tensorflow as tf
import json

LOG_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"


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


@click.command
@click.argument("task")
@click.option("--epochs", default=10)
@click.option("--debug", default=False)
@click.option("--tpu", default=False)
def main(task, epochs, debug, tpu):
    if debug:
        logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

    if task == 'dataset':
        pl = pre_process_model()
        save_dataset(pl)
        return

    if task == 'train':
        device = init_tpu() if tpu else tf.distribute.OneDeviceStrategy('cpu')

        with device.scope():
            p = base_path(tpu)
            train = tf.data.experimental.load(f'{p}/dataset/train')
            val = tf.data.experimental.load(f'{p}/dataset/test')
            metadata = tf.io.read_file(f'{p}/dataset/metadata.json').numpy()
            metadata = json.loads(metadata)
            nn = Classifier(metadata['num_classes'])

            fine_tune(model=nn, epochs=epochs, train_ds=train, val_ds=val)

            print()


if __name__ == "__main__":
    tf.config.set_soft_device_placement(True)
    main()
