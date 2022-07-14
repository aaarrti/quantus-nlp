from __future__ import print_function
from __future__ import with_statement
from __future__ import annotations

import logging
import click

from bert import build_model, fine_tune
from data import dataset
import tensorflow as tf

LOG_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"


def init_tpu():
    print('Connecting to TPU')
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    return strategy


@click.command
@click.option("--epochs", default=10)
@click.option("--debug", default=False)
@click.option("--tpu", default=False)
def main(epochs, debug, tpu):
    if debug:
        # tf.data.experimental.enable_debug_mode()
        # tf.config.run_functions_eagerly(True)
        logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

    device = init_tpu() if tpu else tf.distribute.OneDeviceStrategy("cpu")

    with device.scope():
        train, validation, metadata = dataset()

        nn = build_model(metadata.num_classes)

        fine_tune(model=nn, train_ds=train, val_ds=validation, epochs=epochs)


if __name__ == "__main__":
    main()
