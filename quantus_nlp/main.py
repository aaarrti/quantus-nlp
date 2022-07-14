from __future__ import print_function
from __future__ import with_statement
from __future__ import annotations

import logging
import click

from bert import build_model, fine_tune
from data import dataset

LOG_FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"


@click.command
@click.option("--epochs", default=10)
@click.option("--debug", default=False)
def main(epochs, debug):
    if debug:
        # tf.data.experimental.enable_debug_mode()
        # tf.config.run_functions_eagerly(True)
        logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)
    else:
        logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)

    train, validation, metadata = dataset()

    nn = build_model(metadata.num_classes)

    fine_tune(model=nn, train_ds=train, val_ds=validation, epochs=epochs)


if __name__ == "__main__":
    main()
