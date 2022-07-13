from __future__ import print_function
from __future__ import with_statement
from __future__ import annotations

import click

from bert import build_model, fine_tune
from data import go_emotions_ds


@click.command
@click.option("--epochs", default=10)
def main(epochs):
    nn = build_model(28)

    train_ds, val_ds, test_ds = go_emotions_ds()

    fine_tune(model=nn, train_ds=train_ds, val_ds=val_ds, epochs=epochs)


if __name__ == "__main__":
    main()
