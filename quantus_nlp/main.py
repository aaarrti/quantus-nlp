from __future__ import print_function
from __future__ import with_statement
from __future__ import annotations

import logging
import click

from models import Classifier, fine_tune, pre_process_model
from data import save_dataset
import tensorflow as tf
import json
from xai import LitDatasetAdapter, LitLimeModelAdapter, LitIntegratedGradientModelAdapter
from lit_nlp.components import lime_explainer, gradient_maps

LOG_FORMAT = "[%(filename)s:%(lineno)s:%(funcName)s()] %(message)s"


def init_tpu():
    click.secho("Connecting to TPU", fg='yellow', color=True)
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    click.secho(f"Running on TPU: {tpu.cluster_spec().as_dict()['worker']}", fg='yellow', color=True)
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    return strategy


@click.group(chain=True, invoke_without_command=True)
@click.pass_context
def main(ctx):
    click.secho(f"{ctx = }", fg='yellow', color=True)
    tf.config.set_soft_device_placement(True)
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)


@main.command()
def dataset():
    pl = pre_process_model()
    save_dataset(pl)


@main.command()
@click.option("--tpu", default=False, is_flag=True, help="Enable TPU")
@click.option("--no-jit", default=False, is_flag=True, help="Disable XLA")
@click.option("--epochs", default=10, help="Number of epochs to train")
def train(tpu, no_jit, epochs):
    device = init_tpu() if tpu else tf.distribute.OneDeviceStrategy("cpu")

    with device.scope():
        base_path = (
            "gs://quantus-nlp"
            if tpu
            else "/Users/artemsereda/Documents/PycharmProjects/quantus-nlp"
        )
        _train = tf.data.experimental.load(f"{base_path}/dataset/train")
        val = tf.data.experimental.load(f"{base_path}/dataset/validation")
        metadata = tf.io.read_file(f"{base_path}/dataset/metadata.json").numpy()
        metadata = json.loads(metadata)

        nn = Classifier(metadata["num_classes"])
        fine_tune(model=nn, train_ds=_train, val_ds=val, jit=not no_jit, epochs=epochs)


@main.command("xai-lime")
def xai_lime():
    ds = LitDatasetAdapter()
    nn = LitLimeModelAdapter()

    lime = lime_explainer.LIME()

    lime_results = lime.run(ds.examples[:1], nn, ds)[0]
    click.echo(f'{lime_results = }')


@main.command("xai-ig")
def xai_ig():
    ds = LitDatasetAdapter()
    nn = LitIntegratedGradientModelAdapter()

    ig = gradient_maps.IntegratedGradients()
    integrated_grad_result = ig.run(ds.examples[:1], nn, ds)[0]
    click.echo(f'{integrated_grad_result = }')


if __name__ == "__main__":
    main()
