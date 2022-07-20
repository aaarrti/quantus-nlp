import logging
import click
import tensorflow as tf
import json

from quantus_nlp.xai import LimeExplainer
from quantus_nlp.util import aug_spelling
from quantus_nlp.metrics.ris import RelativeInputStability

from model import TrainableClassifier, pre_process_model, PreTrainedClassifier
from dataset import save_ag_news_dataset


LOG_FORMAT = "[%(filename)s:%(lineno)s:%(funcName)s()] %(message)s"


def init_tpu():
    click.secho("Connecting to TPU", fg="yellow", color=True)
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
    click.secho(
        f"Running on TPU: {tpu.cluster_spec().as_dict()['worker']}",
        fg="yellow",
        color=True,
    )
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    return strategy


@click.group(chain=True, invoke_without_command=True)
@click.pass_context
def main(ctx):
    click.secho(f"{ctx = }", fg="yellow", color=True)
    tf.config.set_soft_device_placement(True)
    logging.basicConfig(format=LOG_FORMAT, level=logging.DEBUG)


@main.command()
def dataset():
    pl = pre_process_model()
    save_ag_news_dataset(pl)


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

        nn = TrainableClassifier(metadata["num_classes"])
        nn.fine_tune(train_ds=_train, val_ds=val, jit=not no_jit, epochs=epochs)


@main.command("xai-lime")
def xai_lime():
    model = PreTrainedClassifier()
    metadata = tf.io.read_file(
        "/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/dataset/metadata.json"
    ).numpy()
    metadata = json.loads(metadata)

    ex = LimeExplainer(model, metadata["class_names"])

    res = ex(
        example="CHARLOTTE, N.C. (Sports Network) - Carolina Panthers  running back Stephen Davis will miss the remainder of the  season after being placed on injured reserve Saturday."
    )
    print(f"LIME results = {res}")


@main.command()
def ris():
    text = "Carolina Panthers  running back Stephen Davis will miss the remainder of the  season after being placed on injured reserve Saturday"
    aug_text = aug_spelling(text)

    metadata = tf.io.read_file(
        "/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/dataset/metadata.json"
    ).numpy()
    metadata = json.loads(metadata)

    model = PreTrainedClassifier()

    ex = LimeExplainer(model, metadata["class_names"])

    _, e = ex(text)
    _, es = ex(aug_text)

    x = model.text_to_vector(text)
    xs = model.text_to_vector(aug_text)

    # slice embedding to match the size of explanations
    # anyway all word ids afterward are 0
    x = x[: len(e)]
    xs = xs[: len(es)]

    metric = RelativeInputStability()
    res = metric(x=x, x_s=xs, ex=e, ex_s=es)

    click.echo(f"RIS = {res}")


if __name__ == "__main__":
    main()
