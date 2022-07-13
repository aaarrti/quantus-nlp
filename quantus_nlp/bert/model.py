import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # noqa
from typing import Dict


def build_model(num_classes: int) -> tf.keras.Model:
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text")
    preprocessing_layer = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        name="preprocessing",
    )
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2",
        trainable=True,
        name="BERT_encoder",
    )
    outputs = encoder(encoder_inputs)
    net = outputs["pooled_output"]
    net = tf.keras.layers.Dropout(0.1)(net)
    net = tf.keras.layers.Dense(num_classes, activation="sigmoid", name="predictions")(
        net
    )
    return tf.keras.Model(text_input, net)


def fine_tune(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
) -> Dict:
    # TODO add weight decay and multi label confusion matrix

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
        loss="binary_crossentropy",
        metrics=["categorical_accuracy"],
        # jit_compile=True
    )

    model.summary()

    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, patience=5, min_lr=1e-6
    )
    terminate_nan = tf.keras.callbacks.TerminateOnNaN()
    early_stop = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3, verbose=1)

    history = model.fit(
        x=train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=[reduce_lr, terminate_nan, early_stop],
        return_dict=True,
    )

    return history
