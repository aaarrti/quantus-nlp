import numpy as np
import tensorflow as tf
import tensorflow_text  # noqa
import tensorflow_hub as hub
from quantus_nlp.interfaces import TextClassificationModel
import json

load_options = tf.saved_model.LoadOptions(experimental_io_device="/job:localhost")


def pre_process_model() -> tf.keras.layers.Layer:
    preprocessing_layer = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
        name="preprocessing",
        load_options=load_options,
        trainable=False,
    )
    return preprocessing_layer


class TrainableClassifier(tf.keras.Model):
    def __init__(self, num_classes):
        super(TrainableClassifier, self).__init__(name="prediction")
        self.transformer = hub.KerasLayer(
            "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2",
            trainable=True,
            name="BERT_encoder",
            load_options=load_options,
        )
        self.dropout = tf.keras.layers.Dropout(0.2)
        self.dense = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, preprocessed_text):
        encoder_outputs = self.encoder(preprocessed_text)
        pooled_output = encoder_outputs["pooled_output"]
        x = self.dropout(pooled_output)
        x = self.dense(x)
        return x

    def fine_tune(
        self,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs=10,
        jit=True,
    ):
        self.transformer.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            jit_compile=jit,
        )

        # models.summary()

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=3, min_lr=1e-6, min_delta=0.01
        )
        terminate_nan = tf.keras.callbacks.TerminateOnNaN()
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="loss", patience=5, verbose=1
        )

        history = self.transformer.fit(
            x=train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=0,
            callbacks=[reduce_lr, terminate_nan, early_stop],
        )

        res = {}
        for i in history.history:
            res[i] = [float(j) for j in history.history[i]]

        s = json.dumps(res)
        tf.io.write_file("gs://quantus-nlp/model/history.json", s)
        tf.saved_model.save(self.transformer, "gs://quantus-nlp/model/encoder")


class PreTrainedClassifier:
    def __init__(self):
        self.transformer = tf.saved_model.load("/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/model/encoder")
        self.emdedder = pre_process_model()

    def call(self, x):
        px = self.emdedder(x)
        res = super(TrainableClassifier).call(px)
        return res

    def text_to_vector(self, text: str) -> np.ndarray:
        x = self.emdedder([text])["input_word_ids"]
        x = x.numpy()[0]
        return x
