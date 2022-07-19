from typing import List, Tuple, Text
import tensorflow as tf

import numpy as np
from lit_nlp.api import model as lit_model, types
from lit_nlp.api.model import JsonDict
from lit_nlp.api import dataset as lit_dataset
from typing import Dict
from lit_nlp.api import types as lit_types
from lit_nlp.components import lime_explainer


class LitLimeModelAdapter(lit_model.Model):
    def __init__(
        self,
        pre_process_model: tf.keras.Model,
        transformer: tf.keras.Model,
        class_names: List[str],
    ):
        self.pre_process_model = pre_process_model
        self.transformer = transformer
        self.class_names = class_names

    def get_embedding_table(self) -> Tuple[List[Text], np.ndarray]:
        return super().get_embedding_table()

    def fit_transform_with_metadata(self, indexed_inputs: List[JsonDict]):
        super().fit_transform_with_metadata(indexed_inputs)

    def output_spec(self) -> types.Spec:
        return {
            "probs": lit_types.MulticlassPreds(vocab=self.class_names, parent="label"),
        }

    def input_spec(self) -> types.Spec:
        return {"text": lit_types.TextSegment()}

    def predict_minibatch(self, inputs: List[JsonDict]) -> List[JsonDict]:
        x = inputs[0]["text"]
        xp = self.pre_process_model([x])
        res = self.transformer(xp)
        return [{"probs": res.numpy()[0]}]


class LitDatasetAdapter(lit_dataset.Dataset):
    def __init__(self, test_ds: tf.data.Dataset, num_data_points=None):
        super().__init__()
        ds = test_ds
        ds = ds.unbatch()
        if num_data_points:
            ds = ds.take(num_data_points)
        ds = ds.map(lambda i, j: {"text": i, "label": j})
        ds = list(ds.as_numpy_iterator())
        for d in ds:
            d["text"] = d["text"].decode("utf-8")
        self._examples = ds

    def spec(self) -> Dict[str, any]:
        return {"text": lit_types.TextSegment()}


def explain_lime(
    pre_process_model: tf.keras.Model,
    transformer: tf.keras.Model,
    class_names: List[str],
    example: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param pre_process_model: embedder model
    :param transformer: transformer model
    :param class_names: corresponding labels
    :param example: sentence which must be explained
    :return: a Tuple of tokens in example + their salience
    """
    lm = LitLimeModelAdapter(pre_process_model, transformer, class_names)

    lime = lime_explainer.LIME()

    lime_results = lime.run([{"text": example}], lm, None)[0]
    # print(f"{lime_results = }")
    tokens = np.asarray(lime_results["text"].tokens)
    salience = lime_results["text"].salience
    return tokens, salience
