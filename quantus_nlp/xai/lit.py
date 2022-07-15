from typing import List, Tuple, Text
import tensorflow as tf

import numpy as np
from lit_nlp.api import model as lit_model, types
from lit_nlp.api.model import JsonDict
from lit_nlp.api import dataset as lit_dataset
from typing import Dict
from lit_nlp.api import types as lit_types
import json

from quantus_nlp.models import pre_process_model
from quantus_nlp.util import log_after, log_before


class LitLimeModelAdapter(lit_model.Model):

    def __init__(self):
        self.pre_process = pre_process_model()
        self.transformer = tf.saved_model.load('/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/model/encoder')
        metadata = tf.io.read_file(
            '/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/dataset/metadata.json'
        ).numpy()
        self.metadata = json.loads(metadata)

    def get_embedding_table(self) -> Tuple[List[Text], np.ndarray]:
        super().get_embedding_table()

    def fit_transform_with_metadata(self, indexed_inputs: List[JsonDict]):
        super().fit_transform_with_metadata(indexed_inputs)

    def output_spec(self) -> types.Spec:
        return {
            'probs': lit_types.MulticlassPreds(vocab=self.metadata['class_names'], parent='label'),
        }

    def input_spec(self) -> types.Spec:
        return {
            'text': lit_types.TextSegment()
        }

    def predict_minibatch(self, inputs: List[JsonDict]) -> List[JsonDict]:
        x = inputs[0]['text']
        xp = self.pre_process([x])
        res = self.transformer(xp)
        return [{
            'probs': res.numpy()[0]
        }]


class LitIntegratedGradientModelAdapter(LitLimeModelAdapter):

    def output_spec(self) -> types.Spec:
        return {
            "tokens": lit_types.Tokens(parent="input_text"),
            "token_embs": lit_types.TokenEmbeddings(align='tokens'),
            "grad_class": lit_types.CategoryLabel(vocab=list(range(self.metadata['num_classes']))),
            "token_grads": lit_types.TokenGradients(align='tokens',
                                                    grad_for="token_embs",
                                                    grad_target_field_key="grad_class"),
        }

    def input_spec(self):
        return {
            "token_embs": lit_types.TokenEmbeddings(align='tokens', required=False),
            "grad_class": lit_types.CategoryLabel(vocab=list(range(self.metadata['num_classes'])), required=False),
        }

    @log_before
    def predict_minibatch(self, inputs: List[JsonDict]) -> List[JsonDict]:
        pass


class LitDatasetAdapter(lit_dataset.Dataset):

    def __init__(self, num_data_points=None):
        # Store as a list of dicts, conforming to self.spec()
        ds = tf.data.experimental.load("/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/dataset/test")
        ds = ds.unbatch()
        if num_data_points:
            ds = ds.take(num_data_points)
        ds = ds.map(lambda i, j: {'text': i, 'label': j})
        ds = list(ds.as_numpy_iterator())
        for i in ds:
            i['text'] = i['text'].decode('utf-8')
        self._examples = ds

    def spec(self) -> Dict[str, any]:
        metadata = tf.io.read_file(
            '/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/dataset/metadata.json'
        ).numpy()
        metadata = json.loads(metadata)
        return {
            'text': lit_types.TextSegment(),
            'label': lit_types.CategoryLabel(vocab=metadata['class_names'])
        }
