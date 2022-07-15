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


class LitModelAdapter(lit_model.Model):

    def __init__(self):
        self.pre_process = pre_process_model()
        self.transformer = tf.saved_model.load('/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/model/encoder')

    def get_embedding_table(self) -> Tuple[List[Text], np.ndarray]:
        super().get_embedding_table()

    def fit_transform_with_metadata(self, indexed_inputs: List[JsonDict]):
        super().fit_transform_with_metadata(indexed_inputs)

    def output_spec(self) -> types.Spec:
        metadata = tf.io.read_file(
            '/Users/artemsereda/Documents/PycharmProjects/quantus-nlp/dataset/metadata.json'
        ).numpy()
        metadata = json.loads(metadata)
        return {
            # The 'parent' keyword tells LIT where to look for gold labels when computing metrics.
            'probs': lit_types.MulticlassPreds(vocab=metadata['class_names'], parent='label'),
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
