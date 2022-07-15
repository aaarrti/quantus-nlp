from typing import List, Tuple, Text

import numpy as np
from lit_nlp.api import model as lit_model, types
from lit_nlp.api.model import JsonDict
from lit_nlp.api import dataset as lit_dataset
from typing import Dict


class LitModelAdapter(lit_model.Model):

    def get_embedding_table(self) -> Tuple[List[Text], np.ndarray]:
        pass

    def fit_transform_with_metadata(self, indexed_inputs: List[JsonDict]):
        pass

    def output_spec(self) -> types.Spec:
        pass

    def input_spec(self) -> types.Spec:
        pass

    def predict_minibatch(self, inputs: List[JsonDict]) -> List[JsonDict]:
        pass


class LitDatasetAdapter(lit_dataset.Dataset):

    def __init__(self):
        # Store as a list of dicts, conforming to self.spec()
        self._examples = [{}]

    def spec(self) -> Dict[str, any]:
        pass
