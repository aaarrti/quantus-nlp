from typing import List, Tuple, Text

import numpy as np
from lit_nlp.api import types
from lit_nlp.api.model import JsonDict, Model
from lit_nlp.api import types as lit_types
from lit_nlp.components import lime_explainer
from quantus_text.interfaces import TextClassificationModel, NlpExplanation


class LitLimeModelAdapter(Model):
    def __init__(
        self,
        model: TextClassificationModel,
        class_names: List[str],
    ):
        self.pre_process_model = model.emdedder
        self.transformer = model.transformer
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


class LimeExplainer(NlpExplanation):
    def __init__(self, model: TextClassificationModel, class_names: List[str]):
        self.lime_adapter = LitLimeModelAdapter(model, class_names)
        self.lime = lime_explainer.LIME()

    def __call__(self, example: str) -> Tuple[np.ndarray, np.ndarray]:
        lime_results = self.lime.run([{"text": example}], self.lime_adapter, None)[0]
        tokens = np.asarray(lime_results["text"].tokens)
        salience = lime_results["text"].salience
        return tokens, salience
