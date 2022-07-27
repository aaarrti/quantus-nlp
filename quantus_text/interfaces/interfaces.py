import numpy as np
import tensorflow as tf
from abc import abstractmethod, ABC

class TextClassificationModel(ABC, tf.keras.Model):
    emdedder: tf.keras.Model
    transformer: tf.keras.Model

    @abstractmethod
    def call(self, x):
        pass

class NlpExplanation(ABC):
    model: TextClassificationModel

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

class XaiMetric(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass
