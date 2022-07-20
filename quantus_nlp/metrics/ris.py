import numpy as np
from quantus_nlp.interfaces import XaiMetric


class RelativeInputStability(XaiMetric):
    def __call__(
        self,
        x: np.ndarray,
        x_s: np.ndarray,
        ex: np.ndarray,
        ex_s: np.ndarray,
        eps_min=0.1e-6,
    ) -> float:

        """
        :param x: original input
        :param x_s: perturbed input
        :param ex: original explanation
        :param ex_s: explanation from the perturbed input
        :param eps_min: value to make sure the denominator is not 0
        :return: float RIS of explanations
        FIXME
        Right now is implemented only for 1-D arrays, need to fix that in future
        """
        if np.any(ex == 0) or np.any(x == 0):
            raise ArithmeticError("Can't divide by 0")
        nominator = np.linalg.norm((ex - ex_s) / ex)
        denominator = np.linalg.norm((x - x_s) / x)
        denominator = np.max([denominator, eps_min])
        return np.max([nominator / denominator])
