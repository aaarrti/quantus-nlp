import numpy as np

from quantus_nlp.util import log_before


#@log_before
def relative_input_stability(
        x: np.ndarray,
        x_s: np.ndarray,
        ex: np.ndarray,
        ex_s: np.ndarray,
        eps_min=0.1e-6
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
    nominator = np.linalg.norm((ex - ex_s) / ex)
    denominator = np.linalg.norm((x - x_s) / x)
    denominator = np.max([denominator, eps_min])
    return np.max([nominator / denominator])
