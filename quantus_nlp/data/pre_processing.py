import re


def remove_punctuation(x: str) -> str:
    x = re.sub(r'\W+', ' ', x)
    x = re.sub(r'\s+', ' ', x)
    return x
