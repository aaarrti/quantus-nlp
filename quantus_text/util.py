import re
import inspect
import functools
import nlpaug.augmenter.word as naw


def log_before(func):
    def wrapper(*args, **kwargs):
        func_args = inspect.signature(func).bind(*args, **kwargs).arguments
        func_args_str = ", ".join(map("{0[0]} = {0[1]!r}".format, func_args.items()))
        print(
            f"Entered {func.__module__}.{func.__qualname__} with args ( {func_args_str} )"
        )
        return func(*args, **kwargs)

    return wrapper


def log_after(func):
    @functools.wraps(func)
    def wrapper(*func_args, **func_kwargs):
        retval = func(*func_args, **func_kwargs)
        print("Exited " + func.__name__ + "() with value: " + repr(retval))
        return retval

    return wrapper


def remove_punctuation(x: str) -> str:
    x = re.sub(r"\W+", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x


def aug_spelling(w: str) -> str:
    aug = naw.SpellingAug()
    augmented_text = aug.augment(w, n=1)
    return augmented_text[0]
