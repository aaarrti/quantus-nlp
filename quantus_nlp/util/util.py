import inspect
from pathlib import Path
import functools
import pickle
from datetime import datetime
from os import environ


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


@log_before
def ensure_dir(file_path):
    Path(file_path).mkdir(exist_ok=True, parents=True)


@log_before
def save_pickle(obj, fpath):
    with open(f"{fpath}.pickle", "wb") as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


@log_before
@log_after
def load_pickle(path):
    with open(f"{path}.pickle", "rb") as file:
        obj = pickle.load(file)
        return obj


@log_after
def time_stamp_tag():
    return datetime.now().strftime("%m.%d:%H.%M")


def print_env():
    print("-" * 50)
    for k, v in sorted(environ.items()):
        print(k + ":", v)
    print("-" * 50)
