import collections
import logging
import sys
from datetime import datetime
from torch.utils.data import Dataset
import numpy as np


def merge_two_dicts(x, y):
    # In case of same key, it keeps the value of y
    return {**x, **y}


def merge_dicts(list_of_dicts):
    from functools import reduce
    return reduce(merge_two_dicts, list_of_dicts)


def flatten_dict(d, parent_key='', sep='_', prefix='eval_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep, prefix=prefix).items())
        else:
            items.append((prefix + new_key, v))
    return dict(items)


def float_format(f: float) -> str:
    return "%+.4e" % f


def my_sign(x):
    return np.sign(x) + (x == 0)
