import numpy as np

CACHE_FLOAT_PRECISION = 3


def cache_value(value):
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    elif isinstance(value, float):
        return np.round(value, CACHE_FLOAT_PRECISION)
    elif isinstance(value, dict):
        return {key: cache_value(val) for key, val in sorted(value.items(), key=lambda x: x[0])}
    elif isinstance(value, list):
        return [cache_value(val) for val in value]
    elif isinstance(value, np.ndarray):
        return value.round(CACHE_FLOAT_PRECISION)
