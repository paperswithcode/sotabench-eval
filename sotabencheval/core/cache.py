import numpy as np

CACHE_FLOAT_PRECISION = 3


def cache_value(value):
    """
    Takes in a value and puts it in a format ready for hashing + caching

    Why? In sotabench we hash the output after the first batch as an indication of whether the model has changed or not.
    If the model hasn't changed, then we don't run the whole evaluation on the server - but return the same results
    as before. This speeds up evaluation - making "continuous evaluation" more feasible...it also means lower
    GPU costs for us :).

    We apply some rounding and reformatting so small low precision changes do not change the hash.

    :param value: example model output
    :return: formatted value (rounded and ready for hashing)
    """
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
