"""Code for numpy arrays from json-numpy under MIT

MIT License

Copyright (c) 2021-2025 Crimson-Crow <github@crimsoncrow.dev>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import json
from base64 import b64decode, b64encode

from numpy import frombuffer, generic, ndarray
from numpy.lib.format import descr_to_dtype, dtype_to_descr


def hint_tuples(item):
    """See https://stackoverflow.com/a/15721641/1745538"""
    if isinstance(item, tuple):
        return {"__tuple__": [hint_tuples(e) for e in item]}
    if isinstance(item, list):
        return [hint_tuples(e) for e in item]
    if isinstance(item, dict):
        return {key: hint_tuples(value) for key, value in item.items()}
    return item


def dehint_tuples(item):
    """See https://stackoverflow.com/a/15721641/1745538"""
    if isinstance(item, tuple):
        return tuple([dehint_tuples(e) for e in item])
    if isinstance(item, list):
        return [dehint_tuples(e) for e in item]
    if isinstance(item, dict) and "__tuple__" in item:
        return tuple([dehint_tuples(e) for e in item["__tuple__"]])
    return item


class _CustomEncoder(json.JSONEncoder):
    """
    See https://stackoverflow.com/a/15721641/1745538
    """

    def encode(self, obj):
        return super().encode(hint_tuples(obj))

    def default(self, o):
        from jax import dtypes
        import jax.random as jrng
        import jax.numpy as jnp
        import numpy as np

        if isinstance(o, jnp.ndarray) and dtypes.issubdtype(o.dtype, dtypes.prng_key):
            o = jrng.key_data(o)
            o = np.array(o)
            data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
            return {
                "__jax_rng_key__": b64encode(data).decode(),
                "dtype": dtype_to_descr(o.dtype),
                "shape": hint_tuples(o.shape),
            }

        if isinstance(o, jnp.ndarray):
            o = np.array(o)
            data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
            return {
                "__jax__": b64encode(data).decode(),
                "dtype": dtype_to_descr(o.dtype),
                "shape": hint_tuples(o.shape),
            }

        if isinstance(o, (ndarray, generic)):
            data = o.data if o.flags["C_CONTIGUOUS"] else o.tobytes()
            return {
                "__numpy__": b64encode(data).decode(),
                "dtype": dtype_to_descr(o.dtype),
                "shape": hint_tuples(o.shape),
            }

        if isinstance(o, np.random.RandomState):
            return {"__numpy_random_state__": hint_tuples(o.get_state())}

        if isinstance(o, np.random.Generator):
            return {"__numpy_random_generator__": hint_tuples(o.bit_generator.state)}

        raise TypeError(
            f"Object of type {o.__class__.__name__} is not JSON serializable"
        )


def _object_hook(dct):
    import jax.random as jrng
    import jax.numpy as jnp
    import numpy as np

    if "__jax_rng_key__" in dct:
        np_obj = frombuffer(
            b64decode(dct["__jax_rng_key__"]), descr_to_dtype(dct["dtype"])
        )
        arr = (
            np_obj.reshape(shape)
            if (shape := dehint_tuples(dct["shape"]))
            else np_obj[0]
        )
        key = jnp.array(arr)
        return jrng.wrap_key_data(key)

    if "__jax__" in dct:
        np_obj = frombuffer(b64decode(dct["__jax__"]), descr_to_dtype(dct["dtype"]))
        arr = (
            np_obj.reshape(shape)
            if (shape := dehint_tuples(dct["shape"]))
            else np_obj[0]
        )
        return jnp.array(arr)

    if "__numpy__" in dct:
        np_obj = frombuffer(b64decode(dct["__numpy__"]), descr_to_dtype(dct["dtype"]))
        return (
            np_obj.reshape(shape)
            if (shape := dehint_tuples(dct["shape"]))
            else np_obj[0]
        )

    if "__tuple__" in dct:
        return dehint_tuples(dct)

    if "__numpy_random_state__" in dct:
        rng = np.random.RandomState()
        rng.set_state(dehint_tuples(dct["__numpy_random_state__"]))
        return rng

    if "__numpy_random_generator__" in dct:
        data = dehint_tuples(dct["__numpy_random_generator__"])
        bg = getattr(np.random, data["bit_generator"])()
        bg.state = data
        return np.random.Generator(bg)

    return dct


def dump(*args, **kwargs):
    return json.dump(*args, cls=_CustomEncoder, **kwargs)


def dumps(*args, **kwargs):
    return json.dumps(*args, cls=_CustomEncoder, **kwargs)


def load(*args, **kwargs):
    return json.load(*args, object_hook=_object_hook, **kwargs)


def loads(*args, **kwargs):
    return json.loads(*args, object_hook=_object_hook, **kwargs)


class EstimatorToFromJSONMixin:
    def to_json(self, out=None):
        """Serialize this estimator to JSON.

        Parameters
        ----------
        out : file-like object, string, or None, optional
            If a file-like object or a string, the data is written
            using the `write` method, creating / overwriting a file
            if a string is given. If None, then only the JSON string
            is returned.

        Returns
        -------
        data : str
            The JSON-serialized data as a string.
        """
        data = {}
        for attr in self.json_attributes_:
            if hasattr(self, attr):
                data[attr] = getattr(self, attr)
        data = dumps(data)

        if out is None:
            pass
        elif hasattr(out, "write"):
            out.write(data)
        else:
            with open(out, "w") as fp:
                fp.write(data)

        return data

    @classmethod
    def from_json(cls, data):
        """Load an estimator from JSON data.

        Parameters
        ----------
        data : str or file-like
            The JSON data.

        Returns
        -------
        estimator
        """
        if hasattr(data, "read"):
            data = load(data)
        else:
            if os.path.exists(str):
                with open(str, "r") as fp:
                    data = loads(fp.read())
            else:
                data = loads(data)

        obj = cls()
        params = {k: data[k] for k in cls.get_params() if k in data}
        obj.set_params(**params)
        for k in cls.get_params():
            if k in data:
                del data[k]

        if getattr(cls, "json_init_method_"):
            getattr(obj, cls.json_init_method_)(**data)
        else:
            for k, v in data.items():
                setattr(obj, k, v)
        return obj
