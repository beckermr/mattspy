import numpy as np
import jax.numpy as jnp
import jax.random as jrng

import pytest

from mattspy.json import dumps, loads


@pytest.mark.parametrize(
    "val",
    [
        np.array(10.0),
        np.array(10.0, dtype=int),
        np.array(10.0, dtype=float),
        np.array(10.0, dtype=np.float32),
        np.array(10.0, dtype=np.int32),
        np.array(10.0, dtype=np.float64),
        np.array(10.0, dtype=np.int64),
        np.array("342524", dtype="U"),
        np.array("dsfcsda", dtype="S"),
        np.arange(10, dtype=int),
        np.arange(10, dtype=np.int32),
        np.arange(10, dtype=np.int64),
        np.arange(10, dtype=np.float64),
        np.arange(10, dtype=np.float32),
        np.array([10, np.nan, np.inf], dtype=np.float32),
        np.array([10, np.nan, np.inf], dtype=np.float64),
        np.arange(10, dtype=np.complex64),
        np.arange(10, dtype=np.complex128),
        np.arange(10, dtype=float),
        np.array(["%s" % i for i in range(10)], dtype="U"),
        np.array(["%s" % i for i in range(10)], dtype="S"),
    ],
)
def test_json_numpy(val):
    sval = loads(dumps([val]))[0]
    if "U" not in val.dtype.descr[0][1] and "S" not in val.dtype.descr[0][1]:
        assert np.array_equal(val, sval, equal_nan=True)
    else:
        assert np.array_equal(val, sval)
    assert val.shape == sval.shape
    assert val.dtype == sval.dtype


@pytest.mark.parametrize(
    "val",
    [
        jnp.array(10.0),
        jnp.array(10.0, dtype=int),
        jnp.array(10.0, dtype=float),
        jnp.array(10.0, dtype=jnp.float32),
        jnp.array(10.0, dtype=jnp.int32),
        jnp.array(10.0, dtype=jnp.float64),
        jnp.array(10.0, dtype=jnp.int64),
        jnp.arange(10, dtype=int),
        jnp.arange(10, dtype=jnp.int32),
        jnp.arange(10, dtype=jnp.int64),
        jnp.arange(10, dtype=jnp.float64),
        jnp.arange(10, dtype=jnp.float32),
        jnp.array([10, jnp.nan, jnp.inf], dtype=jnp.float32),
        jnp.array([10, jnp.nan, jnp.inf], dtype=jnp.float64),
        jnp.arange(10, dtype=jnp.complex64),
        jnp.arange(10, dtype=jnp.complex128),
        jnp.arange(10, dtype=float),
    ],
)
def test_json_jax(val):
    sval = loads(dumps([val]))[0]
    if "U" not in val.dtype.descr[0][1] and "S" not in val.dtype.descr[0][1]:
        assert jnp.array_equal(val, sval, equal_nan=True)
    else:
        assert jnp.array_equal(val, sval)
    assert val.shape == sval.shape
    assert val.dtype == sval.dtype


def test_json_numpy_random():
    rng = np.random.RandomState(seed=10)
    srng = loads(dumps([rng]))[0]
    assert rng.normal() == srng.normal()

    rng = np.random.default_rng(seed=10)
    srng = loads(dumps([rng]))[0]
    assert rng.normal() == srng.normal()

    rng = np.random.Generator(np.random.MT19937(seed=10))
    srng = loads(dumps([rng]))[0]
    assert rng.normal() == srng.normal()


def test_json_tuple():
    val = (10, (10, 4, 5.0, {"a": 10}))
    sval = loads(dumps(val))
    assert val == sval


def test_json_jax_random():
    key = jrng.key(seed=100)
    skey = loads(dumps([key]))[0]
    assert jrng.normal(key) == jrng.normal(skey)
