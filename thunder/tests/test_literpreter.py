from collections.abc import Iterable, Iterator, Sequence
from functools import partial, wraps
from itertools import product

import sys
import dis
from collections.abc import Callable

import pytest
import torch
from torch.testing import assert_close

import thunder
from thunder.core.jit import is_jitting, jit, JITError

import thunder.clang as clang
from thunder.core.options import INTERPRETATION_OPTIONS, CACHE_OPTIONS
import thunder.torch as ltorch
import thunder.core.prims as prims

#
# Test suite for the litjit extension of the Python interpreter
#

# TODO GTC Merge this test file with test_jit.py

tp_jit = partial(thunder.jit, interpretation=INTERPRETATION_OPTIONS.TRANSLATE_PYTHON)


def skipif_python_3_11_plus(f):
    if sys.version_info >= (3, 11):
        return pytest.mark.skip(f, reason=f"not yet implemented for Python 3.11+, got {sys.version_info=}")
    return f


def test_binary_add_tensors():
    def foo(a, b):
        return a + b

    jfoo = tp_jit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_torch_add_tensors():
    def foo(a, b):
        return torch.add(a, b)

    jfoo = tp_jit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_torch_add_tensors_closure():
    def foo(a, b):
        c = a + b

        def bar():
            return torch.add(c, 1)

        return bar()

    jfoo = tp_jit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_torch_add_tensors_closure_external():
    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    def bar(b):
        return torch.add(a, b)

    def foo():
        bar(b)

    jbar = tp_jit(bar)
    actual = jbar(b)
    expected = bar(b)
    assert_close(actual, expected)

    jfoo = tp_jit(foo)
    actual = jfoo()
    expected = foo()
    assert_close(actual, expected)


def test_intermediate_torch_operations():
    def foo(a, b):
        c = a + b
        d = torch.sub(c, b)
        e = torch.mul(d, a)
        f = torch.matmul(e, c)
        g = [e, f]
        return torch.cat(g)

    jfoo = tp_jit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_cache_basic():
    def foo(a, b):
        return a + b

    jfoo = tp_jit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1

    # Tests rank changing
    a = torch.randn((2), device="cpu")

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 1

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 2

    # Tests dtype changing
    a = torch.randn((2, 2), device="cpu", dtype=torch.bfloat16)
    b = torch.randn((2, 2), device="cpu", dtype=torch.bfloat16)

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 3
    assert thunder.cache_hits(jfoo) == 2

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 3
    assert thunder.cache_hits(jfoo) == 3

    # Tests shape changing
    a = torch.randn((2, 1), device="cpu", dtype=torch.bfloat16)
    b = torch.randn((2, 1), device="cpu", dtype=torch.bfloat16)

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 4
    assert thunder.cache_hits(jfoo) == 3

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 4
    assert thunder.cache_hits(jfoo) == 4


def test_cache_always_trace():
    def foo(a, b):
        return a + b

    jfoo = tp_jit(foo, cache=CACHE_OPTIONS.NO_CACHING)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    expected = foo(a, b)
    actual = jfoo(a, b)
    actual = jfoo(a, b)
    actual = jfoo(a, b)
    actual = jfoo(a, b)
    assert_close(expected, actual)
    assert thunder.cache_misses(jfoo) == 4
    assert thunder.cache_hits(jfoo) == 0


def test_cache_equality_contraint():
    x, y = torch.randn(2, 2)

    def fn(b):
        if b:
            return x
        else:
            return y

    jfn = tp_jit(fn)

    assert_close(fn(True), jfn(True))
    assert_close(fn(False), jfn(False))

    assert thunder.cache_misses(jfn) == 2
    assert thunder.cache_hits(jfn) == 0

    assert_close(fn(True), jfn(True))
    assert_close(fn(False), jfn(False))

    assert thunder.cache_misses(jfn) == 2
    assert thunder.cache_hits(jfn) == 2


def test_nn_parameter():
    a = torch.nn.Parameter(torch.randn(2, 3))
    b = torch.tensor(2)

    def fn(a):
        return b * a

    jfn = tp_jit(fn)

    expected = fn(a)
    actual = jfn(a)
    assert_close(expected, actual)


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/1982", raises=BaseException)
def test_nn_module():
    m = torch.nn.Linear(3, 4)
    m2 = torch.nn.Sequential(
        torch.nn.Linear(3, 4),
        torch.nn.Linear(4, 3),
    )

    a = torch.randn(2, 3)

    def fn(a):
        return m(a)

    def fn2(a):
        return m2(a)

    jfn = tp_jit(fn)
    expected = fn(a)
    actual = jfn(a)
    assert_close(expected, actual)

    jfn = tp_jit(fn2)
    expected = fn2(a)
    actual = jfn(a)
    assert_close(expected, actual)

    jm = tp_jit(m.__call__)

    expected = m(a)
    actual = jm(a)
    assert_close(expected, actual)


def test_add_numbers():
    def foo(a, b):
        return torch.add(a, b)

    jfoo = tp_jit(foo)

    # TODO Add test for bool
    # See https://github.com/Lightning-AI/lightning-thunder/issues/1990
    cases = (
        (2, 3),
        (2.1, 3.4),
        (complex(1, 1), complex(-1, 2)),
    )

    for a, b in cases:
        actual = jfoo(a, b)
        expected = a + b

        assert_close(actual, expected)


def test_binary_add_tensor_number():
    # Tests using torch.add
    def foo(a):
        return torch.add(a, 3)

    jfoo = tp_jit(foo)

    a = torch.randn((2, 2), device="cpu")

    actual = jfoo(a)
    expected = foo(a)

    assert_close(actual, expected)

    # Tests using addition operator
    def foo(a):
        return a + 4

    jfoo = tp_jit(foo)

    actual = jfoo(a)
    expected = foo(a)

    assert_close(actual, expected)


def test_binary_add_numbers():
    def foo(a, b):
        return a + b

    jfoo = tp_jit(foo)

    # TODO Add test for bool
    # See https://github.com/Lightning-AI/lightning-thunder/issues/1990
    cases = (
        (2, 3),
        (2.1, 3.4),
        (complex(1, 1), complex(-1, 2)),
    )

    for a, b in cases:
        actual = jfoo(a, b)
        expected = foo(a, b)

        assert_close(actual, expected)


_test_add_global_global = 2


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/1935", raises=BaseException)
def test_global_fails():
    def foo():
        return _test_add_global_global

    jfoo = tp_jit(foo)

    with pytest.raises(NotImplementedError):
        jfoo()


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/1936", raises=BaseException)
def test_nonlocal_outside_interpreter_fails():
    def foo():
        x = 3

        def bar():
            nonlocal x
            x = 4

        jbar = tp_jit(bar)

        jbar()

        return x

    with pytest.raises(NotImplementedError):
        foo()


def test_lookaside_bool():
    def foo(a, b, i):
        if bool(i):
            return a + b
        return a - b

    jfoo = tp_jit(foo)

    a = torch.randn((2, 2), device="cpu")
    b = torch.randn((2, 2), device="cpu")

    expected = foo(a, b, 0)
    actual = jfoo(a, b, 0)
    assert_close(expected, actual)

    expected = foo(a, b, 1)
    actual = jfoo(a, b, 1)
    assert_close(expected, actual)


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2004", raises=BaseException)
def test_litgpt():
    from thunder.benchmarks import LitGPTBenchmark
    from thunder.tests.lit_gpt_model import Config

    cfg: Config = Config.from_name("gpt-neox-like")
    bench = LitGPTBenchmark(config=cfg, device="cpu", dtype=torch.bfloat16, requires_grad=True)
    module = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = tp_jit(module)
    result = jfn(*args, **kwargs)

    assert_close(result, module(*args, **kwargs))


def test_nanogpt_block():
    from thunder.benchmarks import NanoGPTBlockBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTBlockBenchmark(config=config, device="cpu")
    module = bench.fn()

    args, kwargs = bench.make_batch()

    jfn = tp_jit(module)
    result = jfn(*args, **kwargs)

    assert_close(result, module(*args, **kwargs))


def test_nanogpt_attn():
    from thunder.benchmarks import NanoGPTBlockBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTBlockBenchmark(config=config, device="cpu")
    module = bench.fn()
    module = module.attn

    args, kwargs = bench.make_batch()

    jfn = tp_jit(module)
    result = jfn(*args, **kwargs)

    assert_close(result, module(*args, **kwargs))


def test_nanogpt_mlp():
    from thunder.benchmarks import NanoGPTBlockBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0)
    config.update(**_nanogpt_configs["gpt2"])
    bench = NanoGPTBlockBenchmark(config=config, device="cpu")
    module = bench.fn().mlp

    args, kwargs = bench.make_batch()

    jfn = tp_jit(module)
    result = jfn(*args, **kwargs)

    assert_close(result, module(*args, **kwargs))


def test_nanogpt():
    from thunder.benchmarks import NanoGPTBenchmark, NanoGPTConfig, _nanogpt_configs

    config: NanoGPTConfig = NanoGPTConfig(dropout=0, n_layer=2)
    config.update(**_nanogpt_configs["test"])
    bench = NanoGPTBenchmark(config=config, device="cpu")
    fn = bench.fn()

    args, kwargs = bench.make_batch()
    jfn = tp_jit(fn)
    result = jfn(*args, **kwargs)

    assert_close(result, fn(*args, **kwargs))
