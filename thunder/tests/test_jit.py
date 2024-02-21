from collections.abc import Iterable, Iterator, Sequence
from functools import partial, wraps
from itertools import product
import random

import sys
import dis
from collections.abc import Callable

import pytest
import torch
from torch.testing import assert_close

import thunder
from thunder.core.jit import JITError
from thunder.core.jit_ext import minimal_thunder_jit
import thunder.clang as clang
import thunder.core.prims as prims

#
# Test suite for the thunder.jit entrypoint
#

#
# Args, kwargs, varargs, varkwargs tests
#


def test_basic_kwargs():
    def foo(a, b):
        return a + b

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a=a, b=b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_tuple_kwargs():
    def foo(a, b):
        return a + b[0]

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a=a, b=(a, b))
    expected = foo(a, (a, b))

    assert_close(actual, expected)


def test_kwargs_inorder():
    def foo(a, b):
        return a + b

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(b=b, a=a)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_kwonly_args():
    def foo(*, a, b):
        return a + b

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a=a, b=b)
    expected = foo(a=a, b=b)

    assert_close(actual, expected)


def test_posonly_and_kwonly_args():
    def foo(a, /, b, *, c):
        return a + b + c

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))
    c = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b=b, c=c)
    expected = foo(a, b, c=c)

    assert_close(actual, expected)


#
# Binary operation tests
#


def test_binary_add_tensors():
    def foo(a, b):
        return a + b

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_binary_ops_compare_numbers():
    cmp_ops = ["!=", "==", "<=", "<", ">", ">="]

    inps = (
        (5, 9),
        (2, 8),
        (8, 2),
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        (5.5, 2.2),
        (-5.5, 2.2),
        (5, 2.2),
    )

    for op in cmp_ops:
        d = {}
        exec(f"def foo(a, b): return a {op} b", d)
        foo = d["foo"]
        jfoo = thunder.jit(foo)

        for a, b in inps:
            assert jfoo(a, b) == foo(a, b)


def test_binary_ops_int_numbers():
    # Issue https://github.com/Lightning-AI/lightning-thunder/issues/594 for more ops
    # "<<", ">>",
    int_ops = ["+", "&", "//", "*", "%", "|", "**", "-", "/", "^"]

    int_inps = (
        (5, 9),
        (2, 8),
        (8, 2),
    )

    for op in int_ops:
        d = {}
        exec(f"def foo(a, b): return a {op} b", d)
        foo = d["foo"]
        jfoo = thunder.jit(foo)
        exec(f"def bar(a, b):\n a {op}= b\n return a", d)
        bar = d["bar"]
        jbar = thunder.jit(bar)

        for a, b in int_inps:
            assert jfoo(a, b) == foo(a, b)
            assert jbar(a, b) == bar(a, b)


def test_binary_ops_bool_numbers():
    bool_ops = ["+", "&", "//", "*", "%", "|", "**", "-", "/", "^"]

    bool_inps = (
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    )

    for op in bool_ops:
        d = {}
        exec(f"def foo(a, b): return a {op} b", d)
        foo = d["foo"]
        jfoo = thunder.jit(foo)
        exec(f"def bar(a, b):\n a {op}= b\n return a", d)
        bar = d["bar"]
        jbar = thunder.jit(bar)
        for a, b in bool_inps:
            if op not in {"//", "/", "%"} or b:  # could check exceptions for these
                assert jfoo(a, b) == foo(a, b)
                assert jbar(a, b) == bar(a, b)
            else:
                for fn in foo, jfoo, bar, jbar:
                    with pytest.raises(ZeroDivisionError):
                        fn(a, b)


def test_binary_ops_float_numbers():
    float_ops = ["+", "//", "*", "%", "**", "-", "/"]

    float_inps = (
        (5.5, 2.2),
        (-5.5, 2.2),
        (5, 2.2),
    )

    for op in float_ops:
        d = {}
        exec(f"def foo(a, b): return a {op} b", d)
        foo = d["foo"]
        jfoo = thunder.jit(foo)
        exec(f"def bar(a, b):\n a {op}= b\n return a", d)
        bar = d["bar"]
        jbar = thunder.jit(bar)
        for a, b in float_inps:
            assert jfoo(a, b) == foo(a, b)
            assert jbar(a, b) == bar(a, b)


def test_binary_ops_complex_numbers():
    float_ops = ["+", "*", "**", "-", "/"]

    float_inps = (
        (5.5j, 2.2j),
        (-5.5j, 2.2j),
        (5j, 2.2j),
    )

    for op in float_ops:
        d = {}
        exec(f"def foo(a, b): return a {op} b", d)
        foo = d["foo"]
        jfoo = thunder.jit(foo)
        exec(f"def bar(a, b):\n a {op}= b\n return a", d)
        bar = d["bar"]
        jbar = thunder.jit(bar)
        for a, b in float_inps:
            assert jfoo(a, b) == foo(a, b)
            assert jbar(a, b) == bar(a, b)


def test_binary_add_tensor_number():
    def foo(a, b):
        return a + b

    a = torch.randn((2, 2))
    b = 3

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_hasattr_on_proxies():
    def foo(a, b):
        if hasattr(a, "__why_would_it__"):
            raise Exception("Nobody expects the Spanish Inquisition")
        if hasattr(b, "__why_would_it__"):
            raise Exception("Oh well, never happens")
        return a + b

    a = torch.randn((2, 2))
    b = 3

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_clang_add_tensors():
    def foo(a, b):
        return clang.add(a, b)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = a + b

    assert_close(actual, expected)


def test_prim_add_tensors():
    def foo(a, b):
        return prims.add(a, b)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = a + b

    assert_close(actual, expected)


def test_python_fn_binary_add_tensors():
    def bar(a, b):
        return a + b

    def foo(a, b):
        return bar(a, b)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


def test_torch_add_tensors():
    def foo(a, b):
        return torch.add(a, b)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


#
# String tests
#


def test_string_return():
    def foo(s):
        return s

    jfoo = thunder.jit(foo)
    actual = jfoo("hi")
    expected = "hi"

    assert actual == expected


def test_binary_add_strings():
    def foo(a, b):
        return a + b

    jfoo = thunder.jit(foo)
    actual = jfoo("he", "llo")
    expected = "hello"

    assert actual == expected


#
# Tuple tests
#


def test_tuple_len():
    def foo(tup):
        return len(tup) + 5

    jfoo = thunder.jit(foo)

    tup = (0, 1, 2, 3, 4)

    actual = jfoo(tup)
    expected = foo(tup)

    assert actual == expected


def test_tuple_binary_subscr():
    def foo(tup):
        return tup[0], tup[1:3]

    jfoo = thunder.jit(foo)

    tup = (0, 1, 2, 3, 4)

    actual = jfoo(tup)
    expected = foo(tup)

    assert actual == expected


def test_tuple_assignment():
    def foo(tup):
        tup[0] = 5

    jfoo = thunder.jit(foo)

    tup = (0, 1, 2, 3, 4)

    with pytest.raises(JITError):
        jfoo(tup)


def test_tuple_unpack_repack():
    def foo(tup):
        a, b, *_ = tup
        return a, (b, a)

    jfoo = thunder.jit(foo)

    tup = (0, 1, 2, 3, 4)

    actual = jfoo(tup)
    expected = foo(tup)

    assert actual == expected


def test_tuple_list_conversion():
    def foo(tup):
        return list(tup)

    jfoo = thunder.jit(foo)

    tup = (0, 1, 2, 3, 4)

    actual = jfoo(tup)
    expected = foo(tup)

    assert actual == expected


def test_tuple_enumerate():
    def foo(tup):
        accum = 0
        for x in tup:
            accum += x
        return accum

    jfoo = thunder.jit(foo)

    tup = (0, 1, 2, 3, 4)

    actual = jfoo(tup)
    expected = foo(tup)

    assert actual == expected


def test_tuple_addition():
    def foo(tup0, tup1):
        return tup0 + tup1

    jfoo = thunder.jit(foo)

    tup0 = (0, 1, 2, 3, 4)
    tup1 = (5, 6, 7, 8, 9, 10)

    actual = jfoo(tup0, tup1)
    expected = foo(tup0, tup1)

    assert actual == expected


def test_tuple_list_addition():
    def foo(tup0):
        return tup0 + [5, 6, 7]

    jfoo = thunder.jit(foo)

    tup0 = (0, 1, 2, 3, 4)

    with pytest.raises(TypeError):
        jfoo(tup0)


def test_nested_tuples():
    def foo(tup):
        tup0, tup1 = tup
        tup2, tup3 = tup0
        return tup1, tup3

    jfoo = thunder.jit(foo)

    tup3 = (3, 4)
    tup2 = (1, 2)
    tup1 = (-1, 0)
    tup0 = (tup2, tup3)
    tup = (tup0, tup1)

    actual = jfoo(tup)
    expected = foo(tup)

    assert actual == expected


def test_nested_tuples_with_tensors():
    def foo(tup):
        tup0, tup1 = tup
        tup2, tup3 = tup0
        return tup1, tup3, tup2[0] + tup3[1]

    jfoo = thunder.jit(foo)

    a = torch.randn(2, 2)
    b = torch.randn(2, 2)

    tup3 = (3, 4)
    tup2 = (a, b)
    tup1 = (-1, 0)
    tup0 = (tup2, tup3)
    tup = (tup0, tup1)

    actual = jfoo(tup)
    expected = foo(tup)

    assert_close(actual, expected)


#
# Return value tests
#
# Return values must be proxies or (simple) printable objects (or both)


def test_return_number():
    def foo():
        return 5

    jfoo = thunder.jit(foo)

    expected = foo()
    actual = jfoo()

    assert_close(expected, actual)


def test_return_object():
    def foo():
        return object()

    jfoo = thunder.jit(foo)

    with pytest.raises(RuntimeError):
        jfoo()


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2191")
def test_return_tuple():
    def foo(a, b):
        return (a, b)

    jfoo = thunder.jit(foo)

    expected = foo(5, 3)
    actual = jfoo(5, 3)
    assert_close(expected, actual)


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2191")
def test_return_list():
    def foo(a, b):
        return [a, b]

    jfoo = thunder.jit(foo)

    expected = foo(5, 3)
    actual = jfoo(5, 3)
    assert_close(expected, actual)


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2191")
def test_return_list_with_intermediates():
    def foo(a, b):
        l = [a, b]
        l.append(3)
        return l

    jfoo = thunder.jit(foo)

    expected = foo(5, 3)
    actual = jfoo(5, 3)
    assert_close(expected, actual)


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2191")
def test_return_set():
    def foo(a, b):
        return {a, b}

    jfoo = thunder.jit(foo)

    expected = foo(5, 3)
    actual = jfoo(5, 3)
    assert_close(expected, actual)


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2191")
def test_return_dict():
    def foo(a, b):
        return {1: a, 2: b}

    jfoo = thunder.jit(foo)

    expected = foo(5, 3)
    actual = jfoo(5, 3)
    assert_close(expected, actual)


#
# No caching tests
#
# TODO GTC Simple test that the option works and programs run as expected

#
# Same input caching tests
#
# TODO GTC Verify works and fails as expected


#
# Constant values caching tests
#
def test_constant_values_caching():
    def foo(a, b):
        return a + b

    jfoo = thunder.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    expected = foo(a, b)
    actual = jfoo(a, b)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    actual = jfoo(a, b)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1

    c = torch.rand((2, 1))

    expected = foo(a, c)
    actual = jfoo(a, c)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 1

    actual = jfoo(a, c)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 2

    expected = foo(b, c)
    actual = jfoo(b, c)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 3

    expected = foo(a, 1)
    actual = jfoo(a, 1)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 3
    assert thunder.cache_hits(jfoo) == 3

    actual = jfoo(a, 1)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 3
    assert thunder.cache_hits(jfoo) == 4

    expected = foo(a, 2)
    actual = jfoo(a, 2)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 4
    assert thunder.cache_hits(jfoo) == 4

    actual = jfoo(a, 2)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 4
    assert thunder.cache_hits(jfoo) == 5

    expected = foo("he", "llo")
    actual = jfoo("he", "llo")
    assert expected == actual

    assert thunder.cache_misses(jfoo) == 5
    assert thunder.cache_hits(jfoo) == 5

    actual = jfoo("he", "llo")
    assert expected == actual

    assert thunder.cache_misses(jfoo) == 5
    assert thunder.cache_hits(jfoo) == 6


def test_constant_values_caching_with_tuples():
    def foo(tup0, tup1):
        return tup0[0] + tup1[1]

    jfoo = thunder.jit(foo)

    tup0 = (0, 1)
    tup1 = (2, 3)

    actual = jfoo(tup0, tup1)
    expected = foo(tup0, tup1)

    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    jfoo(tup0, tup1)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1

    tup2 = (0, 1)

    jfoo(tup2, tup1)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 2

    a = torch.randn((2, 2))
    tup3 = (a, 1)

    actual = jfoo(tup3, tup1)
    expected = foo(tup3, tup1)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 2

    b = torch.randn((2, 2))
    tup4 = (b, 1)

    actual = jfoo(tup4, tup1)
    expected = foo(tup4, tup1)

    assert thunder.cache_misses(jfoo) == 2
    assert thunder.cache_hits(jfoo) == 3

    c = torch.randn((2, 1))
    tup5 = (c, 1)

    actual = jfoo(tup5, tup1)
    expected = foo(tup5, tup1)

    assert thunder.cache_misses(jfoo) == 3
    assert thunder.cache_hits(jfoo) == 3


def test_constant_values_caching_with_kwargs():
    def foo(a, b):
        return a + b

    jfoo = thunder.jit(foo)

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    expected = foo(a, b)
    actual = jfoo(a, b=b)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 0

    actual = jfoo(a, b=b)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 1

    actual = jfoo(b=b, a=a)
    assert_close(actual, expected)

    assert thunder.cache_misses(jfoo) == 1
    assert thunder.cache_hits(jfoo) == 2


#
# Symbolic values caching tests
#


def test_symbolic_value_warning():
    def foo():
        return

    with pytest.warns(UserWarning):
        thunder.jit(foo, cache="symbolic values")


#
# Sharp edges tests
#
# See thunder/core/options.py for a longer description of sharp edges and the option.


def test_sharp_edges_pass():
    def foo(a, b):
        return torch.add(a, b)

    jfoo = thunder.jit(foo, sharp_edges="error")

    a = torch.randn((2, 2))
    b = torch.randn((2, 2))

    jfoo = thunder.jit(foo)
    actual = jfoo(a, b)
    expected = foo(a, b)

    assert_close(actual, expected)


#
# Sharp edges -- Unexpected inputs
#

_test_load_global_sharp_edge_global = 3


def test_load_global_sharp_edge():
    def foo(a):
        return torch.add(a, _test_load_global_sharp_edge_global)

    jfoo = thunder.jit(foo, sharp_edges="error")

    a = torch.randn((2, 2))

    with pytest.raises(JITError):
        jfoo(a)


_test_store_global_sharp_edge_global = 4


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2186")
def test_store_global_sharp_edge():
    def foo():
        global _test_store_global_sharp_edge_global
        _test_store_global_sharp_edge_global = 5

    jfoo = thunder.jit(foo, sharp_edges="error")

    with pytest.raises(JITError):
        jfoo()


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2177")
def test_calling_globals_sharp_edge():
    def foo():
        g = globals()

    jfoo = thunder.jit(foo, sharp_edges="error")

    with pytest.raises(JITError):
        jfoo()


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2180")
def test_calling_vars_sharp_edge():
    def foo():
        g = vars()

    jfoo = thunder.jit(foo, sharp_edges="error")

    with pytest.raises(JITError):
        jfoo()


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2178")
def test_calling_locals_sharp_edge():
    def foo():
        l = locals()

    jfoo = thunder.jit(foo, sharp_edges="error")

    with pytest.raises(JITError):
        jfoo()


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2188")
def test_accessing_globals_through_function_sharp_edge():
    def foo():
        return foo.__globals__()

    jfoo = thunder.jit(foo, sharp_edges="error")

    with pytest.raises(JITError):
        jfoo()


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2179")
def test_calling_input_sharp_edge():
    def foo():
        inp = input()

    jfoo = thunder.jit(foo, sharp_edges="error")

    with pytest.raises(JITError):
        jfoo()


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2184")
def test_input_closure_sharp_edge():
    x = 5

    def foo():
        return x

    jfoo = thunder.jit(foo, sharp_edges="error")

    with pytest.raises(JITError):
        jfoo()


# NOTE This is NOT a sharp edge because we assume that loading modules and functions (globally or as closures) are
#   not sharp edges
def test_fn_closure_no_sharp_edge():
    def bar(x):
        return x

    def foo():
        return bar(5)

    jfoo = thunder.jit(foo, sharp_edges="error")

    actual = jfoo()
    expected = foo()
    assert_close(expected, actual)


def _test_fn_global_no_sharp_edge_fn():
    return 7


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2189")
def test_fn_global_no_sharp_edge():
    def foo(x):
        return x + _test_fn_global_no_sharp_edge_fn()

    jfoo = thunder.jit(foo, sharp_edges="error")

    actual = jfoo(2)
    expected = foo(2)
    assert_close(expected, actual)


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2187")
def test_nonlocal_write_sharp_edge():
    x = 5

    def foo():
        nonlocal x
        x = 7

    jfoo = thunder.jit(foo, sharp_edges="error")

    with pytest.raises(JITError):
        jfoo()


# NOTE This is NOT a sharp edge -- this test is to ensure this works as expected
def test_intermediate_closure_not_sharp_edge():
    def foo(x):
        def bar():
            return x

        return bar()

    jfoo = thunder.jit(foo, sharp_edges="error")

    expected = foo(5)
    actual = jfoo(5)

    assert_close(expected, actual)


def test_intermediate_nonlocal_not_sharp_edge():
    def foo(x):
        def bar():
            nonlocal x
            x = 9

        return x

        return bar()

    jfoo = thunder.jit(foo, sharp_edges="error")

    expected = foo(5)
    actual = jfoo(5)

    assert_close(expected, actual)


#
# Sharp edges -- Side effects
#


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2181")
def test_calling_open_sharp_edge():
    def foo():
        open("nonexistent file")

    jfoo = thunder.jit(foo, sharp_edges="error")

    with pytest.raises(JITError):
        jfoo()


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2182")
def test_calling_print_sharp_edge():
    def foo(a):
        print(a)

    jfoo = thunder.jit(foo, sharp_edges="error")

    a = torch.randn((2, 2))

    with pytest.raises(JITError):
        jfoo(a)


#
# Sharp edges -- Random module (surprising inputs and side effects)
#


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2183")
def test_calling_random_seed_sharp_edge():
    def foo():
        random.seed(1234)

    jfoo = thunder.jit(foo, sharp_edges="error")

    state = random.getstate()
    try:
        with pytest.raises(JITError):
            jfoo()
    finally:
        random.setstate(state)


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2183")
def test_calling_random_setstate_sharp_edge():
    def foo():
        return random.getstate()

    jfoo = thunder.jit(foo, sharp_edges="error")

    with pytest.raises(JITError):
        jfoo()


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2183")
def test_calling_random_setstate_sharp_edge():
    def foo(state):
        random.setstate(state)

    jfoo = thunder.jit(foo, sharp_edges="error")

    state = random.getstate()
    with pytest.raises(JITError):
        jfoo(state)


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2183")
def test_calling_random_randbytes_sharp_edge():
    def foo():
        return random.randbytes(20)

    jfoo = thunder.jit(foo, sharp_edges="error")

    with pytest.raises(JITError):
        jfoo()


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2183")
def test_calling_random_randrange_sharp_edge():
    def foo():
        return random.randrange(10)

    jfoo = thunder.jit(foo, sharp_edges="error")

    with pytest.raises(JITError):
        jfoo()


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2183")
def test_calling_random_randint_sharp_edge():
    def foo():
        return random.randint(0, 10)

    jfoo = thunder.jit(foo, sharp_edges="error")

    with pytest.raises(JITError):
        jfoo()


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2183")
def test_calling_random_getrandbits_sharp_edge():
    def foo():
        return random.getrandbits(10)

    jfoo = thunder.jit(foo, sharp_edges="error")

    with pytest.raises(JITError):
        jfoo()


@pytest.mark.xfail(reason="https://github.com/Lightning-AI/lightning-thunder/issues/2183")
def test_calling_random_choice_sharp_edge():
    def foo():
        l = [1, 3, 5]
        return random.choice(l)

    jfoo = thunder.jit(foo, sharp_edges="error")

    with pytest.raises(JITError):
        jfoo()


# Additional random functions
# random.choices
# random.shuffle
# random.sample
# random.binomialvariate
# random.random
# random.uniform
# random.triangular
# random.betavariate
# random.expovariate
# random.gammavariate
# random.gauss
# random.lognormvariate
# random.normalvariate
# random.vonmisesvariate
# random.paretovariate
# random.weibullvariate
# random.SystemRandom (class)


# NOTE This use of randomness is OK (and we could add a Proxy for random.Random)
def test_random_Random_class():
    def foo():
        return random.Random(1234).random()

    jfoo = thunder.jit(foo, sharp_edges="error")

    expected = foo()
    actual = jfoo()

    assert_close(expected, actual)
