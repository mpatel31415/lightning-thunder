import numpy as np
import pytest
import torch
from looseversion import LooseVersion
from torch.testing import assert_close
from numbers import Number

import thunder.core.dtypes as dtypes
from thunder.tests.framework import ops, run_snippet, requiresJAX
from thunder.tests.opinfos import opinfos


# Tests for all operators

# TODO: add error inputs tests (like test_elementwise_binary_prim_shape_mismatch and test_elementwise_binary_prim_dtype_mismatch)


def snippet_errors(op, sample, ex_type):
    ex = None
    try:
        op(*sample.args, **sample.kwargs)
    except Exception as e:
        ex = e

    assert ex is not None, f"Expected an exception"
    assert ex_type is type(ex), f"Expected an exception with type {ex_type}, but found ex={ex}"


@ops(tuple(op for op in opinfos if op.error_input_generator is not None))
def test_errors(op, device, _, executor):
    for sample, ex_type in op.error_inputs(device):
        result = run_snippet(snippet_errors, op, device, None, executor.make_callable(op.op), sample, ex_type)
        if result is not None:
            return result


# Snippets run a single test using a single sample
# TODO: should snippets be able to access the original opinfo? -- No?
# TODO: revisit atol/rtol, maybe be more selective about which ops need a more permissive check
def snippet_torch_consistency(op, torch_op, sample):
    thunder_result = op(*sample.args, **sample.kwargs)
    torch_result = torch_op(*sample.args, **sample.kwargs)

    assert_close(thunder_result, torch_result, equal_nan=True, atol=1e-3, rtol=0)


# TODO: consider structuring tests like this to be autogenerated
#   using a snippet and an "extractor" that constructs the args and kwargs for the snippet
# TODO: the name of this test is misleading as it may test operators from a variety of languages,
#   maybe we should cut it up so developers can test just torch operator or just core lang operators
# TODO: extend this test with some reproducible randomness (maybe using hypothesis)
@ops(tuple(op for op in opinfos if op.torch_reference is not None))
def test_core_vs_torch_consistency(op, device, dtype, executor):
    if LooseVersion(torch.__version__) < "1.13" and dtype is dtypes.complex32:
        pytest.skip("complex32 tests on PyTorch versions before 1.13 are skipped!")

    for sample in op.sample_inputs(device, dtype):
        result = run_snippet(
            snippet_torch_consistency,
            op,
            device,
            dtype,
            executor.make_callable(op.op),
            op.torch_reference,
            sample,
        )
        if result is not None:
            return result


def snippet_jax_consistency(op, jax_op, sample):
    jax_sample = sample.jax()

    thunder_result = op(*sample.args, **sample.kwargs)
    jax_result = jax_op(*jax_sample.args, **jax_sample.kwargs)

    # NOTE: this strange unpacking is to handle NumPy's and JAX's sometimes odd
    #   number vs. array representation. In particular, NumPy can mimic
    #   Python numbers, but `asarray` doesn't understand this mimicry
    np_array = np.array(jax_result)
    if np_array.shape == ():
        jax_result = torch.tensor(np_array.item(), device=thunder_result.device)
    else:
        jax_result = torch.asarray(np_array, device=thunder_result.device)

    # NOTE: dtype is not checked because jax will translate int64, float64, and complex128 to int32, float32 and complex64
    assert_close(thunder_result, jax_result, equal_nan=True, check_dtype=False)


# TODO: consider structuring tests like this to be autogenerated
#   using a snippet and an "extractor" that constructs the args and kwargs for the snippet
# TODO: extend this test with some reproducible randomness (maybe using hypothesis)
@ops(tuple(op for op in opinfos if op.jax_reference is not None))
@requiresJAX
def test_core_vs_jax_consistency(op, device, dtype, executor):
    if dtype is dtypes.complex32:
        pytest.skip("jax doesn't support complex32!")
    if dtype is dtypes.bfloat16:
        pytest.skip("jax bfloat16 support is spotty (at least on CPU)")

    for sample in op.sample_inputs(device, dtype):
        result = run_snippet(
            snippet_jax_consistency,
            op,
            device,
            dtype,
            executor.make_callable(op.op),
            op.jax_reference,
            sample,
        )
        if result is not None:
            return result
