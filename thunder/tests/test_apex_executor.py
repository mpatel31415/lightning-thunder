from typing import Any

import pytest
import torch
from torch.testing import assert_close

import thunder
from thunder import dtypes
from thunder.executors.apex_entropyex import deregister_apex_entropyex, register_apex_entropyex
from thunder.tests.framework import instantiate, requiresCUDA, ops, run_snippet
from thunder.tests.opinfos import OpInfo, get_opinfo
import thunder.core.devices as devices

from lightning_utilities.core.imports import package_available

APEX_CROSS_ENTROPY_AVAILABLE = package_available("xentropy_cuda")

xentropy_cuda: None | Any = None
if APEX_CROSS_ENTROPY_AVAILABLE:
    import xentropy_cuda


# NOTE This test modifies the global executor map, so it technically should not
# be run in parallel with other tests
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("device,", ["cuda"])
@requiresCUDA
def test_apex_cross_entropy(device, dtype):
    if not APEX_CROSS_ENTROPY_AVAILABLE:
        pytest.skip("Apex cross entropy is not available")

    try:
        register_apex_entropyex()
        logits = torch.randn([2048, 50257], device=device, dtype=thunder.torch.to_torch_dtype(dtype))
        labels = torch.randint(0, 50257, [2048], device=device)
        expected = torch.nn.functional.cross_entropy(logits, labels, reduction="mean", ignore_index=-1)

        def test(logits, labels):
            return thunder.torch.cross_entropy(logits, labels, reduction="mean", ignore_index=-1)

        ctest = thunder.compile(test, executors_list=["apex_xentropy"])
        actual = ctest(logits, labels)
        torch.testing.assert_close(actual, expected)
        last_trace = thunder.last_traces(ctest)[-1]
        if xentropy_cuda is not None:
            assert any(bsym.sym.name == "apex_cross_entropy" for bsym in last_trace.bound_symbols)
        else:
            assert all(bsym.sym.name != "apex_cross_entropy" for bsym in last_trace.bound_symbols)
    finally:
        deregister_apex_entropyex()


def snippet_torch_consistency(op, torch_op, sample):
    thunder_result = op(*sample.args, **sample.kwargs)
    torch_result = torch_op(*sample.args, **sample.kwargs)
    assert_close(thunder_result, torch_result, equal_nan=True, atol=1e-3, rtol=1e-4)

    last_trace = thunder.last_traces(op)[-1]
    assert any(bsym.sym.name == "apex_cross_entropy" for bsym in last_trace.bound_symbols)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("device,", ["cuda"])
def test_apex_torch_consistency(device, dtype):
    from thunder.executors.apex_entropyex import cross_entropy_checker

    if not APEX_CROSS_ENTROPY_AVAILABLE:
        pytest.skip("Apex cross entropy is not available")

    op = get_opinfo("cross_entropy")

    try:
        register_apex_entropyex()

        def fn(*args, **kwargs):
            return thunder.torch.cross_entropy(*args, **kwargs)

        cfn = thunder.compile(fn, executors_list=["apex_xentropy"])
        at_least_one_supported_input = False
        for sample in op.reference_inputs(device, dtype, requires_grad=False):
            if not cross_entropy_checker(*sample.args, **sample.kwargs):
                continue
            at_least_one_supported_input = True
            result = run_snippet(
                snippet_torch_consistency,
                op,
                device,
                dtype,
                cfn,
                op.torch_reference,
                sample,
            )
            if result is not None:
                return result
        if not at_least_one_supported_input:
            raise ValueError("No supported inputs were generated by the OpInfo")
    finally:
        deregister_apex_entropyex()
