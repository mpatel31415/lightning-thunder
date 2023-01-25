import math
from collections import namedtuple
import pytest
from functools import partial

# TODO: make this import conditional on Torch being available and querying if should test
#   with torch
import torch
from torch.testing import make_tensor

import thunder.core.lang as tlang
import thunder.core.dtypes as datatypes
from thunder.langs.torch import torch_dtype

from .framework import _all_device_types

# Returns a noncontiguous (tensor with the same shape and values as t
# The noncontiguous tensor is constructed such that elements in the innermost
#   dimension are separated by zeros or (whenever possible) nans
# TODO: consider more complicated noncontiguity schemes
def noncontiguous_like(t):
    # Short-circuits if t is already noncontiguous
    if not t.is_contiguous():
        return t

    # Choose a "weird" value that won't be accessed
    if t.dtype.is_floating_point or t.dtype.is_complex:
        value = math.nan
    elif t.dtype == torch.bool:
        value = True
    else:
        value = 12

    result = t.new_empty(t.shape + (2,))
    result[..., 0] = value
    result[..., 1] = t.detach()
    result = result[..., 1]
    result.requires_grad_(t.requires_grad)
    return result


class SampleInput:
    """Represents sample inputs to a function."""

    __slots__ = [
        "args",
        "kwargs",
    ]

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    # TODO: print kwargs
    def __repr__(self):
        arg_string = ", ".join(tuple(str(a) for a in self.args))
        return f"[SampleInput args=({arg_string})]"

    # Applies the transform f(t) -> t to each tensor and dtype in the SampleInput
    def transform(self, f):
        def tt(t):
            def _tt(t):
                with torch.no_grad():
                    return f(t)

            if isinstance(t, torch.Tensor):
                return _tt(t)
            elif isinstance(t, torch.dtype):
                return _tt(t)
            elif isinstance(t, list):
                return list(map(tt, t))
            elif isinstance(t, tuple):
                return tuple(map(tt, t))
            elif isinstance(t, dict):
                return {k: tt(v) for k, v in t.items()}
            else:
                return t

        return SampleInput(tt(self.args), tt(self.kwargs))

    def noncontiguous(self):
        def to_noncontiguous(t):
            if isinstance(t, torch.Tensor):
                return noncontiguous_like(t)
            elif isinstance(t, torch.dtype):
                return t

            return t

        return self.transform(to_noncontiguous)


# TODO: add executor
class DecorateInfo(object):
    """Describes which test, or type of tests, should be wrapped in the given
    decorator when testing an operator. Any test that matches all provided
    arguments will be decorated. The decorator will only be applied if the
    active_if argument is True."""

    __slots__ = [
        "decorator",
        "test_template_name",
        "executors",
        "devicetypes",
        "dtypes",
        "active_if",
    ]

    def __init__(
        self,
        decorator,
        test_template_name=None,
        *,
        executors=None,
        devicetypes=None,
        dtypes=None,
        active_if=True,
    ):
        self.decorator = decorator
        self.test_template_name = test_template_name
        self.executors = executors
        self.devicetypes = devicetypes
        self.dtypes = None if dtypes is None else datatypes.resolve_dtypes(dtypes)
        self.active_if = active_if

    def is_active(self, test_template_name, executor, devicetype, dtype):
        return (
            self.active_if
            and (self.executors is None or executor.name in self.executors)
            and (self.test_template_name is None or self.test_template_name == test_template_name)
            and (self.devicetypes is None or devicetype in self.devicetypes)
            and (self.dtypes is None or dtype in self.dtypes)
        )


Domain = namedtuple("Domain", "low high")
opinfos = []

# TODO: require use of generic Thunder dtypes (once they exist)
class OpInfo:
    """Operator information and helper functions for acquiring it."""

    def __init__(
        self,
        op,
        *,
        name=None,
        devicetypes=None,
        dtypes=None,
        sample_input_generator,
        benchmark_generator=None,
        method_variant=None,
        operator_variant=None,
        torch_reference=None,
        numpy_reference=None,
        test_directives=(),
        domain=(None, None),
    ):
        self.op = op
        self.name = name if name is not None else op.__name__
        self._devicetypes = devicetypes if devicetypes is not None else _all_device_types()
        self._dtypes = dtypes if dtypes is not None else (datatypes.exact, datatypes.inexact)
        self.sample_input_generator = sample_input_generator
        self.benchmark_generator = benchmark_generator
        self.method_variant = method_variant
        self.operator_variant = operator_variant
        self.torch_reference = torch_reference
        self.numpy_reference = numpy_reference
        self.test_directives = test_directives
        self.domain = Domain(*domain)

    def __call__(self, *args, **kwargs):
        """Calls the function variant of the operator."""
        return self.op(*args, **kwargs)

    # TODO: maybe allow sample input generation not using torch
    # NOTE: Today all sample inputs are generated with PyTorch, so Thunder objects,
    #   like dtypes, need to be translated into PyTorch objects
    def sample_inputs(self, device_type, dtype, *, requires_grad=False, **kwargs):
        dtype = torch_dtype(dtype)
        return self.sample_input_generator(self, device_type, dtype, requires_grad, **kwargs)

    # NOTE: Today all benchmarks are generated with PyTorch, so Thunder objects,
    #   like dtypes, need to be translated into PyTorch objects
    def benchmarks(self, device_type, dtype, *, requires_grad=False, **kwargs):
        dtype = torch_dtype(dtype)
        return self.benchmark_generator(self, device_type, dtype, requires_grad, **kwargs)

    def device_types(self):
        return set(self._devicetypes)

    def dtypes(self, device_type=None):
        if device_type is not None:
            raise NotImplementedError

        return datatypes.resolve_dtypes(self._dtypes)

    # TODO: add executor
    def test_decorators(self, test_name, executor, devicetype, dtype):
        return [d.decorator for d in self.test_directives if d.is_active(test_name, executor, devicetype, dtype)]


#
# Elementwise Unary OpInfos
#

# TODO: create elementwise unary OpInfo subclass and maybe auto add to list
elementwise_unary_ops = []


# TODO: add numbers
# TODO: add small value, large value, and extremal-valued samples
def elementwise_unary_generator(op, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, low=op.domain.low, high=op.domain.high)

    shapes = (
        # TODO: restore size zero cases
        # (0, 2, 1),
        # (5, 0, 3),
        (),
        (11,),
        (4, 4),
        (1024, 1024),
        (64, 64, 64),
    )

    # Typical inputs
    for shape in shapes:
        yield SampleInput(make_arg(shape))

    # Noncontiguous inputs
    for shape in shapes:
        yield SampleInput(make_arg(shape, noncontiguous=True))

    # Arbitrarily strided inputs
    # shape, strides, offset
    strided_cases = (
        ((5, 6, 2), (1, 1, 7), 2),
        ((5, 5, 4), (1, 1, 7), 2),
        ((5, 5, 2), (4, 5, 7), 3),
        ((5, 5, 2), (5, 5, 7), 3),
        ((5, 5, 2), (5, 5, 5), 3),
        ((9, 5, 2), (0, 1, 7), 3),
    )

    for shape, strides, offset in strided_cases:
        a = make_arg(
            500,
        ).as_strided(shape, strides, offset)
        yield SampleInput(a)


def elementwise_unary_benchmarks(op, device, dtype, requires_grad, **kwargs):
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    # name x shape
    cases = (
        ("8x8", (8, 8)),
        ("64x64", (64, 64)),
        ("1024x1024", (1024, 1024)),
    )

    for name, shape in cases:
        yield name, SampleInput(make_arg(shape))


class ElementwiseOpInfo(OpInfo):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ElementwiseUnaryOpInfo(ElementwiseOpInfo):
    def __init__(
        self,
        *args,
        sample_input_generator=elementwise_unary_generator,
        benchmark_generator=elementwise_unary_benchmarks,
        **kwargs,
    ):
        super().__init__(
            *args,
            sample_input_generator=sample_input_generator,
            benchmark_generator=elementwise_unary_benchmarks,
            **kwargs,
        )

        elementwise_unary_ops.append(self)


abs_opinfo = ElementwiseUnaryOpInfo(
    tlang.abs,
    torch_reference=torch.abs,
    test_directives=(
        # Torch doesn't support CPU bool abs
        DecorateInfo(
            pytest.mark.xfail, "test_core_vs_torch_consistency", dtypes=(datatypes.bool8,), devicetypes=("cpu",)
        ),
    ),
)

acos_opinfo = OpInfo(
    tlang.acos,
    domain=(-1, 1),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.acos,
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 acos
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=("cpu",),
        ),
    ),
)
elementwise_unary_ops.append(acos_opinfo)

acosh_opinfo = OpInfo(
    tlang.acosh,
    domain=(1, math.inf),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.acosh,
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 acosh
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=("cpu",),
        ),
        DecorateInfo(pytest.mark.xfail, executors=("nvFuser",)),
    ),
)
elementwise_unary_ops.append(acosh_opinfo)

asin_opinfo = OpInfo(
    tlang.asin,
    domain=(-1, 1),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.asin,
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 asin
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=("cpu",),
        ),
    ),
)
elementwise_unary_ops.append(asin_opinfo)

atan_opinfo = OpInfo(
    tlang.atan,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.atan,
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 atan
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=("cpu",),
        ),
    ),
)
elementwise_unary_ops.append(atan_opinfo)

atanh_opinfo = OpInfo(
    tlang.atanh,
    domain=(-1, 1),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.atanh,
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 atanh
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=("cpu",),
        ),
    ),
)
elementwise_unary_ops.append(atanh_opinfo)

bitwise_not_opinfo = OpInfo(
    tlang.bitwise_not,
    dtypes=(datatypes.exact,),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.bitwise_not,
)
elementwise_unary_ops.append(bitwise_not_opinfo)

ceil_opinfo = OpInfo(
    tlang.ceil,
    dtypes=(datatypes.floating, datatypes.exact),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.ceil,
    test_directives=(
        # Torch doesn't support bool ceil
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bool8,),
        ),
        # Torch doesn't support cpu float16 ceil
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=("cpu",),
        ),
    ),
)
elementwise_unary_ops.append(ceil_opinfo)

cos_opinfo = OpInfo(
    tlang.cos,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.cos,
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 cos
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=("cpu",),
        ),
    ),
)
elementwise_unary_ops.append(cos_opinfo)

cosh_opinfo = OpInfo(
    tlang.cosh,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.cosh,
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 cosh
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=("cpu",),
        ),
    ),
)
elementwise_unary_ops.append(cosh_opinfo)

erf_opinfo = OpInfo(
    tlang.erf,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.erf,
    test_directives=(
        # Torch doesn't support CPU float16 erf
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=("cpu",),
        ),
        # Torch doesn't support complex erf
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating,),
        ),
    ),
)
elementwise_unary_ops.append(erf_opinfo)

erfc_opinfo = OpInfo(
    tlang.erfc,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.erfc,
    test_directives=(
        # Torch doesn't support CPU float16 erfc
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=("cpu",),
        ),
        # Torch doesn't support complex erfc
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating,),
        ),
    ),
)
elementwise_unary_ops.append(erfc_opinfo)

exp_opinfo = OpInfo(
    tlang.exp,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.exp,
    test_directives=(
        # Torch doesn't support CPU float16 or complex32 exp
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=("cpu",),
        ),
    ),
)
elementwise_unary_ops.append(exp_opinfo)

expm1_opinfo = OpInfo(
    tlang.expm1,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.expm1,
    test_directives=(
        # Torch doesn't support CPU float16 expm1
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=("cpu",),
        ),
        # Torch doesn't support complex expm1
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.complexfloating,),
        ),
    ),
)
elementwise_unary_ops.append(expm1_opinfo)

floor_opinfo = OpInfo(
    tlang.floor,
    dtypes=(datatypes.floating, datatypes.exact),
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.floor,
    test_directives=(
        # Torch doesn't support bool floor
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.bool8,),
        ),
        # Torch doesn't support cpu float16 floor
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16,),
            devicetypes=("cpu",),
        ),
    ),
)
elementwise_unary_ops.append(floor_opinfo)

isfinite_opinfo = OpInfo(
    tlang.isfinite,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.isfinite,
    test_directives=(
        # nvFuser doesn't correctly return outputs as boolean tensors, and doesn't support full
        DecorateInfo(
            pytest.mark.xfail,
            executors=("nvFuser",),
        ),
        # Torch preserves the uint8 dtype
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.uint8,),
        ),
    ),
)
elementwise_unary_ops.append(isfinite_opinfo)

tanh_opinfo = OpInfo(
    tlang.tanh,
    sample_input_generator=elementwise_unary_generator,
    torch_reference=torch.tanh,
    test_directives=(
        # See https://github.com/csarofeen/pytorch/issues/2360
        DecorateInfo(
            pytest.mark.xfail, "test_core_vs_torch_consistency", executors=("nvFuser",), dtypes=(datatypes.complex64,)
        ),
        # NOTE: Torch doesn't support CPU float16 or complex32 tanh
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            dtypes=(datatypes.float16, datatypes.complex32),
            devicetypes=("cpu",),
        ),
    ),
)
elementwise_unary_ops.append(tanh_opinfo)


# Puts all opinfos into the "opinfos" list
opinfos.extend(elementwise_unary_ops)


#
# Elementwise Binary OpInfos
#

# TODO: create elementwise binary OpInfo subclass and maybe auto add to list
elementwise_binary_ops = []


# TODO: extend this generator
def elementwise_binary_generator(op, device, dtype, requires_grad, **kwargs):
    a = make_tensor((4, 4), device=device, dtype=dtype)
    b = make_tensor((4, 4), device=device, dtype=dtype)

    yield SampleInput(a, b)

    # Tests broadcasting
    c = make_tensor((4, 1), device=device, dtype=dtype)
    yield SampleInput(a, c)


# TODO: update dtypes with Thunder dtypes (when they exist)
add_opinfo = OpInfo(
    tlang.add,
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.add,
)
elementwise_binary_ops.append(add_opinfo)

# NOTE: nvFuser does not currently support uint8, int8, or int16
bitwise_and_opinfo = OpInfo(
    tlang.bitwise_and,
    dtypes=(datatypes.exact,),
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.bitwise_and,
)
elementwise_binary_ops.append(bitwise_and_opinfo)

lt_opinfo = OpInfo(
    tlang.lt,
    # NOTE: less than is only defined for real numbers
    dtypes=(datatypes.exact, datatypes.floating),
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.lt,
)
elementwise_binary_ops.append(lt_opinfo)

mul_opinfo = OpInfo(
    tlang.mul,
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.mul,
)
elementwise_binary_ops.append(mul_opinfo)

pow_opinfo = OpInfo(
    tlang.pow,
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.pow,
    test_directives=(
        # NOTE: PyTorch doesn't support bool pow
        DecorateInfo(pytest.mark.xfail, "test_core_vs_torch_consistency", dtypes=(datatypes.bool8,)),
        # NOTE: PyTorch doesn't support cpu float16 pow
        DecorateInfo(
            pytest.mark.xfail,
            "test_core_vs_torch_consistency",
            devicetypes=("cpu",),
            dtypes=(datatypes.float16, datatypes.complex32),
        ),
        # See https://github.com/csarofeen/pytorch/issues/2361
        DecorateInfo(
            pytest.mark.xfail, "test_core_vs_torch_consistency", executors=("nvFuser,"), dtypes=(datatypes.complex64,)
        ),
    ),
)
elementwise_binary_ops.append(pow_opinfo)

sub_opinfo = OpInfo(
    tlang.sub,
    sample_input_generator=elementwise_binary_generator,
    torch_reference=torch.sub,
    test_directives=(
        # torch doesn't support bool sub
        DecorateInfo(pytest.mark.xfail, "test_core_vs_torch_consistency", dtypes=(datatypes.bool8,)),
    ),
)
elementwise_binary_ops.append(sub_opinfo)


# Puts all opinfos into the "opinfos" list
opinfos.extend(elementwise_binary_ops)
