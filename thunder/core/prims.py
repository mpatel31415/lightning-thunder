from enum import Enum, auto
import operator
from numbers import Number
import builtins
from functools import partial

from .trace import get_trace
from .proxies import NumberProxy, IntegerProxy, TensorProxy, proxy
import thunder.core.utils as utils
from .utils import (
    check,
    check_same_dtype,
    same_shape,
    dtype_to_type,
    get_numberlike_type,
    get_numberlike_value,
)

# This file defines Thunder's "primitive" operations. These are the
#   "building blocks" for all of Thunder's operators.

# Transforms and analysis defined on the primitive operations should
#   be inherited by the operation's they're composed of.

# This file depends on trace.py, proxies.py, and utils.py.

__all__ = [
    # Methods and datastructures for constructing primitive operations
    "make_prim",
    # Data movement and transformation prims
    "convert_element_type",
    # Elementwise unary prims
    "abs",
    # Elementwise binary prims
    "add",
    "div",
    "sub",
    # Shape prims
    "broadcast_in_dim_meta",
    "broadcast_in_dim",
    # Reduction prims
    "reduction_meta",
    "sum_meta",
    "sum",
    "var",
    "var_meta",
]


class Ops(Enum):
    CONVERT_ELEMENT_TYPE = auto()
    ABS = auto()
    ADD = auto()
    DIV = auto()
    SUB = auto()
    BROADCAST_IN_DIM = auto()
    SUM = auto()
    VAR = auto()


# maps from operators to their meta functions
# declared here but updated below
ops_to_meta_functions_map = {}
ops_to_pretty_name_map = {}

# Prim defintions
# A primitive definition needs to do the following:
#   - define the op's meta function
#       The meta function maps proxy inputs to a proxy output that has the same metadata
#       as the result of calling the operation with inputs that have the same metadata as the proxies.
#       Meta functions are called within a tracing context. TODO: relax this.
#   - call make_prim

# TODO: add error context


class Prim(object):
    """
    A call to a primitive.

    Holds the inputs, outputs, and printing information.
    """

    # TODO: support returning multiple values
    def __init__(self, op, name, result, *args, **kwargs):
        self.op = op
        self.name = name
        self.result = result
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        result_string = str(self.result)
        arg_string = ", ".join(str(arg) for arg in self.args)
        kwarg_string = ", ".join(f"{k}={v}" for k, v in self.kwargs.items())
        return f"[Prim {self.name}, \n\tresult=({result_string}), \n\targs=({arg_string}), \n\tkwargs={{{kwarg_string}}}]"


def make_prim(id, name, meta):
    ops_to_meta_functions_map[id] = meta
    ops_to_pretty_name_map[id] = name

    # TODO: update the signature
    # TODO: improve kwarg handling
    def _fn(*_args, **kwargs):
        t = get_trace()

        # Identifies number constants in args
        # TODO: review other types of constants
        def _extract_constant(x):
            if isinstance(x, Number) and not isinstance(x, NumberProxy):
                p = proxy(x)
                t.add_constant(p)
                return p
            return x

        args = tuple(map(_extract_constant, _args))

        result = meta(*args, **kwargs)
        sym = Prim(id, name, result, *args, **kwargs)
        t.add_symbol(sym)
        return result

    return _fn


#
# Data movement and transformation prims
#


def _convert_element_type_meta(a, dtype):
    if isinstance(a, Number):
        typ = utils.dtype_to_type(dtype)
        return proxy(typ(utils.get_numberlike_value(a)))

    # a is a Tensor
    return TensorProxy(tensor=a, dtype=dtype)


convert_element_type = make_prim(
    Ops.CONVERT_ELEMENT_TYPE, "convert_element_type", _convert_element_type_meta
)


#
# Elementwise unary prims
#

# Elementwise unary prims

# Elementwise unary prims to implement:
# "acos",
# "acosh",
# "asin",
# "asinh",
# "atan",
# "atanh",
# "cos",
# "cosh",
# "bitwise_not",
# "cbrt",
# "ceil",
# "digamma",
# "erf",
# "erf_inv",
# "erfc",
# "erfcx",
# "exp",
# "expm1",
# "exp2",
# "fill",
# "floor",
# "imag",
# "isfinite",
# "lgamma",
# "log",
# "log1p",
# "log2",
# "log10",
# "ndtri",
# "neg",
# "real",
# "reciprocal",
# "round",
# "sign",
# "signbit",
# "sin",
# "sinh",
# "sqrt",
# "tan",
# "tanh",
# "trunc",

# nvFuser unary ops (from https://github.com/pytorch/pytorch/blob/master/torch/_prims/nvfuser_prims.py)
# "acos",
# "asin",
# "atan",
# "atanh",
# "cos",
# "cosh",
# "bitwise_not",
# "ceil",
# "erf",
# "erfc",
# "exp",
# "expm1",
# "floor",
# "imag",
# "isfinite",
# "lgamma",
# "log",
# "log1p",
# "log2",
# "log10",
# "reciprocal",
# "neg",
# "real",
# "round",
# "rsqrt",
# "sign",
# "sin",
# "sinh",
# "sqrt",
# "tan",
# "tanh",
# "trunc",


def _elementwise_unary_meta(a, *, name, number_handler=None, **kwargs):
    # Tensor case
    if isinstance(a, TensorProxy):
        return TensorProxy(tensor=a)

    # Number case
    check(
        isinstance(a, Number),
        lambda: f"Elementwise unary primitives don't support inputs of type {type(a)}!",
    )

    check(
        number_handler is not None,
        lambda: f"The elementwise unary primitive {name} doesn't support number inputs!",
    )

    a_typ = get_numberlike_type(a)
    va = get_numberlike_value(a)
    value = number_handler(va)
    return proxy(value)


abs = make_prim(
    Ops.ABS,
    "abs",
    partial(_elementwise_unary_meta, name="abs", number_handler=builtins.abs),
)

#
# Elementwise binary prims
#
# "atan2",
# "bitwise_and",
# "bitwise_or",
# "bitwise_xor",
# # 'complex',  # needs custom meta
# "eq",
# "fmax",
# "fmin",
# "fmod",
# "gcd",
# "ge",
# "gt",
# "hypot",
# "igamma",
# "igammac",
# "le",
# "lt",
# "maximum",
# "minimum",
# "mul",
# "ne",
# "nextafter",
# "pow",
# "remainder",
# "rsqrt",
# "shift_left",
# "shift_right_arithmetic",
# "shift_right_logical",  # not implemented
# "zeta",

# nvFuser binary ops (from https://github.com/pytorch/pytorch/blob/master/torch/_prims/nvfuser_prims.py)
# "atan2",
# "bitwise_and",
# "bitwise_or",
# "bitwise_xor",
# "eq",
# "fmod",
# "ge",
# "gt",
# "le",
# "lt",
# "mul",
# "ne",
# "pow",
# "remainder",

# TODO: add type promotion (ex. abs complex->float type promotion)
# TODO: document elementwise binary meta, incl. stride logic
def _elementwise_binary_meta(a, b, *, name, number_handler=None, **kwargs):

    # Tensors or Number inputs only
    if not isinstance(a, (TensorProxy, Number)):
        raise ValueError(f"Unexpected type {type(a)}!")
    if not isinstance(b, (TensorProxy, Number)):
        raise ValueError(f"Unexpected type {type(b)}!")

    # Inputs must have the same dtype
    utils.check_same_dtype(a, b)

    # tensor x tensor case
    if isinstance(a, TensorProxy) and isinstance(b, TensorProxy):
        check(
            same_shape(a.shape, b.shape),
            lambda: f"Elementwise binary primitives require the shapes of the inputs tensors to be the same! But got shapes {a.shape} and {b.shape}!",
        )
        return TensorProxy(tensor=a)

    # scalar x scalar case
    if isinstance(a, Number) and isinstance(b, Number):
        check(
            number_handler is not None,
            lambda: f"The elementwise binary primitive {name} doesn't support number x number inputs!",
        )

        va, vb = get_numberlike_value(a), get_numberlike_value(b)
        value = number_handler(va, vb)
        return proxy(value)

    # tensor x scalar case
    tensor = a if isinstance(a, TensorProxy) else b
    number = b if tensor is a else a

    return TensorProxy(tensor=tensor)


add = make_prim(
    Ops.ADD,
    "add",
    partial(_elementwise_binary_meta, name="add", number_handler=operator.add),
)


def _div_number_handler(a, b):
    if isinstance(a, (float, complex)):
        return a / b

    # int (and bool) case, performs floor division
    return a // b


div = make_prim(
    Ops.DIV,
    "div",
    partial(_elementwise_binary_meta, name="div", number_handler=_div_number_handler),
)

sub = make_prim(
    Ops.SUB,
    "sub",
    partial(_elementwise_binary_meta, name="sub", number_handler=operator.sub),
)

#
# Shape prims
#


def broadcast_in_dim_meta(a, shape, broadcast_dimensions, **kwargs):
    return TensorProxy(shape=shape, dtype=a.dtype)


broadcast_in_dim = make_prim(
    Ops.BROADCAST_IN_DIM,
    "broadcast_in_dim",
    broadcast_in_dim_meta,
)

#
# Reduction prims
#


def _compute_reduction_output_shape(shape, dims):
    for idx in dims:
        utils.validate_idx(len(shape), idx)

    new_shape = []
    for idx in range(len(shape)):
        if idx in dims:
            continue

        new_shape.append(shape[idx])

    return tuple(new_shape)


def reduction_meta(a, dims, *, output_dtype=None, **kwargs):
    """
    Meta function for single output reduction operations
    """

    if output_dtype is None:
        output_dtype = a.dtype

    output_shape = _compute_reduction_output_shape(a.shape, dims)

    return TensorProxy(
        shape=output_shape,
        dtype=output_dtype,
    )


sum_meta = reduction_meta
sum = make_prim(Ops.SUM, "sum", sum_meta)


def var_meta(a, dims, *, correction, **kwargs):
    if utils.is_complex_dtype(a.dtype):
        output_dtype = utils.corresponding_real_dtype(a.dtype)
    else:
        output_dtype = a.dtype
    return reduction_meta(a, dims, output_dtype=output_dtype)


var = make_prim(Ops.VAR, "var", var_meta)
