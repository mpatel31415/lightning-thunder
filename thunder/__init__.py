import os
from collections import deque
from functools import wraps
from typing import Callable, Sequence, Optional
import time
import inspect

import thunder.langs as langs
import thunder.core.dtypes as dtypes
from thunder.__about__ import *
from thunder.core.pytree import tree_flatten, tree_unflatten
import thunder.core.proxies as proxies

from .core.trace import (
    get_trace,
    new_trace,
    reset_executor_context,
    reset_language_context,
    reset_trace,
    set_executor_context,
    get_executor_context,
    set_language_context,
)

_PACKAGE_ROOT = os.path.dirname(__file__)
_PROJECT_ROOT = os.path.dirname(_PACKAGE_ROOT)

__all__ = [
    # dtype aliases
    "bool8",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "bfloat16",
    "float16",
    "float32",
    "float64",
    "complex32",
    "complex64",
    "complex128",
    # tracing functions
    "make_traced",
]

#
# dtype aliases
#
bool8 = dtypes.bool8
uint8 = dtypes.uint8
int8 = dtypes.int8
int16 = dtypes.int16
int32 = dtypes.int32
int64 = dtypes.int64
bfloat16 = dtypes.bfloat16
float16 = dtypes.float16
float32 = dtypes.float32
float64 = dtypes.float64
complex32 = dtypes.complex32
complex64 = dtypes.complex64
complex128 = dtypes.complex128

#
# tracing functions
#


def _get_executor(executor=None):
    if executor is None:
        ex = get_executor_context()
        if ex is None:
            raise ValueError("No executor specified!")
        return ex

    if executor == "torch":
        try:
            from .executors.torch import torchCtx

            return torchCtx()
        except ModuleNotFoundError:
            raise RuntimeError(
                "The 'torch' executor was requested, but the `torch` package "
                "is not available. Please make sure the `torch` package is installed"
                "in the environment."
            )

    if executor == "nvfuser":
        try:
            from .executors.nvfuser import nvFuserCtx

            return nvFuserCtx()
        except ModuleNotFoundError:
            raise RuntimeError(
                "The 'nvfuser' executor was requested, but NVFuser is not available. "
                "Please make sure the `torch` package is installed and CUDA is available."
            )

    if hasattr(executor, "get_executor_context"):
        return executor.get_executor_context()

    raise ValueError(f"Trying to acquire an executor from unknown object {executor}!")


# TODO: consider how subclasses could be supported
# TODO: consider how proxies are extensible (review JAX's proxy extension mechanism)
# TODO: harvest arg and kwargn names upfront to avoid name collisions with proxies
def _make_proxies(fn, trace, langctx, *args, **kwargs):
    """
    Proxying rules:
        1. All number and tensor inputs are proxied, including if they're in a container.
        2. All other inputs are passed unmodified.
    """

    sig = inspect.signature(fn)
    bound_args = sig.bind_partial(*args)
    varargs_name = inspect.getfullargspec(fn).varargs

    def _convert(x):
        if isinstance(x, (int, float, complex)) or isinstance(x, langctx.tensor_cls):
            # Proxies numbers and tensors
            name = trace.make_proxy_name()
            p = langctx.proxy(x, name=name)
            return p

        if isinstance(x, langctx.dtype_cls):
            # Converts dtypes
            thunder_dtype = langctx.thunder_dtype(x)
            return thunder_dtype

        return x

    proxyargs = []
    for name, arg in bound_args.arguments.items():
        if isinstance(arg, (int, float)) or isinstance(arg, langctx.tensor_cls):
            # NOTE: for numbers or tensors that are passed as positional args,
            #   this just gives them the name of the positional argument
            #   Numbers or tensors in a collection (like a list or dict) are
            #   just given generic names (in the else-block, below)
            p = langctx.proxy(arg, name=name)
            proxyargs.append(p)
        else:
            values, structure = tree_flatten(arg)
            converted_values = list((_convert(v) for v in values))

            packed = tree_unflatten(converted_values, structure)

            # Handles varargs
            if name == varargs_name:
                proxyargs.extend(packed)
            else:
                proxyargs.append(packed)

    proxykwargs = {}
    for name, kwarg in kwargs.items():
        if isinstance(kwarg, (int, float)) or isinstance(kwarg, langctx.tensor_cls):
            # NOTE: for numbers or tensors that are passed as keyword arguments,
            #   this just gives them the name of the argument
            #   Numbers or tensors in a collection (like a list or dict) are
            #   just given generic names (in the else-block, below)
            p = langctx.proxy(kwarg, name=name)
            proxykwargs[name] = p
        else:
            values, structure = tree_flatten(kwarg)
            converted_values = list((_convert(v) for v in values))
            packed = tree_unflatten(converted_values, structure)
            proxykwargs[name] = packed

    return proxyargs, proxykwargs


def _construct_trace(fn, trace, proxyargs, proxykwargs):
    trace.add_args(proxyargs)
    trace.add_kwargs(proxykwargs)
    proxyresult = fn(*proxyargs, **proxykwargs)
    trace.add_outputs(proxyresult)
    return trace


def make_traced(
    fn: Callable, executor: Optional[str] = None, language_ctx=langs.torch, _info=False, _return_fusion=False
) -> Callable:
    """Converts a callable in a callable that will be traced and then executed.

    Example usage:

      def foo(a, b):
        return tlang.add(a, b)

      traced_foo = thunder.make_traced(foo)

      a = torch.randn(2, 2, device='cuda')
      b = torch.randn(2, 1, device='cuda')

      result = traced_foo(a, b)
    """

    ex = _get_executor(executor)
    langctx = language_ctx.ctx()

    @wraps(fn)
    def _fn(*args, **kwargs):
        acquisition_start = time.time_ns()

        # Sets the proper tracing context
        trace_token = new_trace()
        executor_token = set_executor_context(ex)
        lang_token = set_language_context(langctx)

        trace = get_trace()
        proxyargs, proxykwargs = _make_proxies(fn, trace, langctx, *args, **kwargs)

        trace = _construct_trace(fn, trace, proxyargs, proxykwargs)

        acquisition_end = time.time_ns()

        translation_start = time.time_ns()
        fusion = ex.fuse(trace)
        translation_end = time.time_ns()

        invocation_start = time.time_ns()
        result = fusion(*args, **kwargs)
        invocation_end = time.time_ns()

        # Resets the tracing context
        reset_trace(trace_token)
        reset_language_context(lang_token)
        if executor_token is not None:
            reset_executor_context(executor_token)

        meta = None
        if _info:
            meta = {
                "acquisition_time": acquisition_end - acquisition_start,
                "invocation_time": invocation_end - invocation_start,
                "translation_time": translation_end - translation_start,
            }

        if _info and _return_fusion:
            return result, meta, fusion
        if _info:
            return result, meta
        if _return_fusion:
            return result, fusion
        return result

    return _fn
