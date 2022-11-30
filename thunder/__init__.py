import os
from collections import deque
from functools import wraps
from typing import Callable, Sequence, Optional

import thunder.langs as langs
from thunder.__about__ import *

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
    "make_traced",
]


def make_traced(fn: Callable, executor: Optional[str] = None, language_ctx=langs.torch) -> Callable:
    """Converts a callable in a callable that will be traced and then executed.

    Example usage:

      def foo(a, b):
        return tlang.add(a, b)

      traced_foo = thunder.make_traced(foo)

      a = torch.randn(2, 2, device='cuda')
      b = torch.randn(2, 1, device='cuda')

      result = traced_foo(a, b)
    """

    if executor is None:
        exec_ctx = get_executor_context()
        if exec_ctx is None:
            raise RuntimeError("No executor specified!")
    elif executor == "torch":
        try:
            from .executors.torch import torchCtx
        except ModuleNotFoundError:
            raise RuntimeError(
                "The 'torch' executor was requested, but the `torch` package "
                "is not available. Please make sure the `torch` package is installed"
                "in the environment."
            )
    elif executor == "nvfuser":
        try:
            from .executors.nvfuser import nvFuserCtx
        except ModuleNotFoundError:
            raise RuntimeError(
                "The 'nvfuser' executor was requested, but NVFuser is not available. "
                "Please make sure the `torch` package is installed and CUDA is available."
            )

    @wraps(fn)
    def _fn(*args, **kwargs):
        # Acquires a new tracing context
        trace_token = new_trace()

        executor_ctx = None
        executor_token = None
        if executor == "nvfuser":
            executor_ctx = nvFuserCtx()
            executor_token = set_executor_context(executor_ctx)
        elif executor == "torch":
            executor_ctx = torchCtx()
            executor_token = set_executor_context(executor_ctx)
        else:
            # Uses the existing executor context (if it exists)
            executor_ctx = get_executor_context()
            if executor_ctx is None:
                raise RuntimeError("No executor specified and no existing executor context!")

        lang_ctx = language_ctx.ctx()
        lang_token = set_language_context(lang_ctx)

        t = get_trace()

        # Constructs proxies
        proxy_args = deque()
        for arg in args:
            # NOTE: a very limited exception to the requirement we can proxy all inputs
            # TODO: consider more carefully what we proxy vs. don't and how it's modeled
            if isinstance(arg, Sequence):
                proxy_args.append(arg)
                continue

            p = lang_ctx.proxy(arg)
            t.add_input(p)
            proxy_args.append(p)

        proxy_kwargs = {}
        for k, v in kwargs.items():
            # NOTE: two ugly exceptions to what we proxy -- kwarg strings and dtypes are passed through
            # TODO: consider more carefully what we proxy vs. don't
            if isinstance(v, str):
                proxy_kwargs[k] = v
            elif lang_ctx.is_dtype(v):
                proxy_kwargs[k] = lang_ctx.thunder_dtype(v)
            else:
                p = lang_ctx.proxy(v)
                t.add_kwarg_input(k, p)
                proxy_kwargs[k] = p

        # TODO: support multiple return values
        proxy_result = fn(*proxy_args, **proxy_kwargs)
        t.add_output(proxy_result)

        # print(t)

        result, fusion = executor_ctx.execute(t, *args, **kwargs)

        reset_trace(trace_token)
        reset_language_context(lang_token)
        if executor_token is not None:
            reset_executor_context(executor_token)

        # TODO: convert nvFuser output to appropriate object based on language ctx
        # TODO: if the output is a datastructure it will be flattened before being handed to the executor
        #   this needs to re-wrap the executor outputs into the datstructure
        if len(result) == 1:
            # Hack to unwrap singleton results
            return result[0]

        return result

    return _fn
