from __future__ import annotations
import time
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
from collections.abc import Sequence
from itertools import filterfalse
from functools import partial

import thunder
import thunder.core.prims as prims
from thunder.core.baseutils import BoundSymbolInterface
from thunder.core.proxies import Proxy, variableify, Variable, TensorProxy, unvariableify
from thunder.core.pytree import tree_flatten, tree_map, tree_unflatten
from thunder.core.symbol import BoundSymbol, BoundSymbolRHS, has_tags
from thunder.core.trace import from_trace, TraceProvenance, TraceCtx as Trace, tracectx
from thunder.core.utils import ProxyDict, producers, check

if TYPE_CHECKING:
    from thunder.core.proxies import ProxyInterface
    from thunder.core.symbol import Symbol, VariableInterface


#
# Common optimization and transform passes
#
# NOTE This file avoids transforms depending on passes, since passes depend on transforms


# Modifies an existing BoundSymbol, removing its "no-op" subsymbols (recursively) which perform no operations
def _remove_noop_subsymbols(bsym: BoundSymbol) -> None:
    nsbsyms: list[BoundSymbol] = []
    sbsym: BoundSymbol
    for sbsym in bsym.subsymbols:
        if len(sbsym.subsymbols) == 0 and not sbsym.sym.is_prim:
            continue

        _remove_noop_subsymbols(sbsym)
        nsbsyms.append(sbsym)

    bsym.subsymbols = nsbsyms


def _inplace_copy_sanity_check(extrace: Trace):
    """The sanity check is based on the sharp edge of nvfuser's `add_ouput(output, input)` interface,
    it makes sure that the `copy_to` argument of `prims.copy_` is not used as input for any of its subsequent operators in a nvFusion fused operator

    Anti-pattern:

    .. code-block:: python

        [t2] = nvFusion0(x, y)
            # result = prims.mul(x, y)
            # a = prims.copy_(result, x)
            # t2 = prims.add(a, y) or t2 = prims.add(x, y)

    Do not use the `copy_to` variable `x` or `a` after it has been updated, use the `copy_from` variable `result` instead to reflect the dependency:

    .. code-block:: python

        [t2] = nvFusion0(x, y)
            # result = prims.mul(x, y)
            # a = prims.copy_(result, x)
            # t2 = prims.add(result, y)
    """

    from thunder.core.utils import consumers

    nvfuser_symbols = (bsym for bsym in extrace.bound_symbols if bsym.sym.name.startswith("nvFusion"))
    for bsym in nvfuser_symbols:
        consumer_dict = consumers(list(bsym.subsymbols), _map_to_numbers=True)
        inplace_copy_idx = ((idx, sym) for idx, sym in enumerate(bsym.subsymbols) if sym.sym.id == prims.PrimIDs.COPY_)
        for idx, subbsym in inplace_copy_idx:
            copy_to_arg = subbsym.flat_args[1]
            copy_to_out = subbsym.output

            def check(inp, log_str):
                if inp is not None and inp in consumer_dict:
                    last_used_idx = max(consumer_dict[inp])
                    if last_used_idx > idx:
                        raise NotImplementedError(
                            f"{bsym.subsymbols[last_used_idx]} trying to use {inp} (the {log_str} of 'prims.copy_') as input, which is not safe."
                            f" There is a risk of accessing the wrong memory. If you are sure you don't want to use this check, it can be disabled by setting `disable_inplace_copy_check=True` in `thunder.jit`."
                        )

            check(copy_to_arg, "'copy_to' argument")
            check(copy_to_out, "output")


# TODO This calls variableify(), but we could directly construct Variable objects instead, which might slightly
#   improve performance
# Runs a Dead Code Elimination (DCE) pass
# NOTE Today we are only interested in computations that produce proxies, so this will eliminate operations
#   that only produce non-proxy objects
# NOTE needed_proxies is an in/out argument, it takes an initial set of Variables you want to keep, and return
#   all the needed proxies of the input trace
def dce(trace: Trace, needed_proxies: None | set[Variable] = None) -> Trace:
    start_time_ns = time.perf_counter_ns()

    producer_map: ProxyDict = producers(trace)

    flat_trace_outputs, _ = tree_flatten(trace.output)
    if needed_proxies is None:
        needed_proxies: set[Variable] = set(tuple(variableify(x) for x in flat_trace_outputs if isinstance(x, Proxy)))
    else:
        needed_proxies.update(tuple(variableify(x) for x in flat_trace_outputs if isinstance(x, Proxy)))
    dced = []

    bsym: BoundSymbol
    for bsym in reversed(trace.bound_symbols):
        # Preserves symbols that should never be collected
        if has_tags(bsym, {prims.OpTags.DONT_DCE}):
            needed = True
        else:
            needed = False

        # NOTE This block is run even if we know we're preserving the operation, because it
        #   may mark some of the operation's outputs as unused
        some_unused = False
        for out in bsym.flat_proxy_outs:
            if variableify(out) in needed_proxies and producer_map[out] == bsym:
                needed = True
            else:
                some_unused = True

        if needed:
            nbsym: BoundSymbol = bsym

            # Replaces unused Proxy outputs with None
            if some_unused:

                def _helper(x):
                    if isinstance(x, Proxy) and (variableify(x) not in needed_proxies or producer_map[x] != bsym):
                        return None
                    return x

                nbsym_output = tree_map(_helper, bsym.output)
                nbsym = bsym.from_bsym(output=nbsym_output)

            # Eliminates no-op subsymbols
            # NOTE In general editing subsymbols doesn't do anything, but no-op subsymbols are a pain
            #   for transforms to deal with. Transforms typically look for a "flattened" version of an
            #   operator for which they can apply their rules, and no-op subsymbols have no
            #   flattening, requiring each transform handle them explicitly or DCE them themselves
            #   while flattening.
            _remove_noop_subsymbols(nbsym)

            dced.append(nbsym)
            for x in nbsym.flat_proxy_args:
                needed_proxies.add(variableify(x))

    dcetrace = from_trace(trace)
    dcetrace.bound_symbols = list(reversed(dced))

    end_time_ns = time.perf_counter_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000
    dcetrace.set_provenance(TraceProvenance(f"Dead Code Elimination (took {elapsed_time_millis} milliseconds)"))

    return dcetrace


#
# Functions related to the Common Subexpression Elimination (CSE) pass
#
def replace_redundant_inputs(
    redundant_map: dict[Variable, Proxy], bsyms: Sequence[BoundSymbol]
) -> Sequence[BoundSymbol]:
    new_bsyms = []
    for bsym in bsyms:
        # Checks if the bound symbol has redundant inputs (that need to be replaced)
        has_redundant_inputs: bool = False
        for x in bsym.flat_proxy_args:
            if Variable(x) in redundant_map:
                has_redundant_inputs = True
                break

        # Bound symbols without redundant inputs need no modification
        if not has_redundant_inputs:
            new_bsyms.append(bsym)
            continue

        # Bound symbols with redundant inputs have to remap those inputs, and the
        #   inputs of their subsymbols, to the original computations
        new_bsym = bsym.from_bsym_swap_proxies(
            redundant_map,
            skip_inputs=False,
            skip_output=False,
            skip_subsymbols=False,
        )
        new_bsyms.append(new_bsym)

    return new_bsyms


# These are ops that are not referentially transparent. We need to treat such
# ops specially when optimizing; for example, CSE cannot coalesce two calls
# into one for ops in this set.
NON_FUNCTIONAL_OPS: set[prims.PrimIDs | str] = {
    prims.PrimIDs.UNIFORM,
    "torch.uniform",  # this doesn't exist as of the PR
    "torch.uniform_like",  # this doesn't exist as of the PR
    # thunder.core.prims doesn't support. See https://pytorch.org/docs/stable/generated/torch.rand.html.
    # "torch.rand",
    # "torch.rand_like",
    "torch.nn.functional.dropout",
    "torch.nn.functional.scaled_dot_product_attention",
}


# This helper function applies cse transformation to bound symbol that is not a fusion.
def cse_single_bsym(
    redundant_map: dict[Variable, Proxy],
    rhs_to_bsym_map: dict[BoundSymbolRHS, BoundSymbolInterface],
    bsym: BoundSymbolInterface,
) -> BoundSymbolInterface:
    check(
        bsym.sym.is_fusion != True,
        lambda: f"Expected bound symbol not to be a fusion in _cse_single_bsym",
        exception_type=AssertionError,
    )

    # `NON_FUNCTIONAL_OPS` are a op that's not deterministic, for example, `torch.nn.functiona.dropout`
    # and `torch.nn.functional.scaled_dot_product_attention` depending on PRNG.
    if bsym.sym.id in NON_FUNCTIONAL_OPS:
        return bsym

    # We can replace any redundant `bsym.args` and `bsym.kwargs` if it is available in the current context.
    new_bsym = bsym.from_bsym_swap_proxies(
        swap_map=redundant_map,
        skip_inputs=False,
        skip_output=True,
    )

    # Skip appending this bsym to the new bound symbols due to its rhs being a common subexpression.
    rhs = new_bsym.rhs
    if (prior_bsym := rhs_to_bsym_map.get(rhs)) is not None and bsym._executor is prior_bsym._executor:
        for src, dst in zip(bsym.flat_outs, prior_bsym.flat_outs):
            # Detects (and avoids) aliasing
            vsrc, vdst = variableify(src), variableify(dst)
            if vsrc == vdst:
                continue
            redundant_map[vsrc] = dst
        return None
    else:
        rhs_to_bsym_map[rhs] = bsym
        return new_bsym


# TODO Update the replacement of redundant proxies to use a visitor pattern
#   when that architecture is added in the future
def cse(trace: Trace) -> Trace:
    """Remove bound symbols whose right hand side is common expression.

    This does two things:
        1. Removes bound symbols if their right hand side expression is already seen.
        2. Replaces variables of arguments and keyword arguments of the bound symbols to use and
           their subsymbols with the preceding ones using the map of a variable to another with
           the same right hand side expression.

    Say we have `foo` defined in the below code snippet.

    .. code::

        def foo(x, y):
            a = x + y
            b = y - x
            c = x + y  # Expected to be removed in favor of `a`.
            d = y - x  # Expected to be removed in favor of `b`.
            z = a + b  # Expected to be intact.
            w = c + d  # Expected to be converted to `w = a + b` and then removed in favor of `z`.
            m = w + 1  # Expected to be updated to `m = z + 1`.
            return z, w, m  # Expected to be (z, z, z + 1)

        # CPU tensors
        @torch.no_grad()
        def thunder_140374621612752(x, y):
          # x: "cpu f32[2, 2]" =  x  (trivial unpack)
          # y: "cpu f32[2, 2]" =  y  (trivial unpack)
          t0 = torch.add(x, y)  # t0: "cpu f32[2, 2]"
          t1 = torch.sub(y, x)  # t1: "cpu f32[2, 2]"
          del [y, x]
          t4 = torch.add(t0, t1)  # t4: "cpu f32[2, 2]"
          del [t0, t1]
          t6 = torch.add(t4, 1)  # t6: "cpu f32[2, 2]"
          return (t4, t4, t6)

        # CUDA tensors & nvFuser
        @torch.no_grad()
        def thunder_140410131706304(x, y):
          # x: "cuda:0 f32[2, 2]" =  x  (trivial unpack)
          # y: "cuda:0 f32[2, 2]" =  y  (trivial unpack)
          (t4, t6) = nvFusion0(x, y)
            # t0 = prims.add(x, y)  # t0: "cuda:0 f32[2, 2]"
            # t1 = prims.sub(y, x)  # t1: "cuda:0 f32[2, 2]"
            # t4 = prims.add(t0, t1)  # t4: "cuda:0 f32[2, 2]"
            # t6 = prims.add(t4, 1.0)  # t6: "cuda:0 f32[2, 2]"
          del [x, y]
          return (t4, t4, t6)

    Args:
        trace:

    Returns:
        :class:`TraceCtx` with common subexpression eliminated.
    """
    start_time_ns = time.perf_counter_ns()

    cse_trace = from_trace(trace)

    cse_trace_bound_symbols = []
    rhs_to_bsym_map = {}
    redundant_map = {}

    # Identifies redundant operations and maps redundant proxies to originally
    #   computed proxies
    cse_bound_symbols = map(partial(cse_single_bsym, redundant_map, rhs_to_bsym_map), trace.bound_symbols)
    cse_trace_bound_symbols = tuple(filterfalse(lambda a: a is None, cse_bound_symbols))

    # Updates the bound symbols in the trace
    # NOTE This uses the cse_trace_bound_symbols list, which has filtered
    #   redundant symbols
    new_bsyms = replace_redundant_inputs(redundant_map, cse_trace_bound_symbols)
    cse_trace.bound_symbols = new_bsyms

    end_time_ns = time.perf_counter_ns()
    elapsed_time_ns = end_time_ns - start_time_ns
    elapsed_time_millis = elapsed_time_ns // 1000000
    cse_trace.set_provenance(
        TraceProvenance(f"Common Subexpression Elimination (took {elapsed_time_millis} milliseconds)")
    )
    return cse_trace


# Base class for all types of Transform.
class Transform:
    pass


# Below are the types of Transform that user can create and apply to the `jitted` function.
class EarlyTransform(Transform, ABC):
    """
    EarlyTransform enables transforming prologue, computation and epilogue trace.
    Note that the computation trace here is before the autograd transform, so any update to
    the computation trace will also update backward trace.
    """

    def transform_traces(self, prologue_trace: Trace, computation_trace: Trace, epilogue_trace: Trace | None, **kwargs):
        # default to noop
        return prologue_trace, computation_trace, epilogue_trace

    def transform_module(self, model: thunder.ThunderModule):
        """Transforms the ThunderModule. This is executed once on application of the transform"""
        pass

    def transform_state_dict_for_submodule(
        self, model: thunder.ThunderModule, submodule_name: str, state_dict: dict
    ) -> dict:
        """
        Implement this to transform the state dict (mostly parameters and buffers) of a module, e.g. when loading
        from a state dict of the original model.

        Expected to return a state dict (for chaining or populating overrides).

        Note that state dict keys do not include the submodule name as prefix.
        """
        return state_dict


class AdditionalTransform(Transform, ABC):
    """
    AdditionalTransform enables transforming the computation trace before optimization pass.
    Note that this transform is only applicable if autograd is disabled.
    """

    @abstractmethod
    def transform_trace(self, computation_trace: Trace, **kwargs):
        pass


class PostOptimizationTransform(Transform, ABC):
    """
    PostOptimizationTransform EarlyTransform enables transforming computation trace after optimization pass.
    Note that this transform will also be applied to the backward trace if the the autograd transform was enabled.
    """

    @abstractmethod
    def transform_trace(self, computation_trace: Trace, **kwargs):
        pass


def check_inplace_to_views(computation_trace: Trace) -> dict[VariableInterface, TensorProxy]:
    """Error out if in-place op that outputs of different number of elements from the input and the input has other consumers."""
    from thunder.core import utils
    import thunder.torch as ltorch

    producer_bsyms = producers(computation_trace)
    trace_args_set = ProxyDict()
    for a in filter(
        lambda a: isinstance(a, TensorProxy), tree_flatten((computation_trace.args, computation_trace.kwargs))[0]
    ):
        trace_args_set[a] = a

    # note(crcrpar): Why not using :func:`~thunder.core.symbol.has_tags`?
    # Because it looks into `.sym.tags` of the input bsym and its subsymbols,
    # thus even `ltorch.batch_norm` is regarded as `prims.OpTags.IN_PLACE`.
    def has_tag(bsym: BoundSymbol, tag: prims.OpTags) -> bool:
        return bsym.sym.tags and tag in bsym.sym.tags

    swap_map: dict[VariableInterface, TensorProxy] = {}
    consumers = utils.consumers(computation_trace)
    bsym: BoundSymbol
    for bsym in filter(lambda b: has_tag(b, prims.OpTags.IN_PLACE), computation_trace.bound_symbols):
        in_tensor: TensorProxy = list(filter(lambda p: isinstance(p, TensorProxy), bsym.flat_proxy_args))[0]

        if in_tensor in trace_args_set:
            continue
        prod_bsym: BoundSymbol = producer_bsyms[in_tensor]

        flat_tensor_proxy_args = tuple(filter(lambda p: isinstance(p, TensorProxy), prod_bsym.flat_args))
        if not flat_tensor_proxy_args:
            # assuming `prod_bsym` is a tensor factory method such as `torch.empty`, `torch.zeros`, and `torch.ones`
            continue
        orig_tensor = flat_tensor_proxy_args[0]
        consumer_of_orig_tensor = consumers[orig_tensor]
        # When the orig tensor is not used by consumers other than `prod_bsym`, it'd be safe.
        # Otherwise, we'd need to replace the use of ``orig_tensor`` with a view, unless the original
        # is an arg or a kwarg.
        if len(consumer_of_orig_tensor) == 1:
            continue

        utils.check(
            prod_bsym.sym not in ltorch._syms_returning_runtime_dependently_views,
            lambda: (
                f"in-place op of `{bsym.sym.id}` to `{prod_bsym.sym.id}` output `{in_tensor}` is not "
                f"supported. It's unclear if the output of "
                f"{tuple(s.id for s in ltorch._syms_returning_runtime_dependently_views)} is "
                f"a copy, a view, or the input itself, as per https://pytorch.org/docs/stable/tensor_view.html"
            ),
            NotImplementedError,
        )
        if prod_bsym.sym not in ltorch._syms_returning_views:
            continue

        utils.check(
            orig_tensor.numel == in_tensor.numel,
            lambda: (
                f"in-place op of `{bsym.sym.id}` to `{in_tensor}`, a view tensor of "
                f"`{orig_tensor}` is not supported because {in_tensor.numel} != {orig_tensor.numel}"
            ),
            NotImplementedError,
        )

        swap_map[variableify(orig_tensor)] = in_tensor
    return swap_map


def is_functionalizable(bsym: BoundSymbol) -> bool:
    """Has `OpTags.IN_PLACE` and its args are NOT ``computation_trace.args`` nor ``computation_trace.kwargs``."""
    from thunder.torch import _inplace_to_out_of_place

    return (
        bsym.sym in _inplace_to_out_of_place and bsym.subsymbols and bsym.subsymbols[-1].sym.id == prims.PrimIDs.COPY_
    )


def _get_first_tensor_arg(bsym: BoundSymbol) -> TensorProxy | None:
    tensor_args = list(filter(lambda p: isinstance(p, TensorProxy), bsym.flat_args))
    if not tensor_args:
        return None
    return tensor_args[0]


def _get_prod_bsym_with_arg(
    t: TensorProxy,
    producer_map: ProxyDict,
    trace_args: ProxyDict,
) -> BoundSymbol | None:
    from thunder.torch import _syms_returning_views

    def inplace_or_view(bsym) -> bool:
        sym = bsym.sym
        return (sym.tags and prims.OpTags.IN_PLACE in sym.tags) or sym in _syms_returning_views

    if t not in producer_map:
        return None
    prod_bsym: BoundSymbol = producer_map[t]
    first_tensor = _get_first_tensor_arg(prod_bsym)
    if first_tensor is None:
        return None

    if inplace_or_view(prod_bsym):
        if first_tensor in trace_args:
            return prod_bsym
        else:
            return _get_prod_bsym_with_arg(first_tensor, producer_map, trace_args)
    else:
        return None


def collect_required_copy_bsyms_for_args(
    trace: Trace,
    removed_copy_bsyms: list[tuple[int, BoundSymbol]],
) -> tuple[list[BoundSymbol], dict[VariableInterface, TensorProxy]]:
    """Identify required pairs of copy src and copy dst."""

    flat_args: tuple[TensorProxy, ...] = tuple(
        filter(lambda p: isinstance(p, TensorProxy), tree_flatten((trace.args, trace.kwargs))[0])
    )
    trace_args_set = ProxyDict()
    for i, a in enumerate(flat_args):
        trace_args_set[a] = i

    dst_to_copy_bsym: dict[VariableInterface, tuple[int, list[BoundSymbol]]] = {}
    unused_copy_bsyms: list[tuple[int, BoundSymbol]] = []
    for idx, copy_bsym in removed_copy_bsyms:
        if (copy_dst := copy_bsym.flat_proxy_args[1]) in trace_args_set:
            copy_src = _get_first_tensor_arg(copy_bsym)
            dst_to_copy_bsym[variableify(copy_dst)] = idx, [copy_bsym]
        else:
            unused_copy_bsyms.append((idx, copy_bsym))

    producer_map = producers(trace)
    new_dst_to_copy_bsym: dict[VariableInterface, tuple[int, list[BoundSymbol]]] = {}
    for idx, copy_bsym in unused_copy_bsyms:
        copy_src = _get_first_tensor_arg(copy_bsym)
        copy_dst: TensorProxy = copy_bsym.flat_proxy_args[1]
        optional_prod_bsym = _get_prod_bsym_with_arg(copy_dst, producer_map, trace_args_set)
        if optional_prod_bsym is None:
            continue

        new_copy_dst = _get_first_tensor_arg(optional_prod_bsym)

        new_bsyms: list[BoundSymbol] = []
        reshaped_copy_src = copy_src
        with tracectx(trace):
            if copy_src.shape != new_copy_dst.shape:
                dst_shape = new_copy_dst.shape
                reshaped_copy_src = prims.reshape.meta(copy_src, dst_shape)
                reshape_bsym = prims.reshape.bind(copy_src, dst_shape, output=reshaped_copy_src)

                new_bsyms.append(reshape_bsym)
            new_copy_return = prims.copy_.meta(reshaped_copy_src, new_copy_dst)
            new_copy_bsym = prims.copy_.bind(reshaped_copy_src, new_copy_dst, output=new_copy_return)
            new_bsyms.append(new_copy_bsym)

            if (var_t := variableify(new_copy_dst)) not in new_dst_to_copy_bsym:
                new_dst_to_copy_bsym[var_t] = idx, new_bsyms
            else:
                cur_copy_bsym_idx, _ = new_dst_to_copy_bsym[var_t]
                if cur_copy_bsym_idx < idx:
                    new_dst_to_copy_bsym[var_t] = idx, new_bsyms

    required_copy_bsyms = [None for _ in range(len(flat_args))]
    for var_t in dst_to_copy_bsym:
        idx = trace_args_set[unvariableify(var_t)]
        cur_copy_bsym_idx, cur_copy_bsym = dst_to_copy_bsym[var_t]
        if var_t not in new_dst_to_copy_bsym:
            required_copy_bsyms[idx] = cur_copy_bsym
        else:
            new_copy_bsym_idx, new_copy_bsyms = new_dst_to_copy_bsym[var_t]
            if cur_copy_bsym_idx < new_copy_bsym_idx:
                required_copy_bsyms[idx] = new_copy_bsyms
            else:
                required_copy_bsyms[idx] = cur_copy_bsym
    for var_t in new_dst_to_copy_bsym:
        if var_t not in dst_to_copy_bsym:
            _, bsyms = new_dst_to_copy_bsym[var_t]
            required_copy_bsyms[trace_args_set[unvariableify(var_t)]] = bsyms

    sorted_copy_bsyms: list[BoundSymbol] = []
    for list_of_bsyms in required_copy_bsyms:
        if list_of_bsyms is None:
            continue
        sorted_copy_bsyms.extend(list_of_bsyms)

    swap_map_for_return: dict[VariableInterface, TensorProxy] = {}
    for bsym in filter(lambda bsym: bsym.sym.id == prims.PrimIDs.COPY_, sorted_copy_bsyms):
        swap_map_for_return[variableify(bsym.flat_proxy_args[0])] = bsym.flat_proxy_args[1]
    return sorted_copy_bsyms, swap_map_for_return


def functionalize_inplace_ops(
    computation_trace: Trace, orig_to_view_swap_map: dict[VariableInterface, TensorProxy]
) -> list[Trace]:
    r"""Functionalize in-place ops in ``computation_trace``.

    This function is skipped if kwarg of ``skip_inplace_functionalization=True`` is passed to :func:`thunder.jit`.

    In thunder, an in-place is a pair of an out-of-place or functional op followed by :func:`thunder.core.prims.copy_`.
    This function replaces in-place ops with out-of-place ops.

    This function returns an empty list if no in-place ops are found in ``computation_trace``.
    If any are found, functionalization is done in two steps. The first step is to canonicalize the trace
    by making sure that any operands of in-place ops have only one consumer. The second step is to
    replace in-place ops with their out-of-place versions, i.e.,
    to remove ``prims.copy_`` from the trace then inserting required ``prims.copy_``\s. The required ``prims.copy_``\s are ones
    whose destination is ``computation_trace``'s args/kwargs.

    Let's take a look at how the functionalization is working for the following ``f``. This function applies in-place ``exp`` to ``a`` which does not share
    the storage of the argument ``x``, thus the functionalized trace would be
    free from ``prims.copy_``.

    .. code-block:: python
        :name: input-func

        def f(x):
            a = x.sin()
            a.exp_()
            return a.cos()

    The input trace which is the first of many computation traces is as follows. Notice that ``a`` is consumed by not only ``ltorch.exp_`` but also ``ltorch.cos``.

    .. code-block:: python
        :name: initial-trace

        def computation(x):
          # x: "cpu f32[3]"
          a = ltorch.sin(x)  # a: "cpu f32[3]"
            # a = prims.sin(x)  # a: "cpu f32[3]"

          t2 = ltorch.exp_(a)  # t2: "cpu f32[3]"
            # t1 = ltorch.exp(a)  # t1: "cpu f32[3]"
              # t1 = prims.exp(a)  # t1: "cpu f32[3]"
            # t2 = prims.copy_(t1, a)  # t2: "cpu f32[3]"

          t3 = ltorch.cos(a)  # t3: "cpu f32[3]"
            # t3 = prims.cos(a)  # t3: "cpu f32[3]"
          return t3

    The output of the first step ("canonicalization") is as follows. Notice that now ``ltorch.cos`` takes
    ``t2`` which is the return value of ``ltorch.exp_(a)``.

    .. code-block:: python
        :name: canonicalized-traces

        # Constructed by Intermediate trace of `functionalize_inplace_ops`
        def computation(x):
          # x: "cpu f32[3]"
          a = ltorch.sin(x)  # a: "cpu f32[3]"
            # a = prims.sin(x)  # a: "cpu f32[3]"

          t2 = ltorch.exp_(a)  # t2: "cpu f32[3]"
            # t1 = ltorch.exp(a)  # t1: "cpu f32[3]"
              # t1 = prims.exp(a)  # t1: "cpu f32[3]"
            # t2 = prims.copy_(t1, a)  # t2: "cpu f32[3]"

          t3 = ltorch.cos(t2)  # t3: "cpu f32[3]"
            # t3 = prims.cos(t2)  # t3: "cpu f32[3]"
          return t3

    The functionalized trace is as follows. Notice that this trace has no ``prims.copy_``\s
    and the operand of ``ltorch.cos`` is updated to ``t1`` from ``t2``.

    .. code-block:: python
        :name: functionalized-traces

        # Constructed by Functionalize in-place ops
        def computation(x):
          # x: "cpu f32[3]"
          a = ltorch.sin(x)  # a: "cpu f32[3]"
            # a = prims.sin(x)  # a: "cpu f32[3]"

          t1 = ltorch.exp(a)  # t1: "cpu f32[3]"
            # t1 = ltorch.exp(a)  # t1: "cpu f32[3]"
              # t1 = prims.exp(a)  # t1: "cpu f32[3]"

          t3 = ltorch.cos(t1)  # t3: "cpu f32[3]"
            # t3 = prims.cos(t1)  # t3: "cpu f32[3]"
          return t3

    Another example to take a look at would be a function with one or more in-place ops to its argument(s) and/or their views.
    The ``g`` defined below cannot be functionalized with appropriate ``prims.copy_`` to ``x`` because ``a`` shares its storage with ``x``.

    .. code-block:: python
        :name: input-func-with-multiple-inplace

        def g(x):
            a = x.view(-1)
            a.exp_()
            a.sin_()
            return a.cos()

    The input trace to this function is as follows.

    .. code-block:: python
        :name: input-trace-with-multiple-inplace

        def computation(x):
          # x: "cpu f32[2, 2]"
          a = ltorch.view(x, -1)  # a: "cpu f32[4]"
            # a = ltorch.reshape(x, (-1,))  # a: "cpu f32[4]"

          t2 = ltorch.exp_(a)  # t2: "cpu f32[4]"
            # t1 = ltorch.exp(a)  # t1: "cpu f32[4]"
            # t2 = prims.copy_(t1, a)  # t2: "cpu f32[4]"

          t4 = ltorch.sin_(a)  # t4: "cpu f32[4]"
            # t3 = ltorch.sin(a)  # t3: "cpu f32[4]"
            # t4 = prims.copy_(t3, a)  # t4: "cpu f32[4]"

          t5 = ltorch.cos(a)  # t5: "cpu f32[4]"
            # t5 = prims.cos(a)  # t5: "cpu f32[4]"
          return t5

    For ``g``, the copy from ``a.sin_()`` to ``x`` is critical. If it's missing,
    the consumers of ``x`` including ``g`` itself would get broken.
    Below are the outputs of this function. The top is "canonicalized" (see the change
    of ``ltorch.sin_``'s operand).
    The bottom is the functionalized trace. Since ``a`` has a different shape from ``x``,
    there are ``prims.reshape(t3, (2, 2))`` and ``prims.copy_(t8, x)``.

    .. code-block:: python
        :name: functionalization-output-traces

        # Constructed by Intermediate trace of `functionalize_inplace_ops`
        def computation(x):
          # x: "cpu f32[2, 2]"
          a = ltorch.view(x, -1)  # a: "cpu f32[4]"
            # a = ltorch.reshape(x, (-1,))  # a: "cpu f32[4]"

          t2 = ltorch.exp_(a)  # t2: "cpu f32[4]"
            # t1 = ltorch.exp(a)  # t1: "cpu f32[4]"

          t4 = ltorch.sin_(t2)  # t4: "cpu f32[4]"
            # t3 = ltorch.sin(t2)  # t3: "cpu f32[4]"
            # t4 = prims.copy_(t3, t2)  # t4: "cpu f32[4]"

          t5 = ltorch.cos(t4)  # t5: "cpu f32[4]"
            # t5 = prims.cos(t4)  # t5: "cpu f32[4]"
          return t5

        # Constructed by Functionalize in-place ops
        def computation(x):
          # x: "cpu f32[2, 2]"

          a = ltorch.view(x, -1)  # a: "cpu f32[4]"
          t1 = ltorch.exp(a)  # t1: "cpu f32[4]"
          t3 = ltorch.sin(t1)  # t3: "cpu f32[4]"
          t5 = ltorch.cos(t3)  # t5: "cpu f32[4]"

          t8 = prims.reshape(t3, (2, 2))  # t8: "cpu f32[2, 2]"
          t9 = prims.copy_(t8, x)  # t9: "cpu f32[2, 2]"
          return t5

    .. seealso::

        `PyTorch Docs - Tensor Views <https://pytorch.org/docs/stable/tensor_view.html>`_

    Args:
        computation_trace: A computation trace created by ``interpreter``.
        orig_to_view_swap_map:
    """
    from thunder.torch import _inplace_to_out_of_place

    if not any(is_functionalizable(bsym) for bsym in computation_trace.bound_symbols):
        return []

    # Step 1: return the tensors returned from `prims.copy_` as possible not the args for clarity.
    bsym: BoundSymbol
    swap_map: dict[VariableInterface, ProxyInterface] = {}
    bsyms: list[BoundSymbol] = []
    tensors_observed: set[VariableInterface] = set()
    for bsym in computation_trace.bound_symbols:
        new_bsym = bsym.from_bsym_swap_proxies(swap_map)

        cur_orig_to_view_swap_map: dict[VariableInterface, TensorProxy] = {}
        for t in filter(lambda p: isinstance(p, TensorProxy), new_bsym.flat_args):
            if (var_t := variableify(t)) not in tensors_observed:
                tensors_observed.add(var_t)
            else:
                if var_t in orig_to_view_swap_map:
                    var_view_t = variableify(orig_to_view_swap_map[var_t])
                    check(var_view_t in swap_map, lambda: f"{var_view_t} not in {swap_map}, {orig_to_view_swap_map = }")
                    cur_orig_to_view_swap_map[var_t] = swap_map[var_view_t]
        if cur_orig_to_view_swap_map:
            with tracectx(computation_trace):
                for var_orig, view in cur_orig_to_view_swap_map.items():
                    view_of_orig_shape = prims.reshape.meta(view, unvariableify(var_orig).shape)
                    reshape_bsym = prims.reshape.bind(view, unvariableify(var_orig).shape, output=view_of_orig_shape)
                    cur_orig_to_view_swap_map[var_orig] = view_of_orig_shape
                    bsyms.append(reshape_bsym)
        new_bsym = new_bsym.from_bsym_swap_proxies(cur_orig_to_view_swap_map, skip_output=True)

        if not is_functionalizable(new_bsym):
            bsyms.append(new_bsym)
            continue

        copy_bsym = bsym.subsymbols[-1]
        copy_out = copy_bsym.flat_proxy_outs[0]
        copy_dst = copy_bsym.flat_proxy_args[1]
        swap_map[variableify(copy_dst)] = copy_out
        # make sure an in-place bsym returns `prims.copy_` output
        new_bsym = new_bsym.from_bsym_swap_proxies(swap_map, skip_inputs=True, skip_subsymbols=True)
        bsyms.append(new_bsym)

    intermediate_trace = from_trace(computation_trace)
    intermediate_trace.bound_symbols = bsyms
    intermediate_trace.set_provenance(TraceProvenance("Intermediate trace of `functionalize_inplace_ops`"))

    intermediate_trace.bound_symbols[-1] = intermediate_trace.bound_symbols[-1].from_bsym_swap_proxies(swap_map)
    return_bsym = intermediate_trace.bound_symbols[-1]
    for t in filter(lambda p: isinstance(p, TensorProxy), return_bsym.flat_args):
        check(
            (var_t := variableify(t)) not in swap_map,
            lambda: f"{return_bsym.flat_args=}. `{t}` should have been replaced by `{swap_map[var_t]}`",
        )

    # Step 2: Remove `prims.copy_` if it's the last one of `bsym.subsymbols`,
    # unless `copy_to` is `computation_trace.args` or `computation_trace.kwargs`
    bsym_inplace_to_functional = {}
    swap_map.clear()

    new_bsyms: list[BoundSymbol] = []
    # NOTE(crcrpar): The first value of a tuple is an indicator of how the copy src is new.
    removed_copy_bsyms: list[tuple[int, BoundSymbol]] = []
    for idx, bsym in enumerate(intermediate_trace.bound_symbols):
        new_bsym = bsym.from_bsym_swap_proxies(swap_map)

        if not is_functionalizable(new_bsym):
            new_bsyms.append(new_bsym)
            continue

        copy_bsym = bsym.subsymbols[-1]
        copy_return = copy_bsym.flat_proxy_outs[0]
        copy_from = copy_bsym.flat_proxy_args[0]
        removed_copy_bsyms.append((idx, copy_bsym))
        swap_map[variableify(copy_return)] = copy_from
        new_bsym.subsymbols = new_bsym.subsymbols[:-1]
        new_bsym = new_bsym.from_bsym_swap_proxies(swap_map)

        functional_sym: Symbol
        optional_inplace_arg_index: int
        functional_sym, optional_inplace_arg_index = _inplace_to_out_of_place[new_bsym.sym]

        args, kwargs = new_bsym.args, new_bsym.kwargs
        if optional_inplace_arg_index > -1:
            # Update `inplace` from `True` to `False`. e.g. `relu(x, inplace=True)` -> `relu(x, inplace=False)`
            flat_args, flat_args_spec = tree_flatten((args, kwargs))
            flat_args[optional_inplace_arg_index] = False
            args, kwargs = tree_unflatten(flat_args, flat_args_spec)
        new_functional_bsym = functional_sym.bind(
            *args,
            **kwargs,
            output=new_bsym.output,
            subsymbols=new_bsym.subsymbols,
            _call_ctx=new_bsym._call_ctx,
        )
        new_bsyms.append(new_functional_bsym)
        bsym_inplace_to_functional[new_bsym] = new_functional_bsym

    required_copy_bsyms, swap_map_for_return = collect_required_copy_bsyms_for_args(
        intermediate_trace,
        removed_copy_bsyms,
    )
    functionalized_computation_trace = from_trace(computation_trace)
    return_bsym = new_bsyms[-1].from_bsym_swap_proxies(swap_map_for_return)
    functionalized_computation_trace.bound_symbols = new_bsyms[:-1] + required_copy_bsyms + [return_bsym]
    functionalized_computation_trace.set_provenance(TraceProvenance("Functionalize in-place ops"))
    # note(crcrpar): I kind of want to do the following two.
    # functionalized_computation_trace._provenance.swap_map = swap_map
    # functionalized_computation_trace._provenance.bsym_inplace_to_functional = bsym_inplace_to_functional
    return [intermediate_trace, functionalized_computation_trace]
