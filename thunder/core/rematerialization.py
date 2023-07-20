from dataclasses import dataclass, replace
from functools import partial
from itertools import chain
from typing import Callable, Optional, Sequence, Tuple, Union

from igraph import Graph

from thunder.core import prims, utils
from thunder.core.baseutils import BoundSymbolInterface, ProxyInterface
from thunder.core.trace import from_trace, TraceCtx, TraceProvenance
from thunder.core.proxies import TensorProxy
from thunder.core.pytree import tree_flatten, tree_unflatten


def find_external_producer_outputs(
    trace: TraceCtx,
    producer: BoundSymbolInterface,
    consumer: BoundSymbolInterface,
) -> Tuple[ProxyInterface, ...]:
    """Find producer's outputs that must be included in the output of the
    producer because they are used by other consumers.

    Args:
        trace (TraceCtx): Trace object.
        producer (BoundSymbolInterface): Producer node.
        consumer (BoundSymbolInterface): Consumer node.

    Returns:
        Tuple[ProxyInterface, ...]: Producer's outputs that must be included in
        the output of the producer.
    """
    proxy_to_consumers = utils.consumers(trace)

    def filter_func(out: ProxyInterface):
        consumers = proxy_to_consumers.get(out, tuple())
        consumers = tuple(filter(lambda x: x.sym.name != "del", consumers))
        return len(consumers) == 1 and out.name in (x.name for x in consumer.args)

    rematerializable_producer_outputs = tuple(filter(filter_func, producer.output))

    return tuple(x for x in producer.output if x.name not in (y.name for y in rematerializable_producer_outputs))


def find_external_consumer_inputs(
    producer: BoundSymbolInterface,
    consumer: BoundSymbolInterface,
) -> Tuple[ProxyInterface, ...]:
    """Find consumer's inputs that must be included in the input of the
    consumer because they are produced by other producers.

    Args:
        producer (BoundSymbolInterface): Producer node.
        consumer (BoundSymbolInterface): Consumer node.

    Returns:
        Tuple[ProxyInterface, ...]: Consumer's inputs that must be included in
        the input of the consumer.
    """
    external_consumer_inputs_names = tuple(
        sorted(
            set(x.name for x in consumer.args)
            - set(x.name for x in producer.output)
            - set(x.name for x in producer.args)
        )
    )
    return tuple(x for x in consumer.args if x.name in external_consumer_inputs_names)


def apply_rematerialization_for_producer(
    trace: TraceCtx,
    producer: BoundSymbolInterface,
    consumer: BoundSymbolInterface,
    cut: Sequence[Union[ProxyInterface, str]],
) -> BoundSymbolInterface:
    """Update the producer node with the cut information.

    Args:
        producer (BoundSymbolInterface): Producer node.
        cut (Sequence[Union[ProxyInterface, str]]): Cut information.

    Returns:
        BoundSymbolInterface: Updated producer node.
    """
    # It's simple to update the producer node, all we need to do is to update
    # the producer's output with the cut information and the external outputs.
    cut_names = tuple(map(lambda x: x.name, cut)) if isinstance(cut[0], ProxyInterface) else tuple(cut)
    utils.check(isinstance(producer.output, Sequence), lambda: "Producer output must be a sequence")
    external_producer_outputs = find_external_producer_outputs(trace, producer, consumer)
    new_producer_output_names = tuple(x.name for x in external_producer_outputs) + cut_names
    new_producer_output = tuple(x for x in producer.output if x.name in new_producer_output_names)
    new_producer = replace(producer, output=new_producer_output)
    return new_producer


def apply_rematerialization_for_consumer(
    producer: BoundSymbolInterface,
    consumer: BoundSymbolInterface,
    cut: Sequence[Union[ProxyInterface, str]],
) -> BoundSymbolInterface:
    """Update the consumer node with the cut information.

    Args:
        producer (BoundSymbolInterface): Producer node.
        consumer (BoundSymbolInterface): Consumer node.
        cut (Sequence[Union[ProxyInterface, str]]): Cut information.

    Returns:
        BoundSymbolInterface: Updated consumer node.
    """
    # It's a bit more complicated to update the consumer node, we need to
    # update the consumer's input with the cut information.
    # We need to keep consumer's inputs that are not in the cut and are not
    # produced by the producer. We call these inputs "external inputs".
    cut_names = tuple(map(lambda x: x.name, cut)) if isinstance(cut[0], ProxyInterface) else tuple(cut)
    cut_inputs = tuple(filter(lambda x: x.name in cut_names, (*producer.output, *producer.args)))
    external_inputs = find_external_consumer_inputs(producer, consumer)
    new_consumer_args = cut_inputs + external_inputs

    # We need to rematerialize the consumer's inputs that are not in the new consumer's inputs.
    rematerialized_inputs = tuple(
        filter(lambda x: x.name not in map(lambda x: x.name, new_consumer_args), consumer.args)
    )

    # Construct a temporary Trace object with subsymbols from both the producer and the consumer.
    trace = TraceCtx(None)
    trace.bound_symbols = (*producer.subsymbols, *consumer.subsymbols)

    cut_proxies = tuple(filter(lambda x: x.name in cut_names, (*producer.output, *producer.args)))
    recomputing_symbols = utils.find_producer_symbols(trace, rematerialized_inputs, cut_proxies)

    # Some recomputing_symbols might require producer's inputs, so we need to
    # add them to the consumer's inputs.
    # Probably find_min_cut should have returned this information.
    all_args = tuple(
        chain.from_iterable(
            (x.name for x in bsym._flat_args if isinstance(x, ProxyInterface)) for bsym in recomputing_symbols
        )
    )
    new_consumer_args += tuple(
        x for x in producer.args if x.name in all_args and x.name not in (x.name for x in new_consumer_args)
    )

    new_subsymbols = recomputing_symbols + tuple(consumer.subsymbols)
    new_consumer = replace(consumer, args=new_consumer_args, subsymbols=new_subsymbols)
    return new_consumer


def find_filtered_producer_consumer_pairs(
    trace: TraceCtx,
    filter_func: Optional[Callable] = None,
) -> Tuple[Tuple[BoundSymbolInterface, BoundSymbolInterface], ...]:
    """Find producer-consumer pairs among the filtered symbols.

    Args:
        trace (TraceCtx): Trace object.
        filter_func (Optional[Callable], optional): Filter function. Defaults to None.

    Returns:
        Tuple[Tuple[BoundSymbolInterface, BoundSymbolInterface], ...]: Producer-consumer bound symbol pairs.
    """
    filter_func = filter_func or (lambda x: True)
    proxy_to_consumers = utils.consumers(trace)
    producer_consumer_pairs = set()

    # We are looking for special producer-consumer pairs among the filtered symbols
    for producer in filter(filter_func, trace.bound_symbols):
        for out in producer._flat_outs:
            consumers = proxy_to_consumers.get(out, tuple())
            # We only care about producer's outputs with a single fusion consumer
            # prims.del is a special case that we don't care about
            consumers = tuple(filter(lambda x: x.sym.name != "del", consumers))
            if len(consumers) == 1 and filter_func(consumer := consumers[0]):
                producer_consumer_pairs.add((producer, consumer))
    return tuple(sorted(producer_consumer_pairs, key=lambda x: x[0].sym.name))


find_nvfuser_producer_consumer_pairs = partial(
    find_filtered_producer_consumer_pairs, filter_func=lambda x: x.sym.name.startswith("nvFusion")
)


# TODO: Consider moving this to the prims attribute
REMATERIALIZATION_BLOCK_LIST = {
    # We don't want to rematerialize the following primitives
    # Random primitives
    prims.PrimIDs.UNIFORM,
    # Reduction primitives
    prims.PrimIDs.SUM,
    prims.PrimIDs.VAR,
    prims.PrimIDs.VAR_MEAN,
    prims.PrimIDs.AMAX,
    prims.PrimIDs.AMIN,
    prims.PrimIDs.PROD,
    "torch.var_mean",
}


def find_cut(
    trace: TraceCtx,
    producer: BoundSymbolInterface,
    consumer: BoundSymbolInterface,
) -> Sequence[Union[ProxyInterface, str]]:
    """Find the minimal cut between the producer and the consumer.

    Args:
        trace (TraceCtx): Trace object.
        producer (BoundSymbolInterface): Producer node.
        consumer (BoundSymbolInterface): Consumer node.

    Returns:
        Sequence[Union[ProxyInterface, str]]: Cut information.
    """
    # We are going to use the igraph library to find the minimal cut between the
    # producer and the consumer. Minimum cut is a set of edges that, if removed,
    # would disconnect the producer and the consumer. But we are not interested
    # in the edges, we are interested in the nodes that are connected by the
    # edges. These nodes are the cut nodes. So we need to reformulate our node
    # graph into an edge graph.

    # All the nodes from the producer that we connect to a "source" node will
    # not be in the cut. Similarly, all the nodes from the consumer that we
    # connect to a "sink" node will not be in the cut. So we need to add a
    # "source" node and a "sink" node to our graph. We also disallow the cut to
    # be in the consumer's part of the graph to avoid balancing the graph into
    # the producer from the consumer.

    # Determine which producer's outputs can be rematerialized
    external_producer_outputs = find_external_producer_outputs(trace, producer, consumer)

    # Required producer variables. These are the variables that are required to
    # be connected to the "source" node.
    required_producer_vars = tuple(x.name for x in producer.args)
    required_producer_vars += tuple(x.name for x in external_producer_outputs)

    # This is needed to avoid rematerializing random or reduction primitives.
    required_producer_vars += tuple(
        chain.from_iterable(
            (y.name for y in x._flat_outs) for x in producer.subsymbols if x.sym.id in REMATERIALIZATION_BLOCK_LIST
        )
    )

    # Required consumer variables. These are the variables that are required to
    # be connected to the "sink" node.
    required_consumer_vars = tuple(x.name for x in consumer.output)
    external_consumer_inputs = find_external_consumer_inputs(producer, consumer)
    required_consumer_vars += tuple(x.name for x in external_consumer_inputs)

    # To the required consumer variables we also need to add the path from the
    # consumer's output to the external consumer's inputs. This is needed to
    # avoid balancing the graph into the producer from the consumer.
    consumer_trace = TraceCtx(None)
    consumer_trace.bound_symbols = consumer.subsymbols
    required_consumer_symbols = tuple(
        utils.find_producer_symbols(consumer_trace, consumer.output, external_consumer_inputs)
    )
    required_consumer_vars += tuple(
        chain.from_iterable((y.name for y in x._flat_outs) for x in required_consumer_symbols)
    )

    # TODO: Use TensorProxy properties to compute the weights
    WEIGHT = 1.0

    # Create a graph
    edges = []
    name_to_id = {}
    capacities = []

    def add_edge(src, dst, capacity):
        if src not in name_to_id:
            name_to_id[src] = len(name_to_id)
        if dst not in name_to_id:
            name_to_id[dst] = len(name_to_id)
        src, dst = name_to_id[src], name_to_id[dst]
        edges.append((src, dst))
        capacities.append(capacity)

    utils.check(
        len(required_consumer_vars) > 0,
        "The consumer has no outputs. This is not supported by the cut finding algorithm.",
    )
    for var_name in required_consumer_vars:
        add_edge(var_name + "_in", "sink", capacity=float("inf"))

    sym_skip_list = (
        prims.PrimIDs.UNPACK_SEQUENCE,
        prims.PrimIDs.UNPACK_TRIVIAL,
        prims.PrimIDs.UNPACK_KEY,
        prims.PrimIDs.RETURN,
    )

    combined_trace = TraceCtx(None)
    combined_trace.bound_symbols = (*producer.subsymbols, *consumer.subsymbols)
    combined_consumers = utils.consumers(combined_trace)

    def add_edges(var_name):
        weight = WEIGHT / 2.0 if var_name in (x.name for x in producer.args) else WEIGHT
        add_edge(var_name + "_in", var_name + "_out", capacity=weight)
        for user in combined_consumers._dict.get(var_name, tuple()):
            if user.sym.id in sym_skip_list:
                continue
            for out in user._flat_outs:
                user_name = out.name
                add_edge(var_name + "_out", user_name + "_in", capacity=float("inf"))

    if not required_producer_vars:
        # If there are no required producer variables, we need to make sure that
        # the source node is added to the graph.
        add_edge("source", "source", capacity=float("inf"))

    for var_name in required_producer_vars:
        add_edge("source", var_name + "_in", capacity=float("inf"))
        add_edges(var_name)

    for symbol in chain(producer.subsymbols, consumer.subsymbols):
        for var_name in (x.name for x in symbol._flat_outs):
            add_edges(var_name)

    g = Graph(
        n=len(name_to_id),
        edges=edges,
        directed=True,
        edge_attrs={"capacity": capacities},
    )
    source = name_to_id["source"]
    sink = name_to_id["sink"]
    partition = g.mincut(source, sink, "capacity").partition

    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, g.neighbors(n, mode="out")) for n in reachable):
        cutset.update((u, v) for v in nbrs if v in non_reachable)

    id_to_name = dict(map(reversed, name_to_id.items()))
    cutset = ((id_to_name[u], id_to_name[v]) for u, v in cutset)
    cut_nodes = set()
    for node_in, node_out in cutset:
        assert node_in.endswith("_in")
        assert node_out.endswith("_out")
        assert node_in[:-3] == node_out[:-4]
        node_name = node_in[:-3]
        cut_nodes.add(node_name)
    return tuple(sorted(cut_nodes))


# TODO: The following code is a temporary solution to update the call_ctx
# information of the nvFusion BoundSymbol object. This is needed because the
# nvFusion BoundSymbol object is created before the call_ctx information is
# updated. See more details in
# https://github.com/Lightning-AI/lightning-thunder/issues/515
def _update_nvfusion_call_ctx(trace: TraceCtx, bsym: BoundSymbolInterface) -> BoundSymbolInterface:
    """Update the call_ctx information of the nvFusion BoundSymbol object.

    Args:
        trace: The trace context.
        bsym: The nvFusion BoundSymbol object.

    Returns:
        The updated nvFusion BoundSymbol object.
    """
    from thunder.executors.nvfuserex import fuse

    @dataclass
    class BoundSymbolRegion:
        trace: TraceCtx
        inputs: tuple
        outputs: tuple
        bound_symbols: tuple
        counter: int

    def nvfusion_bsym_to_region(trace: TraceCtx, bsym: BoundSymbolInterface):
        counter = int(tuple(bsym._call_ctx.keys())[0].split("nvFusion")[-1])
        return BoundSymbolRegion(
            trace=trace,
            inputs=bsym.args,
            outputs=bsym.output,
            bound_symbols=bsym.subsymbols,
            counter=counter,
        )

    # fuse returns a new BoundSymbol object with correct updated call_ctx
    # information
    return fuse(nvfusion_bsym_to_region(trace, bsym))[0]


def rematerialize(trace: TraceCtx) -> tuple[TraceCtx, list[TraceCtx]]:
    """Rematerialize the trace.

    Args:
        trace (TraceCtx): Trace object.

    Returns:
        tuple[TraceCtx, list[TraceCtx]]: Rematerialized trace and the list of
            rematerialized traces.
    """
    # Find all the producers and consumers
    pairs = find_nvfuser_producer_consumer_pairs(trace)

    # Find the minimal cut between the producer and the consumer
    cuts = tuple(find_cut(trace, producer, consumer) for producer, consumer in pairs)

    # Pairs of producer and consumer are not unique. Each update to the producer
    # or consumer may affect the other. We need to update the producer and
    # consumer sequentially.
    producers = {producer for producer, _ in pairs}
    consumers = {consumer for _, consumer in pairs}
    new_bsyms = {bsym: bsym for bsym in producers | consumers}
    for (producer, consumer), cut in zip(pairs, cuts):
        current_producer = new_bsyms.get(producer, None) or producer
        current_consumer = new_bsyms.get(consumer, None) or consumer
        if cut:
            updated_producer = apply_rematerialization_for_producer(trace, current_producer, current_consumer, cut)
            updated_consumer = apply_rematerialization_for_consumer(current_producer, current_consumer, cut)
            new_bsyms[producer] = updated_producer
            new_bsyms[consumer] = updated_consumer
        else:
            new_bsyms[producer] = None
            new_bsyms[consumer] = None

    rematerialized_trace = from_trace(trace)
    rematerialized_trace.set_provenance(TraceProvenance("Rematerialization"))

    def replace_bound_symbol(bsym):
        new_bsym = new_bsyms.get(bsym, None)
        if new_bsym is not None:
            return _update_nvfusion_call_ctx(trace, new_bsym)
        return bsym

    # TODO: New bound symbols are still incorrect. Its _ctx_call dict points
    # to the old nvFuser fusion. We need to update it to use the new definition.

    new_bound_symbols = tuple(replace_bound_symbol(bsym) for bsym in trace.bound_symbols)
    rematerialized_trace.bound_symbols = new_bound_symbols
    return rematerialized_trace, [rematerialized_trace]


def split_vjp_trace_into_forward_and_backward(
    trace: TraceCtx,
) -> tuple[TraceCtx, TraceCtx]:
    """Split VJP transform applied trace into forward and backward.

    The input trace is supposed to follow the signature of ``func(inputs0, inputs1) -> (outputs0, outputs1)``,
    where
        - ``inputs0``: the inputs for the forward computation (a.k.a. primals)
        - ``inputs1``: the inputs for the backward computation (a.k.a. cotangents)
        - ``outputs0``: the results of the forward
        - ``outputs1``: the results of the backward, i.e. gradients

    This function splits the trace into the forward and the backward. This is feasible only when
    the extracted forward is independent from any of ``inputs1``. This function returns the following
    two functions:
        1. ``forward(*inputs0) -> (outputs0, saved_tensors_for_backward)``
        2. ``backward(*saved_tensors_for_backward, *inputs1) -> outputs1``

    Note that the two traces do not include :class:``BoundSymbols`` of ``prims.PrimIDs.del``.

    Args:
        trace: Trace of a function with VJP transform applied

    Returns:
        Two :class:``TraceCtx``s, one representing forward and the other backward.
        The forward trace has the signature of `forward_trace` with primals spelled out.
        The backward trace's arguments consist of ``*intermediate_values_of_forward`` and ``*cotangents``.
    """
    bwd_trace, tensors_to_save_for_backward = _extract_backward_from_vjp_trace(trace)
    fwd_trace = _extract_forward_from_vjp_trace(trace, tensors_to_save_for_backward)
    return fwd_trace, bwd_trace


def _extract_forward_from_vjp_trace(
    trace: TraceCtx,
    tensors_to_save_for_backward: list[TensorProxy],
) -> TraceCtx:
    """Extract bound symbols defining forward computation from joint trace.

    The generated trace takes ``primals`` as its positional arguments and kwargs, if exists.
    This raises an exception if the trace is not splittable without the forward depending on any of cotangents.
    """
    fwd_trace = from_trace(trace)

    cotangent_args = (a.name for a in trace.args[1])

    fwd_outputs, _ = tree_flatten(fwd_trace.output[0])
    fwd_trace.output = fwd_outputs
    # note(crcrpar): Cotangents are required for unsplittable joint trace. Otherwise, infinite loop.
    # https://github.com/Lightning-AI/lightning-thunder/commit/20505d9772786ac43b1a47811a6846c166f87022
    # seems to fix the infinite loop.
    forward_bsyms = list({
        v: v for v in utils.find_producer_symbols(
            trace, fwd_outputs, tree_flatten((fwd_trace.args, fwd_trace.kwargs))[0])
    }.keys())
    # now that we have collected bound symbols of forward computation, make sure they don't depend on any cotangents
    all_tensor_arg_names = {
        t.name for bsym in forward_bsyms for t in bsym._flat_args if isinstance(t, TensorProxy)
    }
    used_cotangent_args = tuple(a for a in cotangent_args if a in all_tensor_arg_names)
    if used_cotangent_args:
        raise RuntimeError(
            f"Impossible to split the trace due to the dependency on cotangents of {used_cotangent_args}"
        )

    fwd_trace.args = tree_flatten((fwd_trace.args[0], fwd_trace.kwargs))[0]
    fwd_trace.kwargs = {}
    fwd_trace.bound_symbols = forward_bsyms + [prims.python_return.bind(*fwd_outputs, output=())]

    fwd_trace._siginfo = None

    def forward_fn(*args):
        pass

    fwd_trace.fn = forward_fn
    fwd_siginfo = fwd_trace.siginfo()
    fwd_siginfo.args = [(a.name, a) for a in fwd_trace.args]
    fwd_siginfo.varargs = None
    fwd_siginfo.kwargs = {}

    fwd_outputs, fwd_outputs_spec = tree_flatten(trace.output[0])
    fwd_trace.output = fwd_trace.output, tensors_to_save_for_backward
    fwd_trace.bound_symbols[-1].args = (
        tree_unflatten(fwd_trace.bound_symbols[-1].args, fwd_outputs_spec),
        tensors_to_save_for_backward,
    )
    return fwd_trace


def _extract_backward_from_vjp_trace(trace: TraceCtx) -> tuple[TraceCtx, list[TensorProxy]]:
    """Extract backward definition from joint trace.

    The generated trace, the first of two return values, defines the backward computation.
    Its arguments follow the structure of ``*saved_for_backward, *cotangents`` for the sake of
    easy embedding into :class:``torch.autograd.Function``.
    The second return value, a list of :class:``TensorProxy``s, is a list of intermediate and output
    ``Tensor``s that will be used in the backward.
    """
    bwd_trace = from_trace(trace)
    bwd_trace.bound_symbols = trace.bound_symbols
    result, grads = bwd_trace.output
    result, _ = tree_flatten(result)

    forward_args, cotangents = bwd_trace.args
    flat_args, _ = tree_flatten((bwd_trace.args, bwd_trace.kwargs))

    forward_bsyms = list({
        v: v for v in utils.find_producer_symbols(bwd_trace, result, flat_args)
    }.keys())
    forward_intermediate_map = {out.name: out for bsym in forward_bsyms for out in bsym._flat_outs}
    for r in result:
        if r.name not in forward_intermediate_map:
            forward_intermediate_map[r.name] = r
    for a in tree_flatten((bwd_trace.args[1]))[0]:
        if a.name in forward_intermediate_map:
            del forward_intermediate_map[a.name]
    forward_intermediates = list(forward_intermediate_map.values())

    return_bsym = bwd_trace.bound_symbols[-1]
    return_bsym.args = grads
    bound_symbols_for_backward = [
        bsym for bsym in utils.find_producer_symbols(trace, grads, stop_proxies=flat_args)
        if bsym not in forward_bsyms
    ] + [return_bsym]
    backward_trace = from_trace(trace)
    backward_trace._siginfo = None
    backward_trace.bound_symbols = bound_symbols_for_backward
    consumed_vars = set(utils.consumers(backward_trace)._dict.keys())
    used_forward_intermediate_results = tuple(
        a
        for a in (forward_intermediates + tree_flatten((bwd_trace.args[0], bwd_trace.kwargs))[0])
        if a.name in consumed_vars
    )
    # Remove duplicates
    used_forward_intermediate_results = tuple(sorted({x.name: x for x in used_forward_intermediate_results}.values(), key=lambda a: a.name))
    backward_trace.args = tuple(used_forward_intermediate_results + bwd_trace.args[1])
    backward_trace.kwargs = None
    backward_trace.output = grads

    # N.B.(crcrpar): Override `siginfo` so that the trace would spell out arguments.
    # Otherwise, it'd end up `backward_trace(*args)`.
    def backward_fn(*args):
        pass

    backward_trace.fn = backward_fn
    bwd_siginfo = backward_trace.siginfo()
    bwd_siginfo.args = [(a.name, a) for a in bwd_siginfo.varargs[1]]
    bwd_siginfo.varargs = None
    backward_trace._siginfo = bwd_siginfo
    return backward_trace, used_forward_intermediate_results
