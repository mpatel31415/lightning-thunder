import random

from torch.testing import assert_close

import thunder
import thunder.torch as ltorch
from thunder.core import devices, dtypes
from thunder.tests.framework import TorchExecutor, instantiate, NOTHING


@instantiate(
    dtypes=(dtypes.float32, dtypes.float16, dtypes.float64),
    devicetypes=(devices.DeviceType.CUDA,),
)
def test_uniform_philox(executor, device: str, dtype: dtypes.dtype):
    shape = (10, 30)
    rng_seed = random.randint(0, 123902390)
    rng_offset = random.randint(0, 123902390) * (4 if executor == TorchExecutor else 1)

    def func(shape, dtype, device, rng_seed, rng_offset):
        return ltorch.uniform_philox(shape, device=device, dtype=dtype, seed=rng_seed, offset=rng_offset)

    cf = thunder.jit(func, executors_list=executor.executors_list())

    outputs = [cf(shape, dtype, device, rng_seed, rng_offset) for _ in range(3)]
    for o in outputs:
        assert_close(o, outputs[0])


@instantiate(
    dtypes=NOTHING,
    devicetypes=(devices.DeviceType.CUDA,),
)
def test_rng_state_prims(executor, device: str, _):
    import thunder.core.prims as prims
    import torch

    dev = devices.to_device(device)

    def func():
        b = prims.get_rng_state(None, device=dev)
        c = prims.get_rng_state(b, device=dev)
        seed, offset = prims.unpack_rng_state(b)
        new_state1 = prims.update_rng_state(seed, offset)

        new_state1 = prims.set_rng_state(new_state1, dev)
        new_state1_1 = prims.get_rng_state(new_state1, dev)
        state1_seed, state1_offset = prims.unpack_rng_state(new_state1_1)
        return b, c, seed, offset, new_state1_1, state1_seed, state1_offset

    cuda_generator = torch.cuda.default_generators[dev.index]
    jfunc = thunder.jit(func, executors_list=executor.executors_list())
    torch_device = thunder.core.devices.to_torch_device(dev)
    with torch.random.fork_rng(devices=(torch_device,)):
        cuda_generator.manual_seed(2)
        cuda_generator.set_offset(8)
        ori_state, ori_state_1, ori_seed, ori_offset, state1, s1_seed, s1_offset = jfunc()

        cuda_generator.manual_seed(2)
        cuda_generator.set_offset(8)

        assert_close(cuda_generator.get_state(), ori_state)
        assert_close(cuda_generator.get_state(), ori_state_1)
        assert_close(ori_seed, cuda_generator.initial_seed())
        assert_close(cuda_generator.get_offset() // (1 if executor == TorchExecutor else 4), ori_offset)

        cuda_generator.set_offset(cuda_generator.get_offset() + 4)
        assert_close(cuda_generator.get_state(), state1)
        assert_close(cuda_generator.initial_seed(), s1_seed)
        assert_close(cuda_generator.get_offset() // (1 if executor == TorchExecutor else 4), s1_offset)


@instantiate(
    dtypes=(dtypes.float32, dtypes.float16, dtypes.float64),
    devicetypes=(devices.DeviceType.CUDA,),
)
def test_rng_state_uniform_philox_reproducibility(executor, device: str, dtype: dtypes.dtype):
    import torch

    def func(a):
        b = ltorch.uniform_like(a, device=a.device, dtype=a.dtype)
        d = torch.nn.functional.dropout(a, p=0.5)
        c = ltorch.uniform_like(a, device=a.device, dtype=a.dtype)
        return c * b * a * d

    dev = devices.to_torch_device(device)
    cuda_generator = torch.cuda.default_generators[dev.index]
    a = torch.randn(2, 2, device=dev, dtype=dtypes.to_torch_dtype(dtype), requires_grad=True)
    a1 = a.detach().clone()
    a1.requires_grad_()

    jfunc = thunder.jit(func, executors_list=executor.executors_list())

    with torch.random.fork_rng(devices=(dev,)):
        torch.cuda.manual_seed(20)
        expects = []
        for _ in range(4):
            out = jfunc(a)
            out.sum().backward()
            expects.append(out)
            expects.append(a.grad)

        results = []
        torch.cuda.manual_seed(20)
        for _ in range(4):
            out = jfunc(a1)
            out.sum().backward()
            results.append(out)
            results.append(a1.grad)

    for expected, result in zip(expects, results):
        assert_close(expected, result)


@instantiate(
    dtypes=(dtypes.float32, dtypes.float16, dtypes.float64),
    devicetypes=(devices.DeviceType.CUDA,),
    executors=(TorchExecutor,),
)
def test_uniform_philox_vs_uniform(executor, device: str, dtype: dtypes.dtype):
    import torch

    dev = devices.to_torch_device(device)
    cuda_generator = torch.cuda.default_generators[dev.index]

    def func(a):
        b = thunder.torch.uniform_like(a, device=a.device, dtype=a.dtype)
        e = a * b
        c = thunder.torch.uniform_like(a, device=a.device, dtype=a.dtype)
        f = e + c
        d = thunder.torch.uniform_like(a, device=a.device, dtype=a.dtype)
        return f * d

    a = torch.randn(2, 2, device=dev, dtype=dtypes.to_torch_dtype(dtype), requires_grad=True)
    a1 = a.detach().clone().requires_grad_()

    jfunc = thunder.jit(func, executors_list=executor.executors_list())

    with torch.random.fork_rng(devices=(dev,)):
        cuda_generator.manual_seed(20)
        expects = []
        # get the results of uniform_philox with RNG state updates
        for _ in range(4):
            out = jfunc(a)
            expects.append(out)
        assert cuda_generator.get_offset() == 12 * 4
        fwd_trc = [
            t for t in thunder.last_traces(jfunc) if getattr(t.get_provenance(), "pss", "") == "Augmented forward pass"
        ][0]
        from thunder.core.prims import PrimIDs

        uniform_philox_sym = [PrimIDs.UNIFORM_PHILOX, "torch.uniform_philox"]
        uniform_sym = [PrimIDs.UNIFORM, "torch.uniform"]
        assert all(t.sym.id not in uniform_philox_sym for t in fwd_trc.bound_symbols)
        assert all(t not in uniform_sym for t in thunder.last_traces(jfunc)[-1].bound_symbols)

        # get the results of uniform
        results = []
        cuda_generator.manual_seed(20)
        from unittest.mock import patch

        with patch("thunder.core.rematerialization.replace_uniform") as replace_uniform_mock:
            replace_uniform_mock.return_value = fwd_trc
            jfunc = thunder.jit(func, executors_list=executor.executors_list())
            for _ in range(4):
                out = jfunc(a1)
                results.append(out)
            assert cuda_generator.get_offset() == 12 * 4

    for expected, result in zip(expects, results):
        assert_close(expected, result)
