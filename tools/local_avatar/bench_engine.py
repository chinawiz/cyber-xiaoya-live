"""独立进程测 TRT 引擎耗时（不在构建进程里跑，避免插件重复注册）。

用法：python bench_engine.py <engine> [batch ...]
"""
import sys
import time

import tensorrt as trt
import torch

ENGINE = sys.argv[1]
BATCHES = [int(x) for x in sys.argv[2:]] or [1]

LOG = trt.Logger(trt.Logger.ERROR)
rt = trt.Runtime(LOG)
with open(ENGINE, "rb") as f:
    engine = rt.deserialize_cuda_engine(f.read())
if engine is None:
    sys.exit(f"[fail] 无法反序列化 {ENGINE}")

ctx = engine.create_execution_context()
ios = [(engine.get_tensor_name(i), engine.get_tensor_mode(engine.get_tensor_name(i)))
       for i in range(engine.num_io_tensors)]

for bs in BATCHES:
    bufs = {}
    ok = True
    for name, mode in ios:
        shape = list(engine.get_tensor_shape(name))
        dyn = any(d < 0 for d in shape)
        if dyn:
            shape[0] = bs
            if mode == trt.TensorIOMode.INPUT:
                ctx.set_input_shape(name, shape)
        dtype = engine.get_tensor_dtype(name)
        td = torch.int64 if dtype == trt.DataType.INT64 else torch.float32
        bufs[name] = torch.zeros(shape, dtype=td, device="cuda")
        ctx.set_tensor_address(name, bufs[name].data_ptr())
    if not ok:
        continue
    stream = torch.cuda.Stream()
    for _ in range(10):
        ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    n = 40
    t = time.perf_counter()
    for _ in range(n):
        ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / n * 1000
    print(f"[lat] {ENGINE.split('/')[-1]} batch={bs}: {dt:6.1f} ms/次 → {dt / bs:5.1f} ms/帧",
          flush=True)
