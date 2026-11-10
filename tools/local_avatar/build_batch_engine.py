"""把 Ditto 的静态 batch=1 ONNX 改成动态 batch，重建 TRT 引擎并测速。

用法：python build_batch_engine.py <onnx> <out.engine> <out_channels> <in_h> <in_w> <batch>
"""
import sys
import time

import numpy as np
import onnx
import tensorrt as trt
import torch

ONNX = sys.argv[1]
ENGINE = sys.argv[2]
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 4

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ---- 1) ONNX batch 维改动态 ----
m = onnx.load(ONNX)
inp = m.graph.input[0]
out = m.graph.output[0]

def _dims(io):
    return [d.dim_param or d.dim_value for d in io.type.tensor_type.shape.dim]

d_in = _dims(inp)
d_out = _dims(out)
print(f"[onnx] in={inp.name}{d_in} out={out.name}{d_out}", flush=True)

if d_in[0] != "batch":
    inp.type.tensor_type.shape.dim[0].dim_param = "batch"
    out.type.tensor_type.shape.dim[0].dim_param = "batch"
    dyn_path = ONNX.replace(".onnx", f"_dyn.onnx")
    onnx.save(m, dyn_path)
else:
    dyn_path = ONNX
print(f"[onnx] dynamic → {dyn_path}", flush=True)

shape_in = [BATCH] + [int(x) for x in d_in[1:]]
shape_out = [BATCH] + [int(x) for x in d_out[1:]]
min_sh = [1] + [int(x) for x in d_in[1:]]
opt_sh = shape_in
max_sh = shape_in

# ---- 2) 构建引擎 ----
import ctypes
for p in ("/home/dgx/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so",
          "/home/dgx/ditto-talkinghead/checkpoints/ditto_onnx/libgrid_sample_3d_plugin.so"):
    try:
        ctypes.CDLL(p)
        print(f"[plugin] loaded {p}", flush=True)
    except OSError:
        pass

builder = trt.Builder(TRT_LOGGER)
net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(net, TRT_LOGGER)
with open(dyn_path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print("[onnx-error]", parser.get_error(i), flush=True)
        sys.exit(1)

cfg = builder.create_builder_config()
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
cfg.set_flag(trt.BuilderFlag.FP16)
cfg.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS) if hasattr(trt.BuilderFlag, "SPARSE_WEIGHTS") else None

prof = builder.create_optimization_profile()
prof.set_shape(inp.name, min_sh, opt_sh, max_sh)
cfg.add_optimization_profile(prof)

t0 = time.time()
ser = builder.build_serialized_network(net, cfg)
print(f"[build] {time.time() - t0:.1f}s", flush=True)
if ser is None:
    sys.exit("[build] failed")
with open(ENGINE, "wb") as f:
    f.write(ser)

# ---- 3) 测速 ----
rt = trt.Runtime(TRT_LOGGER)
with open(ENGINE, "rb") as f:
    engine = rt.deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()

d_inp = torch.zeros(shape_in, dtype=torch.float32, device="cuda")
d_out = torch.zeros(shape_out, dtype=torch.float32, device="cuda")
ctx.set_tensor_address(inp.name, d_inp.data_ptr())
ctx.set_tensor_address(out.name, d_out.data_ptr())
stream = torch.cuda.Stream()

for bs in [1, 2, BATCH]:
    d_inp2 = torch.zeros([bs] + shape_in[1:], dtype=torch.float32, device="cuda")
    d_out2 = torch.zeros([bs] + shape_out[1:], dtype=torch.float32, device="cuda")
    ctx.set_input_shape(inp.name, [bs] + shape_in[1:])
    ctx.set_tensor_address(inp.name, d_inp2.data_ptr())
    ctx.set_tensor_address(out.name, d_out2.data_ptr())
    for _ in range(10):
        ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    n = 30
    t = time.perf_counter()
    for _ in range(n):
        ctx.execute_async_v3(stream.cuda_stream)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / n * 1000
    print(f"[lat] batch={bs}: {dt:.1f} ms/次 → {dt / bs:.1f} ms/帧", flush=True)
