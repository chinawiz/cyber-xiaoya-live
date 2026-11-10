"""通用 TRT 引擎重建：去掉 hardware_compatibility_level（上游 Ampere_Plus 限制），
可选动态 batch。

对比基线（不改任何模型，仅重建）：
    decoder 旧引擎 52.1ms/帧 → 新引擎 22.8ms（batch1），batch4 为 15.6ms/帧。

用法：
    python build_engine.py <onnx> <out.engine> [--batch 4] [--fp32] [--grid-fp32]
"""
import argparse
import ctypes
import os
import sys
import time

import onnx
import tensorrt as trt
import torch

ap = argparse.ArgumentParser()
ap.add_argument("onnx")
ap.add_argument("out")
ap.add_argument("--batch", type=int, default=1, help="opt/max batch（1 = 保持静态）")
ap.add_argument("--fp32", action="store_true")
ap.add_argument("--grid-fp32", action="store_true", help="GridSample 层强制 fp32")
ap.add_argument("--verbose", action="store_true")
ap.add_argument("--no-bench", action="store_true",
                help="构建完不测速（引擎内嵌插件时，同进程反序列化会撞插件重复注册）")
a = ap.parse_args()

LOG = trt.Logger(trt.Logger.VERBOSE if a.verbose else trt.Logger.WARNING)

PLUGIN_LIBS = []
for p in ("/home/dgx/grid-sample3d-trt-plugin/build/libgrid_sample_3d_plugin.so",
          os.path.join(os.path.dirname(os.path.abspath(a.onnx)), "libgrid_sample_3d_plugin.so")):
    if os.path.isfile(p):
        try:
            ctypes.CDLL(p)
            PLUGIN_LIBS.append(p)
            break          # 只序列化一份：两份同源 so 会让反序列化时插件重复注册
        except OSError:
            pass


def _dims(io):
    return [d.dim_param or d.dim_value for d in io.type.tensor_type.shape.dim]


m = onnx.load(a.onnx)
if a.batch > 1:
    for io in list(m.graph.input) + list(m.graph.output):
        dims = io.type.tensor_type.shape.dim
        if dims[0].dim_value == 1 or dims[0].dim_param == "":
            dims[0].dim_param = "batch"
    path = a.onnx.replace(".onnx", "_dyn.onnx")
    onnx.save(m, path)
    m = onnx.load(path)
else:
    path = a.onnx

inputs = [_dims(i) for i in m.graph.input]
outputs = [_dims(o) for o in m.graph.output]
print(f"[onnx] {os.path.basename(a.onnx)} in={inputs} out={outputs} batch={a.batch}", flush=True)

builder = trt.Builder(LOG)
trt.init_libnvinfer_plugins(LOG, "")
net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(net, LOG)
with open(path, "rb") as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print("[onnx-error]", parser.get_error(i), flush=True)
        sys.exit(1)

cfg = builder.create_builder_config()
cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 6 << 30)
cfg.builder_optimization_level = 5
# 关键：不要设置 hardware_compatibility_level，否则退回 Ampere 内核，损失 2 倍以上
if not a.fp32:
    cfg.set_flag(trt.BuilderFlag.FP16)
    cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)

if a.grid_fp32:
    exclude = {"SHAPE", "ASSERTION", "SHUFFLE", "IDENTITY", "CONSTANT", "CONCATENATION",
               "GATHER", "SLICE", "CONDITION", "CONDITIONAL_INPUT", "CONDITIONAL_OUTPUT",
               "FILL", "NON_ZERO", "ONE_HOT", "PLUGIN_V2"}
    n_set = 0
    for i in range(net.num_layers):
        layer = net.get_layer(i)
        if str(layer.type)[10:] in exclude:
            continue
        if "GridSample" in layer.name or layer.type == trt.LayerType.PLUGIN_V2:
            layer.precision = trt.float32
            n_set += 1
    print(f"[grid] 强制 fp32 层数 {n_set}", flush=True)

if a.batch > 1:
    prof = builder.create_optimization_profile()
    for io, dims in zip(m.graph.input, inputs):
        fixed = [int(x) for x in dims[1:]]
        prof.set_shape(io.name, [1] + fixed, [a.batch] + fixed, [a.batch] + fixed)
    cfg.add_optimization_profile(prof)

# 把插件 so 序列化进引擎，运行时无需再手动指定库路径
if PLUGIN_LIBS:
    cfg.plugins_to_serialize = PLUGIN_LIBS
    print(f"[plugin] serialize {len(PLUGIN_LIBS)}", flush=True)

t0 = time.time()
ser = builder.build_serialized_network(net, cfg)
print(f"[build] {time.time() - t0:.1f}s", flush=True)
if ser is None:
    sys.exit("[build] FAILED")
with open(a.out, "wb") as f:
    f.write(ser)
print(f"[saved] {a.out} {os.path.getsize(a.out) / 1e6:.1f}MB", flush=True)

# ---- 测速 ----
if a.no_bench:
    sys.exit(0)

rt = trt.Runtime(LOG)
with open(a.out, "rb") as f:
    engine = rt.deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()
tnames = [(engine.get_tensor_name(i), engine.get_tensor_mode(engine.get_tensor_name(i)))
          for i in range(engine.num_io_tensors)]

for bs in sorted({1, max(1, a.batch // 2), a.batch}):
    bufs = {}
    for name, mode in tnames:
        shape = list(engine.get_tensor_shape(name))
        if -1 in shape:
            shape[0] = bs
            ctx.set_input_shape(name, shape) if mode == trt.TensorIOMode.INPUT else None
        bufs[name] = torch.zeros(shape, dtype=torch.float32, device="cuda")
        ctx.set_tensor_address(name, bufs[name].data_ptr())
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
    print(f"[lat] batch={bs}: {dt:6.1f} ms/次 → {dt / bs:5.1f} ms/帧", flush=True)
