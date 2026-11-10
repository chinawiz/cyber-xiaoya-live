"""给 decoder.onnx 尾部接上 Transpose + Clip + Mul + Cast(uint8)。

上游的 `DecodeF3D.__call__` 在 Python 侧做：

    pred = np.transpose(pred[0], [1, 2, 0]).clip(0, 1) * 255

对 512×512×3 的 float32 张量要遍历三遍（transpose 产生非连续布局 → clip 复制一份
→ mul 再复制一份），实测每帧 8~10ms，占 25fps 预算(40ms)的四分之一。而且 D2H 要
拷 3MB。

把这段搬进 ONNX 图后：TRT 内部融合掉，输出直接是 [1,512,512,3] uint8，D2H 只拷
0.75MB。视觉上等价（差异只是 float→uint8 的截断 vs 四舍五入，≤1 级）。
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

SRC = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ditto_onnx/decoder.onnx"
DST = sys.argv[2] if len(sys.argv) > 2 else "checkpoints/ditto_onnx/decoder_u8.onnx"

m = onnx.load(SRC)
g = m.graph
old_out = g.output[0].name
oshape = [d.dim_value or d.dim_param for d in g.output[0].type.tensor_type.shape.dim]
assert len(oshape) == 4, f"decoder 输出应为 NCHW，实际 {oshape}"
n, c, h, w = oshape
print(f"[onnx] in={old_out}{oshape} → out_u8 [1,{h},{w},{c}]", flush=True)

t = helper.make_node("Transpose", [old_out], ["out_nhwc"], perm=[0, 2, 3, 1], name="to_nhwc")
cmin = numpy_helper.from_array(np.array(0.0, np.float32), "clip_min")
cmax = numpy_helper.from_array(np.array(1.0, np.float32), "clip_max")
clip = helper.make_node("Clip", ["out_nhwc", "clip_min", "clip_max"], ["out_clip"], name="clip01")
scale = numpy_helper.from_array(np.array(255.0, np.float32), "scale255")
mul = helper.make_node("Mul", ["out_clip", "scale255"], ["out_scaled"], name="x255")
cast = helper.make_node("Cast", ["out_scaled"], ["output_u8"], to=TensorProto.UINT8, name="to_u8")

g.node.extend([t, clip, mul, cast])
g.initializer.extend([cmin, cmax, scale])

out = g.output[0]
out.name = "output_u8"
out.type.tensor_type.elem_type = TensorProto.UINT8
dims = out.type.tensor_type.shape.dim
for i, v in enumerate([n, h, w, c]):
    dims[i].dim_value = int(v)

onnx.checker.check_model(m)
onnx.save(m, DST)
print(f"[saved] {DST}", flush=True)
