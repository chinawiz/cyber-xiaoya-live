"""修补上游 TRTWrapper 的两处性能缺陷。

1. **每次推理都 cudaFree + cudaMalloc 全部 buffer**
   `setup()` 开头无条件释放所有 device buffer，下半段又 cudaMalloc 回来。这段逻辑
   只为动态 shape 准备，对 Ditto 固定 shape 的推理纯属浪费 —— warp 输出 4MB、
   decoder 输出 3MB，每帧白花约 10~20ms。

2. **同步 cudaMemcpy 把两条流水线串行化**
   `infer()` 用 `cudart.cudaMemcpy`（同步版），它会等待 device 上**所有**工作完成。
   Ditto 的 warp 和 decode 分别跑在两个线程里，本该并行，但因为彼此同步拷贝互相
   等待，实际退化成串行：单测 warp 16.8ms + decode 23.1ms，管线里却变成 37.6+46.8。
   改成每实例一个 stream + 全程 async + 只 sync 自己的 stream。

不改上游文件，import 后调用 patch() 即可。
"""
from __future__ import annotations

import numpy as np
import tensorrt as trt


def patch() -> bool:
    """返回是否执行了补丁（重复调用安全）。"""
    from core.utils import tensorrt_utils as tu

    wrapper = tu.TRTWrapper
    if getattr(wrapper, "_buffer_cached", False):
        return False

    trt.init_libnvinfer_plugins(tu.logger, "")
    orig_init = wrapper.__init__

    def __init__(self, *a, **k):
        orig_init(self, *a, **k)
        self._meta = None          # [(name, is_input, is_device, dtype)]
        self._shape_cache = {}     # name -> shape tuple（复用判定）
        # 每个实例独占一条 stream：warp / decode 在不同线程，只有各用各的 stream
        # 才能真正并行，否则同步拷贝会互相牵制
        self._stream = tu.checkCudaErrors(tu.cudart.cudaStreamCreate())

    def setup(self, input_data=None):
        input_data = input_data or {}
        if self._meta is None:
            self._meta = []
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                self._meta.append((
                    name,
                    self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT,
                    self.engine.get_tensor_location(name) == trt.TensorLocation.DEVICE,
                    self.engine.get_tensor_dtype(name),
                ))
            # 上游 infer() 依赖这几个字段
            self.tensor_name_list = [m[0] for m in self._meta]
            self.n_input = sum(1 for m in self._meta if m[1])
            self.n_output = len(self._meta) - self.n_input

        # 1) 先告诉 context 本次的输入 shape / 宿主内存地址
        for name, _is_in, is_dev, _dt in self._meta:
            if name not in input_data:
                continue
            data = input_data[name]
            if is_dev:
                self.context.set_input_shape(name, data.shape)
            else:
                self.context.set_tensor_address(name, data.ctypes.data)

        # 2) shape 变了才（重新）分配 buffer
        for name, _is_in, is_dev, dtype in self._meta:
            rshape = tuple(self.context.get_tensor_shape(name))
            if -1 in rshape:                      # 数据相关 shape，走 allocator
                if name not in self.output_allocator_map:
                    alloc = tu.MyOutputAllocator()
                    self.output_allocator_map[name] = alloc
                    self.context.set_output_allocator(name, alloc)
                    self.buffer[name] = [np.empty(0, dtype=trt.nptype(dtype)), None, 0]
                continue

            if self._shape_cache.get(name) == rshape:
                continue                          # ★ 复用：稳态下完全不分配

            old = self.buffer.get(name)
            if old and old[1]:
                tu.checkCudaErrors(tu.cudart.cudaFree(old[1]))
            n_bytes = trt.volume(rshape) * dtype.itemsize
            dev = (tu.checkCudaErrors(tu.cudart.cudaMalloc(n_bytes)) if is_dev else None)
            self.buffer[name] = [np.empty(rshape, dtype=trt.nptype(dtype)), dev, n_bytes]
            self._shape_cache[name] = rshape

        # 3) 输入数据拷进 host buffer（cudaMemcpyAsync 在 infer 里做）
        for name, data in input_data.items():
            if name in self.buffer:
                self.buffer[name][0] = np.ascontiguousarray(data)

        # 4) 绑定地址
        for name, _is_in, is_dev, _dt in self._meta:
            if is_dev:
                if self.buffer[name][1]:
                    self.context.set_tensor_address(name, self.buffer[name][1])
            elif _is_in is False:
                self.context.set_tensor_address(name, self.buffer[name][0].ctypes.data)

    def infer(self, stream=None) -> None:
        s = self._stream if stream in (None, 0) else stream
        sh = int(s)
        cudart = tu.cudart

        for name, is_in, is_dev, _dt in self._meta:
            if is_in and is_dev:
                cudart.cudaMemcpyAsync(self.buffer[name][1], self.buffer[name][0].ctypes.data,
                                      self.buffer[name][2],
                                      cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                                      stream=s)

        self.context.execute_async_v3(sh)

        for name in self.output_allocator_map:
            alloc = self.context.get_output_allocator(name)
            dtype = self.engine.get_tensor_dtype(name)
            rshape = alloc.shape
            n_bytes = trt.volume(rshape) * dtype.itemsize
            self.buffer[name] = [np.empty(rshape, dtype=trt.nptype(dtype)),
                                 alloc.address, n_bytes]

        for name, is_in, is_dev, _dt in self._meta:
            if (not is_in) and is_dev:
                cudart.cudaMemcpyAsync(self.buffer[name][0].ctypes.data, self.buffer[name][1],
                                      self.buffer[name][2],
                                      cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                                      stream=s)

        tu.checkCudaErrors(cudart.cudaStreamSynchronize(s))

    wrapper.__init__ = __init__
    wrapper.setup = setup
    wrapper.infer = infer
    wrapper._buffer_cached = True

    _patch_decoder()
    return True


def _patch_decoder() -> None:
    """decoder 若用 u8 引擎（输出已在图内做完 NHWC + 0~255 + uint8），
    跳过 Python 侧的 transpose/clip/mul —— 那三步对 512×512×3 要遍历三遍，
    实测每帧 8~10ms。旧引擎（输出 float NCHW）自动走原逻辑。

    注意补丁打在 `Decoder.__call__`（真正持有 model / model_type 的那一层），
    DecodeF3D 只是转发。
    """
    from core.models.decoder import Decoder

    if getattr(Decoder, "_u8_fast", False):
        return

    def __call__(self, feature):
        if self.model_type == "tensorrt":
            self.model.setup({"feature": feature})
            self.model.infer()
            buf = self.model.buffer
            if "output_u8" in buf and buf["output_u8"][0].dtype == np.uint8:
                # buffer[name] = [host, device, nbytes]；host 是 (1,H,W,C)，剥掉 batch
                return buf["output_u8"][0][0].astype(np.float32)
            pred = buf["output"][0]
        elif self.model_type == "onnx":
            pred = self.model.run(None, {"feature": feature})[0]
        elif self.model_type == "pytorch":
            import torch
            with torch.no_grad(), torch.autocast(device_type=self.device[:4],
                                                 dtype=torch.float16, enabled=True):
                pred = self.model(torch.from_numpy(feature).to(self.device)).float().cpu().numpy()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return np.transpose(pred[0], [1, 2, 0]).clip(0, 1) * 255

    Decoder.__call__ = __call__
    Decoder._u8_fast = True


if __name__ == "__main__":
    print("patched:", patch())
