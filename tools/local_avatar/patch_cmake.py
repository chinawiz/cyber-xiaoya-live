"""把 grid-sample3d-trt-plugin 的 CMakeLists 改成 ARM64 (sm_121) 可编译的版本。

原始配置有三处问题：
  1. CUDA_ARCHITECTURES 写死 "70;80;86;89"，sm_121 的 nvcc 不支持 compute_70；
  2. target_include_directories 没带 TensorRT 头文件路径，找不到 NvInfer.h；
  3. add_subdirectory(test) 会编译可选的测试目标。
"""
import pathlib
import sys

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                    else "/home/dgx/grid-sample3d-trt-plugin/CMakeLists.txt")
text = path.read_text()

pairs = [
    ('CUDA_ARCHITECTURES "70;80;86;89"', 'CUDA_ARCHITECTURES "121"'),
    ('PRIVATE "./src" ${CUDAToolkit_INCLUDE_DIRS})',
     'PRIVATE "./src" ${CUDAToolkit_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIR})'),
    ("add_subdirectory(test)", "# add_subdirectory(test)"),
    ("enable_testing()", "# enable_testing()"),
]
for old, new in pairs:
    if old not in text:
        print("WARN 未找到:", old)
    text = text.replace(old, new)

path.write_text(text)
print(text[-600:])
