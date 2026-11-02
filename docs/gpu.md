# GPU Roadmap

## 现状（0.1.0）

**AnTorch 0.1.0 不调用 GPU。** 训练与推理使用：

- Kotlin 参考实现；
- C++/JNI + OpenMP CPU kernel（`antorch-native`）。

这样做的原因是：Android 设备 GPU 碎片化严重，小模型（MLP/小 CNN）在 CPU 上往往已经足够快，
而 GPU 后端会显著增加显存管理、驱动兼容和测试成本。

## 能不能调用 GPU？

**可以，需要单独实现一个 GPU 后端。** Android 上可行的路线：

| 方案 | 训练支持 | 可移植性 | 说明 |
| --- | --- | --- | --- |
| **Vulkan Compute** | 可以 | 高（Android 7+ 大多数设备） | 推荐；需要 GLSL/SPIR-V shader、command buffer、显存/缓冲区管理 |
| OpenCL | 可以 | 中（厂商驱动差异大） | 部分 Adreno/Mali 支持，兼容性风险高 |
| NNAPI | 主要推理 | 高 | 面向推理加速，不提供通用训练反向图 |
| LiteRT GPU delegate | 仅推理 | 高 | 不能直接用于 AnTorch 训练 |
| RenderScript | 已弃用 | 低 | 不建议新项目使用 |

结论：

- **训练**：推荐 Vulkan Compute；OpenCL 作为可选厂商路径。
- **推理**：可以接 NNAPI / LiteRT GPU delegate，但它们是推理后端，不是训练后端。

## Vulkan 后端设计草案

```text
antorch-gpu-vulkan/
  VulkanContext      instance / device / queue / allocator
  GpuBuffer          host-visible / device-local buffer
  PipelineCache      SPIR-V shader + specialization
  kernels/
    matmul.comp
    conv2d.comp
    relu.comp
    softmax.comp
    elementwise.comp
```

接入点：

1. `Device` 增加 `CPU` / `VULKAN`；
2. `Tensor` 增加 device 与 buffer 句柄；
3. `NativeOps` 增加 GPU dispatch；
4. autograd 仍由 CPU 图调度，反向 kernel 复用同一套 shader；
5. 小 Tensor 自动回退 CPU，避免传输开销。

首版建议只做 **matmul + elementwise + softmax**，先覆盖 Transformer/MLP；Conv2d 第二阶段。

## 性能预期

GPU 的收益取决于：

- 模型规模：小 MLP 在 CPU 上可能更快；
- batch size：大 batch 更能摊薄传输；
- 设备：Adreno/Mali 差异明显；
- 内存带宽：移动 GPU 与 CPU 共享内存，但仍需 staging buffer。

因此必须做**分设备、分模型**的基准，不能承诺“GPU 一定更快”。

## 计划

| 阶段 | 内容 |
| --- | --- |
| 0.2 | Vulkan context、buffer、matmul shader、CPU/GPU 一致性测试 |
| 0.3 | elementwise/softmax/Conv2d、autograd 反向 kernel |
| 0.4 | FP16 存储、显存复用、真实设备基准（Adreno/Mali） |

在此之前，`Trainer` 与所有模型均可在 CPU/JNI 路径运行。
