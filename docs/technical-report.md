# AnTorch 技术报告（中文）

## 摘要

AnTorch 是一个面向 Android 的、PyTorch 风格的轻量深度学习训练框架。它以 Kotlin 实现
动态计算图与反向自动微分，以 C++/JNI + OpenMP 实现热点算子，提供类型化 Tensor、
MLP/CNN/Transformer 模型库、现代损失函数与优化器、真实小数据集加载器，以及可在手机上
训练、保存并接受用户输入推理的 Demo。

0.1.0 的目标不是复刻完整 PyTorch，而是验证一条可运行的端侧训练链路：**数据加载 → 前向 →
反向 → 优化 → 测试 → 保存 → 用户输入推理**。

## 1. 设计目标

- Android 优先，minSdk 26，Kotlin API；
- PyTorch 风格命令式训练：`model(x)`、`loss.backward()`、`optimizer.step()`；
- 支持 `FLOAT16/FLOAT32/INT8/INT32/INT64`；
- CPU 训练优先，JNI + OpenMP 加速；
- JVM 单测可运行，Android 自动使用原生库；
- 真实数据、可复现实验、可安装 Demo。

## 2. 架构

```text
annotations -> core -> nn -> optim -> data -> serialization -> android
                         \-> JNI (C++/OpenMP)
```

- `core`：`DType`、`Tensor`、`Parameter`、`NativeOps`、自动微分；
- `nn`：`Module`、层、损失、Transformer、`ModelZoo`；
- `optim`：SGD/Adam/AdamW/RMSprop/Adagrad/Adamax/Lion 与调度器；
- `data`：`Dataset`/`DataLoader` 与 Iris/MNIST/SMS 加载器；
- `serialization`：dtype-aware 权重格式 v2；
- `android`：协程 `Trainer`、检查点；
- `cpp`：JNI/OpenMP kernel。

## 3. Tensor 与自动微分

`Tensor` 由连续存储 + shape + dtype + `requiresGrad` 构成。每个算子生成结果张量，并挂接
反向闭包；`backward()` 对计算图做拓扑排序，逆序累加梯度。

标量/同形状/2D+1D bias 的加法、矩阵乘法、ReLU、Sigmoid、Tanh、Softmax、Sum、Mean、
Reshape、Transpose 等均支持反向。`FLOAT32` 参数参与训练；其他 dtype 用于标签、token、
推理或量化实验。

## 4. JNI 与 OpenMP

热点 kernel 位于 `antorch/src/main/cpp`：

| Kernel | 并行方式 |
| --- | --- |
| matmul | 外层循环 OpenMP |
| add/sub/mul/div/scalar | OpenMP for |
| transpose | OpenMP collapse(2) |
| relu/sigmoid/tanh | OpenMP for |
| softmax | 按行并行 |
| sum | OpenMP reduction |
| conv2d | OpenMP collapse(2) over N/O |

`NativeOps.available` 在运行时探测 `.so`；`Tensor` 仅在 `FLOAT32` 且规模足够时走 JNI，
小 Tensor 留在 Kotlin，避免 JNI 调用开销。JVM 单测与桌面工具使用 Kotlin 回退。

## 5. 模型与训练

模型库包含 MLP、CNN、MNIST-MLP、MNIST-CNN、Transformer。Transformer 使用
Embedding + 可学习位置编码 + 多头自注意力 + LayerNorm + FFN + 均值池化 + 线性分类头。
`AttentionOp` 手写 Q/K/V 反向。

训练循环支持任意 `Loss` 与 `Optimizer`，按 epoch 返回 `loss/accuracy/elapsedMs`，可选
`ModelIO.saveCheckpoint`。

## 6. 数据

- **Iris**：150 条，4 特征，3 类；按列标准化；Demo 80/20 划分。
- **MNIST**：2,000 训练 + 500 测试，28×28，像素归一化到 `[0,1]`；Demo 800/200。
- **SMS Spam**：1,200 条，161 条 spam；小写 + 正则分词 + 1,200 词表 + 24 token 截断；
  Demo 80/20 划分。

数据来源与许可见 [Datasets](datasets.md)。

## 7. 实验

### 7.1 真实 MNIST

Demo M4 使用 `mnistCnn(4)`：Conv2d(1→4, 3×3) + ReLU + Flatten + Linear(2704→10)，
Adam(lr=0.005)，batch=32，训练 4 epochs。设备模拟器实测：

```text
MNIST · train=800 test=200
train: 98.8%
test: 90.5% (held-out)
```

该结果说明端侧小 CNN 可以在数秒级完成一次真实数据训练，并在独立测试集上得到可用的
泛化精度；它不是 MNIST 全量 SOTA，也不用于对外承诺模型质量。

### 7.2 其他 Demo

- Iris MLP：真实数据 80/20，3 类；
- SMS Transformer：真实短信 80/20，二分类；
- M8：独立 MNIST 测试集复测；
- 推理：用户在 28×28 画布写数字调用 CNN，或输入短信调用 Transformer。

## 8. 限制

- 训练参数仅 FLOAT32；FLOAT16/INT8 尚未进入反向计算；
- Conv2d 仅 stride=1、padding=0；
- 无 GPU 后端；见 [GPU Roadmap](gpu.md)；
- 尚未实现完整 PyTorch 算子/分布式训练/大模型；
- Demo 精度与耗时依赖设备，不能直接外推。

## 9. 结论

AnTorch 0.1.0 验证了 Android 端“可训练、可测试、可保存、可交互推理”的最小完整闭环，
并以 JNI/OpenMP 覆盖主要热点算子。后续工作包括 Vulkan GPU 后端、FLOAT16 训练、
更多算子与真实设备基准。

## 参考

- [API Reference](api.md)
- [Architecture](architecture.md)
- [JNI & OpenMP](jni.md)
- [GitHub Repository](https://github.com/trueWangSyutung/AnTorch)
