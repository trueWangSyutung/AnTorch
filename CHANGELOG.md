# Changelog

## 0.1.0 — 2026-11-02

- Typed tensors: `FLOAT16`, `FLOAT32`, `INT8`, `INT32`, `INT64`.
- Dynamic autograd with `backward()` and gradient accumulation.
- Model zoo: MLP, CNN, MNIST-MLP, MNIST-CNN, Transformer classifier.
- Modern losses: BCEWithLogits, Huber, SmoothL1, Focal, NLL, LabelSmoothing.
- Optimizers: SGD, Adam, AdamW, RMSprop, Adagrad, Adamax, Lion; LR schedulers.
- JNI + OpenMP kernels: matmul, add/sub/mul/div, scalar ops, transpose,
  ReLU/Sigmoid/Tanh, softmax, sum, Conv2d.
- Real datasets: Iris, MNIST subset, SMS Spam subset.
- Compose demo M1–M9 with held-out tests, digit drawing inference and SMS inference.
- GitBook docs, API reference and Chinese technical report.

Known limitations:

- Training parameters are FLOAT32; FLOAT16/INT8 are storage/quantization dtypes.
- Conv2d supports stride=1, padding=0.
- No GPU backend yet; see [GPU Roadmap](gpu.md).
