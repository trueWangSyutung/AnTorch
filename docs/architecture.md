# Architecture

```text
antorch/
  annotation/     @AnTorchApi, @NativeAccelerated, registry
  core/           DType, Tensor, autograd, NativeOps
  nn/             Module, layers, losses, Transformer, model zoo
  optim/          SGD/Adam/AdamW/RMSprop/Adagrad/Adamax/Lion, schedulers
  data/           Dataset, DataLoader, real dataset loaders
  serialization/  ModelIO v2 (dtype-aware)
  android/        Trainer, TrainConfig, AnTorch
  cpp/            JNI + OpenMP kernels
demo/             Compose M1–M9 + drawing/text inference
```

## Tensor and autograd

A `Tensor` owns a contiguous storage (`FloatArray`, `ShortArray`, `ByteArray`, `IntArray` or
`LongArray`) selected by `DType`. Operations create result tensors and attach a backward closure.
`backward()` performs a reverse topological traversal and accumulates gradients.

`FLOAT32` is the training dtype; `FLOAT16`, `INT8`, `INT32`, `INT64` are storage/logical dtypes
for inference, labels, token IDs and quantization experiments. Arithmetic promotes to `FLOAT32`.

## Native layer

`Tensor` dispatches hot ops to `NativeOps` when the JNI library is present and the tensors are
`FLOAT32`; otherwise Kotlin fallbacks are used. This keeps JVM tests and desktop tools functional.

## Android trainer

`Trainer.train` runs on `Dispatchers.Default`, reports `EpochStats`, and can write checkpoints via
`ModelIO.saveCheckpoint`. Long training should be hosted in a foreground service by the app.
