# AnTorch

Android-first, PyTorch-style deep learning library: typed tensors, reverse-mode autograd,
MLP/CNN/Transformer/model zoo, modern losses and optimizers, JNI + OpenMP kernels, real
small datasets, and a runnable Compose demo.

[![Android](https://img.shields.io/badge/Android-26%2B-3DDC84)](https://developer.android.com)
[![Kotlin](https://img.shields.io/badge/Kotlin-2.2.20-7F52FF)](https://kotlinlang.org)
[![JNI](https://img.shields.io/badge/JNI-OpenMP-orange)]()
[![Release](https://img.shields.io/github/v/release/trueWangSyutung/AnTorch)](https://github.com/trueWangSyutung/AnTorch/releases)

## Features

| Area | What is inside |
| --- | --- |
| Tensor | `FLOAT16`, `FLOAT32`, `INT8`, `INT32`, `INT64`; reshape/transpose/matmul/broadcast |
| Autograd | dynamic graph, `backward()`, gradient accumulation, `noGrad`-style inference |
| NN | `Linear`, `Conv2d`, `ReLU`, `GELU`, `LayerNorm`, `Embedding`, `MultiHeadAttention`, `TransformerBlock` |
| Model zoo | MLP, CNN, MNIST-MLP, MNIST-CNN, Transformer classifier |
| Losses | MSE, CrossEntropy, BCEWithLogits, Huber, SmoothL1, Focal, NLL, LabelSmoothing |
| Optimizers | SGD, Adam, AdamW, RMSprop, Adagrad, Adamax, Lion, StepLR/ExponentialLR/CosineAnnealingLR |
| JNI | matmul, add/sub/mul/div, scalar ops, transpose, ReLU/Sigmoid/Tanh, softmax, sum, Conv2d (OpenMP) |
| Data | TensorDataset, DataLoader, SubsetDataset, Iris/MNIST/SMS loaders |
| Android | coroutine Trainer, checkpoints, model format v2, demo APK |

## Demo

The demo is a single runnable app with **M1–M9** and interactive inference:

- M1 autograd / XOR
- M2 MLP on **real Iris** (80/20 held-out)
- M3 save/load
- M4 CNN on **real MNIST** (800 train / 200 test)
- M5 JNI/OpenMP benchmark
- M6 training benchmark
- M7 Transformer on **real SMS Spam** (80/20 held-out)
- M8 MNIST held-out evaluation
- M9 Model Zoo parameter counts
- **Draw a digit** on a 28×28 canvas and call the trained CNN
- **Type an SMS** and call the trained Transformer

Download the APK from the [latest release](https://github.com/trueWangSyutung/AnTorch/releases).

## Quick start

```kotlin
val model = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10)
)
val optimizer = Adam(model.parameters, learningRate = 1e-3f)
val lossFn = CrossEntropyLoss()

for (batch in DataLoader(trainSet, batchSize = 32)) {
    optimizer.zeroGrad()
    val loss = lossFn(model(batch.x), batch.y)
    loss.backward()
    optimizer.step()
}
```

Using the coroutine trainer:

```kotlin
Trainer.train(
    model = model,
    dataset = data,
    loss = CrossEntropyLoss(),
    optimizer = Adam(model.parameters),
    config = TrainConfig(epochs = 10, batchSize = 32)
) { stats ->
    println("epoch=${stats.epoch} loss=${stats.loss} acc=${stats.accuracy}")
}
```

### JitPack

```kotlin
implementation("com.github.trueWangSyutung:AnTorch:0.1.0")
```

## Real datasets

Bundled subsets (see [datasets/README.md](datasets/README.md)):

| Dataset | Use | Size |
| --- | --- | --- |
| Iris | tabular classification | 150 rows, 4 features, 3 classes |
| MNIST | digit classification | 2,000 train + 500 test, 28×28 |
| SMS Spam Collection | text classification | 1,200 messages |

The demo trains and evaluates on held-out data; the MNIST card reports train/test accuracy,
and the inference cards call the trained models with user input.

## JNI / OpenMP

`antorch-native` implements the hot kernels in C++17 and parallelizes loops with OpenMP:

```text
matmul, add/sub/mul/div, scalar add/mul, transpose,
relu/sigmoid/tanh, softmax, sum, conv2d
```

Kotlin fallbacks keep JVM unit tests and non-Android environments usable. Enable/disable OpenMP
is automatic in CMake (`find_package(OpenMP QUIET)`).

## Build

```bash
./gradlew :antorch:testDebugUnitTest
./gradlew :demo:assembleDebug
./gradlew :demo:assembleRelease
```

Requirements: JDK 21, Android SDK 36, NDK 27+, minSdk 26.

## Docs

- [GitHub Pages](https://truewangsyutung.github.io/AnTorch/)
- [Quick start](docs/quickstart.md)
- [Architecture](docs/architecture.md)
- [API reference](docs/api.md)
- [JNI & OpenMP](docs/jni.md)
- [Datasets](datasets/README.md)

## License

Apache-2.0. Dataset licenses are listed in `datasets/README.md`.
