# API Reference

AnTorch 0.1.0 · package root `cn.ac.oac.libs.antorch`.

All public APIs are annotated with `@AnTorchApi`; experimental APIs use
`@ExperimentalAnTorch` and native kernels use `@NativeAccelerated`.

---

## 1. Annotations

Package `cn.ac.oac.libs.antorch.annotation`.

| API | Description |
| --- | --- |
| `enum ApiStability` | `STABLE`, `EXPERIMENTAL`, `DEPRECATED`, `INTERNAL`. |
| `@AnTorchApi(stability, since, note)` | Public API stability and introduction version. Target: class, function, property. |
| `@ExperimentalAnTorch(note)` | Experimental API; may change between minor versions. |
| `@InternalAnTorch` | Internal implementation detail; not a compatibility contract. |
| `@NativeAccelerated(fallback)` | Marks a JNI/OpenMP kernel and its Kotlin fallback. |
| `@AnTorchFunction(name, description, differentiable)` | Marks a custom autograd function. |
| `@RegisterLayer(name)` | Marks a custom layer for the registry. |
| `@RegisterOptimizer(name)` | Marks a custom optimizer. |
| `@RegisterLoss(name)` | Marks a custom loss. |
| `object AnTorchRegistry` | Runtime registry: `registerLayer`, `registerOptimizer`, `registerLoss`, `create(kind, name, config)`, `names(kind)`, `clear()`. |

```kotlin
AnTorchRegistry.registerLayer("linear") { config ->
    Linear(config["in"] as Int, config["out"] as Int)
}
val layer = AnTorchRegistry.create("layer", "linear", mapOf("in" to 4, "out" to 2))
```

---

## 2. Core

Package `cn.ac.oac.libs.antorch.core`.

### 2.1 `DType`

`FLOAT16`, `FLOAT32`, `INT8`, `INT32`, `INT64`.

| Property | Description |
| --- | --- |
| `bytes` | Storage bytes per element. |
| `isFloating` | `FLOAT16` or `FLOAT32`. |
| `isInteger` | `INT8`, `INT32`, `INT64`. |

`FLOAT32` is the default training dtype. Only `FLOAT32` tensors support `requiresGrad`.
Arithmetic promotes results to `FLOAT32`.

### 2.2 `Tensor`

```kotlin
Tensor(data: FloatArray, shape: IntArray, requiresGrad: Boolean = false)
```

| Member | Description |
| --- | --- |
| `data: FloatArray` | Compute view. For non-FLOAT32 tensors, values are converted/cached as floats. |
| `shape: IntArray` | Row-major shape. |
| `dtype: DType` | Storage dtype. |
| `requiresGrad: Boolean` | Enables autograd (FLOAT32 only). |
| `grad: Tensor?` | Accumulated gradient after `backward()`. |
| `ndim`, `size` | Rank and element count. |
| `item(): Float` | Value of a single-element tensor. |
| `zeroGrad()` | Clears `grad`. |
| `backward()` | Reverse-mode autograd from this tensor. |
| `detach()` | Copy without graph/requiresGrad. |
| `copy(requiresGrad)` | Copy preserving dtype. |
| `reshape(vararg shape)` | View with a new shape; shares storage. |
| `flatten()` | 1-D view. |
| `transpose()` | 2-D transpose; returns a copy. |
| `matmul(other)` | 2-D matrix multiplication. |
| `plus/minus/times/div` | Same-shape elementwise ops; `+` also supports scalar and 2-D+1-D bias. |
| `relu()`, `sigmoid()`, `tanh()`, `softmax()` | Activations; `softmax` requires `[batch, classes]`. |
| `sum()`, `mean()` | Full reduction to shape `[1]`. |
| `log()` | Natural log. |
| `get(index)`, `get(vararg indices)` | Read a value. |
| `toFloatArray()`, `toIntArray()`, `toLongArray()` | Typed copies. |
| `astype(dtype)` | Convert storage/dtype. |

Companion factories:

| Factory | Description |
| --- | --- |
| `zeros(*shape)`, `ones(*shape)`, `full(value, *shape)` | FLOAT32 initialization. |
| `zeros(dtype, *shape)`, `full(value, dtype, *shape)` | Typed initialization. |
| `of(vararg values)`, `ofType(dtype, vararg values)` | 1-D factory. |
| `fromArray(data, *shape)` | FLOAT32 tensor. |
| `fromArray(data, shape, dtype)` | Typed tensor. |
| `fromFloat16Array(ShortArray, *shape)` | Raw half bits. |
| `fromInt8Array(IntArray, *shape)` | INT8 tensor. |
| `fromInt32Array(IntArray, *shape)` | INT32 tensor. |
| `fromLongArray(LongArray, *shape)` | INT64 tensor. |
| `fromValues(data, dtype, *shape)` | Convert float values into a dtype. |
| `randn(*shape, seed)` | Normal random tensor. |
| `arange(start, end, step)` | 1-D range. |

```kotlin
val x = Tensor.fromLongArray(longArrayOf(1, 2, 3), 3)
val f = x.astype(DType.FLOAT32)
val logits = Tensor.randn(4, 10, seed = 42)
```

### 2.3 `Parameter`

`Parameter(data: FloatArray, shape: IntArray)` or `Parameter(vararg shape: Int)`.
A `Tensor` with `requiresGrad = true`; used by modules/optimizers.

### 2.4 `NativeOps`

Package `cn.ac.oac.libs.antorch.core`.

| Function | Description |
| --- | --- |
| `available: Boolean` | Whether `libantorch_native` loaded. |
| `matmul(a,b,m,k,n)` | Float matrix multiply. |
| `add/sub/mul/div(a,b)` | Elementwise float arrays. |
| `addScalar/mulScalar(input,value)` | Scalar ops. |
| `transpose(input,rows,cols)` | 2-D transpose. |
| `relu/sigmoid/tanh(input)` | Activations. |
| `softmaxRows(input,rows,cols)` | Row softmax. |
| `sum(input)` | Sum. |
| `conv2d(input,weight,bias,N,C,H,W,O,K)` | Valid 3×3/ K×K convolution. |

---

## 3. Neural networks

Package `cn.ac.oac.libs.antorch.nn`.

### 3.1 `Module`

| Member | Description |
| --- | --- |
| `training` | Training/eval flag. |
| `parameters` | Parameters owned by this module. |
| `forward(input)` | Abstract forward. |
| `invoke(input)` | Alias for `forward`. |
| `train()/eval()` | Switch mode recursively where implemented. |
| `stateDict(prefix)` | Named parameter map for serialization. |
| `loadStateDict(state, prefix)` | Copy values into existing parameters. |

### 3.2 Layers

| Class | Signature / Description |
| --- | --- |
| `Sequential(vararg modules)` | Chains modules; flattens parameters and state dict. |
| `Linear(inFeatures, outFeatures)` | `x @ Wᵀ + b`; supports N-D input with last dim `inFeatures`. |
| `ReLU`, `Sigmoid`, `Tanh` | Elementwise activations. |
| `Flatten` | `[N, ...] → [N, product]`. |
| `Dropout(probability = 0.5f)` | Inverted dropout; only in training mode. |
| `Conv2d(inChannels, outChannels, kernelSize, stride=1, padding=0)` | 2-D convolution; first version requires stride=1, padding=0. |
| `GELU` | Tanh approximation. |
| `LayerNorm(features, epsilon=1e-5f)` | Normalizes the last dimension; supports N-D input. |
| `Embedding(numEmbeddings, embeddingDim)` | Integer `[B,S] → [B,S,D]`; input must be INT8/INT32/INT64. |
| `PositionalEncoding(maxSeq, dim)` | Learnable additive positional encoding. |
| `MultiHeadAttention(embedDim, heads)` | Q/K/V projections + scaled dot-product attention + output projection. |
| `TransformerBlock(dim, heads, ffnDim=dim*4)` | Pre-LN attention + FFN with residuals. |
| `SequenceMean` | Mean-pools `[B,S,D] → [B,D]`. |
| `TransformerClassifier(vocabSize,maxSeq,dim,heads,classes,layers=1,ffnDim=dim*4)` | Embedding + positional + transformer blocks + mean pool + linear head. |

### 3.3 Model Zoo

`object ModelZoo`:

| Function | Description |
| --- | --- |
| `mlp(input, hidden=128, classes=10)` | One hidden layer MLP. |
| `mlp(input, hidden: List<Int>, classes, activation)` | Configurable MLP. |
| `cnn(inChannels, imageSize, classes, channels=8)` | Conv 3×3 + ReLU + flatten + linear. |
| `mnistMlp(hidden=128)` | `[N,1,28,28] → 10` MLP. |
| `mnistCnn(channels=8)` | `[N,1,28,28] → 10` CNN. |
| `transformer(vocabSize,maxSeq,dim,heads,classes,layers,ffnDim)` | Transformer classifier. |
| `tinyTransformer(vocabSize,maxSeq=8,classes=2)` | Small demo Transformer. |

### 3.4 Losses

| Class | Description |
| --- | --- |
| `MSELoss` | Mean squared error; same-shape input/target. |
| `CrossEntropyLoss` | Logits `[N,C]` + integer labels `[N]`; stable softmax + NLL. |
| `BCEWithLogitsLoss` | Stable binary cross entropy from logits; same shape or `[N,1]` vs `[N]`. |
| `HuberLoss(delta=1f)` | Quadratic near zero, linear outside. |
| `SmoothL1Loss(beta=1f)` | Alias of Huber behavior. |
| `FocalLoss(alpha=0.25f, gamma=2f)` | Class-balanced focal loss; integer labels. |
| `NLLLoss` | Negative log likelihood; input must be log-probabilities. |
| `LabelSmoothingCrossEntropy(smoothing=0.1f)` | CE with uniform label smoothing. |

All losses return a scalar tensor and implement `backward`.

---

## 4. Optimizers

Package `cn.ac.oac.libs.antorch.optim`.

| Class | Key arguments | Description |
| --- | --- | --- |
| `Optimizer(parameters, learningRate)` | base | `step()`, `zeroGrad()`. |
| `SGD` | `momentum`, `weightDecay` | Stochastic gradient descent with optional momentum. |
| `Adam` | `beta1`, `beta2`, `epsilon`, `weightDecay` | Adam. |
| `AdamW` | `beta1`, `beta2`, `epsilon`, `weightDecay` | Decoupled weight decay Adam. |
| `RMSprop` | `alpha`, `epsilon`, `weightDecay` | RMS propagation. |
| `Adagrad` | `epsilon`, `weightDecay` | Adaptive gradient. |
| `Adamax` | `beta1`, `beta2`, `epsilon` | Infinity-norm Adam variant. |
| `Lion` | `beta1`, `beta2`, `weightDecay` | Sign-momentum optimizer. |
| `StepLR(optimizer, stepSize, gamma)` | | Multiply LR every N steps. |
| `ExponentialLR(optimizer, gamma)` | | Multiply LR each `step()`. |
| `CosineAnnealingLR(optimizer, maxEpochs, minLR)` | | Cosine schedule. |

```kotlin
val optimizer = AdamW(model.parameters, learningRate = 1e-3f, weightDecay = 0.01f)
optimizer.zeroGrad()
loss.backward()
optimizer.step()
```

---

## 5. Data

Package `cn.ac.oac.libs.antorch.data`.

| API | Description |
| --- | --- |
| `Sample(x, y)` | One input/label pair. |
| `Batch(x, y)` | Batched tensors. |
| `Dataset` | `size`, `get(index)`. |
| `TensorDataset(x, y)` | Tensor-backed dataset; first dimension is sample count. |
| `SubsetDataset(source, indices)` | Indexed subset (train/test split). |
| `DataLoader(dataset, batchSize=32, shuffle=true, seed=42)` | Iterable batches; preserves dtype. |
| `Datasets.xor()` | 4-sample XOR. |
| `Datasets.spiral(samplesPerClass, noise, seed)` | Synthetic 2-D spiral. |
| `Datasets.bars(samples, noise, seed)` | 8×8 bar images. |
| `Datasets.mnistLike(samples, noise, seed)` | 28×28 synthetic MNIST-shaped images. |
| `Datasets.sequences(samples, vocab, length, seed)` | Token sequences for Transformer tests. |
| `RealDatasets.iris(input)` | Real Iris CSV, standardized, 150×4, 3 classes. |
| `RealDatasets.mnist(images, labels, limit)` | IDX loader, `[N,1,28,28]` normalized, INT32 labels. |
| `RealDatasets.sms(input, maxLength=32, vocabSize=2000)` | SMS spam TSV; returns `TextDataset`. |
| `RealDatasets.TextDataset` | `dataset`, `vocabulary`, `maxLength`, `encode(text): Tensor`. |

```kotlin
val (train, test) = split(iris, 0.8, seed = 7)
val loader = DataLoader(train, batchSize = 16)
for (batch in loader) {
    val logits = model(batch.x)
    val loss = CrossEntropyLoss()(logits, batch.y)
}
```

---

## 6. Serialization

Package `cn.ac.oac.libs.antorch.serialization`.

`object ModelIO` (format v2, dtype-aware):

| Function | Description |
| --- | --- |
| `save(model, file)` | Write magic/version/state dict with dtype. |
| `load(model, file)` | Read and copy into an existing model; shape checks. |
| `saveCheckpoint(model, epoch, loss, file)` | Save weights + `epoch/loss` sidecar metadata. |

```kotlin
ModelIO.save(model, File(context.filesDir, "mnist.antorch"))
val restored = ModelZoo.mnistCnn(4)
ModelIO.load(restored, File(context.filesDir, "mnist.antorch"))
```

---

## 7. Android training

Package `cn.ac.oac.libs.antorch.android`.

| API | Description |
| --- | --- |
| `TrainConfig(epochs, batchSize, shuffle, checkpointEvery, checkpointFile)` | Training loop configuration. |
| `EpochStats(epoch, loss, accuracy, elapsedMs)` | Per-epoch result. |
| `Trainer.train(model, dataset, loss, optimizer, config, onEpoch)` | Suspending training loop on `Dispatchers.Default`. |
| `AnTorch.VERSION`, `AnTorch.init()`, `AnTorch.isInitialized()` | Library metadata/initialization. |
| `Tensor.argmaxRows()` | Per-row argmax for `[N,C]`. |

```kotlin
val history = Trainer.train(
    model, trainSet, CrossEntropyLoss(), Adam(model.parameters),
    TrainConfig(epochs = 4, batchSize = 32)
) { stats -> Log.i("AnTorch", "epoch=${stats.epoch} loss=${stats.loss}") }
```

---

## 8. Package map

```text
core/           DType, Tensor, Parameter, NativeOps
nn/             Module, layers, losses, Transformer, ModelZoo
optim/          optimizers and schedulers
data/           datasets, loaders, real data
serialization/  ModelIO
android/        Trainer and metadata
annotation/     API annotations and registry
cpp/            JNI + OpenMP kernels
```
