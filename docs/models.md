# Models & Training

## 模型库

```kotlin
val mlp = ModelZoo.mlp(4, 16, 3)
val cnn = ModelZoo.cnn(1, 8, 2)
val mnistMlp = ModelZoo.mnistMlp(128)
val mnistCnn = ModelZoo.mnistCnn(8)
val transformer = ModelZoo.transformer(
    vocabSize = 1201, maxSeq = 24, dim = 16, heads = 2, classes = 2
)
```

| 模型 | 输入 | 输出 |
| --- | --- | --- |
| `mlp` | `[N, features]` | `[N, classes]` |
| `cnn` | `[N, C, H, W]` | `[N, classes]` |
| `mnistMlp` | `[N, 1, 28, 28]` | `[N, 10]` |
| `mnistCnn` | `[N, 1, 28, 28]` | `[N, 10]` |
| `TransformerClassifier` | `[N, seq]` INT32/INT64 tokens | `[N, classes]` |

## Transformer 结构

```text
Embedding -> PositionalEncoding
  -> TransformerBlock × N
       LayerNorm -> MultiHeadAttention -> residual
       LayerNorm -> Linear -> GELU -> Linear -> residual
  -> SequenceMean -> Linear
```

注意力使用缩放点积：

```text
Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k)) V
```

反向传播由自定义 `AttentionOp` 实现，支持 Q/K/V 梯度。

## 训练循环

```kotlin
val optimizer = AdamW(model.parameters, learningRate = 1e-3f, weightDecay = 0.01f)
repeat(epochs) {
    for (batch in DataLoader(trainSet, batchSize = 32)) {
        optimizer.zeroGrad()
        val loss = CrossEntropyLoss()(model(batch.x), batch.y)
        loss.backward()
        optimizer.step()
    }
}
```

Android 协程封装：

```kotlin
Trainer.train(
    model, trainSet, CrossEntropyLoss(), optimizer,
    TrainConfig(epochs = 4, batchSize = 32, checkpointEvery = 1, checkpointFile = file)
) { stats -> /* UI */ }
```

## 损失与优化器

- 损失：MSE、CrossEntropy、BCEWithLogits、Huber、SmoothL1、Focal、NLL、LabelSmoothing。
- 优化器：SGD、Adam、AdamW、RMSprop、Adagrad、Adamax、Lion。
- 调度器：StepLR、ExponentialLR、CosineAnnealingLR。
