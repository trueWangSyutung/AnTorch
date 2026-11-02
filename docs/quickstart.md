# Quick start

## Add the dependency

```kotlin
implementation("com.github.trueWangSyutung:AnTorch:0.1.0")
```

## Tensor + autograd

```kotlin
val x = Tensor.fromArray(floatArrayOf(1f, 2f, 3f, 4f), 2, 2)
val w = Parameter(FloatArray(4) { 0.1f }, intArrayOf(2, 2))
val loss = x.matmul(w).sum()
loss.backward()
println(w.grad)
```

## Train a classifier

```kotlin
val model = Sequential(Linear(4, 16), ReLU(), Linear(16, 3))
val optimizer = Adam(model.parameters, learningRate = 0.01f)
val loader = DataLoader(iris, batchSize = 16)

repeat(200) {
    for (batch in loader) {
        optimizer.zeroGrad()
        val loss = CrossEntropyLoss()(model(batch.x), batch.y)
        loss.backward()
        optimizer.step()
    }
}
```

## Load real data

```kotlin
val iris = RealDatasets.iris(context.assets.open("iris.csv"))
val mnist = RealDatasets.mnist(
    context.assets.open("mnist_train_images.idx"),
    context.assets.open("mnist_train_labels.idx"),
    limit = 800
)
val sms = RealDatasets.sms(context.assets.open("sms.tsv"), maxLength = 24)
```
