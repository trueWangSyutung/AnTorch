package cn.ac.oac.libs.antorch.nn

import cn.ac.oac.libs.antorch.annotation.AnTorchApi

/** 常见模型工厂：MLP、CNN、MNIST、Transformer。 */
@AnTorchApi(since = "0.1.0")
object ModelZoo {

    fun mlp(input: Int, hidden: Int = 128, classes: Int = 10): Sequential =
        Sequential(Linear(input, hidden), ReLU(), Linear(hidden, classes))

    fun mlp(input: Int, hidden: List<Int>, classes: Int, activation: () -> Module = { ReLU() }): Sequential {
        val modules = ArrayList<Module>()
        var last = input
        for (size in hidden) {
            modules.add(Linear(last, size))
            modules.add(activation())
            last = size
        }
        modules.add(Linear(last, classes))
        return Sequential(*modules.toTypedArray())
    }

    /** 简单 CNN：Conv 3x3 -> ReLU -> Flatten -> Linear。 */
    fun cnn(inChannels: Int, imageSize: Int, classes: Int, channels: Int = 8): Sequential {
        val outSize = imageSize - 2
        require(outSize > 0) { "imageSize 必须大于 2" }
        return Sequential(
            Conv2d(inChannels, channels, 3),
            ReLU(),
            Flatten(),
            Linear(channels * outSize * outSize, classes)
        )
    }

    /** MNIST MLP：输入 [N,1,28,28]，输出 10 类。 */
    fun mnistMlp(hidden: Int = 128): Sequential =
        Sequential(Flatten(), Linear(28 * 28, hidden), ReLU(), Linear(hidden, 10))

    /** MNIST CNN：输入 [N,1,28,28]，输出 10 类。 */
    fun mnistCnn(channels: Int = 8): Sequential =
        Sequential(Conv2d(1, channels, 3), ReLU(), Flatten(), Linear(channels * 26 * 26, 10))

    fun transformer(
        vocabSize: Int,
        maxSeq: Int = 16,
        dim: Int = 32,
        heads: Int = 4,
        classes: Int = 2,
        layers: Int = 1,
        ffnDim: Int = dim * 4
    ): TransformerClassifier = TransformerClassifier(vocabSize, maxSeq, dim, heads, classes, layers, ffnDim)

    fun tinyTransformer(
        vocabSize: Int,
        maxSeq: Int = 8,
        classes: Int = 2
    ): TransformerClassifier = TransformerClassifier(vocabSize, maxSeq, 16, 2, classes, layers = 1, ffnDim = 32)
}
