package cn.ac.oac.libs.antorch.nn

import cn.ac.oac.libs.antorch.annotation.AnTorchApi
import cn.ac.oac.libs.antorch.core.Tensor
import kotlin.math.exp
import kotlin.math.ln

abstract class Loss {
    abstract fun forward(input: Tensor, target: Tensor): Tensor
    operator fun invoke(input: Tensor, target: Tensor): Tensor = forward(input, target)
}

@AnTorchApi(since = "0.1.0")
class MSELoss : Loss() {
    override fun forward(input: Tensor, target: Tensor): Tensor {
        require(input.shape.contentEquals(target.shape)) { "MSE 形状必须一致" }
        return (input - target).square().mean()
    }
}

@AnTorchApi(since = "0.1.0")
class CrossEntropyLoss : Loss() {
    override fun forward(input: Tensor, target: Tensor): Tensor {
        require(input.ndim == 2) { "CrossEntropy 需要 [batch, classes]" }
        val batch = input.shape[0]
        val classes = input.shape[1]
        val labels = IntArray(batch) { target.data[it].toInt() }
        for (label in labels) require(label in 0 until classes) { "标签越界: $label" }

        val probabilities = Array(batch) { FloatArray(classes) }
        var loss = 0.0
        for (n in 0 until batch) {
            var max = Float.NEGATIVE_INFINITY
            for (c in 0 until classes) max = maxOf(max, input.data[n * classes + c])
            var sum = 0f
            for (c in 0 until classes) {
                val value = exp((input.data[n * classes + c] - max).toDouble()).toFloat()
                probabilities[n][c] = value
                sum += value
            }
            for (c in 0 until classes) probabilities[n][c] /= sum
            loss += -ln(probabilities[n][labels[n]].toDouble().coerceAtLeast(1e-12))
        }
        val meanLoss = (loss / batch).toFloat()
        val output = Tensor(floatArrayOf(meanLoss), intArrayOf(1), input.requiresGrad)
        output.parents = listOf(input)
        output.backwardFn = { g ->
            val gradient = FloatArray(input.size)
            val scale = g.data[0] / batch
            for (n in 0 until batch) for (c in 0 until classes) {
                val indicator = if (c == labels[n]) 1f else 0f
                gradient[n * classes + c] = (probabilities[n][c] - indicator) * scale
            }
            Tensor.accumulate(input, Tensor(gradient, input.shape))
        }
        return output
    }
}
