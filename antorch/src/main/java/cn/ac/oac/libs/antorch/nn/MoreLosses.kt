package cn.ac.oac.libs.antorch.nn

import cn.ac.oac.libs.antorch.core.Tensor
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/** 数值稳定的 BCEWithLogits。 */
class BCEWithLogitsLoss : Loss() {
    override fun forward(input: Tensor, target: Tensor): Tensor {
        require(input.shape.contentEquals(target.shape) || (input.ndim == 2 && input.shape[1] == 1 && target.ndim == 1 && target.shape[0] == input.shape[0]))
        val n = input.size
        val t = if (target.size == n) target.toFloatArray() else FloatArray(n) { target.data[it] }
        val x = input.toFloatArray()
        var loss = 0.0
        val probabilities = FloatArray(n)
        for (i in 0 until n) {
            val value = x[i]
            loss += max(value, 0f) - value * t[i] + ln(1 + exp(-abs(value).toDouble()))
            probabilities[i] = (1.0 / (1.0 + exp(-value.toDouble()))).toFloat()
        }
        val output = Tensor(floatArrayOf((loss / n).toFloat()), intArrayOf(1), input.requiresGrad)
        output.parents = listOf(input)
        output.backwardFn = { g ->
            val gradient = FloatArray(n) { (probabilities[it] - t[it]) * g.data[0] / n }
            Tensor.accumulate(input, Tensor(gradient, input.shape.copyOf()))
        }
        return output
    }
}

class HuberLoss(val delta: Float = 1f) : Loss() {
    override fun forward(input: Tensor, target: Tensor): Tensor {
        require(input.shape.contentEquals(target.shape))
        val n = input.size
        val difference = FloatArray(n) { input.data[it] - target.data[it] }
        var loss = 0.0
        for (d in difference) {
            loss += if (abs(d) <= delta) 0.5 * d * d else delta * (abs(d) - 0.5 * delta)
        }
        val output = Tensor(floatArrayOf((loss / n).toFloat()), intArrayOf(1), input.requiresGrad)
        output.parents = listOf(input)
        output.backwardFn = { g ->
            val gradient = FloatArray(n) { i ->
                val d = difference[i]
                (if (abs(d) <= delta) d else delta * if (d > 0) 1f else -1f) * g.data[0] / n
            }
            Tensor.accumulate(input, Tensor(gradient, input.shape.copyOf()))
        }
        return output
    }
}

class SmoothL1Loss(val beta: Float = 1f) : Loss() {
    override fun forward(input: Tensor, target: Tensor): Tensor =
        HuberLoss(beta).forward(input, target)
}

/** Focal Loss（二分类/多分类，标签为类别索引）。 */
class FocalLoss(val alpha: Float = 0.25f, val gamma: Float = 2f) : Loss() {
    override fun forward(input: Tensor, target: Tensor): Tensor {
        require(input.ndim == 2)
        val batch = input.shape[0]
        val classes = input.shape[1]
        val labels = IntArray(batch) { target.data[it].toInt() }
        val probabilities = Array(batch) { FloatArray(classes) }
        var loss = 0.0
        for (n in 0 until batch) {
            var maxValue = Float.NEGATIVE_INFINITY
            for (c in 0 until classes) maxValue = maxOf(maxValue, input.data[n * classes + c])
            var sum = 0f
            for (c in 0 until classes) {
                val e = exp((input.data[n * classes + c] - maxValue).toDouble()).toFloat()
                probabilities[n][c] = e
                sum += e
            }
            for (c in 0 until classes) probabilities[n][c] /= sum
            val p = probabilities[n][labels[n]].coerceAtLeast(1e-7f)
            loss += -alpha * Math.pow(1.0 - p, gamma.toDouble()) * ln(p.toDouble())
        }
        val output = Tensor(floatArrayOf((loss / batch).toFloat()), intArrayOf(1), input.requiresGrad)
        output.parents = listOf(input)
        output.backwardFn = { g ->
            val gradient = FloatArray(input.size)
            val scale = g.data[0] / batch
            for (n in 0 until batch) {
                val pTarget = probabilities[n][labels[n]].coerceAtLeast(1e-7f)
                val oneMinus = (1.0 - pTarget).coerceAtLeast(1e-7)
                val dLdp = alpha * (gamma * Math.pow(oneMinus, gamma - 1.0) * ln(pTarget) - Math.pow(oneMinus, gamma.toDouble()) / pTarget)
                for (c in 0 until classes) {
                    val dz = if (c == labels[n]) pTarget * (1 - pTarget) * dLdp
                    else -probabilities[n][c] * pTarget * dLdp
                    gradient[n * classes + c] = dz.toFloat() * scale
                }
            }
            Tensor.accumulate(input, Tensor(gradient, input.shape.copyOf()))
        }
        return output
    }
}

/** 负对数似然，输入必须是 log-probabilities。 */
class NLLLoss : Loss() {
    override fun forward(input: Tensor, target: Tensor): Tensor {
        require(input.ndim == 2)
        val batch = input.shape[0]
        val classes = input.shape[1]
        val labels = IntArray(batch) { target.data[it].toInt() }
        var loss = 0.0
        for (n in 0 until batch) loss += -input.data[n * classes + labels[n]]
        val output = Tensor(floatArrayOf((loss / batch).toFloat()), intArrayOf(1), input.requiresGrad)
        output.parents = listOf(input)
        output.backwardFn = { g ->
            val gradient = FloatArray(input.size)
            for (n in 0 until batch) gradient[n * classes + labels[n]] = -g.data[0] / batch
            Tensor.accumulate(input, Tensor(gradient, input.shape.copyOf()))
        }
        return output
    }
}

class LabelSmoothingCrossEntropy(val smoothing: Float = 0.1f) : Loss() {
    override fun forward(input: Tensor, target: Tensor): Tensor {
        require(input.ndim == 2)
        val batch = input.shape[0]
        val classes = input.shape[1]
        val labels = IntArray(batch) { target.data[it].toInt() }
        val probabilities = Array(batch) { FloatArray(classes) }
        var loss = 0.0
        for (n in 0 until batch) {
            var maxValue = Float.NEGATIVE_INFINITY
            for (c in 0 until classes) maxValue = maxOf(maxValue, input.data[n * classes + c])
            var sum = 0f
            for (c in 0 until classes) {
                val e = exp((input.data[n * classes + c] - maxValue).toDouble()).toFloat()
                probabilities[n][c] = e
                sum += e
            }
            for (c in 0 until classes) probabilities[n][c] /= sum
            for (c in 0 until classes) {
                val targetProbability = if (c == labels[n]) 1f - smoothing + smoothing / classes else smoothing / classes
                loss += -targetProbability * ln(probabilities[n][c].coerceAtLeast(1e-7f).toDouble())
            }
        }
        val output = Tensor(floatArrayOf((loss / batch).toFloat()), intArrayOf(1), input.requiresGrad)
        output.parents = listOf(input)
        output.backwardFn = { g ->
            val gradient = FloatArray(input.size)
            for (n in 0 until batch) for (c in 0 until classes) {
                val targetProbability = if (c == labels[n]) 1f - smoothing + smoothing / classes else smoothing / classes
                gradient[n * classes + c] = (probabilities[n][c] - targetProbability) * g.data[0] / batch
            }
            Tensor.accumulate(input, Tensor(gradient, input.shape.copyOf()))
        }
        return output
    }
}
