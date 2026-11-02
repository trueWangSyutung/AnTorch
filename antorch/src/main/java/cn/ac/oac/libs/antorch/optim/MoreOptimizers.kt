package cn.ac.oac.libs.antorch.optim

import cn.ac.oac.libs.antorch.core.Parameter
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sqrt

class RMSprop(
    parameters: List<Parameter>,
    learningRate: Float = 1e-3f,
    val alpha: Float = 0.99f,
    val epsilon: Float = 1e-8f,
    val weightDecay: Float = 0f
) : Optimizer(parameters, learningRate) {
    private val squareAverage = HashMap<Parameter, FloatArray>()

    override fun step() {
        forEachGradient { parameter, gradient ->
            val average = squareAverage.getOrPut(parameter) { FloatArray(parameter.size) }
            for (i in parameter.data.indices) {
                val g = gradient[i] + weightDecay * parameter.data[i]
                average[i] = alpha * average[i] + (1 - alpha) * g * g
                parameter.data[i] -= learningRate * g / (sqrt(average[i]) + epsilon)
            }
        }
    }
}

class Adagrad(
    parameters: List<Parameter>,
    learningRate: Float = 0.01f,
    val epsilon: Float = 1e-8f,
    val weightDecay: Float = 0f
) : Optimizer(parameters, learningRate) {
    private val accumulated = HashMap<Parameter, FloatArray>()

    override fun step() {
        forEachGradient { parameter, gradient ->
            val sum = accumulated.getOrPut(parameter) { FloatArray(parameter.size) }
            for (i in parameter.data.indices) {
                val g = gradient[i] + weightDecay * parameter.data[i]
                sum[i] += g * g
                parameter.data[i] -= learningRate * g / (sqrt(sum[i]) + epsilon)
            }
        }
    }
}

class Adamax(
    parameters: List<Parameter>,
    learningRate: Float = 2e-3f,
    private val beta1: Float = 0.9f,
    private val beta2: Float = 0.999f,
    private val epsilon: Float = 1e-8f
) : Optimizer(parameters, learningRate) {
    private val first = HashMap<Parameter, FloatArray>()
    private val infinity = HashMap<Parameter, FloatArray>()
    private var stepCount = 0

    override fun step() {
        stepCount++
        val bias1 = 1f - beta1.toDouble().pow(stepCount).toFloat()
        forEachGradient { parameter, gradient ->
            val m = first.getOrPut(parameter) { FloatArray(parameter.size) }
            val u = infinity.getOrPut(parameter) { FloatArray(parameter.size) }
            for (i in parameter.data.indices) {
                val g = gradient[i]
                m[i] = beta1 * m[i] + (1 - beta1) * g
                u[i] = maxOf(beta2 * u[i], abs(g))
                parameter.data[i] -= learningRate * (m[i] / bias1) / (u[i] + epsilon)
            }
        }
    }
}

/** Lion 优化器（符号动量），新颖优化器之一。 */
class Lion(
    parameters: List<Parameter>,
    learningRate: Float = 1e-4f,
    private val beta1: Float = 0.9f,
    private val beta2: Float = 0.99f,
    val weightDecay: Float = 0.0f
) : Optimizer(parameters, learningRate) {
    private val momentum = HashMap<Parameter, FloatArray>()

    override fun step() {
        forEachGradient { parameter, gradient ->
            val m = momentum.getOrPut(parameter) { FloatArray(parameter.size) }
            for (i in parameter.data.indices) {
                val update = beta1 * m[i] + (1 - beta1) * gradient[i]
                parameter.data[i] -= learningRate * (if (update >= 0) 1f else -1f)
                if (weightDecay != 0f) parameter.data[i] -= learningRate * weightDecay * parameter.data[i]
                m[i] = beta2 * m[i] + (1 - beta2) * gradient[i]
            }
        }
    }
}

class ExponentialLR(private val optimizer: Optimizer, private val gamma: Float) {
    fun step() { optimizer.learningRate *= gamma }
}

class CosineAnnealingLR(
    private val optimizer: Optimizer,
    private val maxEpochs: Int,
    private val minimumLearningRate: Float = 0f
) {
    private var epoch = 0
    fun step() {
        epoch++
        val progress = (epoch % maxEpochs).toDouble() / maxEpochs
        optimizer.learningRate = (minimumLearningRate + 0.5 * (optimizer.learningRate - minimumLearningRate) * (1 + kotlin.math.cos(Math.PI * progress))).toFloat()
    }
}
