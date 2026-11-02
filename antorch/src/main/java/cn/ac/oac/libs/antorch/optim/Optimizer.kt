package cn.ac.oac.libs.antorch.optim

import cn.ac.oac.libs.antorch.annotation.AnTorchApi
import cn.ac.oac.libs.antorch.core.Parameter
import kotlin.math.pow
import kotlin.math.sqrt

@AnTorchApi(since = "0.1.0")
abstract class Optimizer(
    val parameters: List<Parameter>,
    var learningRate: Float
) {
    abstract fun step()

    open fun zeroGrad() = parameters.forEach { it.zeroGrad() }

    protected fun forEachGradient(block: (Parameter, FloatArray) -> Unit) {
        for (parameter in parameters) {
            val gradient = parameter.grad ?: continue
            block(parameter, gradient.data)
        }
    }
}

class SGD(
    parameters: List<Parameter>,
    learningRate: Float = 0.01f,
    val momentum: Float = 0f,
    val weightDecay: Float = 0f
) : Optimizer(parameters, learningRate) {
    private val velocity = HashMap<Parameter, FloatArray>()

    override fun step() {
        forEachGradient { parameter, gradient ->
            val velocityValue = velocity.getOrPut(parameter) { FloatArray(parameter.size) }
            for (i in parameter.data.indices) {
                var g = gradient[i] + weightDecay * parameter.data[i]
                if (momentum > 0f) {
                    velocityValue[i] = momentum * velocityValue[i] + g
                    g = velocityValue[i]
                }
                parameter.data[i] -= learningRate * g
            }
        }
    }
}

@AnTorchApi(since = "0.1.0")
class Adam(
    parameters: List<Parameter>,
    learningRate: Float = 1e-3f,
    val beta1: Float = 0.9f,
    val beta2: Float = 0.999f,
    val epsilon: Float = 1e-8f,
    val weightDecay: Float = 0f
) : Optimizer(parameters, learningRate) {
    private val first = HashMap<Parameter, FloatArray>()
    private val second = HashMap<Parameter, FloatArray>()
    private var stepCount = 0

    override fun step() {
        stepCount++
        val bias1 = 1f - beta1.toDouble().pow(stepCount).toFloat()
        val bias2 = 1f - beta2.toDouble().pow(stepCount).toFloat()
        forEachGradient { parameter, gradient ->
            val m = first.getOrPut(parameter) { FloatArray(parameter.size) }
            val v = second.getOrPut(parameter) { FloatArray(parameter.size) }
            for (i in parameter.data.indices) {
                val g = gradient[i] + weightDecay * parameter.data[i]
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * g * g
                val mHat = m[i] / bias1
                val vHat = v[i] / bias2
                parameter.data[i] -= learningRate * mHat / (sqrt(vHat) + epsilon)
            }
        }
    }
}

class AdamW(
    parameters: List<Parameter>,
    learningRate: Float = 1e-3f,
    private val beta1: Float = 0.9f,
    private val beta2: Float = 0.999f,
    private val epsilon: Float = 1e-8f,
    val weightDecay: Float = 0.01f
) : Optimizer(parameters, learningRate) {
    private val first = HashMap<Parameter, FloatArray>()
    private val second = HashMap<Parameter, FloatArray>()
    private var stepCount = 0

    override fun step() {
        stepCount++
        val bias1 = 1f - beta1.toDouble().pow(stepCount).toFloat()
        val bias2 = 1f - beta2.toDouble().pow(stepCount).toFloat()
        forEachGradient { parameter, gradient ->
            val m = first.getOrPut(parameter) { FloatArray(parameter.size) }
            val v = second.getOrPut(parameter) { FloatArray(parameter.size) }
            for (i in parameter.data.indices) {
                parameter.data[i] *= (1f - learningRate * weightDecay)
                val g = gradient[i]
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * g * g
                parameter.data[i] -= learningRate * (m[i] / bias1) / (sqrt(v[i] / bias2) + epsilon)
            }
        }
    }
}

class StepLR(private val optimizer: Optimizer, private val stepSize: Int, private val gamma: Float = 0.1f) {
    private var epoch = 0
    fun step() {
        epoch++
        if (epoch % stepSize == 0) optimizer.learningRate *= gamma
    }
}
