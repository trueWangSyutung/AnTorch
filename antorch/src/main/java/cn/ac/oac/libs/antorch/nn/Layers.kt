package cn.ac.oac.libs.antorch.nn

import cn.ac.oac.libs.antorch.annotation.AnTorchApi
import cn.ac.oac.libs.antorch.core.Parameter
import cn.ac.oac.libs.antorch.core.Tensor
import kotlin.math.sqrt

@AnTorchApi(since = "0.1.0")
class Linear(val inFeatures: Int, val outFeatures: Int) : Module() {
    val weight: Parameter
    val bias: Parameter

    init {
        val bound = (1.0 / sqrt(inFeatures.toDouble())).toFloat()
        weight = Parameter(FloatArray(outFeatures * inFeatures) {
            (kotlin.random.Random.nextFloat() * 2f - 1f) * bound
        }, intArrayOf(outFeatures, inFeatures))
        bias = Parameter(FloatArray(outFeatures), intArrayOf(outFeatures))
    }

    override val parameters: List<Parameter> get() = listOf(weight, bias)

    override fun forward(input: Tensor): Tensor {
        require(input.shape.last() == inFeatures) { "Linear 输入最后一维必须是 $inFeatures" }
        if (input.ndim == 2) return input.matmul(weight.transpose()) + bias
        val rows = input.size / inFeatures
        val flattened = input.reshape(rows, inFeatures)
        val output = flattened.matmul(weight.transpose()) + bias
        val newShape = input.shape.copyOf()
        newShape[newShape.size - 1] = outFeatures
        return output.reshape(*newShape)
    }

    override fun stateDict(prefix: String): Map<String, Tensor> = mapOf(
        "${prefix}weight" to weight,
        "${prefix}bias" to bias
    )
}

class ReLU : Module() {
    override fun forward(input: Tensor): Tensor = input.relu()
}

class Sigmoid : Module() {
    override fun forward(input: Tensor): Tensor = input.sigmoid()
}

class Tanh : Module() {
    override fun forward(input: Tensor): Tensor = input.tanh()
}

class Flatten : Module() {
    override fun forward(input: Tensor): Tensor {
        require(input.ndim >= 2)
        return input.reshape(input.shape[0], input.size / input.shape[0])
    }
}

class Dropout(private val probability: Float = 0.5f) : Module() {
    override fun forward(input: Tensor): Tensor {
        if (!training || probability <= 0f) return input
        require(probability < 1f)
        val mask = FloatArray(input.size) {
            if (kotlin.random.Random.nextFloat() >= probability) 1f / (1f - probability) else 0f
        }
        return input * Tensor.fromArray(mask, *input.shape)
    }
}

/**
 * 2D 卷积（stride=1, padding=0），首版面向 8x8/28x28 小图。
 */
@AnTorchApi(since = "0.1.0")
class Conv2d(
    val inChannels: Int,
    val outChannels: Int,
    val kernelSize: Int,
    val stride: Int = 1,
    val padding: Int = 0
) : Module() {
    val weight: Parameter
    val bias: Parameter

    init {
        require(stride == 1 && padding == 0) { "首版 Conv2d 仅支持 stride=1, padding=0" }
        val bound = (1.0 / sqrt((inChannels * kernelSize * kernelSize).toDouble())).toFloat()
        weight = Parameter(FloatArray(outChannels * inChannels * kernelSize * kernelSize) {
            (kotlin.random.Random.nextFloat() * 2f - 1f) * bound
        }, intArrayOf(outChannels, inChannels, kernelSize, kernelSize))
        bias = Parameter(FloatArray(outChannels), intArrayOf(outChannels))
    }

    override val parameters: List<Parameter> get() = listOf(weight, bias)

    override fun forward(input: Tensor): Tensor {
        require(input.ndim == 4) { "Conv2d 输入必须是 [N,C,H,W]" }
        require(input.shape[1] == inChannels) { "通道数不匹配" }
        val batch = input.shape[0]
        val height = input.shape[2]
        val width = input.shape[3]
        val outH = height - kernelSize + 1
        val outW = width - kernelSize + 1
        require(outH > 0 && outW > 0) { "卷积核大于输入" }
        val outData = if (input.dtype == cn.ac.oac.libs.antorch.core.DType.FLOAT32 &&
            weight.dtype == cn.ac.oac.libs.antorch.core.DType.FLOAT32 &&
            cn.ac.oac.libs.antorch.core.NativeOps.available
        ) {
            cn.ac.oac.libs.antorch.core.NativeOps.conv2d(
                input.data, weight.data, bias.data, batch, inChannels, height, width, outChannels, kernelSize
            )
        } else {
            val result = FloatArray(batch * outChannels * outH * outW)
            for (n in 0 until batch) for (o in 0 until outChannels) for (oh in 0 until outH) for (ow in 0 until outW) {
                var sum = bias.data[o]
                for (c in 0 until inChannels) for (kh in 0 until kernelSize) for (kw in 0 until kernelSize) {
                    val ih = oh + kh
                    val iw = ow + kw
                    sum += input.data[((n * inChannels + c) * height + ih) * width + iw] *
                        weight.data[((o * inChannels + c) * kernelSize + kh) * kernelSize + kw]
                }
                result[((n * outChannels + o) * outH + oh) * outW + ow] = sum
            }
            result
        }
        val output = Tensor(outData, intArrayOf(batch, outChannels, outH, outW), input.requiresGrad || weight.requiresGrad || bias.requiresGrad)
        output.parents = listOf(input, weight, bias)
        output.backwardFn = { g ->
            val gradInput = FloatArray(input.size)
            val gradWeight = FloatArray(weight.size)
            val gradBias = FloatArray(outChannels)
            for (n in 0 until batch) for (o in 0 until outChannels) for (oh in 0 until outH) for (ow in 0 until outW) {
                val delta = g.data[((n * outChannels + o) * outH + oh) * outW + ow]
                gradBias[o] += delta
                for (c in 0 until inChannels) for (kh in 0 until kernelSize) for (kw in 0 until kernelSize) {
                    val ih = oh + kh
                    val iw = ow + kw
                    val inputIndex = ((n * inChannels + c) * height + ih) * width + iw
                    val weightIndex = ((o * inChannels + c) * kernelSize + kh) * kernelSize + kw
                    gradInput[inputIndex] += delta * weight.data[weightIndex]
                    gradWeight[weightIndex] += delta * input.data[inputIndex]
                }
            }
            Tensor.accumulate(input, Tensor(gradInput, input.shape))
            Tensor.accumulate(weight, Tensor(gradWeight, weight.shape))
            Tensor.accumulate(bias, Tensor(gradBias, bias.shape))
        }
        return output
    }

    override fun stateDict(prefix: String): Map<String, Tensor> = mapOf(
        "${prefix}weight" to weight,
        "${prefix}bias" to bias
    )
}
