package cn.ac.oac.libs.antorch.data

import cn.ac.oac.libs.antorch.core.Tensor
import kotlin.random.Random

data class Sample(val x: Tensor, val y: Tensor)
data class Batch(val x: Tensor, val y: Tensor)

interface Dataset {
    val size: Int
    fun get(index: Int): Sample
}

class TensorDataset(private val x: Tensor, private val y: Tensor) : Dataset {
    init {
        require(x.shape[0] == y.shape[0]) { "x/y 样本数不一致" }
    }
    override val size: Int get() = x.shape[0]
    override fun get(index: Int): Sample = Sample(DataLoader.select(x, intArrayOf(index)), DataLoader.select(y, intArrayOf(index)))
}

class SubsetDataset(private val source: Dataset, private val indices: IntArray) : Dataset {
    override val size: Int get() = indices.size
    override fun get(index: Int): Sample = source.get(indices[index])
}

class DataLoader(
    private val dataset: Dataset,
    private val batchSize: Int = 32,
    private val shuffle: Boolean = true,
    seed: Long = 42
) : Iterable<Batch> {
    private val random = Random(seed)

    override fun iterator(): Iterator<Batch> = object : Iterator<Batch> {
        private val indices = IntArray(dataset.size) { it }
        private var position = 0

        init {
            if (shuffle) {
                for (i in indices.indices.reversed()) {
                    val j = random.nextInt(i + 1)
                    val t = indices[i]; indices[i] = indices[j]; indices[j] = t
                }
            }
        }

        override fun hasNext(): Boolean = position < indices.size

        override fun next(): Batch {
            val end = minOf(position + batchSize, indices.size)
            val batchIndices = indices.copyOfRange(position, end)
            position = end
            val samples = batchIndices.map { dataset.get(it) }
            return Batch(concat(samples.map { it.x }), concat(samples.map { it.y }))
        }
    }

    companion object {
        fun select(tensor: Tensor, indices: IntArray): Tensor {
            if (tensor.ndim == 1) {
                return Tensor.fromValues(FloatArray(indices.size) { tensor.data[indices[it]] }, tensor.dtype, indices.size)
            }
            val rowSize = tensor.size / tensor.shape[0]
            val data = FloatArray(indices.size * rowSize)
            for (i in indices.indices) {
                System.arraycopy(tensor.data, indices[i] * rowSize, data, i * rowSize, rowSize)
            }
            val shape = tensor.shape.copyOf()
            shape[0] = indices.size
            return Tensor.fromValues(data, tensor.dtype, *shape)
        }

        fun concat(tensors: List<Tensor>): Tensor {
            require(tensors.isNotEmpty())
            val first = tensors.first()
            val rowSize = first.size / first.shape[0]
            val data = FloatArray(tensors.sumOf { it.size })
            var offset = 0
            for (tensor in tensors) {
                tensor.data.copyInto(data, offset)
                offset += tensor.size
            }
            val shape = first.shape.copyOf()
            shape[0] = tensors.sumOf { it.shape[0] }
            return Tensor.fromValues(data, first.dtype, *shape)
        }
    }
}

/** 内置小数据集，便于 Demo 与单测。 */
object Datasets {
    fun xor(): TensorDataset {
        val x = Tensor.fromArray(floatArrayOf(0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f), 4, 2)
        val y = Tensor.fromArray(floatArrayOf(0f, 1f, 1f, 0f), 4)
        return TensorDataset(x, y)
    }

    fun spiral(samplesPerClass: Int = 200, noise: Float = 0.12f, seed: Long = 7): TensorDataset {
        val random = Random(seed)
        val x = FloatArray(samplesPerClass * 2 * 2)
        val y = FloatArray(samplesPerClass * 2)
        for (c in 0..1) for (i in 0 until samplesPerClass) {
            val r = i.toFloat() / samplesPerClass * 2.5f
            val theta = i.toFloat() / samplesPerClass * 3.2f + c * Math.PI.toFloat()
            val index = c * samplesPerClass + i
            x[index * 2] = r * kotlin.math.cos(theta) + (random.nextFloat() - 0.5f) * noise
            x[index * 2 + 1] = r * kotlin.math.sin(theta) + (random.nextFloat() - 0.5f) * noise
            y[index] = c.toFloat()
        }
        return TensorDataset(Tensor.fromArray(x, samplesPerClass * 2, 2), Tensor.fromArray(y, samplesPerClass * 2))
    }

    /** 序列分类：label = 第一个 token 是否为偶数；用于 Transformer 快速验证。 */
    fun sequences(samples: Int = 240, vocab: Int = 16, length: Int = 8, seed: Long = 13): TensorDataset {
        val random = Random(seed)
        val tokens = LongArray(samples * length)
        val labels = FloatArray(samples)
        for (n in 0 until samples) {
            for (s in 0 until length) tokens[n * length + s] = random.nextInt(vocab).toLong()
            labels[n] = if (tokens[n * length] % 2L == 0L) 1f else 0f
        }
        return TensorDataset(
            Tensor.fromLongArray(tokens, samples, length),
            Tensor.fromInt32Array(IntArray(samples) { labels[it].toInt() }, samples)
        )
    }

    /** 28x28 MNIST 形状的二分类合成图，用于端侧 CNN 快速验证。 */
    fun mnistLike(samples: Int = 160, noise: Float = 0.15f, seed: Long = 17): TensorDataset {
        val random = Random(seed)
        val x = FloatArray(samples * 28 * 28)
        val y = FloatArray(samples)
        for (n in 0 until samples) {
            val label = n % 2
            y[n] = label.toFloat()
            for (r in 0 until 28) for (c in 0 until 28) {
                val base = if (label == 0) {
                    if (r in 12..15) 1f else 0f
                } else {
                    if (c in 12..15) 1f else 0f
                }
                x[n * 784 + r * 28 + c] = base + (random.nextFloat() - 0.5f) * noise
            }
        }
        return TensorDataset(Tensor.fromArray(x, samples, 1, 28, 28), Tensor.fromInt32Array(IntArray(samples) { y[it].toInt() }, samples))
    }

    /** 8x8 二分类“横条/竖条”图像，用于 CNN 快速验证。 */
    fun bars(samples: Int = 240, noise: Float = 0.15f, seed: Long = 11): TensorDataset {
        val random = Random(seed)
        val x = FloatArray(samples * 64)
        val y = FloatArray(samples)
        for (n in 0 until samples) {
            val label = n % 2
            y[n] = label.toFloat()
            for (r in 0 until 8) for (c in 0 until 8) {
                val base = if (label == 0) {
                    if (r in 3..4) 1f else 0f
                } else {
                    if (c in 3..4) 1f else 0f
                }
                x[n * 64 + r * 8 + c] = base + (random.nextFloat() - 0.5f) * noise
            }
        }
        return TensorDataset(Tensor.fromArray(x, samples, 1, 8, 8), Tensor.fromArray(y, samples))
    }
}
