package cn.ac.oac.libs.antorch.core

import cn.ac.oac.libs.antorch.annotation.AnTorchApi
import java.util.IdentityHashMap
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.sqrt
import kotlin.random.Random

/**
 * PyTorch 风格张量：连续行主序存储，float32，动态计算图，反向自动微分。
 */
@AnTorchApi(since = "0.1.0")
open class Tensor internal constructor(
    internal val storage: Any,
    val shape: IntArray,
    val requiresGrad: Boolean = false,
    val dtype: DType = DType.FLOAT32
) {
    constructor(data: FloatArray, shape: IntArray, requiresGrad: Boolean = false) :
        this(data as Any, shape, requiresGrad, DType.FLOAT32)

    private var floatCache: FloatArray? = null

    /** 计算视图：FLOAT32 直接返回底层数组；其他 dtype 转为 FloatArray 缓存。 */
    val data: FloatArray
        get() = (storage as? FloatArray) ?: floatCache ?: FloatArray(size) { valueAt(it) }.also { floatCache = it }

    init {
        require(shape.isNotEmpty()) { "shape 不能为空" }
        require(shape.all { it > 0 }) { "shape 必须为正: ${shape.contentToString()}" }
        require(storageSize(storage) == shape.fold(1) { a, b -> a * b }) {
            "storage.size=${storageSize(storage)} 与 shape=${shape.contentToString()} 不匹配"
        }
        require(requiresGrad == false || dtype == DType.FLOAT32) {
            "首版仅 FLOAT32 支持 requiresGrad"
        }
    }

    private fun storageSize(value: Any): Int = when (value) {
        is FloatArray -> value.size
        is ShortArray -> value.size
        is ByteArray -> value.size
        is IntArray -> value.size
        is LongArray -> value.size
        else -> 0
    }

    private fun valueAt(index: Int): Float = when (val value = storage) {
        is FloatArray -> value[index]
        is ShortArray -> Half.toFloat(value[index])
        is ByteArray -> value[index].toFloat()
        is IntArray -> value[index].toFloat()
        is LongArray -> value[index].toFloat()
        else -> 0f
    }

    fun toFloatArray(): FloatArray = FloatArray(size) { valueAt(it) }
    fun toIntArray(): IntArray = IntArray(size) {
        when (val value = storage) {
            is LongArray -> value[it].toInt()
            is IntArray -> value[it]
            is ByteArray -> value[it].toInt()
            else -> valueAt(it).toInt()
        }
    }
    fun toLongArray(): LongArray = LongArray(size) {
        when (val value = storage) {
            is LongArray -> value[it]
            is IntArray -> value[it].toLong()
            is ByteArray -> value[it].toLong()
            else -> valueAt(it).toLong()
        }
    }

    fun astype(target: DType): Tensor {
        if (target == dtype) return copy()
        return when (target) {
            DType.FLOAT16 -> fromValues(toFloatArray(), DType.FLOAT16, *shape.copyOf())
            DType.FLOAT32 -> fromValues(toFloatArray(), DType.FLOAT32, *shape.copyOf())
            DType.INT8 -> fromInt8Array(IntArray(size) { valueAt(it).toInt() }, *shape.copyOf())
            DType.INT32 -> fromValues(toFloatArray(), DType.INT32, *shape.copyOf())
            DType.INT64 -> fromLongArray(toLongArray(), *shape.copyOf())
        }
    }

    val ndim: Int get() = shape.size
    val size: Int get() = shape.fold(1) { a, b -> a * b }
    var grad: Tensor? = null
        internal set
    internal var backwardFn: ((Tensor) -> Unit)? = null
    internal var parents: List<Tensor> = emptyList()

    fun item(): Float {
        require(size == 1) { "item() 仅适用于单元素张量" }
        return data[0]
    }

    fun zeroGrad() { grad = null }

    fun backward() {
        if (!requiresGrad) return
        val topo = ArrayList<Tensor>()
        val visited = IdentityHashMap<Tensor, Boolean>()
        fun visit(node: Tensor) {
            if (visited.put(node, true) != null) return
            node.parents.forEach { visit(it) }
            topo.add(node)
        }
        visit(this)
        if (grad == null) grad = ones(*shape)
        for (node in topo.asReversed()) {
            val g = node.grad ?: continue
            node.backwardFn?.invoke(g)
        }
    }

    fun detach(): Tensor = Tensor(data.copyOf(), shape.copyOf(), false)

    fun copy(requiresGrad: Boolean = this.requiresGrad): Tensor =
        Tensor(copyStorage(), shape.copyOf(), requiresGrad, dtype)

    private fun copyStorage(): Any = when (val value = storage) {
        is FloatArray -> value.copyOf()
        is ShortArray -> value.copyOf()
        is ByteArray -> value.copyOf()
        is IntArray -> value.copyOf()
        is LongArray -> value.copyOf()
        else -> FloatArray(size)
    }

    fun reshape(vararg newShape: Int): Tensor {
        require(newShape.fold(1) { a, b -> a * b } == size) { "reshape 元素数不一致" }
        val out = Tensor(data, newShape.copyOf(), requiresGrad)
        out.parents = listOf(this)
        out.backwardFn = { g -> accumulate(this, g.reshape(*shape)) }
        return out
    }

    fun flatten(): Tensor = reshape(size)

    fun transpose(): Tensor {
        require(ndim == 2) { "transpose 仅支持二维" }
        val rows = shape[0]
        val cols = shape[1]
        val outData = if (dtype == DType.FLOAT32 && NativeOps.available && size > 64) {
            NativeOps.transpose(data, rows, cols)
        } else {
            val result = FloatArray(size)
            for (r in 0 until rows) for (c in 0 until cols) result[c * rows + r] = data[r * cols + c]
            result
        }
        val out = Tensor(outData, intArrayOf(cols, rows), requiresGrad)
        out.parents = listOf(this)
        out.backwardFn = { g -> accumulate(this, g.transpose()) }
        return out
    }

    fun matmul(other: Tensor): Tensor {
        require(ndim == 2 && other.ndim == 2) { "matmul 仅支持二维" }
        require(shape[1] == other.shape[0]) {
            "矩阵维度不匹配: ${shape.contentToString()} x ${other.shape.contentToString()}"
        }
        val m = shape[0]
        val k = shape[1]
        val n = other.shape[1]
        val outData = if (NativeOps.available && m * n * k > 256) {
            NativeOps.matmul(data, other.data, m, k, n)
        } else {
            val result = FloatArray(m * n)
            for (i in 0 until m) for (j in 0 until n) {
                var sum = 0f
                for (p in 0 until k) sum += data[i * k + p] * other.data[p * n + j]
                result[i * n + j] = sum
            }
            result
        }
        val out = Tensor(outData, intArrayOf(m, n), requiresGrad || other.requiresGrad)
        out.parents = listOf(this, other)
        out.backwardFn = { g ->
            accumulate(this, g.matmul(other.transpose()))
            accumulate(other, this.transpose().matmul(g))
        }
        return out
    }

    operator fun plus(other: Tensor): Tensor {
        if (shape.contentEquals(other.shape)) {
            val outData = if (dtype == DType.FLOAT32 && other.dtype == DType.FLOAT32 && NativeOps.available && size > 64) {
                NativeOps.add(data, other.data)
            } else {
                FloatArray(size) { data[it] + other.data[it] }
            }
            val out = Tensor(outData, shape.copyOf(), requiresGrad || other.requiresGrad)
            out.parents = listOf(this, other)
            out.backwardFn = { g -> accumulate(this, g); accumulate(other, g) }
            return out
        }
        if (other.size == 1) {
            val scalar = other.data[0]
            val outData = if (dtype == DType.FLOAT32 && NativeOps.available && size > 64) NativeOps.addScalar(data, scalar)
            else FloatArray(size) { data[it] + scalar }
            val out = Tensor(outData, shape.copyOf(), requiresGrad || other.requiresGrad)
            out.parents = listOf(this, other)
            out.backwardFn = { g ->
                accumulate(this, g)
                accumulate(other, Tensor(floatArrayOf(g.data.sum()), intArrayOf(1)))
            }
            return out
        }
        // 2D + 1D bias 广播
        if (ndim == 2 && other.ndim == 1 && other.shape[0] == shape[1]) {
            val rows = shape[0]
            val cols = shape[1]
            val outData = FloatArray(size)
            for (r in 0 until rows) for (c in 0 until cols) {
                outData[r * cols + c] = data[r * cols + c] + other.data[c]
            }
            val out = Tensor(outData, shape.copyOf(), requiresGrad || other.requiresGrad)
            out.parents = listOf(this, other)
            out.backwardFn = { g ->
                accumulate(this, g)
                val biasGrad = FloatArray(cols)
                for (r in 0 until rows) for (c in 0 until cols) biasGrad[c] += g.data[r * cols + c]
                accumulate(other, Tensor(biasGrad, intArrayOf(cols)))
            }
            return out
        }
        throw IllegalArgumentException("不支持的广播: ${shape.contentToString()} + ${other.shape.contentToString()}")
    }

    operator fun minus(other: Tensor): Tensor {
        if (shape.contentEquals(other.shape) && dtype == DType.FLOAT32 && other.dtype == DType.FLOAT32 && NativeOps.available && size > 64) {
            val outData = NativeOps.sub(data, other.data)
            val out = Tensor(outData, shape.copyOf(), requiresGrad || other.requiresGrad)
            out.parents = listOf(this, other)
            out.backwardFn = { g ->
                accumulate(this, g)
                accumulate(other, Tensor(FloatArray(size) { -g.data[it] }, shape.copyOf()))
            }
            return out
        }
        return plus(other * -1f)
    }

    operator fun plus(scalar: Float): Tensor = plus(Tensor(floatArrayOf(scalar), intArrayOf(1)))
    operator fun minus(scalar: Float): Tensor = plus(-scalar)
    operator fun div(scalar: Float): Tensor = this * (1f / scalar)

    operator fun div(other: Tensor): Tensor {
        if (shape.contentEquals(other.shape) && dtype == DType.FLOAT32 && other.dtype == DType.FLOAT32 && NativeOps.available && size > 64) {
            val outData = NativeOps.div(data, other.data)
            val out = Tensor(outData, shape.copyOf(), requiresGrad || other.requiresGrad)
            out.parents = listOf(this, other)
            out.backwardFn = { g ->
                accumulate(this, Tensor(FloatArray(size) { g.data[it] / other.data[it] }, shape.copyOf()))
                accumulate(other, Tensor(FloatArray(size) { -g.data[it] * data[it] / (other.data[it] * other.data[it]) }, shape.copyOf()))
            }
            return out
        }
        throw IllegalArgumentException("除法仅支持同形状张量")
    }

    operator fun times(other: Tensor): Tensor {
        if (shape.contentEquals(other.shape)) {
            val outData = if (dtype == DType.FLOAT32 && other.dtype == DType.FLOAT32 && NativeOps.available && size > 64) {
                NativeOps.mul(data, other.data)
            } else {
                FloatArray(size) { data[it] * other.data[it] }
            }
            val out = Tensor(outData, shape.copyOf(), requiresGrad || other.requiresGrad)
            out.parents = listOf(this, other)
            out.backwardFn = { g ->
                accumulate(this, Tensor(FloatArray(size) { g.data[it] * other.data[it] }, shape.copyOf()))
                accumulate(other, Tensor(FloatArray(size) { g.data[it] * data[it] }, shape.copyOf()))
            }
            return out
        }
        if (other.size == 1) return this * other.data[0]
        throw IllegalArgumentException("乘法仅支持同形状或标量")
    }

    operator fun times(scalar: Float): Tensor {
        val outData = if (dtype == DType.FLOAT32 && NativeOps.available && size > 64) NativeOps.mulScalar(data, scalar)
        else FloatArray(size) { data[it] * scalar }
        val out = Tensor(outData, shape.copyOf(), requiresGrad)
        out.parents = listOf(this)
        out.backwardFn = { g -> accumulate(this, Tensor(FloatArray(size) { g.data[it] * scalar }, shape.copyOf())) }
        return out
    }

    fun square(): Tensor = this * this

    fun relu(): Tensor {
        val outData = if (NativeOps.available && size > 1024) NativeOps.relu(data) else FloatArray(size) { if (data[it] > 0f) data[it] else 0f }
        val out = Tensor(outData, shape.copyOf(), requiresGrad)
        out.parents = listOf(this)
        out.backwardFn = { g ->
            accumulate(this, Tensor(FloatArray(size) { if (data[it] > 0f) g.data[it] else 0f }, shape.copyOf()))
        }
        return out
    }

    fun sigmoid(): Tensor {
        val outData = if (dtype == DType.FLOAT32 && NativeOps.available && size > 64) NativeOps.sigmoid(data)
        else FloatArray(size) { 1f / (1f + exp(-data[it].toDouble()).toFloat()) }
        val out = Tensor(outData, shape.copyOf(), requiresGrad)
        out.parents = listOf(this)
        out.backwardFn = { g ->
            accumulate(this, Tensor(FloatArray(size) { g.data[it] * outData[it] * (1f - outData[it]) }, shape.copyOf()))
        }
        return out
    }

    fun tanh(): Tensor {
        val outData = if (dtype == DType.FLOAT32 && NativeOps.available && size > 64) NativeOps.tanh(data)
        else FloatArray(size) { kotlin.math.tanh(data[it].toDouble()).toFloat() }
        val out = Tensor(outData, shape.copyOf(), requiresGrad)
        out.parents = listOf(this)
        out.backwardFn = { g ->
            accumulate(this, Tensor(FloatArray(size) { g.data[it] * (1f - outData[it] * outData[it]) }, shape.copyOf()))
        }
        return out
    }

    fun softmax(): Tensor {
        require(ndim == 2) { "softmax 仅支持二维 [batch, classes]" }
        val rows = shape[0]
        val cols = shape[1]
        val outData = if (NativeOps.available && size > 1024) NativeOps.softmaxRows(data, rows, cols) else {
            val result = FloatArray(size)
            for (r in 0 until rows) {
                var max = Float.NEGATIVE_INFINITY
                for (c in 0 until cols) max = maxOf(max, data[r * cols + c])
                var sum = 0f
                for (c in 0 until cols) {
                    val e = exp((data[r * cols + c] - max).toDouble()).toFloat()
                    result[r * cols + c] = e
                    sum += e
                }
                for (c in 0 until cols) result[r * cols + c] /= sum
            }
            result
        }
        val out = Tensor(outData, shape.copyOf(), requiresGrad)
        out.parents = listOf(this)
        out.backwardFn = { g ->
            val dx = FloatArray(size)
            for (r in 0 until rows) {
                var dot = 0f
                for (c in 0 until cols) dot += g.data[r * cols + c] * outData[r * cols + c]
                for (c in 0 until cols) {
                    dx[r * cols + c] = outData[r * cols + c] * (g.data[r * cols + c] - dot)
                }
            }
            accumulate(this, Tensor(dx, shape.copyOf()))
        }
        return out
    }

    fun sum(): Tensor {
        var total = 0f
        if (dtype == DType.FLOAT32 && NativeOps.available && size > 64) total = NativeOps.sum(data)
        else for (v in data) total += v
        val out = Tensor(floatArrayOf(total), intArrayOf(1), requiresGrad)
        out.parents = listOf(this)
        out.backwardFn = { g -> accumulate(this, Tensor(FloatArray(size) { g.data[0] }, shape.copyOf())) }
        return out
    }

    fun mean(): Tensor = sum() * (1f / size)

    fun log(): Tensor {
        val outData = FloatArray(size) { ln(data[it].toDouble()).toFloat() }
        val out = Tensor(outData, shape.copyOf(), requiresGrad)
        out.parents = listOf(this)
        out.backwardFn = { g -> accumulate(this, Tensor(FloatArray(size) { g.data[it] / data[it] }, shape.copyOf())) }
        return out
    }

    operator fun get(index: Int): Float = data[index]
    operator fun get(vararg indices: Int): Float {
        var offset = 0
        var stride = 1
        for (d in indices.indices.reversed()) {
            offset += indices[d] * stride
            stride *= shape[d]
        }
        return data[offset]
    }

    override fun toString(): String = "Tensor(shape=${shape.contentToString()}, requiresGrad=$requiresGrad)"

    companion object {
        fun zeros(vararg shape: Int): Tensor = zeros(DType.FLOAT32, *shape)
        fun zeros(dtype: DType, vararg shape: Int): Tensor = full(0f, dtype, *shape)
        fun ones(vararg shape: Int): Tensor = full(1f, *shape)
        fun full(value: Float, vararg shape: Int): Tensor = full(value, DType.FLOAT32, *shape)
        fun full(value: Float, dtype: DType, vararg shape: Int): Tensor =
            fromValues(FloatArray(shape.fold(1) { a, b -> a * b }) { value }, dtype, *shape)

        fun of(vararg values: Float): Tensor = fromValues(values.copyOf(), DType.FLOAT32, values.size)
        fun ofType(dtype: DType, vararg values: Float): Tensor = fromValues(values.copyOf(), dtype, values.size)

        fun fromArray(data: FloatArray, vararg shape: Int): Tensor =
            fromValues(data, DType.FLOAT32, *shape)

        fun fromArray(data: FloatArray, shape: IntArray, dtype: DType = DType.FLOAT32): Tensor =
            fromValues(data, dtype, *shape)

        fun fromFloat16Array(data: ShortArray, vararg shape: Int): Tensor =
            Tensor(data.copyOf(), shape.copyOf(), false, DType.FLOAT16)

        fun fromInt8Array(data: IntArray, vararg shape: Int): Tensor =
            Tensor(ByteArray(data.size) { data[it].toByte() }, shape.copyOf(), false, DType.INT8)

        fun fromInt32Array(data: IntArray, vararg shape: Int): Tensor =
            Tensor(data.copyOf(), shape.copyOf(), false, DType.INT32)

        fun fromLongArray(data: LongArray, vararg shape: Int): Tensor =
            Tensor(data.copyOf(), shape.copyOf(), false, DType.INT64)

        fun fromValues(data: FloatArray, dtype: DType, vararg shape: Int): Tensor {
            val count = shape.fold(1) { a, b -> a * b }
            require(data.size == count) { "数据长度与 shape 不匹配" }
            val storage: Any = when (dtype) {
                DType.FLOAT32 -> data.copyOf()
                DType.FLOAT16 -> ShortArray(count) { Half.toHalf(data[it]) }
                DType.INT8 -> ByteArray(count) { data[it].toInt().toByte() }
                DType.INT32 -> IntArray(count) { data[it].toInt() }
                DType.INT64 -> LongArray(count) { data[it].toLong() }
            }
            return Tensor(storage, shape.copyOf(), false, dtype)
        }

        fun randn(vararg shape: Int, seed: Long = System.nanoTime()): Tensor {
            val random = Random(seed)
            val count = shape.fold(1) { a, b -> a * b }
            val data = FloatArray(count)
            var i = 0
            while (i < count) {
                val u1 = random.nextDouble().coerceAtLeast(1e-12)
                val u2 = random.nextDouble()
                val r = sqrt(-2.0 * ln(u1))
                data[i++] = (r * kotlin.math.cos(2 * Math.PI * u2)).toFloat()
                if (i < count) data[i++] = (r * kotlin.math.sin(2 * Math.PI * u2)).toFloat()
            }
            return Tensor(data, shape.copyOf())
        }

        fun arange(start: Float, end: Float, step: Float = 1f): Tensor {
            val count = kotlin.math.max(0, kotlin.math.ceil((end - start) / step).toInt())
            return Tensor(FloatArray(count) { start + it * step }, intArrayOf(count))
        }

        internal fun accumulate(target: Tensor, g: Tensor) {
            if (!target.requiresGrad) return
            require(target.shape.contentEquals(g.shape)) {
                "梯度形状不匹配: ${target.shape.contentToString()} vs ${g.shape.contentToString()}"
            }
            val existing = target.grad
            target.grad = if (existing == null) {
                Tensor(g.data.copyOf(), g.shape.copyOf())
            } else {
                Tensor(FloatArray(g.size) { existing.data[it] + g.data[it] }, g.shape.copyOf())
            }
        }
    }
}

/** 可训练参数。 */
class Parameter(data: FloatArray, shape: IntArray) : Tensor(data, shape, requiresGrad = true) {
    constructor(vararg shape: Int) : this(FloatArray(shape.fold(1) { a, b -> a * b }), shape.copyOf())
}
