package cn.ac.oac.libs.antorch.nn

import cn.ac.oac.libs.antorch.annotation.AnTorchApi
import cn.ac.oac.libs.antorch.core.Parameter
import cn.ac.oac.libs.antorch.core.Tensor
import kotlin.math.sqrt

/** GELU（tanh 近似），由现有 Tensor 算子组合，自动微分自然成立。 */
class GELU : Module() {
    override fun forward(input: Tensor): Tensor {
        val inner = (input + input * input * input * 0.044715f) * 0.7978845608f
        return input * 0.5f * (inner.tanh() + 1f)
    }
}

@AnTorchApi(since = "0.1.0")
class LayerNorm(val features: Int, val epsilon: Float = 1e-5f) : Module() {
    val gamma = Parameter(FloatArray(features) { 1f }, intArrayOf(features))
    val beta = Parameter(FloatArray(features), intArrayOf(features))

    override val parameters: List<Parameter> get() = listOf(gamma, beta)

    override fun forward(input: Tensor): Tensor {
        require(input.shape.last() == features) { "LayerNorm 最后一维必须是 $features" }
        val rows = input.size / features
        val output = FloatArray(input.size)
        val normalized = FloatArray(input.size)
        val means = FloatArray(rows)
        val stds = FloatArray(rows)
        for (r in 0 until rows) {
            val base = r * features
            var mean = 0f
            for (c in 0 until features) mean += input.data[base + c]
            mean /= features
            var variance = 0f
            for (c in 0 until features) {
                val d = input.data[base + c] - mean
                variance += d * d
            }
            variance /= features
            val std = sqrt(variance + epsilon)
            means[r] = mean
            stds[r] = std
            for (c in 0 until features) {
                val n = (input.data[base + c] - mean) / std
                normalized[base + c] = n
                output[base + c] = n * gamma.data[c] + beta.data[c]
            }
        }
        val out = Tensor(output, input.shape.copyOf(), input.requiresGrad || gamma.requiresGrad || beta.requiresGrad)
        out.parents = listOf(input, gamma, beta)
        out.backwardFn = { g ->
            val dx = FloatArray(input.size)
            val dGamma = FloatArray(features)
            val dBeta = FloatArray(features)
            for (r in 0 until rows) {
                val base = r * features
                var meanD = 0f
                var meanDN = 0f
                for (c in 0 until features) {
                    val dy = g.data[base + c]
                    dGamma[c] += dy * normalized[base + c]
                    dBeta[c] += dy
                    val dn = dy * gamma.data[c]
                    meanD += dn
                    meanDN += dn * normalized[base + c]
                }
                meanD /= features
                meanDN /= features
                for (c in 0 until features) {
                    val dn = g.data[base + c] * gamma.data[c]
                    dx[base + c] = (dn - meanD - normalized[base + c] * meanDN) / stds[r]
                }
            }
            Tensor.accumulate(input, Tensor(dx, input.shape.copyOf()))
            Tensor.accumulate(gamma, Tensor(dGamma, gamma.shape.copyOf()))
            Tensor.accumulate(beta, Tensor(dBeta, beta.shape.copyOf()))
        }
        return out
    }

    override fun stateDict(prefix: String): Map<String, Tensor> = mapOf(
        "${prefix}gamma" to gamma,
        "${prefix}beta" to beta
    )
}

@AnTorchApi(since = "0.1.0")
class Embedding(val numEmbeddings: Int, val embeddingDim: Int) : Module() {
    val weight = Parameter(
        FloatArray(numEmbeddings * embeddingDim) { (kotlin.random.Random.nextFloat() - 0.5f) * 0.1f },
        intArrayOf(numEmbeddings, embeddingDim)
    )

    override val parameters: List<Parameter> get() = listOf(weight)

    override fun forward(input: Tensor): Tensor {
        require(input.dtype.isInteger && input.ndim == 2) { "Embedding 输入必须是 [batch, seq] 整数张量" }
        val batch = input.shape[0]
        val sequence = input.shape[1]
        val ids = input.toIntArray()
        val output = FloatArray(batch * sequence * embeddingDim)
        for (i in ids.indices) {
            require(ids[i] in 0 until numEmbeddings) { "token 越界: ${ids[i]}" }
            System.arraycopy(weight.data, ids[i] * embeddingDim, output, i * embeddingDim, embeddingDim)
        }
        val out = Tensor(output, intArrayOf(batch, sequence, embeddingDim), weight.requiresGrad)
        out.parents = listOf(weight)
        out.backwardFn = { g ->
            val dWeight = FloatArray(weight.size)
            for (i in ids.indices) {
                val token = ids[i]
                for (d in 0 until embeddingDim) dWeight[token * embeddingDim + d] += g.data[i * embeddingDim + d]
            }
            Tensor.accumulate(weight, Tensor(dWeight, weight.shape.copyOf()))
        }
        return out
    }

    override fun stateDict(prefix: String): Map<String, Tensor> = mapOf("${prefix}weight" to weight)
}

/** 可学习的位置编码 [maxSeq, dim]。 */
class PositionalEncoding(val maxSeq: Int, val dim: Int) : Module() {
    val weight = Parameter(FloatArray(maxSeq * dim) { i ->
        val position = i / dim
        val d = i % dim
        if (d % 2 == 0) kotlin.math.sin(position / Math.pow(10000.0, d.toDouble() / dim)).toFloat()
        else kotlin.math.cos(position / Math.pow(10000.0, (d - 1).toDouble() / dim)).toFloat()
    }, intArrayOf(maxSeq, dim))

    override val parameters: List<Parameter> get() = listOf(weight)

    override fun forward(input: Tensor): Tensor {
        require(input.ndim == 3 && input.shape[1] <= maxSeq && input.shape[2] == dim)
        val batch = input.shape[0]
        val sequence = input.shape[1]
        val output = FloatArray(input.size)
        for (b in 0 until batch) for (s in 0 until sequence) for (d in 0 until dim) {
            val index = (b * sequence + s) * dim + d
            output[index] = input.data[index] + weight.data[s * dim + d]
        }
        val out = Tensor(output, input.shape.copyOf(), input.requiresGrad || weight.requiresGrad)
        out.parents = listOf(input, weight)
        out.backwardFn = { g ->
            Tensor.accumulate(input, Tensor(g.data.copyOf(), input.shape.copyOf()))
            val dWeight = FloatArray(weight.size)
            for (b in 0 until batch) for (s in 0 until sequence) for (d in 0 until dim) {
                dWeight[s * dim + d] += g.data[(b * sequence + s) * dim + d]
            }
            Tensor.accumulate(weight, Tensor(dWeight, weight.shape.copyOf()))
        }
        return out
    }

    override fun stateDict(prefix: String): Map<String, Tensor> = mapOf("${prefix}weight" to weight)
}

/** 多头自注意力；自定义反向，支持 [B,S,D]。 */
@AnTorchApi(since = "0.1.0")
class MultiHeadAttention(val embedDim: Int, val heads: Int) : Module() {
    init { require(embedDim % heads == 0) { "embedDim 必须能被 heads 整除" } }

    private val headDim = embedDim / heads
    val query = Linear(embedDim, embedDim)
    val key = Linear(embedDim, embedDim)
    val value = Linear(embedDim, embedDim)
    val output = Linear(embedDim, embedDim)

    override val parameters: List<Parameter>
        get() = query.parameters + key.parameters + value.parameters + output.parameters

    override fun forward(input: Tensor): Tensor {
        require(input.ndim == 3 && input.shape[2] == embedDim)
        val batch = input.shape[0]
        val sequence = input.shape[1]
        val flat = input.reshape(batch * sequence, embedDim)
        val q = query(flat).reshape(batch, sequence, embedDim)
        val k = key(flat).reshape(batch, sequence, embedDim)
        val v = value(flat).reshape(batch, sequence, embedDim)
        val attended = AttentionOp.forward(q, k, v, heads)
        return output(attended.reshape(batch * sequence, embedDim)).reshape(batch, sequence, embedDim)
    }

    override fun stateDict(prefix: String): Map<String, Tensor> {
        val result = LinkedHashMap<String, Tensor>()
        result.putAll(query.stateDict("${prefix}query."))
        result.putAll(key.stateDict("${prefix}key."))
        result.putAll(value.stateDict("${prefix}value."))
        result.putAll(output.stateDict("${prefix}output."))
        return result
    }
}

internal object AttentionOp {
    fun forward(q: Tensor, k: Tensor, v: Tensor, heads: Int): Tensor {
        val batch = q.shape[0]
        val sequence = q.shape[1]
        val dim = q.shape[2]
        val headDim = dim / heads
        val scale = 1f / sqrt(headDim.toFloat())
        val output = FloatArray(batch * sequence * dim)
        val probabilities = Array(batch * heads) { Array(sequence) { FloatArray(sequence) } }
        for (b in 0 until batch) for (h in 0 until heads) {
            val offset = h * headDim
            for (i in 0 until sequence) {
                var max = Float.NEGATIVE_INFINITY
                for (j in 0 until sequence) {
                    var score = 0f
                    for (d in 0 until headDim) {
                        score += q.data[(b * sequence + i) * dim + offset + d] * k.data[(b * sequence + j) * dim + offset + d]
                    }
                    score *= scale
                    probabilities[b * heads + h][i][j] = score
                    max = maxOf(max, score)
                }
                var sum = 0f
                for (j in 0 until sequence) {
                    val e = kotlin.math.exp((probabilities[b * heads + h][i][j] - max).toDouble()).toFloat()
                    probabilities[b * heads + h][i][j] = e
                    sum += e
                }
                for (j in 0 until sequence) probabilities[b * heads + h][i][j] /= sum
                for (d in 0 until headDim) {
                    var value = 0f
                    for (j in 0 until sequence) value += probabilities[b * heads + h][i][j] * v.data[(b * sequence + j) * dim + offset + d]
                    output[(b * sequence + i) * dim + offset + d] = value
                }
            }
        }
        val result = Tensor(output, intArrayOf(batch, sequence, dim), q.requiresGrad || k.requiresGrad || v.requiresGrad)
        result.parents = listOf(q, k, v)
        result.backwardFn = { g ->
            val dQ = FloatArray(q.size)
            val dK = FloatArray(k.size)
            val dV = FloatArray(v.size)
            for (b in 0 until batch) for (h in 0 until heads) {
                val offset = h * headDim
                val p = probabilities[b * heads + h]
                for (i in 0 until sequence) {
                    val dP = FloatArray(sequence)
                    var rowDot = 0f
                    for (j in 0 until sequence) {
                        var dot = 0f
                        for (d in 0 until headDim) {
                            dot += g.data[(b * sequence + i) * dim + offset + d] * v.data[(b * sequence + j) * dim + offset + d]
                        }
                        dP[j] = dot
                        rowDot += dot * p[i][j]
                    }
                    for (j in 0 until sequence) {
                        val dScore = p[i][j] * (dP[j] - rowDot) * scale
                        for (d in 0 until headDim) {
                            dQ[(b * sequence + i) * dim + offset + d] += dScore * k.data[(b * sequence + j) * dim + offset + d]
                            dK[(b * sequence + j) * dim + offset + d] += dScore * q.data[(b * sequence + i) * dim + offset + d]
                            dV[(b * sequence + j) * dim + offset + d] += p[i][j] * g.data[(b * sequence + i) * dim + offset + d]
                        }
                    }
                }
            }
            Tensor.accumulate(q, Tensor(dQ, q.shape.copyOf()))
            Tensor.accumulate(k, Tensor(dK, k.shape.copyOf()))
            Tensor.accumulate(v, Tensor(dV, v.shape.copyOf()))
        }
        return result
    }
}

class TransformerBlock(
    val dim: Int,
    val heads: Int,
    val ffnDim: Int = dim * 4
) : Module() {
    val norm1 = LayerNorm(dim)
    val attention = MultiHeadAttention(dim, heads)
    val norm2 = LayerNorm(dim)
    val ffn = Sequential(Linear(dim, ffnDim), GELU(), Linear(ffnDim, dim))

    override val parameters: List<Parameter>
        get() = norm1.parameters + attention.parameters + norm2.parameters + ffn.parameters

    override fun forward(input: Tensor): Tensor {
        val afterAttention = input + attention(norm1(input))
        return afterAttention + ffn(norm2(afterAttention))
    }

    override fun train() {
        super.train(); norm1.train(); attention.train(); norm2.train(); ffn.train()
    }

    override fun eval() {
        super.eval(); norm1.eval(); attention.eval(); norm2.eval(); ffn.eval()
    }

    override fun stateDict(prefix: String): Map<String, Tensor> {
        val result = LinkedHashMap<String, Tensor>()
        result.putAll(norm1.stateDict("${prefix}norm1."))
        result.putAll(attention.stateDict("${prefix}attention."))
        result.putAll(norm2.stateDict("${prefix}norm2."))
        result.putAll(ffn.stateDict("${prefix}ffn."))
        return result
    }
}

/** 序列平均池化 [B,S,D] -> [B,D]。 */
class SequenceMean : Module() {
    override fun forward(input: Tensor): Tensor {
        require(input.ndim == 3)
        val batch = input.shape[0]
        val sequence = input.shape[1]
        val dim = input.shape[2]
        val output = FloatArray(batch * dim)
        for (b in 0 until batch) for (s in 0 until sequence) for (d in 0 until dim) {
            output[b * dim + d] += input.data[(b * sequence + s) * dim + d] / sequence
        }
        val out = Tensor(output, intArrayOf(batch, dim), input.requiresGrad)
        out.parents = listOf(input)
        out.backwardFn = { g ->
            val dx = FloatArray(input.size)
            for (b in 0 until batch) for (s in 0 until sequence) for (d in 0 until dim) {
                dx[(b * sequence + s) * dim + d] = g.data[b * dim + d] / sequence
            }
            Tensor.accumulate(input, Tensor(dx, input.shape.copyOf()))
        }
        return out
    }
}

@AnTorchApi(since = "0.1.0")
class TransformerClassifier(
    val vocabSize: Int,
    val maxSeq: Int,
    val dim: Int,
    val heads: Int,
    val classes: Int,
    layers: Int = 1,
    ffnDim: Int = dim * 4
) : Module() {
    val embedding = Embedding(vocabSize, dim)
    val positional = PositionalEncoding(maxSeq, dim)
    val blocks = List(layers) { TransformerBlock(dim, heads, ffnDim) }
    val pool = SequenceMean()
    val head = Linear(dim, classes)

    override val parameters: List<Parameter>
        get() = embedding.parameters + positional.parameters + blocks.flatMap { it.parameters } + pool.parameters + head.parameters

    override fun forward(input: Tensor): Tensor {
        var value = positional(embedding(input))
        blocks.forEach { value = it(value) }
        return head(pool(value))
    }

    override fun train() {
        super.train(); embedding.train(); positional.train(); blocks.forEach { it.train() }; pool.train(); head.train()
    }

    override fun eval() {
        super.eval(); embedding.eval(); positional.eval(); blocks.forEach { it.eval() }; pool.eval(); head.eval()
    }

    override fun stateDict(prefix: String): Map<String, Tensor> {
        val result = LinkedHashMap<String, Tensor>()
        result.putAll(embedding.stateDict("${prefix}embedding."))
        result.putAll(positional.stateDict("${prefix}positional."))
        blocks.forEachIndexed { index, block -> result.putAll(block.stateDict("${prefix}blocks.$index.")) }
        result.putAll(head.stateDict("${prefix}head."))
        return result
    }
}
