package cn.ac.oac.libs.antorch.data

import cn.ac.oac.libs.antorch.core.Tensor
import java.io.DataInputStream
import java.io.InputStream
import kotlin.math.sqrt

/** 真实小数据集加载器：Iris / MNIST IDX / SMS Spam。 */
object RealDatasets {

    /** Iris：4 特征、3 类；按列标准化。 */
    fun iris(input: InputStream): TensorDataset {
        val lines = input.bufferedReader().useLines { it.toList() }
            .filter { it.isNotBlank() && !it.startsWith("sepal_length") }
        require(lines.size >= 30) { "Iris 数据为空" }
        val features = Array(lines.size) { FloatArray(4) }
        val labels = FloatArray(lines.size)
        lines.forEachIndexed { index, line ->
            val parts = line.split(',')
            require(parts.size >= 5)
            for (i in 0 until 4) features[index][i] = parts[i].toFloat()
            labels[index] = parts[4].trim().toFloat()
        }
        val means = FloatArray(4)
        val stds = FloatArray(4)
        for (c in 0 until 4) {
            var mean = 0f
            for (r in features.indices) mean += features[r][c]
            mean /= features.size
            var variance = 0f
            for (r in features.indices) variance += (features[r][c] - mean).pow2()
            means[c] = mean
            stds[c] = sqrt(variance / features.size).coerceAtLeast(1e-6f)
        }
        val flat = FloatArray(features.size * 4)
        for (r in features.indices) for (c in 0 until 4) flat[r * 4 + c] = (features[r][c] - means[c]) / stds[c]
        return TensorDataset(Tensor.fromArray(flat, features.size, 4), Tensor.fromInt32Array(IntArray(labels.size) { labels[it].toInt() }, labels.size))
    }

    /** MNIST IDX：图像归一化到 [0,1]，输出 [N,1,28,28] 与 INT32 标签。 */
    fun mnist(images: InputStream, labels: InputStream, limit: Int = Int.MAX_VALUE): TensorDataset {
        val imageInput = DataInputStream(images.buffered())
        val labelInput = DataInputStream(labels.buffered())
        val imageMagic = imageInput.readInt()
        require(imageMagic == 2051) { "不是 MNIST 图像 IDX" }
        val count = imageInput.readInt()
        val rows = imageInput.readInt()
        val cols = imageInput.readInt()
        val labelMagic = labelInput.readInt()
        require(labelMagic == 2049) { "不是 MNIST 标签 IDX" }
        val labelCount = labelInput.readInt()
        require(labelCount == count) { "图像/标签数量不一致" }
        val used = minOf(limit, count)
        val data = FloatArray(used * rows * cols)
        val labelData = IntArray(used)
        val row = ByteArray(rows * cols)
        for (n in 0 until count) {
            imageInput.readFully(row)
            if (n < used) {
                for (i in row.indices) data[n * row.size + i] = (row[i].toInt() and 0xFF) / 255f
            }
        }
        for (n in 0 until used) labelData[n] = labelInput.readUnsignedByte()
        return TensorDataset(
            Tensor.fromArray(data, used, 1, rows, cols),
            Tensor.fromInt32Array(labelData, used)
        )
    }

    /** SMS Spam：label=0 ham / 1 spam；返回词表与 token 序列。 */
    fun sms(input: InputStream, maxLength: Int = 32, vocabSize: Int = 2000): TextDataset {
        val rows = input.bufferedReader().useLines { lines ->
            lines.filter { it.isNotBlank() && !it.startsWith("label\t") }
                .mapNotNull { line ->
                    val index = line.indexOf('\t')
                    if (index <= 0) null else line.substring(0, index).toInt() to line.substring(index + 1)
                }.toList()
        }
        require(rows.isNotEmpty()) { "SMS 数据为空" }
        val tokenized = rows.map { (label, text) -> label to tokenize(text) }
        val frequencies = HashMap<String, Int>()
        for ((_, tokens) in tokenized) for (token in tokens) frequencies[token] = (frequencies[token] ?: 0) + 1
        val vocabulary = frequencies.entries.sortedByDescending { it.value }.take(vocabSize).map { it.key }
        val index = vocabulary.withIndex().associate { it.value to it.index + 1 }
        val tokens = LongArray(rows.size * maxLength)
        val labels = IntArray(rows.size)
        tokenized.forEachIndexed { n, (label, words) ->
            labels[n] = label
            words.take(maxLength).forEachIndexed { position, word ->
                tokens[n * maxLength + position] = (index[word] ?: 0).toLong()
            }
        }
        return TextDataset(
            TensorDataset(Tensor.fromLongArray(tokens, rows.size, maxLength), Tensor.fromInt32Array(labels, labels.size)),
            vocabulary,
            maxLength,
            index
        )
    }

    private fun tokenize(text: String): List<String> =
        Regex("[a-z0-9']+").findAll(text.lowercase()).map { it.value }.toList()

    private fun Float.pow2(): Float = this * this

    data class TextDataset(
        val dataset: TensorDataset,
        val vocabulary: List<String>,
        val maxLength: Int,
        private val vocabularyIndex: Map<String, Int>
    ) {
        fun encode(text: String): Tensor {
            val tokens = LongArray(maxLength)
            Regex("[a-z0-9']+").findAll(text.lowercase()).take(maxLength).forEachIndexed { index, match ->
                tokens[index] = (vocabularyIndex[match.value] ?: 0).toLong()
            }
            return Tensor.fromLongArray(tokens, 1, maxLength)
        }
    }
}
