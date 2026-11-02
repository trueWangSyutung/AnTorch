package cn.ac.oac.libs.antorch.serialization

import cn.ac.oac.libs.antorch.core.DType
import cn.ac.oac.libs.antorch.core.Half
import cn.ac.oac.libs.antorch.core.Tensor
import cn.ac.oac.libs.antorch.nn.Module
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File

/** AnTorch 自定义权重格式 v2：魔数 + 版本 + dtype + 命名参数。 */
object ModelIO {
    private const val MAGIC = 0x414E544F // ANTO
    private const val VERSION = 2

    fun save(model: Module, file: File) {
        file.parentFile?.mkdirs()
        DataOutputStream(file.outputStream().buffered()).use { output ->
            output.writeInt(MAGIC)
            output.writeInt(VERSION)
            val state = model.stateDict()
            output.writeInt(state.size)
            for ((name, tensor) in state) {
                output.writeUTF(name)
                output.writeUTF(tensor.dtype.name)
                output.writeInt(tensor.ndim)
                tensor.shape.forEach { output.writeInt(it) }
                writeValues(output, tensor)
            }
        }
    }

    fun load(model: Module, file: File) {
        require(file.exists()) { "模型文件不存在: ${file.absolutePath}" }
        DataInputStream(file.inputStream().buffered()).use { input ->
            require(input.readInt() == MAGIC) { "不是 AnTorch 模型文件" }
            require(input.readInt() == VERSION) { "不支持的模型版本" }
            val count = input.readInt()
            val state = LinkedHashMap<String, Tensor>()
            repeat(count) {
                val name = input.readUTF()
                val dtype = DType.valueOf(input.readUTF())
                val dims = IntArray(input.readInt()) { input.readInt() }
                val size = dims.fold(1) { a, b -> a * b }
                state[name] = readTensor(input, dtype, dims, size)
            }
            model.loadStateDict(state)
        }
    }

    fun saveCheckpoint(model: Module, epoch: Int, loss: Float, file: File) {
        val weights = File(file.parentFile, "${file.nameWithoutExtension}.antorch")
        save(model, weights)
        file.writeText("epoch=$epoch\nloss=$loss\nweights=${weights.name}\n")
    }

    private fun writeValues(output: DataOutputStream, tensor: Tensor) {
        when (tensor.dtype) {
            DType.FLOAT16 -> tensor.toFloatArray().forEach { output.writeShort(cn.ac.oac.libs.antorch.core.Half.toHalf(it).toInt()) }
            DType.FLOAT32 -> tensor.toFloatArray().forEach { output.writeFloat(it) }
            DType.INT8 -> tensor.toIntArray().forEach { output.writeByte(it) }
            DType.INT32 -> tensor.toIntArray().forEach { output.writeInt(it) }
            DType.INT64 -> tensor.toLongArray().forEach { output.writeLong(it) }
        }
    }

    private fun readTensor(input: DataInputStream, dtype: DType, dims: IntArray, size: Int): Tensor =
        when (dtype) {
            DType.FLOAT16 -> Tensor.fromFloat16Array(ShortArray(size) { input.readShort() }, *dims)
            DType.FLOAT32 -> Tensor.fromArray(FloatArray(size) { input.readFloat() }, *dims)
            DType.INT8 -> Tensor.fromInt8Array(IntArray(size) { input.readByte().toInt() }, *dims)
            DType.INT32 -> Tensor.fromInt32Array(IntArray(size) { input.readInt() }, *dims)
            DType.INT64 -> Tensor.fromLongArray(LongArray(size) { input.readLong() }, *dims)
        }
}
