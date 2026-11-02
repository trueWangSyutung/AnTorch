package cn.ac.oac.libs.antorch.core

import cn.ac.oac.libs.antorch.annotation.NativeAccelerated

/** JNI 内核；不可用时由 Tensor 的 Kotlin 实现回退。 */
@NativeAccelerated(fallback = "Kotlin")
object NativeOps {
    init {
        try {
            System.loadLibrary("antorch_native")
        } catch (_: Throwable) {
            // JVM 单测/桌面环境没有 Android .so，自动回退。
        }
    }

    external fun matmul(a: FloatArray, b: FloatArray, m: Int, k: Int, n: Int): FloatArray
    external fun add(a: FloatArray, b: FloatArray): FloatArray
    external fun sub(a: FloatArray, b: FloatArray): FloatArray
    external fun mul(a: FloatArray, b: FloatArray): FloatArray
    external fun div(a: FloatArray, b: FloatArray): FloatArray
    external fun addScalar(input: FloatArray, value: Float): FloatArray
    external fun mulScalar(input: FloatArray, value: Float): FloatArray
    external fun transpose(input: FloatArray, rows: Int, cols: Int): FloatArray
    external fun relu(input: FloatArray): FloatArray
    external fun sigmoid(input: FloatArray): FloatArray
    external fun tanh(input: FloatArray): FloatArray
    external fun softmaxRows(input: FloatArray, rows: Int, cols: Int): FloatArray
    external fun sum(input: FloatArray): Float
    external fun conv2d(
        input: FloatArray,
        weight: FloatArray,
        bias: FloatArray,
        batch: Int,
        channels: Int,
        height: Int,
        width: Int,
        outChannels: Int,
        kernel: Int
    ): FloatArray

    val available: Boolean by lazy {
        try {
            matmul(floatArrayOf(1f, 2f, 3f, 4f), floatArrayOf(1f, 0f, 0f, 1f), 2, 2, 2).contentEquals(floatArrayOf(1f, 2f, 3f, 4f))
        } catch (_: Throwable) {
            false
        }
    }
}
