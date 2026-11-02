package cn.ac.oac.libs.antorch.core

/** Tensor 存储类型；训练参数默认 FLOAT32，其他类型用于标签/索引/量化。 */
enum class DType(val bytes: Int) {
    FLOAT16(2),
    FLOAT32(4),
    INT8(1),
    INT32(4),
    INT64(8);

    val isFloating: Boolean get() = this == FLOAT16 || this == FLOAT32
    val isInteger: Boolean get() = !isFloating
}

internal object Half {
    fun toHalf(value: Float): Short {
        val bits = java.lang.Float.floatToIntBits(value)
        val sign = (bits ushr 16) and 0x8000
        var exponent = ((bits ushr 23) and 0xFF) - 127 + 15
        var mantissa = bits and 0x7FFFFF
        if (exponent <= 0) {
            if (exponent < -10) return sign.toShort()
            mantissa = (mantissa or 0x800000) ushr (1 - exponent)
            return (sign or ((mantissa + 0x1000) ushr 13)).toShort()
        }
        if (exponent >= 31) {
            return (sign or 0x7C00 or if (mantissa != 0) 0x200 else 0).toShort()
        }
        mantissa = (mantissa + 0x1000) ushr 13
        if (mantissa == 0x400) {
            mantissa = 0
            exponent++
            if (exponent >= 31) return (sign or 0x7C00).toShort()
        }
        return (sign or (exponent shl 10) or mantissa).toShort()
    }

    fun toFloat(value: Short): Float {
        val bits = value.toInt() and 0xFFFF
        val sign = (bits and 0x8000) shl 16
        val exponent = (bits ushr 10) and 0x1F
        var mantissa = bits and 0x3FF
        return when {
            exponent == 0 -> {
                if (mantissa == 0) java.lang.Float.intBitsToFloat(sign)
                else {
                    var e = -1
                    do {
                        mantissa = mantissa shl 1
                        e++
                    } while ((mantissa and 0x400) == 0)
                    java.lang.Float.intBitsToFloat(sign or ((e + 127 - 15) shl 23) or ((mantissa and 0x3FF) shl 13))
                }
            }
            exponent == 0x1F -> java.lang.Float.intBitsToFloat(sign or 0x7F800000 or (mantissa shl 13))
            else -> java.lang.Float.intBitsToFloat(sign or ((exponent - 15 + 127) shl 23) or (mantissa shl 13))
        }
    }
}
