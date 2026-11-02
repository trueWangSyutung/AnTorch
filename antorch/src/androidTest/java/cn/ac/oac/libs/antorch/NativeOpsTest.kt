package cn.ac.oac.libs.antorch

import androidx.test.ext.junit.runners.AndroidJUnit4
import cn.ac.oac.libs.antorch.core.NativeOps
import cn.ac.oac.libs.antorch.core.Tensor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class NativeOpsTest {
    @Test
    fun jniKernelsMatchKotlin() {
        assertTrue(NativeOps.available)
        val a = floatArrayOf(1f, 2f, 3f, 4f)
        val b = floatArrayOf(5f, 6f, 7f, 8f)
        val product = NativeOps.matmul(a, b, 2, 2, 2)
        assertEquals(19f, product[0], 1e-5f)
        assertEquals(50f, product[3], 1e-5f)

        val relu = NativeOps.relu(floatArrayOf(-1f, 0f, 2f))
        assertEquals(0f, relu[0], 0f)
        assertEquals(2f, relu[2], 0f)

        val softmax = NativeOps.softmaxRows(floatArrayOf(1f, 2f, 3f, 1f, 1f, 1f), 2, 3)
        assertEquals(1f, softmax.copyOfRange(0, 3).sum(), 1e-5f)
        assertEquals(1f / 3f, softmax[3], 1e-5f)

        assertTrue(NativeOps.add(floatArrayOf(1f, 2f), floatArrayOf(3f, 4f)).contentEquals(floatArrayOf(4f, 6f)))
        assertTrue(NativeOps.sub(floatArrayOf(3f, 4f), floatArrayOf(1f, 2f)).contentEquals(floatArrayOf(2f, 2f)))
        assertTrue(NativeOps.mul(floatArrayOf(2f, 3f), floatArrayOf(4f, 5f)).contentEquals(floatArrayOf(8f, 15f)))
        assertTrue(NativeOps.div(floatArrayOf(8f, 9f), floatArrayOf(2f, 3f)).contentEquals(floatArrayOf(4f, 3f)))
        assertEquals(3f, NativeOps.sum(floatArrayOf(0f, 1f, 2f)), 1e-5f)
        assertEquals(0.5f, NativeOps.sigmoid(floatArrayOf(0f))[0], 1e-5f)
        assertEquals(0f, NativeOps.tanh(floatArrayOf(0f))[0], 1e-5f)

        val conv = NativeOps.conv2d(
            floatArrayOf(1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f), // 1x1x3x3
            floatArrayOf(1f, 0f, 0f, 1f), // 1x1x2x2
            floatArrayOf(0f), 1, 1, 3, 3, 1, 2
        )
        assertEquals(6f, conv[0], 1e-5f)
        assertEquals(8f, conv[1], 1e-5f)
        assertEquals(12f, conv[2], 1e-5f)
        assertEquals(14f, conv[3], 1e-5f)
    }

    @Test
    fun tensorUsesNativePathAndGradients() {
        val a = Tensor(floatArrayOf(1f, 2f, 3f, 4f), intArrayOf(2, 2), requiresGrad = true)
        val b = Tensor(floatArrayOf(5f, 6f, 7f, 8f), intArrayOf(2, 2), requiresGrad = true)
        val loss = a.matmul(b).sum()
        loss.backward()
        assertEquals(11f, a.grad!!.data[0], 1e-5f)
        assertEquals(4f, b.grad!!.data[0], 1e-5f)
    }
}
