package cn.ac.oac.libs.antorch

import cn.ac.oac.libs.antorch.core.DType
import cn.ac.oac.libs.antorch.core.Parameter
import cn.ac.oac.libs.antorch.core.Tensor
import cn.ac.oac.libs.antorch.data.Datasets
import cn.ac.oac.libs.antorch.data.RealDatasets
import cn.ac.oac.libs.antorch.nn.Conv2d
import cn.ac.oac.libs.antorch.nn.CrossEntropyLoss
import cn.ac.oac.libs.antorch.nn.Linear
import cn.ac.oac.libs.antorch.nn.MSELoss
import cn.ac.oac.libs.antorch.nn.ReLU
import cn.ac.oac.libs.antorch.nn.Sequential
import cn.ac.oac.libs.antorch.nn.Tanh
import cn.ac.oac.libs.antorch.nn.ModelZoo
import cn.ac.oac.libs.antorch.nn.BCEWithLogitsLoss
import cn.ac.oac.libs.antorch.nn.FocalLoss
import cn.ac.oac.libs.antorch.nn.HuberLoss
import cn.ac.oac.libs.antorch.nn.LabelSmoothingCrossEntropy
import cn.ac.oac.libs.antorch.optim.Adam
import cn.ac.oac.libs.antorch.optim.Adagrad
import cn.ac.oac.libs.antorch.optim.Lion
import cn.ac.oac.libs.antorch.optim.RMSprop
import cn.ac.oac.libs.antorch.optim.SGD
import cn.ac.oac.libs.antorch.serialization.ModelIO
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import kotlin.math.abs

class AnTorchTest {

    @Test
    fun matmulAndAutograd() {
        val a = Tensor(floatArrayOf(1f, 2f, 3f, 4f), intArrayOf(2, 2), requiresGrad = true)
        val b = Tensor(floatArrayOf(5f, 6f, 7f, 8f), intArrayOf(2, 2), requiresGrad = true)
        val c = a.matmul(b)
        assertEquals(19f, c.data[0], 1e-5f)
        assertEquals(50f, c.data[3], 1e-5f)
        val loss = c.sum()
        loss.backward()
        // d(sum(A@B))/dA = ones @ B^T
        assertEquals(11f, a.grad!!.data[0], 1e-5f)
        assertEquals(15f, a.grad!!.data[1], 1e-5f)
        assertEquals(4f, b.grad!!.data[0], 1e-5f)
    }

    @Test
    fun numericalGradientCheck() {
        val x = Tensor.fromArray(floatArrayOf(0.2f, -0.4f, 0.7f), 1, 3)
        val w = Parameter(floatArrayOf(0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f), intArrayOf(3, 2))
        val b = Parameter(floatArrayOf(0.05f, -0.1f), intArrayOf(2))
        fun lossValue(): Float = ((x.matmul(w) + b).tanh().square().mean()).item()
        val loss = (x.matmul(w) + b).tanh().square().mean()
        loss.backward()
        val analytic = w.grad!!.data.copyOf()
        val epsilon = 1e-3f
        for (i in w.data.indices) {
            val original = w.data[i]
            w.data[i] = original + epsilon
            val plus = lossValue()
            w.data[i] = original - epsilon
            val minus = lossValue()
            w.data[i] = original
            val numerical = (plus - minus) / (2 * epsilon)
            assertTrue("gradient[$i] analytic=${analytic[i]} numerical=$numerical", abs(analytic[i] - numerical) < 2e-2f)
        }
    }

    @Test
    fun xorTrains() {
        val model = Sequential(
            Linear(2, 8), Tanh(), Linear(8, 1), Tanh()
        )
        val optimizer = SGD(model.parameters, learningRate = 0.2f, momentum = 0.9f)
        val lossFn = MSELoss()
        val dataset = Datasets.xor()
        repeat(1200) {
            optimizer.zeroGrad()
            val output = model(dataset.get(0).x)
            // full batch
            val x = dataset.get(0).x
            val y = dataset.get(0).y
            val batchX = Tensor.fromArray(
                floatArrayOf(
                    dataset.get(0).x.data[0], dataset.get(0).x.data[1],
                    dataset.get(1).x.data[0], dataset.get(1).x.data[1],
                    dataset.get(2).x.data[0], dataset.get(2).x.data[1],
                    dataset.get(3).x.data[0], dataset.get(3).x.data[1]
                ), 4, 2
            )
            val batchY = Tensor.fromArray(floatArrayOf(0f, 1f, 1f, 0f), 4, 1)
            val loss = lossFn(model(batchX), batchY)
            loss.backward()
            optimizer.step()
        }
        val output = model(Tensor.fromArray(floatArrayOf(0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f), 4, 2))
        assertTrue(output.data[0] < 0.25f)
        assertTrue(output.data[1] > 0.75f)
        assertTrue(output.data[2] > 0.75f)
        assertTrue(output.data[3] < 0.25f)
    }

    @Test
    fun linearRegressionAndSaveLoad() {
        val x = Tensor.fromArray(FloatArray(20) { it.toFloat() / 20f }, 20, 1)
        val y = Tensor.fromArray(FloatArray(20) { 3f * (it.toFloat() / 20f) + 1f }, 20, 1)
        val model = Sequential(Linear(1, 1))
        val optimizer = SGD(model.parameters, learningRate = 0.1f)
        repeat(800) {
            optimizer.zeroGrad()
            val loss = MSELoss()(model(x), y)
            loss.backward()
            optimizer.step()
        }
        val prediction = model(Tensor.fromArray(floatArrayOf(0.5f), 1, 1)).item()
        assertTrue("prediction=$prediction", abs(prediction - 2.5f) < 0.2f)

        val file = File.createTempFile("antorch", ".model")
        try {
            ModelIO.save(model, file)
            val restored = Sequential(Linear(1, 1))
            ModelIO.load(restored, file)
            assertEquals(prediction, restored(Tensor.fromArray(floatArrayOf(0.5f), 1, 1)).item(), 1e-4f)
        } finally {
            file.delete()
        }
    }

    @Test
    fun convAndCrossEntropy() {
        val model = Sequential(
            Conv2d(1, 4, 3), ReLU(), cn.ac.oac.libs.antorch.nn.Flatten(), Linear(4 * 6 * 6, 2)
        )
        val data = Datasets.bars(samples = 4)
        val optimizer = SGD(model.parameters, learningRate = 0.05f)
        val lossFn = CrossEntropyLoss()
        repeat(120) {
            optimizer.zeroGrad()
            val x = Tensor.fromArray(data.get(0).x.data + data.get(1).x.data + data.get(2).x.data + data.get(3).x.data, 4, 1, 8, 8)
            val y = Tensor.fromArray(floatArrayOf(0f, 1f, 0f, 1f), 4)
            val loss = lossFn(model(x), y)
            loss.backward()
            optimizer.step()
        }
        val all = Tensor.fromArray(data.get(0).x.data + data.get(1).x.data, 2, 1, 8, 8)
        val output = model(all)
        assertEquals(2, output.shape[1])
    }

    @Test
    fun dtypesAndCasts() {
        val i32 = Tensor.fromInt32Array(intArrayOf(1, 2, 3, 4), 2, 2)
        assertEquals(DType.INT32, i32.dtype)
        assertEquals(4, i32.toIntArray()[3])
        val i64 = Tensor.fromLongArray(longArrayOf(10L, 20L, 30L), 3)
        assertEquals(DType.INT64, i64.dtype)
        assertEquals(30L, i64.toLongArray()[2])
        val i8 = Tensor.fromInt8Array(intArrayOf(-1, 2, 3), 3)
        assertEquals(DType.INT8, i8.dtype)
        assertEquals(-1, i8.toIntArray()[0])
        val f16 = Tensor.fromValues(floatArrayOf(1.5f, -2.25f), DType.FLOAT16, 2)
        assertEquals(DType.FLOAT16, f16.dtype)
        assertEquals(-2.25f, f16.toFloatArray()[1], 1e-6f)
        assertEquals(DType.FLOAT32, f16.astype(DType.FLOAT32).dtype)
        assertEquals(2.0f, i32.astype(DType.FLOAT16).toFloatArray()[1], 0f)
        assertEquals(DType.INT64, i32.astype(DType.INT64).dtype)
    }


    @Test
    fun modelZooForwardShapes() {
        val mlp = ModelZoo.mlp(4, 8, 3)
        assertEquals(3, mlp(Tensor.randn(5, 4)).shape[1])
        val cnn = ModelZoo.cnn(1, 8, 2)
        assertEquals(2, cnn(Tensor.randn(3, 1, 8, 8)).shape[1])
        val mnist = ModelZoo.mnistMlp(16)
        assertEquals(10, mnist(Tensor.randn(3, 1, 28, 28)).shape[1])
        val mnistCnn = ModelZoo.mnistCnn(4)
        assertEquals(10, mnistCnn(Tensor.randn(2, 1, 28, 28)).shape[1])
        val transformer = ModelZoo.tinyTransformer(16, 8, 2)
        assertEquals(2, transformer(Tensor.fromLongArray(longArrayOf(1, 2, 3, 4, 5, 6, 7, 8), 1, 8)).shape[1])
    }

    @Test
    fun transformerTrainsOnToySequences() {
        val data = Datasets.sequences(samples = 160, vocab = 16, length = 8)
        val model = ModelZoo.tinyTransformer(16, 8, 2)
        val optimizer = Adam(model.parameters, learningRate = 0.01f)
        val history = kotlinx.coroutines.runBlocking {
            cn.ac.oac.libs.antorch.android.Trainer.train(
                model, data, CrossEntropyLoss(), optimizer,
                cn.ac.oac.libs.antorch.android.TrainConfig(epochs = 60, batchSize = 32)
            )
        }
        var correct = 0
        for (i in 0 until data.size) {
            val sample = data.get(i)
            val output = model(sample.x)
            val best = if (output.data[1] > output.data[0]) 1 else 0
            if (best == sample.y.data[0].toInt()) correct++
        }
        val accuracy = correct.toFloat() / data.size
        assertTrue("accuracy=$accuracy history=${history.last().loss}", accuracy > 0.7f)
    }


    @Test
    fun modernLossesAndOptimizers() {
        val logits = Parameter(floatArrayOf(0.2f, -0.3f, 0.8f, 0.1f), intArrayOf(2, 2))
        val labels = Tensor.fromInt32Array(intArrayOf(0, 1), 2)
        assertTrue(BCEWithLogitsLoss()(Tensor.fromArray(floatArrayOf(0f, 0.2f), 2), Tensor.fromArray(floatArrayOf(0f, 1f), 2)).item() > 0f)
        FocalLoss()(logits, labels).backward()
        assertTrue(logits.grad != null)
        val smooth = LabelSmoothingCrossEntropy(0.1f)(logits, labels)
        assertTrue(smooth.item() > 0f)
        val huber = HuberLoss(0.5f)(Tensor.fromArray(floatArrayOf(1f, 3f), 2), Tensor.fromArray(floatArrayOf(0f, 0f), 2))
        assertTrue(huber.item() > 0f)

        for (optimizer in listOf(
            RMSprop(listOf(logits), learningRate = 0.01f),
            Adagrad(listOf(logits), learningRate = 0.01f),
            Lion(listOf(logits), learningRate = 0.01f)
        )) {
            val before = logits.data[0]
            val loss = logits.square().sum()
            loss.backward()
            optimizer.step()
            assertTrue(logits.data[0] != before)
            optimizer.zeroGrad()
        }
    }


    @Test
    fun realDatasetsLoad() {
        val iris = RealDatasets.iris(javaClass.getResourceAsStream("/iris.csv")!!)
        assertEquals(150, iris.size)
        assertEquals(4, iris.get(0).x.size)
        assertEquals(2, iris.get(149).y.data[0].toInt())

        val mnist = RealDatasets.mnist(
            javaClass.getResourceAsStream("/mnist_train_images.idx")!!,
            javaClass.getResourceAsStream("/mnist_train_labels.idx")!!,
            limit = 200
        )
        assertEquals(200, mnist.size)
        assertEquals(784, mnist.get(0).x.size)
        assertTrue(mnist.get(0).x.data.all { it in 0f..1f })

        val sms = RealDatasets.sms(javaClass.getResourceAsStream("/sms.tsv")!!, maxLength = 16, vocabSize = 500)
        assertEquals(1200, sms.dataset.size)
        assertTrue(sms.vocabulary.isNotEmpty())
        assertEquals(16, sms.dataset.get(0).x.shape[1])
    }

}
