package cn.ac.oac.demo.antorch

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import cn.ac.oac.libs.antorch.core.NativeOps
import cn.ac.oac.libs.antorch.core.Tensor
import cn.ac.oac.libs.antorch.data.DataLoader
import cn.ac.oac.libs.antorch.data.Datasets
import cn.ac.oac.libs.antorch.data.RealDatasets
import cn.ac.oac.libs.antorch.data.SubsetDataset
import cn.ac.oac.libs.antorch.nn.Conv2d
import cn.ac.oac.libs.antorch.nn.CrossEntropyLoss
import cn.ac.oac.libs.antorch.nn.Flatten
import cn.ac.oac.libs.antorch.nn.Linear
import cn.ac.oac.libs.antorch.nn.MSELoss
import cn.ac.oac.libs.antorch.nn.ModelZoo
import cn.ac.oac.libs.antorch.nn.Module
import cn.ac.oac.libs.antorch.nn.ReLU
import cn.ac.oac.libs.antorch.nn.Sequential
import cn.ac.oac.libs.antorch.nn.Tanh
import cn.ac.oac.libs.antorch.optim.Adam
import cn.ac.oac.libs.antorch.optim.SGD
import cn.ac.oac.libs.antorch.serialization.ModelIO
import cn.ac.oac.libs.antorch.android.Trainer
import cn.ac.oac.libs.antorch.android.TrainConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.util.Locale
import kotlin.math.abs
import kotlin.random.Random

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme(colorScheme = lightColorScheme(primary = Color(0xFF2563EB))) {
                Surface(color = Color(0xFFF8FAFC)) {
                    DemoScreen()
                }
            }
        }
    }
}

private data class RunState(
    val running: Boolean = false,
    val result: String = "未运行",
    val losses: List<Float> = emptyList()
)

@Composable
private fun DemoScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var xor by remember { mutableStateOf(RunState()) }
    var spiral by remember { mutableStateOf(RunState()) }
    var persist by remember { mutableStateOf(RunState()) }
    var cnn by remember { mutableStateOf(RunState()) }
    var jni by remember { mutableStateOf(RunState()) }
    var bench by remember { mutableStateOf(RunState()) }
    var transformer by remember { mutableStateOf(RunState()) }
    var mnist by remember { mutableStateOf(RunState()) }
    var zoo by remember { mutableStateOf(RunState()) }
    var mnistModel by remember { mutableStateOf<Module?>(null) }
    var smsModel by remember { mutableStateOf<Module?>(null) }
    var smsEncoder by remember { mutableStateOf<RealDatasets.TextDataset?>(null) }

    fun launchRun(
        setState: (RunState) -> Unit,
        getState: () -> RunState,
        block: suspend () -> Pair<String, List<Float>>
    ) {
        if (getState().running) return
        setState(getState().copy(running = true, result = "运行中…"))
        scope.launch {
            val result = runCatching { withContext(Dispatchers.Default) { block() } }
            setState(
                result.fold(
                    onSuccess = { (text, losses) -> RunState(false, text, losses) },
                    onFailure = { RunState(false, "失败: ${it.message}", emptyList()) }
                )
            )
        }
    }

    LazyColumn(
        modifier = Modifier.fillMaxSize().padding(horizontal = 16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        item {
            Spacer(Modifier.height(18.dp))
            Text("AnTorch", style = MaterialTheme.typography.headlineLarge, fontWeight = FontWeight.Bold, color = Color(0xFF1D4ED8))
            Text("Android 端 PyTorch 风格训练框架 · M1–M9", style = MaterialTheme.typography.bodyMedium, color = Color(0xFF475569))
            Spacer(Modifier.height(10.dp))
            Row(
                modifier = Modifier.horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(6.dp)
            ) {
                listOf("M1 Autograd", "M2 MLP", "M3 Save", "M4 CNN", "M5 JNI", "M6 Bench", "M7 Transformer", "M8 MNIST", "M9 Zoo").forEach { tag ->
                    Text(
                        tag,
                        style = MaterialTheme.typography.labelSmall,
                        maxLines = 1,
                        modifier = Modifier.background(Color(0xFFDBEAFE), RoundedCornerShape(20.dp)).padding(horizontal = 8.dp, vertical = 4.dp)
                    )
                }
            }
            Spacer(Modifier.height(4.dp))
            Text("JNI: ${if (NativeOps.available) "已启用" else "Kotlin 回退"} · 全部在设备本地运行",
                style = MaterialTheme.typography.labelSmall, color = Color(0xFF64748B))
        }

        item {
            DemoCard(
                tag = "M1", title = "自动微分 · XOR",
                description = "MLP 2→8→1，Tanh + MSE + SGD(momentum)，观察 loss 与四个 XOR 预测。",
                state = xor, onRun = {
                    launchRun({ xor = it }, { xor }) {
                        val model = Sequential(Linear(2, 8), Tanh(), Linear(8, 1), Tanh())
                        val optimizer = SGD(model.parameters, learningRate = 0.2f, momentum = 0.9f)
                        val lossFn = MSELoss()
                        val x = Tensor.fromArray(floatArrayOf(0f, 0f, 0f, 1f, 1f, 0f, 1f, 1f), 4, 2)
                        val y = Tensor.fromArray(floatArrayOf(0f, 1f, 1f, 0f), 4, 1)
                        val losses = ArrayList<Float>()
                        repeat(1200) {
                            optimizer.zeroGrad()
                            val loss = lossFn(model(x), y)
                            loss.backward()
                            optimizer.step()
                            if (it % 20 == 0) losses.add(loss.item())
                        }
                        val output = model(x)
                        val predictions = (0 until 4).joinToString("  ") { i ->
                            String.format(Locale.US, "%.2f", output.data[i])
                        }
                        "predictions: $predictions\nfinal loss: ${String.format(Locale.US, "%.5f", losses.last())}" to losses
                    }
                }
            )
        }

        item {
            DemoCard(
                tag = "M2", title = "MLP · Iris 分类",
                description = "真实 Iris 150 条（4 特征、3 类），80/20 划分；MLP + ReLU + CrossEntropy + Adam。",
                state = spiral, onRun = {
                    launchRun({ spiral = it }, { spiral }) {
                        val all = RealDatasets.iris(context.assets.open("iris.csv"))
                        val (train, test) = trainTestSplit(all, 0.8, seed = 7)
                        val model = Sequential(Linear(4, 16), ReLU(), Linear(16, 3))
                        val optimizer = Adam(model.parameters, learningRate = 0.01f)
                        val history = runBlockingTrainer(model, train, optimizer, epochs = 200)
                        val trainAccuracy = evaluate(model, train)
                        val testAccuracy = evaluate(model, test)
                        "Iris · train=${train.size} test=${test.size}\n" +
                            "train: ${String.format(Locale.US, "%.1f", trainAccuracy * 100)}%\n" +
                            "test: ${String.format(Locale.US, "%.1f", testAccuracy * 100)}% (held-out)\n" +
                            "final loss: ${String.format(Locale.US, "%.4f", history.lastOrNull()?.loss ?: 0f)}" to history.map { it.loss }
                    }
                }
            )
        }

        item {
            DemoCard(
                tag = "M3", title = "保存 / 加载 · 线性回归",
                description = "训练 y=3x+1，保存 AnTorch 权重，重新实例化后加载并比较预测。",
                state = persist, onRun = {
                    launchRun({ persist = it }, { persist }) {
                        val x = Tensor.fromArray(FloatArray(20) { it / 20f }, 20, 1)
                        val y = Tensor.fromArray(FloatArray(20) { 3f * (it / 20f) + 1f }, 20, 1)
                        val model = Sequential(Linear(1, 1))
                        val optimizer = SGD(model.parameters, learningRate = 0.1f)
                        repeat(600) {
                            optimizer.zeroGrad()
                            val loss = MSELoss()(model(x), y)
                            loss.backward()
                            optimizer.step()
                        }
                        val before = model(Tensor.fromArray(floatArrayOf(0.5f), 1, 1)).item()
                        val file = File(context.cacheDir, "linear.antorch")
                        ModelIO.save(model, file)
                        val restored = Sequential(Linear(1, 1))
                        ModelIO.load(restored, file)
                        val after = restored(Tensor.fromArray(floatArrayOf(0.5f), 1, 1)).item()
                        "before=${String.format(Locale.US, "%.4f", before)}  after=${String.format(Locale.US, "%.4f", after)}\n" +
                            "file=${file.name} size=${file.length()} bytes" to emptyList()
                    }
                }
            )
        }

        item {
            DemoCard(
                tag = "M4", title = "CNN · MNIST 手写数字",
                description = "真实 MNIST 800 张训练 / 200 张测试；Conv2d + ReLU + Flatten + Linear 训练 10 类。",
                state = cnn, onRun = {
                    launchRun({ cnn = it }, { cnn }) {
                        val train = RealDatasets.mnist(
                            context.assets.open("mnist_train_images.idx"),
                            context.assets.open("mnist_train_labels.idx"),
                            limit = 800
                        )
                        val test = RealDatasets.mnist(
                            context.assets.open("mnist_test_images.idx"),
                            context.assets.open("mnist_test_labels.idx"),
                            limit = 200
                        )
                        val model = ModelZoo.mnistCnn(4)
                        val optimizer = Adam(model.parameters, learningRate = 0.005f)
                        val history = runBlockingTrainer(model, train, optimizer, epochs = 4, batchSize = 32)
                        withContext(Dispatchers.Main) { mnistModel = model }
                        val trainAccuracy = evaluate(model, train)
                        val testAccuracy = evaluate(model, test)
                        "MNIST · train=${train.size} test=${test.size}\n" +
                            "train: ${String.format(Locale.US, "%.1f", trainAccuracy * 100)}%\n" +
                            "test: ${String.format(Locale.US, "%.1f", testAccuracy * 100)}% (held-out)\n" +
                            "final loss: ${String.format(Locale.US, "%.4f", history.last().loss)}" to history.map { it.loss }
                    }
                }
            )
        }

        item {
            DemoCard(
                tag = "M7", title = "Transformer · SMS 文本分类",
                description = "真实 SMS Spam 1200 条，80/20 划分；Embedding + 位置编码 + 多头自注意力 + LayerNorm + FFN。",
                state = transformer, onRun = {
                    launchRun({ transformer = it }, { transformer }) {
                        val sms = RealDatasets.sms(context.assets.open("sms.tsv"), maxLength = 24, vocabSize = 1200)
                        val (train, test) = trainTestSplit(sms.dataset, 0.8, seed = 11)
                        val model = ModelZoo.transformer(
                            vocabSize = sms.vocabulary.size + 1,
                            maxSeq = 24,
                            dim = 16,
                            heads = 2,
                            classes = 2,
                            layers = 1,
                            ffnDim = 32
                        )
                        val optimizer = Adam(model.parameters, learningRate = 0.005f)
                        val history = runBlockingTrainer(model, train, optimizer, epochs = 12, batchSize = 32)
                        withContext(Dispatchers.Main) { smsModel = model; smsEncoder = sms }
                        val trainAccuracy = evaluate(model, train)
                        val testAccuracy = evaluate(model, test)
                        "SMS Spam · train=${train.size} test=${test.size}, vocab=${sms.vocabulary.size}\n" +
                            "train: ${String.format(Locale.US, "%.1f", trainAccuracy * 100)}%\n" +
                            "test: ${String.format(Locale.US, "%.1f", testAccuracy * 100)}% (held-out)\n" +
                            "final loss: ${String.format(Locale.US, "%.4f", history.last().loss)}" to history.map { it.loss }
                    }
                }
            )
        }

        item {
            DemoCard(
                tag = "M8", title = "MNIST · held-out 测试",
                description = "真实 MNIST 训练后，在独立测试集上评估，输出 train/test 准确率与 loss。",
                state = mnist, onRun = {
                    launchRun({ mnist = it }, { mnist }) {
                        val train = RealDatasets.mnist(
                            context.assets.open("mnist_train_images.idx"),
                            context.assets.open("mnist_train_labels.idx"),
                            limit = 800
                        )
                        val test = RealDatasets.mnist(
                            context.assets.open("mnist_test_images.idx"),
                            context.assets.open("mnist_test_labels.idx"),
                            limit = 200
                        )
                        val model = ModelZoo.mnistCnn(4)
                        val optimizer = Adam(model.parameters, learningRate = 0.005f)
                        val history = runBlockingTrainer(model, train, optimizer, epochs = 3, batchSize = 32)
                        withContext(Dispatchers.Main) { mnistModel = model }
                        val trainAccuracy = evaluate(model, train)
                        val testAccuracy = evaluate(model, test)
                        "train: ${String.format(Locale.US, "%.1f", trainAccuracy * 100)}%\n" +
                            "test: ${String.format(Locale.US, "%.1f", testAccuracy * 100)}% (held-out)\n" +
                            "params: ${paramCount(model)}\n" +
                            "final loss: ${String.format(Locale.US, "%.4f", history.last().loss)}" to history.map { it.loss }
                    }
                }
            )
        }

        item {
            DemoCard(
                tag = "M9", title = "模型库 · Model Zoo",
                description = "MLP / CNN / MNIST-MLP / MNIST-CNN / Transformer 的参数量与结构。",
                state = zoo, onRun = {
                    launchRun({ zoo = it }, { zoo }) {
                        val models = linkedMapOf(
                            "MLP" to ModelZoo.mlp(4, 8, 3),
                            "CNN" to ModelZoo.cnn(1, 8, 2),
                            "MNIST-MLP" to ModelZoo.mnistMlp(32),
                            "MNIST-CNN" to ModelZoo.mnistCnn(4),
                            "Transformer" to ModelZoo.tinyTransformer(16, 8, 2)
                        )
                        models.entries.joinToString("\n") { "${it.key}: ${paramCount(it.value)} params" } to emptyList()
                    }
                }
            )
        }

        item {
            DemoCard(
                tag = "M5", title = "JNI · 原生内核",
                description = "对比 JNI matmul 与 Kotlin 回退的 256×256 矩阵乘法耗时，并验证 softmax。",
                state = jni, onRun = {
                    launchRun({ jni = it }, { jni }) {
                        val n = 256
                        val random = Random(7)
                        val a = FloatArray(n * n) { random.nextFloat() }
                        val b = FloatArray(n * n) { random.nextFloat() }
                        repeat(2) { NativeOps.matmul(a, b, n, n, n) }
                        val nativeStart = System.nanoTime()
                        val nativeResult = NativeOps.matmul(a, b, n, n, n)
                        val nativeMs = (System.nanoTime() - nativeStart) / 1_000_000.0
                        val kotlinStart = System.nanoTime()
                        val kotlinResult = FloatArray(n * n)
                        for (i in 0 until n) for (j in 0 until n) {
                            var sum = 0f
                            for (p in 0 until n) sum += a[i * n + p] * b[p * n + j]
                            kotlinResult[i * n + j] = sum
                        }
                        val kotlinMs = (System.nanoTime() - kotlinStart) / 1_000_000.0
                        var maxDiff = 0f
                        for (i in nativeResult.indices) maxDiff = maxOf(maxDiff, abs(nativeResult[i] - kotlinResult[i]))
                        "JNI available: ${NativeOps.available}\n" +
                            "JNI: ${String.format(Locale.US, "%.2f", nativeMs)} ms\n" +
                            "Kotlin: ${String.format(Locale.US, "%.2f", kotlinMs)} ms\n" +
                            "max diff: ${String.format(Locale.US, "%.2e", maxDiff)}" to emptyList()
                    }
                }
            )
        }

        item {
            DemoCard(
                tag = "M6", title = "训练基准 · 内存/耗时",
                description = "跑一轮 200 epochs MLP，记录单轮耗时、总耗时和 JVM 堆占用。",
                state = bench, onRun = {
                    launchRun({ bench = it }, { bench }) {
                        val data = Datasets.spiral(120)
                        val model = Sequential(Linear(2, 16), ReLU(), Linear(16, 2))
                        val optimizer = Adam(model.parameters, learningRate = 0.01f)
                        val start = System.currentTimeMillis()
                        val history = runBlockingTrainer(model, data, optimizer, epochs = 200, batchSize = 32)
                        val total = System.currentTimeMillis() - start
                        val runtime = Runtime.getRuntime()
                        "total: $total ms\navg epoch: ${total / 200} ms\n" +
                            "heap: ${(runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024} MB / ${runtime.maxMemory() / 1024 / 1024} MB\n" +
                            "final loss: ${String.format(Locale.US, "%.4f", history.last().loss)}" to history.map { it.loss }
                    }
                }
            )
        }

        item { MnistInferenceCard(mnistModel) }
        item { SmsInferenceCard(smsModel, smsEncoder) }

        item { Spacer(Modifier.height(30.dp)) }
    }
}

private suspend fun runBlockingTrainer(
    model: cn.ac.oac.libs.antorch.nn.Module,
    data: cn.ac.oac.libs.antorch.data.Dataset,
    optimizer: Adam,
    epochs: Int,
    batchSize: Int = 64
): List<cn.ac.oac.libs.antorch.android.EpochStats> =
    Trainer.train(
        model = model,
        dataset = data,
        loss = CrossEntropyLoss(),
        optimizer = optimizer,
        config = TrainConfig(epochs = epochs, batchSize = batchSize, shuffle = true)
    )

private fun trainTestSplit(
    dataset: cn.ac.oac.libs.antorch.data.Dataset,
    trainRatio: Double,
    seed: Long
): Pair<cn.ac.oac.libs.antorch.data.Dataset, cn.ac.oac.libs.antorch.data.Dataset> {
    val random = kotlin.random.Random(seed)
    val indices = IntArray(dataset.size) { it }
    for (i in indices.indices.reversed()) {
        val j = random.nextInt(i + 1)
        val t = indices[i]; indices[i] = indices[j]; indices[j] = t
    }
    val cut = (dataset.size * trainRatio).toInt().coerceIn(1, dataset.size - 1)
    return cn.ac.oac.libs.antorch.data.SubsetDataset(dataset, indices.copyOfRange(0, cut)) to
        cn.ac.oac.libs.antorch.data.SubsetDataset(dataset, indices.copyOfRange(cut, indices.size))
}

private fun paramCount(model: cn.ac.oac.libs.antorch.nn.Module): Int = model.parameters.sumOf { it.size }

private fun evaluate(model: cn.ac.oac.libs.antorch.nn.Module, data: cn.ac.oac.libs.antorch.data.Dataset): Float {
    var correct = 0
    var total = 0
    model.eval()
    for (i in 0 until data.size) {
        val sample = data.get(i)
        val output = model(sample.x)
        var best = 0
        for (c in 1 until output.shape[1]) if (output.data[c] > output.data[best]) best = c
        if (best == sample.y.data[0].toInt()) correct++
        total++
    }
    model.train()
    return if (total == 0) 0f else correct.toFloat() / total
}

@Composable
private fun MnistInferenceCard(model: Module?) {
    var pixels by remember { mutableStateOf(FloatArray(28 * 28)) }
    var result by remember { mutableStateOf(if (model == null) "先运行 M4 或 M8 训练 MNIST 模型" else "在黑色画布上写一个数字") }

    fun paint(position: Offset, canvasSize: IntSize) {
        val cx = (position.x / canvasSize.width * 28).toInt().coerceIn(0, 27)
        val cy = (position.y / canvasSize.height * 28).toInt().coerceIn(0, 27)
        val next = pixels.copyOf()
        for (dy in -1..1) for (dx in -1..1) {
            val x = cx + dx
            val y = cy + dy
            if (x in 0 until 28 && y in 0 until 28) next[y * 28 + x] = 1f
        }
        pixels = next
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text("推理测试 · MNIST 手写数字", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            Text("训练完成后，在画布上写 0–9，点击预测调用刚训练的 CNN。",
                style = MaterialTheme.typography.bodySmall, color = Color(0xFF64748B))
            Spacer(Modifier.height(10.dp))
            Canvas(
                modifier = Modifier
                    .size(280.dp)
                    .background(Color.Black)
                    .pointerInput(Unit) {
                        detectDragGestures(
                            onDragStart = { offset -> paint(offset, size) },
                            onDrag = { change, _ -> change.consume(); paint(change.position, size) }
                        )
                    }
                    .pointerInput(Unit) {
                        detectTapGestures { offset -> paint(offset, size) }
                    }
            ) {
                val cell = size.width / 28f
                for (r in 0 until 28) for (c in 0 until 28) {
                    if (pixels[r * 28 + c] > 0f) {
                        drawRect(Color.White, topLeft = Offset(c * cell, r * cell), size = Size(cell, cell))
                    }
                }
            }
            Spacer(Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(enabled = model != null, onClick = {
                    if (model != null) {
                        val logits = model(Tensor.fromArray(pixels, 1, 1, 28, 28))
                        var max = 0
                        for (i in 1 until logits.size) if (logits.data[i] > logits.data[max]) max = i
                        val exps = FloatArray(logits.size) { kotlin.math.exp((logits.data[it] - logits.data[max]).toDouble()).toFloat() }
                        val sum = exps.sum()
                        result = "预测：${max}   置信度 ${String.format(Locale.US, "%.1f", exps[max] / sum * 100)}%"
                    }
                }) { Text("预测") }
                OutlinedButton(onClick = {
                    pixels = FloatArray(28 * 28)
                    result = "已清空，重新写一个数字"
                }) { Text("清空") }
            }
            Text(result, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(top = 8.dp))
        }
    }
}

@Composable
private fun SmsInferenceCard(model: Module?, encoder: RealDatasets.TextDataset?) {
    var text by remember { mutableStateOf("Congratulations! You have won a free prize, call now") }
    var result by remember { mutableStateOf(if (model == null || encoder == null) "先运行 M7 训练 SMS 模型" else "输入短信内容后点击预测") }
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White)
    ) {
        Column(Modifier.padding(16.dp)) {
            Text("推理测试 · SMS 垃圾短信", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
            OutlinedTextField(
                value = text,
                onValueChange = { text = it },
                label = { Text("短信文本") },
                minLines = 3,
                modifier = Modifier.fillMaxWidth().padding(top = 8.dp)
            )
            Button(
                enabled = model != null && encoder != null,
                modifier = Modifier.padding(top = 8.dp),
                onClick = {
                    if (model != null && encoder != null) {
                        val input = encoder.encode(text)
                        val output = model(input)
                        val spam = output.data[1] > output.data[0]
                        result = if (spam) "预测：spam（垃圾短信）" else "预测：ham（正常短信）"
                    }
                }
            ) { Text("预测") }
            Text(result, fontFamily = FontFamily.Monospace, style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.padding(top = 8.dp))
        }
    }
}

@Composable
private fun DemoCard(
    tag: String,
    title: String,
    description: String,
    state: RunState,
    onRun: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White)
    ) {
        Column(Modifier.padding(16.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    tag,
                    style = MaterialTheme.typography.labelMedium,
                    color = Color.White,
                    modifier = Modifier.background(Color(0xFF2563EB), RoundedCornerShape(6.dp)).padding(horizontal = 7.dp, vertical = 3.dp)
                )
                Spacer(Modifier.size(8.dp))
                Text(title, style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                Spacer(Modifier.weight(1f))
                if (state.running) CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
            }
            Spacer(Modifier.height(6.dp))
            Text(description, style = MaterialTheme.typography.bodySmall, color = Color(0xFF64748B))
            if (state.losses.size > 1) {
                LossChart(state.losses)
            }
            Text(
                state.result,
                style = MaterialTheme.typography.bodySmall,
                fontFamily = FontFamily.Monospace,
                modifier = Modifier.padding(top = 8.dp)
            )
            Spacer(Modifier.height(8.dp))
            Button(onClick = onRun, enabled = !state.running) {
                Text(if (state.running) "运行中" else "运行")
            }
        }
    }
}

@Composable
private fun LossChart(losses: List<Float>) {
    val min = losses.minOrNull() ?: 0f
    val max = losses.maxOrNull() ?: 1f
    Canvas(Modifier.fillMaxWidth().height(72.dp).padding(top = 8.dp)) {
        val path = Path()
        losses.forEachIndexed { index, value ->
            val x = index.toFloat() / (losses.size - 1) * size.width
            val ratio = if (max == min) 0.5f else (value - min) / (max - min)
            val y = size.height - ratio * size.height
            if (index == 0) path.moveTo(x, y) else path.lineTo(x, y)
        }
        drawPath(path, Color(0xFF2563EB), style = Stroke(width = 2.dp.toPx()))
    }
}
