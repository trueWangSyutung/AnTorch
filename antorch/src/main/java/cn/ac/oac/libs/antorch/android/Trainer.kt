package cn.ac.oac.libs.antorch.android

import cn.ac.oac.libs.antorch.core.Tensor
import cn.ac.oac.libs.antorch.data.Batch
import cn.ac.oac.libs.antorch.data.DataLoader
import cn.ac.oac.libs.antorch.data.Dataset
import cn.ac.oac.libs.antorch.nn.Loss
import cn.ac.oac.libs.antorch.nn.Module
import cn.ac.oac.libs.antorch.optim.Optimizer
import cn.ac.oac.libs.antorch.serialization.ModelIO
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

data class TrainConfig(
    val epochs: Int = 10,
    val batchSize: Int = 32,
    val shuffle: Boolean = true,
    val checkpointEvery: Int = 0,
    val checkpointFile: File? = null
)

data class EpochStats(
    val epoch: Int,
    val loss: Float,
    val accuracy: Float,
    val elapsedMs: Long
)

/**
 * 轻量训练循环：前台协程中运行，支持按 epoch 回调与权重检查点。
 * 长训练建议由调用方放入前台 Service / WorkManager，并持有 WakeLock。
 */
object Trainer {
    suspend fun train(
        model: Module,
        dataset: Dataset,
        loss: Loss,
        optimizer: Optimizer,
        config: TrainConfig = TrainConfig(),
        onEpoch: (EpochStats) -> Unit = {}
    ): List<EpochStats> = withContext(Dispatchers.Default) {
        model.train()
        val history = ArrayList<EpochStats>()
        repeat(config.epochs) { epoch ->
            val start = System.currentTimeMillis()
            var totalLoss = 0.0
            var correct = 0
            var total = 0
            var batches = 0
            val loader = DataLoader(dataset, config.batchSize, config.shuffle, seed = epoch.toLong() + 42)
            for (batch in loader) {
                optimizer.zeroGrad()
                val output = model(batch.x)
                val lossValue = loss(output, batch.y)
                lossValue.backward()
                optimizer.step()
                totalLoss += lossValue.item()
                batches++
                if (output.ndim == 2) {
                    val classes = output.shape[1]
                    for (i in 0 until output.shape[0]) {
                        var best = 0
                        var bestValue = Float.NEGATIVE_INFINITY
                        for (c in 0 until classes) {
                            val value = output.data[i * classes + c]
                            if (value > bestValue) { bestValue = value; best = c }
                        }
                        if (best == batch.y.data[i].toInt()) correct++
                        total++
                    }
                }
            }
            val stats = EpochStats(
                epoch = epoch + 1,
                loss = if (batches > 0) (totalLoss / batches).toFloat() else Float.NaN,
                accuracy = if (total > 0) correct.toFloat() / total else 0f,
                elapsedMs = System.currentTimeMillis() - start
            )
            history.add(stats)
            if (config.checkpointEvery > 0 && config.checkpointFile != null && (epoch + 1) % config.checkpointEvery == 0) {
                ModelIO.saveCheckpoint(model, epoch + 1, stats.loss, config.checkpointFile)
            }
            onEpoch(stats)
        }
        history
    }
}

object AnTorch {
    const val VERSION = "0.1.0"
    @Volatile private var initialized = false
    fun init() { initialized = true }
    fun isInitialized(): Boolean = initialized
}

fun Tensor.argmaxRows(): IntArray {
    require(ndim == 2)
    return IntArray(shape[0]) { row ->
        var best = 0
        for (c in 1 until shape[1]) if (data[row * shape[1] + c] > data[row * shape[1] + best]) best = c
        best
    }
}
