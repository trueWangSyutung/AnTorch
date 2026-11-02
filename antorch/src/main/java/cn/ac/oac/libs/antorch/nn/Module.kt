package cn.ac.oac.libs.antorch.nn

import cn.ac.oac.libs.antorch.annotation.AnTorchApi
import cn.ac.oac.libs.antorch.core.Parameter
import cn.ac.oac.libs.antorch.core.Tensor

@AnTorchApi(since = "0.1.0")
abstract class Module {
    var training: Boolean = true
        protected set

    open val parameters: List<Parameter> get() = emptyList()

    abstract fun forward(input: Tensor): Tensor

    operator fun invoke(input: Tensor): Tensor = forward(input)

    open fun train() { training = true }
    open fun eval() { training = false }

    open fun stateDict(prefix: String = ""): Map<String, Tensor> = emptyMap()

    open fun loadStateDict(state: Map<String, Tensor>, prefix: String = "") {
        val current = stateDict(prefix)
        for ((name, tensor) in current) {
            val source = state[name] ?: throw IllegalArgumentException("缺少参数: $name")
            require(source.shape.contentEquals(tensor.shape)) {
                "参数形状不匹配 $name: ${source.shape.contentToString()} vs ${tensor.shape.contentToString()}"
            }
            source.data.copyInto(tensor.data)
        }
    }
}

class Sequential(vararg val modules: Module) : Module() {
    override val parameters: List<Parameter> get() = modules.flatMap { it.parameters }

    override fun forward(input: Tensor): Tensor = modules.fold(input) { value, module -> module(value) }

    override fun train() {
        super.train()
        modules.forEach { it.train() }
    }

    override fun eval() {
        super.eval()
        modules.forEach { it.eval() }
    }

    override fun stateDict(prefix: String): Map<String, Tensor> {
        val result = LinkedHashMap<String, Tensor>()
        modules.forEachIndexed { index, module ->
            result.putAll(module.stateDict(if (prefix.isEmpty()) "$index." else "$prefix$index."))
        }
        return result
    }

    override fun loadStateDict(state: Map<String, Tensor>, prefix: String) {
        modules.forEachIndexed { index, module ->
            module.loadStateDict(state, if (prefix.isEmpty()) "$index." else "$prefix$index.")
        }
    }
}
