package cn.ac.oac.libs.antorch.annotation

import java.util.concurrent.ConcurrentHashMap

enum class ApiStability { STABLE, EXPERIMENTAL, DEPRECATED, INTERNAL }

@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION, AnnotationTarget.PROPERTY)
@Retention(AnnotationRetention.RUNTIME)
@MustBeDocumented
annotation class AnTorchApi(
    val stability: ApiStability = ApiStability.STABLE,
    val since: String = "0.1.0",
    val note: String = ""
)

@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION, AnnotationTarget.PROPERTY)
@Retention(AnnotationRetention.RUNTIME)
@MustBeDocumented
annotation class ExperimentalAnTorch(val note: String = "")

@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION, AnnotationTarget.PROPERTY)
@Retention(AnnotationRetention.RUNTIME)
@MustBeDocumented
annotation class InternalAnTorch

@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
@MustBeDocumented
annotation class NativeAccelerated(val fallback: String = "Kotlin")

@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
@MustBeDocumented
annotation class AnTorchFunction(
    val name: String = "",
    val description: String = "",
    val differentiable: Boolean = true
)

@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
@MustBeDocumented
annotation class RegisterLayer(val name: String)

@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
@MustBeDocumented
annotation class RegisterOptimizer(val name: String)

@Target(AnnotationTarget.CLASS, AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.RUNTIME)
@MustBeDocumented
annotation class RegisterLoss(val name: String)

data class RegistryEntry(val kind: String, val name: String, val factory: (Map<String, Any>) -> Any)

/** 运行时扩展注册表：不依赖反射扫描，适合 Android/R8。 */
object AnTorchRegistry {
    private val entries = ConcurrentHashMap<String, RegistryEntry>()

    fun registerLayer(name: String, factory: (Map<String, Any>) -> Any) = register("layer", name, factory)
    fun registerOptimizer(name: String, factory: (Map<String, Any>) -> Any) = register("optimizer", name, factory)
    fun registerLoss(name: String, factory: (Map<String, Any>) -> Any) = register("loss", name, factory)

    fun create(kind: String, name: String, config: Map<String, Any> = emptyMap()): Any =
        entries[key(kind, name)]?.factory?.invoke(config)
            ?: throw NoSuchElementException("未注册 $kind: $name")

    fun names(kind: String): List<String> = entries.values.filter { it.kind == kind }.map { it.name }.sorted()
    fun clear() = entries.clear()

    private fun register(kind: String, name: String, factory: (Map<String, Any>) -> Any) {
        entries[key(kind, name)] = RegistryEntry(kind, name, factory)
    }

    private fun key(kind: String, name: String) = "$kind:$name"
}
