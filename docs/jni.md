# JNI and OpenMP kernels

The native library is built from `antorch/src/main/cpp` with C++17 and optional OpenMP.

| Kernel | Kotlin fallback | Native |
| --- | --- | --- |
| matmul | triple loop | blocked triple loop, OpenMP |
| add/sub/mul/div | elementwise | OpenMP for |
| scalar add/mul | map | OpenMP for |
| transpose | copy | OpenMP collapse(2) |
| relu/sigmoid/tanh | map | OpenMP for |
| softmax | rows | OpenMP-friendly row loops |
| sum | loop | OpenMP reduction |
| conv2d | direct loops | OpenMP collapse(2) over N/O |

`NativeOps.available` probes the library at runtime. `Tensor` only takes the native path for
`FLOAT32` and sufficiently large tensors; small tensors stay in Kotlin to avoid JNI overhead.

Build:

```bash
./gradlew :antorch:assembleDebug
# .so at anttorch/build/intermediates/cxx/Debug/*/obj/arm64-v8a/libantorch_native.so
```

Instrumentation test:

```bash
./gradlew :antorch:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class=cn.ac.oac.libs.antorch.NativeOpsTest
```
