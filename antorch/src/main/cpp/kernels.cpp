#include <jni.h>
#include <cmath>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

inline void requireSame(JNIEnv* env, jfloatArray a, jfloatArray b) {
    if (env->GetArrayLength(a) != env->GetArrayLength(b)) {
        jclass exception = env->FindClass("java/lang/IllegalArgumentException");
        env->ThrowNew(exception, "array length mismatch");
    }
}

} // namespace

extern "C" JNIEXPORT jfloatArray JNICALL
Java_cn_ac_oac_libs_antorch_core_NativeOps_add(JNIEnv* env, jobject, jfloatArray a, jfloatArray b) {
    requireSame(env, a, b);
    const jsize n = env->GetArrayLength(a);
    jfloat* x = env->GetFloatArrayElements(a, nullptr);
    jfloat* y = env->GetFloatArrayElements(b, nullptr);
    jfloatArray out = env->NewFloatArray(n);
    jfloat* z = env->GetFloatArrayElements(out, nullptr);
    #pragma omp parallel for schedule(static)
    for (jsize i = 0; i < n; ++i) z[i] = x[i] + y[i];
    env->ReleaseFloatArrayElements(a, x, JNI_ABORT);
    env->ReleaseFloatArrayElements(b, y, JNI_ABORT);
    env->ReleaseFloatArrayElements(out, z, 0);
    return out;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_cn_ac_oac_libs_antorch_core_NativeOps_sub(JNIEnv* env, jobject, jfloatArray a, jfloatArray b) {
    requireSame(env, a, b);
    const jsize n = env->GetArrayLength(a);
    jfloat* x = env->GetFloatArrayElements(a, nullptr);
    jfloat* y = env->GetFloatArrayElements(b, nullptr);
    jfloatArray out = env->NewFloatArray(n);
    jfloat* z = env->GetFloatArrayElements(out, nullptr);
    #pragma omp parallel for schedule(static)
    for (jsize i = 0; i < n; ++i) z[i] = x[i] - y[i];
    env->ReleaseFloatArrayElements(a, x, JNI_ABORT);
    env->ReleaseFloatArrayElements(b, y, JNI_ABORT);
    env->ReleaseFloatArrayElements(out, z, 0);
    return out;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_cn_ac_oac_libs_antorch_core_NativeOps_mul(JNIEnv* env, jobject, jfloatArray a, jfloatArray b) {
    requireSame(env, a, b);
    const jsize n = env->GetArrayLength(a);
    jfloat* x = env->GetFloatArrayElements(a, nullptr);
    jfloat* y = env->GetFloatArrayElements(b, nullptr);
    jfloatArray out = env->NewFloatArray(n);
    jfloat* z = env->GetFloatArrayElements(out, nullptr);
    #pragma omp parallel for schedule(static)
    for (jsize i = 0; i < n; ++i) z[i] = x[i] * y[i];
    env->ReleaseFloatArrayElements(a, x, JNI_ABORT);
    env->ReleaseFloatArrayElements(b, y, JNI_ABORT);
    env->ReleaseFloatArrayElements(out, z, 0);
    return out;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_cn_ac_oac_libs_antorch_core_NativeOps_div(JNIEnv* env, jobject, jfloatArray a, jfloatArray b) {
    requireSame(env, a, b);
    const jsize n = env->GetArrayLength(a);
    jfloat* x = env->GetFloatArrayElements(a, nullptr);
    jfloat* y = env->GetFloatArrayElements(b, nullptr);
    jfloatArray out = env->NewFloatArray(n);
    jfloat* z = env->GetFloatArrayElements(out, nullptr);
    #pragma omp parallel for schedule(static)
    for (jsize i = 0; i < n; ++i) z[i] = x[i] / y[i];
    env->ReleaseFloatArrayElements(a, x, JNI_ABORT);
    env->ReleaseFloatArrayElements(b, y, JNI_ABORT);
    env->ReleaseFloatArrayElements(out, z, 0);
    return out;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_cn_ac_oac_libs_antorch_core_NativeOps_addScalar(JNIEnv* env, jobject, jfloatArray a, jfloat value) {
    const jsize n = env->GetArrayLength(a);
    jfloat* x = env->GetFloatArrayElements(a, nullptr);
    jfloatArray out = env->NewFloatArray(n);
    jfloat* z = env->GetFloatArrayElements(out, nullptr);
    #pragma omp parallel for schedule(static)
    for (jsize i = 0; i < n; ++i) z[i] = x[i] + value;
    env->ReleaseFloatArrayElements(a, x, JNI_ABORT);
    env->ReleaseFloatArrayElements(out, z, 0);
    return out;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_cn_ac_oac_libs_antorch_core_NativeOps_mulScalar(JNIEnv* env, jobject, jfloatArray a, jfloat value) {
    const jsize n = env->GetArrayLength(a);
    jfloat* x = env->GetFloatArrayElements(a, nullptr);
    jfloatArray out = env->NewFloatArray(n);
    jfloat* z = env->GetFloatArrayElements(out, nullptr);
    #pragma omp parallel for schedule(static)
    for (jsize i = 0; i < n; ++i) z[i] = x[i] * value;
    env->ReleaseFloatArrayElements(a, x, JNI_ABORT);
    env->ReleaseFloatArrayElements(out, z, 0);
    return out;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_cn_ac_oac_libs_antorch_core_NativeOps_transpose(JNIEnv* env, jobject, jfloatArray a, jint rows, jint cols) {
    jfloat* x = env->GetFloatArrayElements(a, nullptr);
    jfloatArray out = env->NewFloatArray(rows * cols);
    jfloat* z = env->GetFloatArrayElements(out, nullptr);
    #pragma omp parallel for collapse(2) schedule(static)
    for (jint r = 0; r < rows; ++r) for (jint c = 0; c < cols; ++c) z[c * rows + r] = x[r * cols + c];
    env->ReleaseFloatArrayElements(a, x, JNI_ABORT);
    env->ReleaseFloatArrayElements(out, z, 0);
    return out;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_cn_ac_oac_libs_antorch_core_NativeOps_sigmoid(JNIEnv* env, jobject, jfloatArray a) {
    const jsize n = env->GetArrayLength(a);
    jfloat* x = env->GetFloatArrayElements(a, nullptr);
    jfloatArray out = env->NewFloatArray(n);
    jfloat* z = env->GetFloatArrayElements(out, nullptr);
    #pragma omp parallel for schedule(static)
    for (jsize i = 0; i < n; ++i) z[i] = 1.0f / (1.0f + std::exp(-x[i]));
    env->ReleaseFloatArrayElements(a, x, JNI_ABORT);
    env->ReleaseFloatArrayElements(out, z, 0);
    return out;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_cn_ac_oac_libs_antorch_core_NativeOps_tanh(JNIEnv* env, jobject, jfloatArray a) {
    const jsize n = env->GetArrayLength(a);
    jfloat* x = env->GetFloatArrayElements(a, nullptr);
    jfloatArray out = env->NewFloatArray(n);
    jfloat* z = env->GetFloatArrayElements(out, nullptr);
    #pragma omp parallel for schedule(static)
    for (jsize i = 0; i < n; ++i) z[i] = std::tanh(x[i]);
    env->ReleaseFloatArrayElements(a, x, JNI_ABORT);
    env->ReleaseFloatArrayElements(out, z, 0);
    return out;
}

extern "C" JNIEXPORT jfloat JNICALL
Java_cn_ac_oac_libs_antorch_core_NativeOps_sum(JNIEnv* env, jobject, jfloatArray a) {
    const jsize n = env->GetArrayLength(a);
    jfloat* x = env->GetFloatArrayElements(a, nullptr);
    float total = 0.0f;
    #pragma omp parallel for reduction(+:total) schedule(static)
    for (jsize i = 0; i < n; ++i) total += x[i];
    env->ReleaseFloatArrayElements(a, x, JNI_ABORT);
    return total;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_cn_ac_oac_libs_antorch_core_NativeOps_conv2d(
        JNIEnv* env, jobject, jfloatArray input, jfloatArray weight, jfloatArray bias,
        jint batch, jint channels, jint height, jint width, jint outChannels, jint kernel) {
    const jint outH = height - kernel + 1;
    const jint outW = width - kernel + 1;
    jfloat* x = env->GetFloatArrayElements(input, nullptr);
    jfloat* w = env->GetFloatArrayElements(weight, nullptr);
    jfloat* b = env->GetFloatArrayElements(bias, nullptr);
    jfloatArray out = env->NewFloatArray(batch * outChannels * outH * outW);
    jfloat* z = env->GetFloatArrayElements(out, nullptr);
    #pragma omp parallel for collapse(2) schedule(static)
    for (jint n = 0; n < batch; ++n) {
        for (jint o = 0; o < outChannels; ++o) {
            for (jint oh = 0; oh < outH; ++oh) {
                for (jint ow = 0; ow < outW; ++ow) {
                    float sum = b[o];
                    for (jint c = 0; c < channels; ++c) {
                        for (jint kh = 0; kh < kernel; ++kh) {
                            for (jint kw = 0; kw < kernel; ++kw) {
                                sum += x[((n * channels + c) * height + oh + kh) * width + ow + kw] *
                                       w[((o * channels + c) * kernel + kh) * kernel + kw];
                            }
                        }
                    }
                    z[((n * outChannels + o) * outH + oh) * outW + ow] = sum;
                }
            }
        }
    }
    env->ReleaseFloatArrayElements(input, x, JNI_ABORT);
    env->ReleaseFloatArrayElements(weight, w, JNI_ABORT);
    env->ReleaseFloatArrayElements(bias, b, JNI_ABORT);
    env->ReleaseFloatArrayElements(out, z, 0);
    return out;
}
