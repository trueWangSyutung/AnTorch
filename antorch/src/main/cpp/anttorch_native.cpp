#include <jni.h>
#include <cmath>
#include <vector>
#include <algorithm>

extern "C" JNIEXPORT jfloatArray JNICALL
Java_cn_ac_oac_libs_antorch_core_NativeOps_matmul(
        JNIEnv* env, jobject, jfloatArray a, jfloatArray b, jint m, jint k, jint n) {
    if (env->GetArrayLength(a) != m * k || env->GetArrayLength(b) != k * n) return nullptr;
    jfloat* left = env->GetFloatArrayElements(a, nullptr);
    jfloat* right = env->GetFloatArrayElements(b, nullptr);
    jfloatArray output = env->NewFloatArray(m * n);
    jfloat* result = env->GetFloatArrayElements(output, nullptr);
    for (jint i = 0; i < m; ++i) {
        for (jint j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (jint p = 0; p < k; ++p) sum += left[i * k + p] * right[p * n + j];
            result[i * n + j] = sum;
        }
    }
    env->ReleaseFloatArrayElements(a, left, JNI_ABORT);
    env->ReleaseFloatArrayElements(b, right, JNI_ABORT);
    env->ReleaseFloatArrayElements(output, result, 0);
    return output;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_cn_ac_oac_libs_antorch_core_NativeOps_relu(
        JNIEnv* env, jobject, jfloatArray input) {
    const jsize length = env->GetArrayLength(input);
    jfloat* values = env->GetFloatArrayElements(input, nullptr);
    jfloatArray output = env->NewFloatArray(length);
    jfloat* result = env->GetFloatArrayElements(output, nullptr);
    for (jsize i = 0; i < length; ++i) result[i] = values[i] > 0.0f ? values[i] : 0.0f;
    env->ReleaseFloatArrayElements(input, values, JNI_ABORT);
    env->ReleaseFloatArrayElements(output, result, 0);
    return output;
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_cn_ac_oac_libs_antorch_core_NativeOps_softmaxRows(
        JNIEnv* env, jobject, jfloatArray input, jint rows, jint cols) {
    if (env->GetArrayLength(input) != rows * cols) return nullptr;
    jfloat* values = env->GetFloatArrayElements(input, nullptr);
    jfloatArray output = env->NewFloatArray(rows * cols);
    jfloat* result = env->GetFloatArrayElements(output, nullptr);
    for (jint r = 0; r < rows; ++r) {
        float maximum = -INFINITY;
        for (jint c = 0; c < cols; ++c) maximum = std::max(maximum, values[r * cols + c]);
        float sum = 0.0f;
        for (jint c = 0; c < cols; ++c) {
            const float value = std::exp(values[r * cols + c] - maximum);
            result[r * cols + c] = value;
            sum += value;
        }
        for (jint c = 0; c < cols; ++c) result[r * cols + c] /= sum;
    }
    env->ReleaseFloatArrayElements(input, values, JNI_ABORT);
    env->ReleaseFloatArrayElements(output, result, 0);
    return output;
}
