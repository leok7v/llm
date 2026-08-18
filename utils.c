#include "utils.h"
#include <math.h>

// NOTE: only used by llm_sandbox.c (kept for building the article). If the
// sandbox is ever removed, this function is dead and should be removed too.
float dot_product(float v[], float u[], int n) {
    float result = 0.0;
    for (int i=0; i<n; i++) {
        result += v[i] * u[i];
    }
    return result;
}

// Note: in real production code, use PyTorch's built-in softmax for numerical stability.
void softmax(float *input, int size) {
    // subtract max for numerical stability: shift-invariant, prevents expf overflow
    float max = input[0];
    for (int i = 1; i < size; i++) if (input[i] > max) max = input[i];
    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        input[i] = expf(input[i] - max);
        sum += input[i];
    }
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

float mean(float *input, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += input[i];
    }

    return sum / size;
}