#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

Tensor tensor_create(const int *shape, const int ndim) {
    // The function caller's might be stack allocated meaning t.shape = shape can disappear, so I malloc one. Same with ndim (I suppose haha)
    int *shape_copy = (int *)malloc(ndim * sizeof(int));
    int ndim_copy = ndim;

    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
        shape_copy[i] = shape[i];
    }

    float *data = (float *)calloc(size, sizeof(float));

    // precomputing strides -> flat_index = i*(B*C) + j*(C) + k
    int *strides = (int *)malloc(ndim * sizeof(int));
    strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    Tensor t = { data, shape_copy, strides, ndim_copy, size };
    return t;
}

Tensor tensor_create_zeros(const int *shape, const int ndim) {
    Tensor t = tensor_create(shape, ndim);
    
    for (int i = 0; i < t.size; i++) {
        t.data[i] = 0.0f;
    }

    return t;
}

Tensor tensor_create_ones(const int *shape, const int ndim) {
    Tensor t = tensor_create(shape, ndim);
    
    for (int i = 0; i < t.size; i++) {
        t.data[i] = 1.0f;
    }

    return t;
}

Tensor tensor_create_random(const int *shape, const int ndim, const float scale) {
    Tensor t = tensor_create(shape, ndim);

    // Raplace data elements with random values
    for (int i = 0; i<t.size; i++) {
        t.data[i] = ((float)rand() / (float)RAND_MAX) * 2.0f * scale - scale;
    }

    return t;    
}

Tensor tensor_create_dropout(const int *shape, const int ndim, const float rate) {
    Tensor t = tensor_create(shape, ndim);
    const float keep_scale = 1.0f / (1.0f - rate);

    for (int i = 0; i < t.size; i++) {
        float r = (double) rand() / RAND_MAX;
        t.data[i] = (r > rate) ? keep_scale : 0.0f;
    }

    return t;
}

void tensor_dropout_inplace(Tensor *t, float rate) {
    Tensor dropout = tensor_create_dropout(t->shape, t->ndim, rate);
    for (int i = 0; i < t->size; i++) {
        t->data[i] *= dropout.data[i];
    }
    tensor_free(&dropout);
}

float tensor_get(Tensor *t, const int *coords, const int num_coords) {
    int flat_index = 0;
    for (int i = 0; i < num_coords; i++) {
        flat_index += coords[i] * t->strides[i];
    }

    return t->data[flat_index];
}

void tensor_set(Tensor *t, const int *coords, const float value) {
    int flat_index = 0;
    for (int i = 0; i < t->ndim; i++) {
        flat_index += coords[i] * t->strides[i];
    }

    t->data[flat_index] = value;
}

Tensor tensor_add(Tensor *a, Tensor*b) {
    if (a->ndim != b->ndim) {
        fprintf(stderr, "tensor_add: ndim mismatch\n");
        exit(1);
    }

    for (int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "tensor_add: shape mismatch\n");
            exit(1);
        }
    }

    Tensor result = tensor_create(a->shape, a->ndim);
    
    for (int i = 0; i < a->size; i++) {
        result.data[i] = a->data[i] + b->data[i];
    }

    return result;
}

// Lowkey, this func was made with AI, haha.
// Matrix multiplication over the last two dims; everything before is treated
// as a batch axis. Supports 2D @ 2D, 3D @ 2D (b broadcast over batch),
// 2D @ 3D (a broadcast), and 3D @ 3D (matched batch).
//   2D @ 2D : [M, K] @ [K, N]       -> [M, N]
//   3D @ 2D : [B, M, K] @ [K, N]    -> [B, M, N]
//   2D @ 3D : [M, K] @ [B, K, N]    -> [B, M, N]
//   3D @ 3D : [B, M, K] @ [B, K, N] -> [B, M, N]
Tensor tensor_matmul(Tensor *a, Tensor *b) {
    int M = a->shape[a->ndim - 2];
    int K = a->shape[a->ndim - 1];
    int Kb = b->shape[b->ndim - 2];
    int N = b->shape[b->ndim - 1];

    if (K != Kb) {
        fprintf(stderr, "tensor_matmul: shape mismatch — A's last dim (%d) != B's second-to-last dim (%d)\n", K, Kb);
        exit(1);
    }
    if (a->ndim != 2 && a->ndim != 3) {
        fprintf(stderr, "tensor_matmul: A must be 2D or 3D, got %dD\n", a->ndim);
        exit(1);
    }
    if (b->ndim != 2 && b->ndim != 3) {
        fprintf(stderr, "tensor_matmul: B must be 2D or 3D, got %dD\n", b->ndim);
        exit(1);
    }
    if (a->ndim == 3 && b->ndim == 3 && a->shape[0] != b->shape[0]) {
        fprintf(stderr, "tensor_matmul: batch mismatch — A batch (%d) != B batch (%d)\n", a->shape[0], b->shape[0]);
        exit(1);
    }

    int batch = 1;
    if (a->ndim == 3) batch = a->shape[0];
    else if (b->ndim == 3) batch = b->shape[0];

    int out_ndim = (a->ndim == 3 || b->ndim == 3) ? 3 : 2;
    int out_shape[3];
    if (out_ndim == 3) { out_shape[0] = batch; out_shape[1] = M; out_shape[2] = N; }
    else                { out_shape[0] = M;     out_shape[1] = N; }

    Tensor c = tensor_create(out_shape, out_ndim);

    {
        const float *A = a->data;
        const float *Bd = b->data;
        float *C = c.data;
        const int a3 = (a->ndim == 3);
        const int b3 = (b->ndim == 3);
        const int aStride = a3 ? M * K : 0; // per-batch stride into A (0 = broadcast)
        const int bStride = b3 ? K * N : 0; // per-batch stride into B
        for (int bat = 0; bat < batch; bat++) {
            const float *Ab = A + (long)bat * aStride;
            const float *Bb = Bd + (long)bat * bStride;
            float *Cb = C + (long)bat * M * N;
            for (int i = 0; i < M; i++) {
                float *Crow = Cb + i * N;
                for (int j = 0; j < N; j++) Crow[j] = 0.0f;
                const float *Arow = Ab + i * K;
                for (int k = 0; k < K; k++) {
                    const float aik = Arow[k];
                    const float *Brow = Bb + k * N;
                    for (int j = 0; j < N; j++) Crow[j] += aik * Brow[j];
                }
            }
        }
        return c;
    }

    for (int bat = 0; bat < batch; bat++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    float a_val, b_val;
                    if (a->ndim == 3) {
                        const int a_coords[3] = {bat, i, k};
                        a_val = tensor_get(a, a_coords, 3);
                    } else {
                        const int a_coords[2] = {i, k};
                        a_val = tensor_get(a, a_coords, 2);
                    }
                    if (b->ndim == 3) {
                        const int b_coords[3] = {bat, k, j};
                        b_val = tensor_get(b, b_coords, 3);
                    } else {
                        const int b_coords[2] = {k, j};
                        b_val = tensor_get(b, b_coords, 2);
                    }
                    sum += a_val * b_val;
                }
                if (out_ndim == 3) {
                    const int c_coords[3] = {bat, i, j};
                    tensor_set(&c, c_coords, sum);
                } else {
                    const int c_coords[2] = {i, j};
                    tensor_set(&c, c_coords, sum);
                }
            }
        }
    }
    return c;
}

// Lowkey, this func was also made with AI, hehe.
// Transpose the last two dimensions; copies the data into a new tensor.
// (For attention use: K.transpose(-1, -2) — PyTorch idiom.)
//   2D: [M, N]    -> [N, M]
//   3D: [B, M, N] -> [B, N, M]
// 4D+ would need a permutation argument; not handled here yet.
Tensor tensor_transpose(Tensor *t) {
    if (t->ndim != 2 && t->ndim != 3) {
        fprintf(stderr, "tensor_transpose: only 2D or 3D supported, got %dD\n", t->ndim);
        exit(1);
    }

    int M = t->shape[t->ndim - 2];
    int N = t->shape[t->ndim - 1];
    int batch = (t->ndim == 3) ? t->shape[0] : 1;

    int out_shape[3];
    if (t->ndim == 3) { out_shape[0] = batch; out_shape[1] = N; out_shape[2] = M; }
    else              { out_shape[0] = N;     out_shape[1] = M; }

    Tensor result = tensor_create(out_shape, t->ndim);

    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                if (t->ndim == 3) {
                    const int src[3] = {b, i, j};
                    const int dst[3] = {b, j, i};
                    tensor_set(&result, dst, tensor_get(t, src, 3));
                } else {
                    const int src[2] = {i, j};
                    const int dst[2] = {j, i};
                    tensor_set(&result, dst, tensor_get(t, src, 2));
                }
            }
        }
    }
    return result;
}

// Jesus, matmul, transpose, and tensor_reshape needed days studying the perfect shape, but honestly, they were not
// the point of this practice, so, yes, another one made with AI.
// Reshape in place: rewrites shape and strides for a new layout of the same total
// size. Data buffer is NOT touched — the underlying memory order is preserved, the
// new shape/strides just reinterpret it. Cheap (metadata-only).
//   precondition: prod(new_shape) == t->size  (otherwise fatal)
void tensor_reshape(Tensor *t, const int *new_shape, int new_ndim) {
    int new_size = 1;
    for (int i = 0; i < new_ndim; i++) new_size *= new_shape[i];
    if (new_size != t->size) {
        fprintf(stderr, "tensor_reshape: size mismatch (new=%d, current=%d)\n", new_size, t->size);
        exit(1);
    }

    int *shape_copy = (int *)malloc(new_ndim * sizeof(int));
    int *new_strides = (int *)malloc(new_ndim * sizeof(int));
    for (int i = 0; i < new_ndim; i++) shape_copy[i] = new_shape[i];
    new_strides[new_ndim - 1] = 1;
    for (int i = new_ndim - 2; i >= 0; i--) {
        new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
    }

    free(t->shape);
    free(t->strides);
    t->shape = shape_copy;
    t->strides = new_strides;
    t->ndim = new_ndim;
}

Tensor tensor_permute(Tensor *t, const int *perm, int n) {
    if (n != t->ndim) {
        fprintf(stderr, "tensor_permute: perm length %d != tensor ndim %d\n", n, t->ndim);
        exit(1);
    }

    int new_shape[8];
    for (int i = 0; i < n; i++) new_shape[i] = t->shape[perm[i]];

    Tensor result = tensor_create(new_shape, n);

    int coords[8] = {0};
    int new_coords[8];
    while (1) {
        for (int i = 0; i < n; i++) new_coords[i] = coords[perm[i]];
        tensor_set(&result, new_coords, tensor_get(t, coords, n));

        int axis = n - 1;
        while (axis >= 0) {
            coords[axis]++;
            if (coords[axis] < t->shape[axis]) break;
            coords[axis] = 0;
            axis--;
        }
        if (axis < 0) break;
    }
    return result;
}

Tensor tensor_cat(Tensor *a, Tensor *b, int dim) {
    if (a->ndim != b->ndim) {
        fprintf(stderr, "tensor_cat: ndim mismatch\n");
        exit(1);
    }

    if (dim < 0 || dim >= a->ndim) {
        fprintf(stderr, "tensor_cat: invalid dim %d\n", dim);
        exit(1);
    }

    for (int i = 0; i < a->ndim; i++) {
        if (i == dim) continue;

        if (a->shape[i] != b->shape[i]) {
            fprintf(stderr, "tensor_cat: shape mismatch at dim %d (%d != %d)\n",
                    i, a->shape[i], b->shape[i]);
            exit(1);
        }
    }

    int out_shape[8];
    for (int i = 0; i < a->ndim; i++)
        out_shape[i] = a->shape[i];

    out_shape[dim] += b->shape[dim];

    Tensor out = tensor_create(out_shape, a->ndim);

    int outer = 1;
    for (int i = 0; i < dim; i++)
        outer *= a->shape[i];

    int inner = 1;
    for (int i = dim + 1; i < a->ndim; i++)
        inner *= a->shape[i];

    int a_chunk = a->shape[dim] * inner;
    int b_chunk = b->shape[dim] * inner;
    int out_chunk = out.shape[dim] * inner;

    for (int i = 0; i < outer; i++) {
        memcpy(out.data + i * out_chunk,
               a->data + i * a_chunk,
               a_chunk * sizeof(float));

        memcpy(out.data + i * out_chunk + a_chunk,
               b->data + i * b_chunk,
               b_chunk * sizeof(float));
    }

    return out;
}

float tensor_mean(Tensor *t) {
    float sum = 0.0f;
    for (int i = 0; i < t->size; i++) {
        sum += t->data[i];
    }

    return sum / t->size;
}

float tensor_var(Tensor *t) {
    float sum = 0.0f;
    float mean = tensor_mean(t);
    for (int i = 0; i < t->size; i++) {
        sum += powf(t->data[i] - mean, 2);
    }

    return sum / t->size;
}

Tensor tensor_slice_rows(Tensor *tensor, int start_row, int num_rows) {
    const int shape[2] = {num_rows, tensor->shape[1]};
    Tensor slice = tensor_create(shape, 2);

    for (int r = 0; r < num_rows; r++) {
        for (int c = 0; c < tensor->shape[1]; c++) {

            float value = tensor_get(
                tensor,
                (int[]){start_row + r, c},
                2
            );

            tensor_set(
                &slice,
                (int[]){r, c},
                value
            );
        }
    }

    return slice;
}

void tensor_free(Tensor *t) {
    free(t->shape);
    t->shape = NULL;
    free(t->data);
    t->data = NULL;
    free(t->strides);
    t->strides = NULL;
    free(t->grad);
    t->grad = NULL;
}

void tensor_reshape2d(Tensor *t, int rows, int cols) {
    const int s[2] = {rows, cols};
    tensor_reshape(t, s, 2);
}

void tensor_add_inplace(Tensor *a, Tensor *b) {
    for (int i = 0; i < a->size; i++) a->data[i] += b->data[i];
}

Tensor tensor_linear_backward(Tensor *X2d, Tensor *W, Tensor *dY2d) {
    Tensor Xt = tensor_transpose(X2d); // [K,N]
    Tensor dW = tensor_matmul(&Xt, dY2d); // [K,M]
    for (int i = 0; i < dW.size; i++) W->grad[i] += dW.data[i];
    Tensor Wt = tensor_transpose(W); // [M,K]
    Tensor dX = tensor_matmul(dY2d, &Wt); // [N,K]
    tensor_free(&Xt);
    tensor_free(&dW);
    tensor_free(&Wt);
    return dX;
}

void tensor_bias_backward(Tensor *dY2d, Tensor *b) {
    const int N = dY2d->shape[0];
    const int M = dY2d->shape[1];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < M; j++)
            b->grad[j] += dY2d->data[i * M + j];
}

