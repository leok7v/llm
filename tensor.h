#ifndef TENSOR_H
#define TENSOR_H

// tensors are only a 1D array in memory, with metadata
typedef struct {
    float *data; // flat array
    int *shape; // e.g {2, 4, 768}
    int *strides; // precomputed strides -> flat_index = i*(B*C) + j*(C) + k . e.g 3d example
    int ndim; // number of dimensions e.g 3
    int size; // total elements = 2*4*768 = 6144
    float *grad;
} Tensor;

Tensor tensor_create(const int *shape, const int ndim);
Tensor tensor_create_zeros(const int *shape, const int ndim);
Tensor tensor_create_ones(const int *shape, const int ndim);
Tensor tensor_create_random(const int *shape, const int ndim, const float scale);
Tensor tensor_create_dropout(const int *shape, const int ndim, const float rate);
void tensor_dropout_inplace(Tensor *t, float rate);
float tensor_get(Tensor *t, const int *coords, const int num_coords);
void tensor_set(Tensor *t, const int *coords, const float value);
Tensor tensor_add(Tensor*a, Tensor*b);
Tensor tensor_matmul(Tensor *a, Tensor *b);
Tensor tensor_transpose(Tensor *t);  // swaps the last two dims (2D or 3D)
void tensor_reshape(Tensor *t, const int *new_shape, int new_ndim); // in-place metadata change; total size must match
Tensor tensor_permute(Tensor *t, const int *perm, int n); // general N-D axis permutation (allocates new tensor)
Tensor tensor_cat(Tensor *a, Tensor *b, int dim);
float tensor_mean(Tensor *t);
float tensor_var(Tensor *t);
void tensor_free(Tensor *t);
Tensor tensor_slice_rows(Tensor *tensor, int start_row, int num_rows);
void tensor_reshape2d(Tensor *t, int rows, int cols);          
void tensor_add_inplace(Tensor *a, Tensor *b);             
Tensor tensor_linear_backward(Tensor *X2d, Tensor *W, Tensor *dY2d);
void tensor_bias_backward(Tensor *dY2d, Tensor *b);

#endif