#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"
#include <stdbool.h>

// shared across model/train/main
#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#define CONTEXT_SIZE 64 // context length (window width in tokens)
#define STRIDE 64 // dataloader stride (== CONTEXT_SIZE -> non-overlapping)
#define EMBEDDIGS_OUTPUT_DIM 128 // embedding dimension
#define N_HEADS 8 // attention heads (emb_dim must divide by this)
#define N_LAYERS 4 // transformer blocks

typedef struct {
    int vocab_size;
    int context_length; // CONTEXT_SIZE
    int emb_dim;
    int n_heads;
    int n_layers;
    float drop_rate;
    bool qkv_bias;
} GPT_CONFIG;

typedef struct { Tensor lin1, b1, lin2, b2; } FeedForward;
typedef struct { float eps; Tensor scale, shift; } LayerNorm;
typedef struct { Tensor W_query, W_key, W_value; } MultiHeadAttention;
typedef struct { MultiHeadAttention mha; FeedForward ffn; LayerNorm ln1, ln2; } TransformerBlock;
typedef struct {
    Tensor tok_emb;
    Tensor pos_emb;
    TransformerBlock *trf_blocks;
    LayerNorm final_norm;
    Tensor out_head;
} GPTModel;

void GELU(Tensor *x);
float gelu_prime(float x);

void FeedForward_init(FeedForward *self, GPT_CONFIG *cfg);
Tensor FeedForward_forward(FeedForward *self, Tensor *x);
void LayerNorm_init(LayerNorm *self, int emb_dim);
Tensor LayerNorm_forward(
    LayerNorm *self, 
    GPT_CONFIG *cfg, 
    Tensor *x,
    Tensor *cached_xhat, 
    Tensor *cached_r
);
void MultiHeadAttention_init(MultiHeadAttention *self, GPT_CONFIG *cfg);
Tensor MultiHeadAttention_forward(MultiHeadAttention *self, GPT_CONFIG *cfg, Tensor *x);
void TransformerBlock_init(TransformerBlock *self, GPT_CONFIG *cfg);
Tensor TransformerBlock_forward(TransformerBlock *self, GPT_CONFIG *cfg, Tensor *x);

void GPTModel_init(GPTModel *self, GPT_CONFIG *cfg);
void GPTModel_free(GPTModel *self, GPT_CONFIG *cfg);
Tensor GPTModel_forward(GPTModel *self, GPT_CONFIG *cfg, Tensor in_idx);
Tensor generate_text(GPTModel *model, GPT_CONFIG *cfg, Tensor idx, int max_new_tokens);

#endif