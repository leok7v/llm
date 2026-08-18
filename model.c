// The GPT network: activations, modules, forward pass, generation.
#include "model.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void GELU(Tensor *x) {
    // Hyperbolic Tangent (Tanh) Approximation
    for (int i = 0; i < x->size; i++) {
        // Approximate Cumulative Distribution Function (CDF). The expression that gets multiplied by 0.5 * x
        float cdf = 1 + tanhf(sqrtf(2 / M_PI) * (x->data[i] + (0.044715 * powf(x->data[i], 3))));
        x->data[i] = 0.5 * x->data[i] * cdf;
    }
}

// derivative of the tanh-approx GELU above (used by the FFN backward).
// GELU(x) = 0.5x(1+tanh(g)), g = c(x + 0.044715 x^3), c = sqrt(2/pi).
float gelu_prime(float x) {
    const float c = sqrtf(2.0f / M_PI);
    const float g = c * (x + 0.044715f * x * x * x);
    const float t = tanhf(g);
    const float sech2 = 1.0f - t * t; // sech^2(g)
    const float gp = c * (1.0f + 3.0f * 0.044715f * x * x); // g'(x)
    return 0.5f * (1.0f + t) + 0.5f * x * sech2 * gp;
}

void FeedForward_init(FeedForward *self, GPT_CONFIG *cfg) {
    const int lin1_dim = 2;
    const int lin1_shape[2] = {cfg->emb_dim, 4 * cfg->emb_dim};
    self->lin1 = tensor_create_random(
        lin1_shape,
        lin1_dim,
        0.02
    );

    const int b1_dim = 1;
    const int b1_shape[1] = {4 * cfg->emb_dim};
    self->b1 = tensor_create_zeros(b1_shape, b1_dim);

    const int lin2_dim = 2;
    const int lin2_shape[2] = {cfg->emb_dim * 4, cfg->emb_dim};
    self->lin2 = tensor_create_random(
        lin2_shape,
        lin2_dim,
        0.02
    );

    const int b2_dim = 1;
    const int b2_shape[1] = {cfg->emb_dim};
    self->b2 = tensor_create_zeros(b2_shape, b2_dim);
}

Tensor FeedForward_forward(FeedForward *self, Tensor *x) {
    Tensor hidden = tensor_matmul(x, &self->lin1);
    for (int i = 0; i < hidden.size; i++) {
        int j = i % self->lin1.shape[1];
        hidden.data[i] += self->b1.data[j];
    }
    GELU(&hidden);
    Tensor out = tensor_matmul(&hidden, &self->lin2);
    for (int i = 0; i < out.size; i++) {
        int j = i % self->lin2.shape[1];
        out.data[i] += self->b2.data[j];
    }

    tensor_free(&hidden);
    return out;
}

void LayerNorm_init(LayerNorm *self, int emb_dim) {
    self->eps = 1e-5; // prevents division by zero

    const int scale_and_shift_shape[1] = {emb_dim};
    self->scale = tensor_create_ones(scale_and_shift_shape, 1);
    self->shift = tensor_create_zeros(scale_and_shift_shape, 1);
}

Tensor LayerNorm_forward(
    LayerNorm *self, 
    GPT_CONFIG *cfg, 
    Tensor *x,
    Tensor *cached_xhat, 
    Tensor *cached_r
) {
    const int batch_size = x->shape[0];
    const int seq_len = x->shape[1];

    const int normed_x_dim = 3;
    const int normed_x_shape[3] = {batch_size, seq_len, cfg->emb_dim};
    Tensor normed_x = tensor_create(normed_x_shape, normed_x_dim);

    if (cached_xhat) {
        const int xhat_shape[2] = {batch_size * seq_len, cfg->emb_dim};
        *cached_xhat = tensor_create(xhat_shape, 2);
    }
    if (cached_r) {
        const int r_shape[1] = {batch_size * seq_len};
        *cached_r = tensor_create(r_shape, 1);
    }

    for (int b = 0; b < batch_size; b++) {
        for (int pos = 0; pos < seq_len; pos++) {
            const int token_vec_shape[1] = {x->shape[2]};
            Tensor token_vec = tensor_create(token_vec_shape, 1);
            for (int d = 0; d < x->shape[2]; d++) {
                const int coords[3] = {b, pos, d};
                token_vec.data[d] = tensor_get(x, coords, 3);
            }

            float mean = tensor_mean(&token_vec);
            float var = tensor_var(&token_vec);
            float r = 1.0f / sqrtf(var + self->eps);

            const int tok_flat = b * seq_len + pos;
            if (cached_r) cached_r->data[tok_flat] = r;

            for (int d = 0; d < x->shape[2]; d++) {
                float xhat = (token_vec.data[d] - mean) * r;
                if (cached_xhat) {
                    cached_xhat->data[tok_flat * cfg->emb_dim + d] = xhat;
                }
                const int coords[3] = {b, pos, d};
                tensor_set(&normed_x, coords,
                           (self->scale.data[d] * xhat) + self->shift.data[d]);
            }

            tensor_free(&token_vec);
        }
    }

    return normed_x;
}

void MultiHeadAttention_init(MultiHeadAttention *self, GPT_CONFIG *cfg) {
    const int weight_matrices_shape[2] = {cfg->emb_dim, cfg->emb_dim};
    self->W_query = tensor_create_random(weight_matrices_shape, 2, 0.02);
    self->W_key = tensor_create_random(weight_matrices_shape, 2, 0.02);
    self->W_value = tensor_create_random(weight_matrices_shape, 2, 0.02);
}

Tensor MultiHeadAttention_forward(MultiHeadAttention *self, GPT_CONFIG *cfg, Tensor *x) {
    const int batch_size = x->shape[0];
    const int seq_len = x->shape[1];
    const int d_head = cfg->emb_dim / cfg->n_heads;
    const int batch_heads = batch_size * cfg->n_heads;

    Tensor queries_mh = tensor_matmul(x, &self->W_query);
    Tensor keys_mh = tensor_matmul(x, &self->W_key);
    Tensor values_mh = tensor_matmul(x, &self->W_value);

    const int split_shape[4] = {batch_size, seq_len, cfg->n_heads, d_head};
    tensor_reshape(&queries_mh, split_shape, 4);
    tensor_reshape(&keys_mh, split_shape, 4);
    tensor_reshape(&values_mh, split_shape, 4);

    const int perm_fwd[4] = {0, 2, 1, 3};
    Tensor queries_BHTD = tensor_permute(&queries_mh, perm_fwd, 4);
    Tensor keys_BHTD = tensor_permute(&keys_mh, perm_fwd, 4);
    Tensor values_BHTD = tensor_permute(&values_mh, perm_fwd, 4);
    tensor_free(&queries_mh);
    tensor_free(&keys_mh);
    tensor_free(&values_mh);

    const int flat_shape[3] = {batch_heads, seq_len, d_head};
    tensor_reshape(&queries_BHTD, flat_shape, 3);
    tensor_reshape(&keys_BHTD, flat_shape, 3);
    tensor_reshape(&values_BHTD, flat_shape, 3);

    /*
        the reason for normalization by embedding size is to improve the training 
        performance by avoiding small gradients. As dot products increase, the softmax
        function behaves more like a step function, resulting in gradients nearing zero...

        the scaling by the square root of the embdeding dimension is the reason why this self-attention 
        mechanism is also called scaled-dot product attention.
    */

    const float scale_mh = 1.0f / sqrtf((float)d_head);

    Tensor keys_BHTD_T = tensor_transpose(&keys_BHTD); // [B*H, d_head, T]
    Tensor attention_scores_mh = tensor_matmul(&queries_BHTD, &keys_BHTD_T); // [B*H, T, T]
    tensor_free(&queries_BHTD);
    tensor_free(&keys_BHTD);
    tensor_free(&keys_BHTD_T);

    const int weights_mh_shape[3] = {batch_heads, seq_len, seq_len};
    Tensor attention_weights_mh = tensor_create(weights_mh_shape, 3);

    float *row_buffer_mh = (float *)malloc(seq_len * sizeof(float));
    for (int bh = 0; bh < batch_heads; bh++) {
        for (int query_pos = 0; query_pos < seq_len; query_pos++) {
            for (int key_pos = 0; key_pos < seq_len; key_pos++) {
                const int coords[3] = {bh, query_pos, key_pos};
                float s = tensor_get(&attention_scores_mh, coords, 3) * scale_mh;
                if (key_pos > query_pos) s = -INFINITY;  // causal mask
                row_buffer_mh[key_pos] = s;
            }
            softmax(row_buffer_mh, seq_len);
            for (int key_pos = 0; key_pos < seq_len; key_pos++) {
                const int coords[3] = {bh, query_pos, key_pos};
                tensor_set(&attention_weights_mh, coords, row_buffer_mh[key_pos]);
            }
        }
    }
    tensor_free(&attention_scores_mh);
    free(row_buffer_mh);

    Tensor heads_flat = tensor_matmul(&attention_weights_mh, &values_BHTD);
    tensor_free(&values_BHTD);
    tensor_free(&attention_weights_mh);

    const int unflatten_shape[4] = {batch_size, cfg->n_heads, seq_len, d_head};
    tensor_reshape(&heads_flat, unflatten_shape, 4);

    Tensor heads_BTHD = tensor_permute(&heads_flat, perm_fwd, 4);
    tensor_free(&heads_flat);

    const int final_shape[3] = {batch_size, seq_len, cfg->emb_dim};
    tensor_reshape(&heads_BTHD, final_shape, 3);

    return heads_BTHD;
}

void TransformerBlock_init(TransformerBlock *self, GPT_CONFIG *cfg) {
    MultiHeadAttention mha;
    MultiHeadAttention_init(&mha, cfg);
    self->mha = mha;

    FeedForward ffn;
    FeedForward_init(&ffn, cfg);
    self->ffn = ffn;

    LayerNorm ln1;
    LayerNorm_init(&ln1, cfg->emb_dim);
    self->ln1 = ln1;

    LayerNorm ln2;
    LayerNorm_init(&ln2, cfg->emb_dim);
    self->ln2 = ln2;
}

Tensor TransformerBlock_forward(TransformerBlock *self, GPT_CONFIG *cfg, Tensor *x) {
    Tensor *shortcut1 = x;
    Tensor normed1_x = LayerNorm_forward(&self->ln1, cfg, x, NULL, NULL);
    Tensor mha_x = MultiHeadAttention_forward(&self->mha, cfg, &normed1_x);
    tensor_free(&normed1_x);
    tensor_dropout_inplace(&mha_x, cfg->drop_rate);
    Tensor attn_residual = tensor_add(&mha_x, shortcut1);
    tensor_free(&mha_x);

    Tensor *shortcut2 = &attn_residual;
    Tensor normed2_x = LayerNorm_forward(&self->ln2, cfg, &attn_residual, NULL, NULL);
    Tensor ffn_x = FeedForward_forward(&self->ffn, &normed2_x);
    tensor_free(&normed2_x);
    tensor_dropout_inplace(&ffn_x, cfg->drop_rate);
    Tensor block_out = tensor_add(&ffn_x, shortcut2);
    tensor_free(&ffn_x);
    tensor_free(&attn_residual);
    
    return block_out;
}

void GPTModel_init(GPTModel *self, GPT_CONFIG *cfg) {
    const int tok_emb_dim = 2;
    const int tok_emb_shape[2] = {cfg->vocab_size, cfg->emb_dim};
    self->tok_emb = tensor_create_random(
        tok_emb_shape,
        tok_emb_dim,
        0.02
    );

    const int pos_emb_dim = 2;
    const int pos_emb_shape[2] = {cfg->context_length, cfg->emb_dim};
    self->pos_emb = tensor_create_random(
        pos_emb_shape,
        pos_emb_dim,
        0.02
    );

    TransformerBlock *trf_blocks = (TransformerBlock *)malloc(cfg->n_layers * sizeof(TransformerBlock));
    for (int i = 0; i < cfg->n_layers; i++) {
        TransformerBlock trf_block;
        TransformerBlock_init(&trf_block, cfg);
        trf_blocks[i] = trf_block;
    }
    self->trf_blocks = trf_blocks; 

    LayerNorm final_norm;
    LayerNorm_init(&final_norm, cfg->emb_dim);
    self->final_norm = final_norm; 

    const int out_head_dim = 2;
    const int out_head_shape[2] = {cfg->emb_dim, cfg->vocab_size};
    self->out_head = tensor_create_random(
        out_head_shape,
        out_head_dim,
        0.02
    );
}

void GPTModel_free(GPTModel *self, GPT_CONFIG *cfg) {
    tensor_free(&self->tok_emb);
    tensor_free(&self->pos_emb);
    for (int i = 0; i < cfg->n_layers; i++) {
        TransformerBlock *b = &self->trf_blocks[i];
        tensor_free(&b->mha.W_query);
        tensor_free(&b->mha.W_key);
        tensor_free(&b->mha.W_value);
        tensor_free(&b->ffn.lin1);
        tensor_free(&b->ffn.b1);
        tensor_free(&b->ffn.lin2);
        tensor_free(&b->ffn.b2);
        tensor_free(&b->ln1.scale);
        tensor_free(&b->ln1.shift);
        tensor_free(&b->ln2.scale);
        tensor_free(&b->ln2.shift);
    }
    free(self->trf_blocks);
    tensor_free(&self->out_head);
    tensor_free(&self->final_norm.scale);
    tensor_free(&self->final_norm.shift);
}

Tensor GPTModel_forward(GPTModel *self, GPT_CONFIG *cfg, Tensor in_idx) {
    const int batch_size = in_idx.shape[0];
    const int seq_len = in_idx.shape[1];

    const int x_dim = 3;
    const int x_shape[3] = {batch_size, seq_len, cfg->emb_dim};
    Tensor x = tensor_create(x_shape, x_dim);

    for (int b = 0; b < batch_size; b++) {
        for (int pos = 0; pos < seq_len; pos++) {

            const int coords_x[2] = {b, pos};
            int token_id = (int)tensor_get(&in_idx, coords_x, 2);

            for (int d = 0; d < cfg->emb_dim; d++) {
                // fetch from token embedding matrix
                const int tok_coords[2]  = {token_id, d};
                float tok_val = tensor_get(&self->tok_emb, tok_coords, 2);

                // fetch from positional embedding matrix
                const int pos_coords[2]  = {pos, d};
                float pos_val = tensor_get(&self->pos_emb, pos_coords, 2);

                // token_emb + pos_emb = input_emb
                const int out_coords[3] = {b, pos, d};
                tensor_set(&x, out_coords, tok_val + pos_val);
            }
        }
    }

    tensor_dropout_inplace(&x, cfg->drop_rate);

    for (int i = 0; i < cfg->n_layers; i++) {
        Tensor next_block = TransformerBlock_forward(&self->trf_blocks[i], cfg, &x);
        tensor_free(&x);
        x = next_block;
    }

    Tensor normed_x = LayerNorm_forward(&self->final_norm, cfg, &x, NULL, NULL);
    tensor_free(&x);

    Tensor logits = tensor_matmul(&normed_x, &self->out_head);
    tensor_free(&normed_x);

    return logits;
}

Tensor generate_text(GPTModel *model, GPT_CONFIG *cfg, Tensor idx, int max_new_tokens) {
    for (int i = 0; i < max_new_tokens; i++) {
        const int bound = MIN(idx.shape[1], cfg->context_length);
        const int idx_cond_dim = 2;
        const int idx_cond_shape[2] = {idx.shape[0], bound};
        Tensor idx_cond = tensor_create(idx_cond_shape, idx_cond_dim);
        for (int t = 0; t < bound; t++) {
            const int write_coords[2] = {0, t};
            const int read_coords[2] = {0, (idx.shape[1] - bound) + t};
            tensor_set(&idx_cond, write_coords, tensor_get(&idx, read_coords, 2));
        }

        Tensor logits = GPTModel_forward(model, cfg, idx_cond);
        tensor_free(&idx_cond);
        float *last_logit_vec = (float *)malloc(logits.shape[2] * sizeof(float));
        for (int d = 0; d < logits.shape[2]; d++) {
            const int coords[3] = {0, logits.shape[1] - 1, d};
            last_logit_vec[d] = tensor_get(&logits, coords, 3);
        }
        softmax(last_logit_vec, logits.shape[2]);

        int token_id = 0;
        for (int d = 0; d < logits.shape[2]; d++) {
            if (last_logit_vec[d] > last_logit_vec[token_id]) {
                token_id = d;
            }
        }
        tensor_free(&logits);
        free(last_logit_vec);

        const int idx_next_dim = 2;
        const int idx_next_shape[2] = {1, 1}; // will hold next token_id for idx (token_id = max idx of last_logit_vec)
        Tensor idx_next = tensor_create(idx_next_shape, idx_next_dim);

        const int next_coords[2] = {0, 0};
        tensor_set(&idx_next, next_coords, (float)token_id);

        Tensor new_idx = tensor_cat(&idx, &idx_next, 1);
        tensor_free(&idx);
        tensor_free(&idx_next);
        idx = new_idx;
    }

    return idx;
}
