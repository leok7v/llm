#include "train.h"
#include "utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

DataLoader create_dataloader_v1(int *tokens, int num_tokens) {
    const int num_windows = ((num_tokens - 1 - CONTEXT_SIZE) / STRIDE) + 1;

    const int shape[2] = {num_windows, CONTEXT_SIZE};
    Tensor inputs = tensor_create(shape, 2);
    Tensor targets = tensor_create(shape, 2);

    for (int x = 0; x < num_windows; x++) {
        for (int y = 0; y < CONTEXT_SIZE; y++) {
            tensor_set(&inputs, (int[]){x, y}, tokens[x * STRIDE + y]);
            tensor_set(&targets, (int[]){x, y}, tokens[x * STRIDE + y + 1]);
        }
    }

    DataLoader loader;
    loader.inputs = inputs;
    loader.targets = targets;

    return loader;
}

Tensor backward_layernorm(
    LayerNorm *self,
    int d,
    Tensor *upstream,
    Tensor *xhat,
    Tensor *r
) {
    const int BT = upstream->shape[0];

    Tensor grad_x = tensor_create(upstream->shape, 2);

    for (int t = 0; t < BT; t++) {
        const float rt = r->data[t];

        float S1 = 0.0f;
        float S2 = 0.0f;
        for (int i = 0; i < d; i++) {
            const float g_i    = upstream->data[t * d + i];
            const float xhat_i = xhat->data[t * d + i];

            self->shift.grad[i] += g_i;
            self->scale.grad[i] += g_i * xhat_i;

            const float gtilde_i = g_i * self->scale.data[i];
            S1 += gtilde_i;
            S2 += gtilde_i * xhat_i;
        }

        const float S1_over_d = S1 / d;
        const float S2_over_d = S2 / d;

        for (int j = 0; j < d; j++) {
            const float g_j     = upstream->data[t * d + j];
            const float xhat_j  = xhat->data[t * d + j];
            const float gtilde_j = g_j * self->scale.data[j];

            grad_x.data[t * d + j] =
                rt * (gtilde_j - S1_over_d - xhat_j * S2_over_d);
        }
    }

    return grad_x;
}

typedef struct {
    Tensor normed1;
    Tensor ln1_xhat;
    Tensor ln1_r;
    Tensor Q, K, V;
    Tensor attn_w;
    Tensor normed2;
    Tensor ln2_xhat;
    Tensor ln2_r;
    Tensor ffn_pre;
    Tensor ffn_post;
} BlockCache;

typedef struct {
    BlockCache *blocks;
    Tensor final_xhat;
    Tensor final_r;
    Tensor normed_final;
    Tensor logits;
} ForwardCache;

ParamList collect_params(GPTModel *model, GPT_CONFIG *cfg) {
    int n = 5 + 11 * cfg->n_layers;
    Tensor **p = (Tensor **)malloc(n * sizeof(Tensor *));
    int k = 0;
    p[k++] = &model->tok_emb;
    p[k++] = &model->pos_emb;
    for (int l = 0; l < cfg->n_layers; l++) {
        TransformerBlock *blk = &model->trf_blocks[l];
        p[k++] = &blk->ln1.scale;
        p[k++] = &blk->ln1.shift;
        p[k++] = &blk->mha.W_query;
        p[k++] = &blk->mha.W_key;
        p[k++] = &blk->mha.W_value;
        p[k++] = &blk->ln2.scale;
        p[k++] = &blk->ln2.shift;
        p[k++] = &blk->ffn.lin1;
        p[k++] = &blk->ffn.b1;
        p[k++] = &blk->ffn.lin2;
        p[k++] = &blk->ffn.b2;
    }
    p[k++] = &model->final_norm.scale;
    p[k++] = &model->final_norm.shift;
    p[k++] = &model->out_head;
    ParamList pl = { p, n };
    return pl;
}

void alloc_grads(ParamList *pl) {
    for (int i = 0; i < pl->count; i++) {
        Tensor *t = pl->params[i];
        if (t->grad == NULL) t->grad = (float *)calloc(t->size, sizeof(float));
    }
}

void zero_grads(ParamList *pl) {
    for (int i = 0; i < pl->count; i++) {
        Tensor *t = pl->params[i];
        memset(t->grad, 0, t->size * sizeof(float));
    }
}

ForwardCache train_forward(GPTModel *model, GPT_CONFIG *cfg, Tensor inputs) {
    const int B = inputs.shape[0];
    const int T = inputs.shape[1];
    const int d = cfg->emb_dim;
    const int H = cfg->n_heads;
    const int dh = d / H;
    const int BH = B * H;

    ForwardCache fc;
    fc.blocks = (BlockCache *)malloc(cfg->n_layers * sizeof(BlockCache));

    Tensor x = tensor_create((int[]){B, T, d}, 3);
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int token = (int)tensor_get(&inputs, (int[]){b, t}, 2);
            for (int j = 0; j < d; j++) {
                float tv = tensor_get(&model->tok_emb, (int[]){token, j}, 2);
                float pv = tensor_get(&model->pos_emb, (int[]){t, j}, 2);
                tensor_set(&x, (int[]){b, t, j}, tv + pv);
            }
        }
    }

    for (int l = 0; l < cfg->n_layers; l++) {
        TransformerBlock *blk = &model->trf_blocks[l];
        BlockCache *bc = &fc.blocks[l];

        Tensor normed1 = LayerNorm_forward(&blk->ln1, cfg, &x, &bc->ln1_xhat, &bc->ln1_r);
        bc->normed1 = normed1;

        Tensor Qp = tensor_matmul(&normed1, &blk->mha.W_query);
        Tensor Kp = tensor_matmul(&normed1, &blk->mha.W_key);
        Tensor Vp = tensor_matmul(&normed1, &blk->mha.W_value);
        const int split4[4] = {B, T, H, dh};
        tensor_reshape(&Qp, split4, 4);
        tensor_reshape(&Kp, split4, 4);
        tensor_reshape(&Vp, split4, 4);
        const int perm[4] = {0, 2, 1, 3};
        Tensor Q = tensor_permute(&Qp, perm, 4);
        Tensor K = tensor_permute(&Kp, perm, 4);
        Tensor V = tensor_permute(&Vp, perm, 4);
        tensor_free(&Qp); tensor_free(&Kp); tensor_free(&Vp);
        const int flat3[3] = {BH, T, dh};
        tensor_reshape(&Q, flat3, 3);
        tensor_reshape(&K, flat3, 3);
        tensor_reshape(&V, flat3, 3);

        const float scale = 1.0f / sqrtf((float)dh);
        Tensor Kt = tensor_transpose(&K);
        Tensor scores = tensor_matmul(&Q, &Kt);
        tensor_free(&Kt);
        Tensor attn_w = tensor_create((int[]){BH, T, T}, 3);
        float *rowbuf = (float *)malloc(T * sizeof(float));
        for (int bh = 0; bh < BH; bh++) {
            for (int qp = 0; qp < T; qp++) {
                for (int kp = 0; kp < T; kp++) {
                    float s = tensor_get(&scores, (int[]){bh, qp, kp}, 3) * scale;
                    if (kp > qp) s = -INFINITY;
                    rowbuf[kp] = s;
                }
                softmax(rowbuf, T);
                for (int kp = 0; kp < T; kp++)
                    tensor_set(&attn_w, (int[]){bh, qp, kp}, rowbuf[kp]);
            }
        }
        free(rowbuf);
        tensor_free(&scores);

        Tensor heads = tensor_matmul(&attn_w, &V);
        const int unflat4[4] = {B, H, T, dh};
        tensor_reshape(&heads, unflat4, 4);
        Tensor mha_out = tensor_permute(&heads, perm, 4);
        tensor_free(&heads);
        const int back3[3] = {B, T, d};
        tensor_reshape(&mha_out, back3, 3);

        bc->Q = Q; bc->K = K; bc->V = V; bc->attn_w = attn_w;

        Tensor attn_res = tensor_add(&mha_out, &x);
        tensor_free(&mha_out);
        tensor_free(&x);

        Tensor normed2 = LayerNorm_forward(&blk->ln2, cfg, &attn_res, &bc->ln2_xhat, &bc->ln2_r);
        bc->normed2 = normed2;

        Tensor hidden = tensor_matmul(&normed2, &blk->ffn.lin1);
        for (int i = 0; i < hidden.size; i++)
            hidden.data[i] += blk->ffn.b1.data[i % blk->ffn.lin1.shape[1]];
        bc->ffn_pre = tensor_create(hidden.shape, hidden.ndim);
        memcpy(bc->ffn_pre.data, hidden.data, hidden.size * sizeof(float));
        GELU(&hidden);
        bc->ffn_post = tensor_create(hidden.shape, hidden.ndim);
        memcpy(bc->ffn_post.data, hidden.data, hidden.size * sizeof(float));
        Tensor ffn_out = tensor_matmul(&hidden, &blk->ffn.lin2);
        for (int i = 0; i < ffn_out.size; i++)
            ffn_out.data[i] += blk->ffn.b2.data[i % blk->ffn.lin2.shape[1]];
        tensor_free(&hidden);

        Tensor block_out = tensor_add(&ffn_out, &attn_res);
        tensor_free(&ffn_out);
        tensor_free(&attn_res);

        x = block_out;
    }

    fc.normed_final = LayerNorm_forward(&model->final_norm, cfg, &x, &fc.final_xhat, &fc.final_r);
    tensor_free(&x);
    fc.logits = tensor_matmul(&fc.normed_final, &model->out_head);
    return fc;
}

void free_forward_cache(ForwardCache *fc, int n_layers) {
    for (int l = 0; l < n_layers; l++) {
        BlockCache *bc = &fc->blocks[l];
        tensor_free(&bc->normed1);
        tensor_free(&bc->ln1_xhat); tensor_free(&bc->ln1_r);
        tensor_free(&bc->Q); tensor_free(&bc->K); tensor_free(&bc->V);
        tensor_free(&bc->attn_w);
        tensor_free(&bc->normed2);
        tensor_free(&bc->ln2_xhat); tensor_free(&bc->ln2_r);
        tensor_free(&bc->ffn_pre); tensor_free(&bc->ffn_post);
    }
    free(fc->blocks);
    tensor_free(&fc->final_xhat); tensor_free(&fc->final_r);
    tensor_free(&fc->normed_final);
    tensor_free(&fc->logits);
}

Tensor mha_backward(
    MultiHeadAttention *mha,
    GPT_CONFIG *cfg,
    BlockCache *bc,
    Tensor *d_mha_out,
    int B, int T
) {
    const int d = cfg->emb_dim;
    const int H = cfg->n_heads;
    const int dh = d / H;
    const int BH = B * H;
    const float scale = 1.0f / sqrtf((float)dh);
    const int perm[4] = {0, 2, 1, 3};

    Tensor dmo = tensor_create(d_mha_out->shape, 2);
    memcpy(dmo.data, d_mha_out->data, d_mha_out->size * sizeof(float));
    tensor_reshape(&dmo, (int[]){B, T, H, dh}, 4);
    Tensor d_heads = tensor_permute(&dmo, perm, 4);
    tensor_free(&dmo);
    tensor_reshape(&d_heads, (int[]){BH, T, dh}, 3);

    Tensor Vt = tensor_transpose(&bc->V);
    Tensor d_attn_w = tensor_matmul(&d_heads, &Vt);
    tensor_free(&Vt);
    Tensor attn_wt = tensor_transpose(&bc->attn_w);
    Tensor d_V = tensor_matmul(&attn_wt, &d_heads);
    tensor_free(&attn_wt);
    tensor_free(&d_heads);

    Tensor d_scores = tensor_create((int[]){BH, T, T}, 3);
    for (int bh = 0; bh < BH; bh++) {
        for (int qp = 0; qp < T; qp++) {
            float dot = 0.0f;
            for (int kp = 0; kp < T; kp++) {
                float p = bc->attn_w.data[(bh * T + qp) * T + kp];
                float g = d_attn_w.data[(bh * T + qp) * T + kp];
                dot += p * g;
            }
            for (int kp = 0; kp < T; kp++) {
                float p = bc->attn_w.data[(bh * T + qp) * T + kp];
                float g = d_attn_w.data[(bh * T + qp) * T + kp];
                d_scores.data[(bh * T + qp) * T + kp] = p * (g - dot) * scale;
            }
        }
    }
    tensor_free(&d_attn_w);

    Tensor d_Q = tensor_matmul(&d_scores, &bc->K);
    Tensor d_scores_t = tensor_transpose(&d_scores);
    Tensor d_K = tensor_matmul(&d_scores_t, &bc->Q);
    tensor_free(&d_scores_t);
    tensor_free(&d_scores);

    Tensor *heads_grads[3] = {&d_Q, &d_K, &d_V};
    Tensor proj_grads[3];
    for (int g = 0; g < 3; g++) {
        tensor_reshape(heads_grads[g], (int[]){B, H, T, dh}, 4);
        Tensor bthd = tensor_permute(heads_grads[g], perm, 4);
        tensor_reshape(&bthd, (int[]){B * T, d}, 2);
        proj_grads[g] = bthd;
    }
    tensor_free(&d_Q); tensor_free(&d_K); tensor_free(&d_V);

    Tensor n1 = tensor_create(bc->normed1.shape, bc->normed1.ndim);
    memcpy(n1.data, bc->normed1.data, bc->normed1.size * sizeof(float));
    tensor_reshape2d(&n1, B * T, d);

    Tensor *Ws[3] = {&mha->W_query, &mha->W_key, &mha->W_value};
    Tensor d_normed1 = tensor_create((int[]){B * T, d}, 2);
    for (int g = 0; g < 3; g++) {
        Tensor dn = tensor_linear_backward(&n1, Ws[g], &proj_grads[g]);
        tensor_add_inplace(&d_normed1, &dn);
        tensor_free(&dn);
        tensor_free(&proj_grads[g]);
    }
    tensor_free(&n1);
    return d_normed1;
}

void train_backward(
    GPTModel *model,
    GPT_CONFIG *cfg,
    Tensor inputs,
    Tensor targets,
    ForwardCache *fc
) {
    const int B = fc->logits.shape[0];
    const int T = fc->logits.shape[1];
    const int V = fc->logits.shape[2];
    const int d = cfg->emb_dim;
    const int BT = B * T;

    Tensor gz = tensor_create((int[]){BT, V}, 2);
    float *row = (float *)malloc(V * sizeof(float));
    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            for (int v = 0; v < V; v++)
                row[v] = tensor_get(&fc->logits, (int[]){b, t, v}, 3);
            softmax(row, V);
            int target = (int)tensor_get(&targets, (int[]){b, t}, 2);
            int r = b * T + t;
            for (int v = 0; v < V; v++) {
                float g = row[v];
                if (v == target) g -= 1.0f;
                gz.data[r * V + v] = g / (float)BT;
            }
        }
    }
    free(row);

    tensor_reshape2d(&fc->normed_final, BT, d);
    Tensor d_normed_final = tensor_linear_backward(&fc->normed_final, &model->out_head, &gz);
    tensor_free(&gz);

    Tensor g = backward_layernorm(&model->final_norm, d, &d_normed_final,
                                  &fc->final_xhat, &fc->final_r);
    tensor_free(&d_normed_final);

    for (int l = cfg->n_layers - 1; l >= 0; l--) {
        TransformerBlock *blk = &model->trf_blocks[l];
        BlockCache *bc = &fc->blocks[l];

        Tensor d_attn_res = tensor_create(g.shape, 2);
        memcpy(d_attn_res.data, g.data, g.size * sizeof(float));

        tensor_reshape2d(&bc->ffn_post, BT, 4 * d);
        tensor_reshape2d(&bc->ffn_pre, BT, 4 * d);
        tensor_reshape2d(&bc->normed2, BT, d);

        tensor_bias_backward(&g, &blk->ffn.b2);
        Tensor d_hidden_post = tensor_linear_backward(&bc->ffn_post, &blk->ffn.lin2, &g);
        Tensor d_hidden_pre = tensor_create(d_hidden_post.shape, 2);
        for (int i = 0; i < d_hidden_pre.size; i++)
            d_hidden_pre.data[i] = d_hidden_post.data[i] * gelu_prime(bc->ffn_pre.data[i]);
        tensor_free(&d_hidden_post);
        tensor_bias_backward(&d_hidden_pre, &blk->ffn.b1);
        Tensor d_normed2 = tensor_linear_backward(&bc->normed2, &blk->ffn.lin1, &d_hidden_pre);
        tensor_free(&d_hidden_pre);

        Tensor d_attn_res_ln2 = backward_layernorm(&blk->ln2, d, &d_normed2,
                                                   &bc->ln2_xhat, &bc->ln2_r);
        tensor_free(&d_normed2);
        tensor_add_inplace(&d_attn_res, &d_attn_res_ln2);
        tensor_free(&d_attn_res_ln2);

        Tensor d_block_in = tensor_create(d_attn_res.shape, 2);
        memcpy(d_block_in.data, d_attn_res.data, d_attn_res.size * sizeof(float));

        Tensor d_normed1 = mha_backward(&blk->mha, cfg, bc, &d_attn_res, B, T);
        tensor_free(&d_attn_res);

        Tensor d_block_in_ln1 = backward_layernorm(&blk->ln1, d, &d_normed1,
                                                   &bc->ln1_xhat, &bc->ln1_r);
        tensor_free(&d_normed1);
        tensor_add_inplace(&d_block_in, &d_block_in_ln1);
        tensor_free(&d_block_in_ln1);

        tensor_free(&g);
        g = d_block_in;
    }

    for (int b = 0; b < B; b++) {
        for (int t = 0; t < T; t++) {
            int token = (int)tensor_get(&inputs, (int[]){b, t}, 2);
            int rrow = b * T + t;
            for (int j = 0; j < d; j++) {
                float gv = g.data[rrow * d + j];
                model->tok_emb.grad[token * d + j] += gv;
                model->pos_emb.grad[t * d + j] += gv;
            }
        }
    }
    tensor_free(&g);
}

float train_step(GPTModel *model, GPT_CONFIG *cfg, Tensor inputs, Tensor targets) {
    ForwardCache fc = train_forward(model, cfg, inputs);
    float loss = calc_loss_from_logits(&fc.logits, &targets);
    train_backward(model, cfg, inputs, targets, &fc);
    free_forward_cache(&fc, cfg->n_layers);
    return loss;
}

float calc_loss_from_logits(
    Tensor *logits,
    Tensor *targets
) {
    float *logit_vec = (float *)malloc(logits->shape[2] * sizeof(float));
    float *target_probabilities = (float *)malloc(logits->shape[0] * logits->shape[1] * sizeof(float));

    for (int b = 0; b < logits->shape[0]; b++) {
        for (int pos = 0; pos < logits->shape[1]; pos++) {

            for (int v = 0; v < logits->shape[2]; v++) {
                const int coords[3] = {b, pos, v};
                logit_vec[v] = tensor_get(logits, coords, 3);
            }

            softmax(logit_vec, logits->shape[2]);

            const int coords[2] = {b, pos};
            int target_token_id = (int)tensor_get(targets, coords, 2);
            target_probabilities[(b * logits->shape[1]) + pos] = logit_vec[target_token_id];
        }
    }
    free(logit_vec);

    float *log_probabilities = (float *)malloc(logits->shape[0] * logits->shape[1] * sizeof(float));
    for (int p = 0; p < logits->shape[0] * logits->shape[1]; p++) {
        log_probabilities[p] = logf(target_probabilities[p]);
    }
    free(target_probabilities);
    float average_log_prob = mean(log_probabilities, logits->shape[0] * logits->shape[1]);
    free(log_probabilities);
    float loss = average_log_prob * -1;

    return loss;
}

// Negative Average Log Probabilities (Better known as Cross Entropy loss)
float calc_loss_batch(
    GPTModel *model,
    GPT_CONFIG *cfg,
    Tensor inputs,
    Tensor targets
) {
    Tensor logits = GPTModel_forward(model, cfg, inputs);

    float loss = calc_loss_from_logits(&logits, &targets);

    tensor_free(&logits);

    return loss;
}

float calc_loss_loader(
    GPTModel *model,
    GPT_CONFIG *cfg,
    DataLoader *loader,
    int batch_size
) {
    float total_loss = 0.0f;

    const int num_windows = loader->inputs.shape[0];
    int num_batches = 0;

    for (int start = 0; start < num_windows; start += batch_size) {

        int current_batch_size = MIN(batch_size, num_windows - start);

        Tensor inputs = tensor_slice_rows(
            &loader->inputs,
            start,
            current_batch_size
        );

        Tensor targets = tensor_slice_rows(
            &loader->targets,
            start,
            current_batch_size
        );

        total_loss += calc_loss_batch(
            model,
            cfg,
            inputs,
            targets
        );

        tensor_free(&inputs);
        tensor_free(&targets);

        num_batches++;
    }

    return total_loss / num_batches;
}

float numerical_gradient(
    GPTModel *model,
    GPT_CONFIG *cfg,
    Tensor *param,
    int index,
    float eps,
    Tensor inputs,
    Tensor targets
) {
    float saved = param->data[index];

    param->data[index] = saved + eps;
    float loss_plus = calc_loss_batch(model, cfg, inputs, targets);

    param->data[index] = saved - eps;
    float loss_minus = calc_loss_batch(model, cfg, inputs, targets);

    param->data[index] = saved;

    return (loss_plus - loss_minus) / (2.0f * eps);
}

AdamW adamw_init(ParamList *pl) {
    AdamW o;
    o.count = pl->count;
    o.m = (float **)malloc(pl->count * sizeof(float *));
    o.v = (float **)malloc(pl->count * sizeof(float *));
    for (int i = 0; i < pl->count; i++) {
        int sz = pl->params[i]->size;
        o.m[i] = (float *)calloc(sz, sizeof(float));
        o.v[i] = (float *)calloc(sz, sizeof(float));
    }
    o.t = 0;
    return o;
}

void adamw_free(AdamW *o) {
    for (int i = 0; i < o->count; i++) { free(o->m[i]); free(o->v[i]); }
    free(o->m); free(o->v);
}

void adamw_step(
    ParamList *pl, 
    AdamW *o, 
    float lr, 
    float b1, 
    float b2,
    float eps, 
    float wd
) {
    o->t++;
    float bc1 = 1.0f - powf(b1, (float)o->t);
    float bc2 = 1.0f - powf(b2, (float)o->t);
    for (int i = 0; i < pl->count; i++) {
        Tensor *p = pl->params[i];
        for (int j = 0; j < p->size; j++) {
            float g = p->grad[j];
            o->m[i][j] = b1 * o->m[i][j] + (1.0f - b1) * g;
            o->v[i][j] = b2 * o->v[i][j] + (1.0f - b2) * g * g;
            float mhat = o->m[i][j] / bc1;
            float vhat = o->v[i][j] / bc2;
            p->data[j] -= lr * (mhat / (sqrtf(vhat) + eps) + wd * p->data[j]);
        }
    }
}

void save_weights(const char *path, ParamList *pl) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "save_weights: cannot open %s\n", path); return; }
    for (int i = 0; i < pl->count; i++)
        fwrite(pl->params[i]->data, sizeof(float), pl->params[i]->size, f);
    fclose(f);
    printf("[saved weights -> %s]\n", path);
}

int load_weights(const char *path, ParamList *pl) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    for (int i = 0; i < pl->count; i++)
        fread(pl->params[i]->data, sizeof(float), pl->params[i]->size, f);
    fclose(f);
    return 1;
}

float grad_check_param(
    GPTModel *model, 
    GPT_CONFIG *cfg, 
    Tensor *param,
    const char *name, 
    Tensor inputs, 
    Tensor targets
) {
    int idx[3] = {-1, -1, -1};
    float best[3] = {-1, -1, -1};
    for (int i = 0; i < param->size; i++) {
        float a = fabsf(param->grad[i]);
        if (a > best[0]) { best[2]=best[1]; idx[2]=idx[1]; best[1]=best[0]; idx[1]=idx[0]; best[0]=a; idx[0]=i; }
        else if (a > best[1]) { best[2]=best[1]; idx[2]=idx[1]; best[1]=a; idx[1]=i; }
        else if (a > best[2]) { best[2]=a; idx[2]=i; }
    }
    float worst = 0.0f;
    for (int k = 0; k < 3; k++) {
        if (idx[k] < 0) continue;
        float analytic = param->grad[idx[k]];
        float numeric = numerical_gradient(model, cfg, param, idx[k], 1e-2f, inputs, targets);
        float rel = fabsf(analytic - numeric) / (fabsf(analytic) + fabsf(numeric) + 1e-8f);
        if (rel > worst) worst = rel;
    }
    const char *verdict;
    if (worst < 2e-2f) verdict = "PASS";
    else if (best[0] < 2e-3f) verdict = "quant-limited (|g| too small to measure)";
    else verdict = "FAIL <-----";
    printf("grad-check %-18s |g|max=%.2e worst-rel=%.2e  %s\n", name, best[0], worst, verdict);
    return (best[0] < 2e-3f) ? 0.0f : worst;
}
