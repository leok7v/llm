#ifndef TRAIN_H
#define TRAIN_H

#include "tensor.h"
#include "model.h"

typedef struct {
    Tensor inputs;
    Tensor targets;
} DataLoader;
DataLoader create_dataloader_v1(int *tokens, int num_tokens);

float calc_loss_from_logits(Tensor *logits, Tensor *targets);
float calc_loss_batch(GPTModel *model, GPT_CONFIG *cfg, Tensor inputs, Tensor targets);
float calc_loss_loader(GPTModel *model, GPT_CONFIG *cfg, DataLoader *loader, int batch_size);

Tensor backward_layernorm(LayerNorm *self, int d, Tensor *upstream, Tensor *xhat, Tensor *r);

float numerical_gradient(
    GPTModel *model, 
    GPT_CONFIG *cfg, 
    Tensor *param, 
    int index,
    float eps, 
    Tensor inputs, 
    Tensor targets
);
float grad_check_param(
    GPTModel *model, 
    GPT_CONFIG *cfg, 
    Tensor *param,
    const char *name, 
    Tensor inputs, 
    Tensor targets
);

typedef struct { Tensor **params; int count; } ParamList;
ParamList collect_params(GPTModel *model, GPT_CONFIG *cfg);
void alloc_grads(ParamList *pl);
void zero_grads(ParamList *pl);

float train_step(GPTModel *model, GPT_CONFIG *cfg, Tensor inputs, Tensor targets);

typedef struct { float **m; float **v; int count; long t; } AdamW;
AdamW adamw_init(ParamList *pl);
void  adamw_free(AdamW *o);
void  adamw_step(ParamList *pl, AdamW *o, float lr, float b1, float b2, float eps, float wd);

void save_weights(const char *path, ParamList *pl);
int  load_weights(const char *path, ParamList *pl);

#endif
