// Requires: merges.bin produced by ./train_bpe
// Compile: gcc llm.c model.c train.c tensor.c utils.c bpe_runtime.c -o llm -lm
// Run (run): ./llm run <merges.bin path> "<text>"
// Run (train): ./llm train <merges.bin path>

#include "tensor.h"
#include "bpe.h"
#include "utils.h"
#include "uthash.h"
#include "model.h"
#include "train.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdbool.h>

void run(const char *merges_path, char *text) {
    int num_merges;
    Merge *merges = load_merges(merges_path, &num_merges);
    const int vocab_size = 256 + num_merges;
    Vocab *vocab = build_vocab(merges, num_merges);

    int enc_len;
    int *enc_sample = encode(text, merges, num_merges, &enc_len);

    const int batch_dim = 2;
    const int batch_shape[2] = {1, enc_len};  // 1 because of a single text sample
    Tensor batch = tensor_create(batch_shape, batch_dim);

    for (int x = 0; x < 1; x++) { // later, we will add more samples
        for (int y = 0; y < enc_len; y++) {
            const int coords[2] = {x, y};
            tensor_set(&batch, coords, enc_sample[y]);
        }
    }

    GPT_CONFIG gpt_config_1_26m;
    gpt_config_1_26m.vocab_size = vocab_size;
    gpt_config_1_26m.context_length = CONTEXT_SIZE;
    gpt_config_1_26m.emb_dim = EMBEDDIGS_OUTPUT_DIM;
    gpt_config_1_26m.n_heads = N_HEADS;
    gpt_config_1_26m.n_layers = N_LAYERS;
    gpt_config_1_26m.drop_rate = 0.0f;
    gpt_config_1_26m.qkv_bias = false;

    GPTModel model;
    GPTModel_init(&model, &gpt_config_1_26m);

    ParamList params = collect_params(&model, &gpt_config_1_26m);
    if (load_weights("weights.bin", &params))
        printf("[loaded trained weights from weights.bin]\n");
    else
        printf("[no weights.bin found, using random init]\n");
    free(params.params);

    int max_new_tokens = 10;
    Tensor idx = generate_text(&model, &gpt_config_1_26m, batch, max_new_tokens);
    int *ids = (int *)malloc(idx.shape[1] * sizeof(int));
    for (int i = 0; i < idx.shape[1]; i++) {
        const int coords[2] = {0, i};
        ids[i] = (int)tensor_get(&idx, coords, 2);
    }
    char *decoded = decode(ids, idx.shape[1], vocab);
    printf("%s\n", decoded);
    tensor_free(&idx);
    free(ids);
    free(decoded);

    GPTModel_free(&model, &gpt_config_1_26m);

    Vocab *vc, *vtmp;
    HASH_ITER(hh, vocab, vc, vtmp) {
        HASH_DEL(vocab, vc);
        free(vc->bytes);
        free(vc);
    }

    free(enc_sample);
    free(merges);
}

void train(const char *merges_path) {
    int num_merges;
    Merge *merges = load_merges(merges_path, &num_merges);
    const int vocab_size = 256 + num_merges;
    Vocab *vocab = build_vocab(merges, num_merges);

    char *text = load_corpus("tinyshakespeare.txt"); // if good computer
    // char *text = load_corpus("the-verdict.txt"); // if bad computer

    int enc_len;
    int *enc_sample = encode(text, merges, num_merges, &enc_len);

    int train_len = (int)(enc_len * 0.9f);
    int val_len = enc_len - train_len;

    DataLoader train_loader = create_dataloader_v1(enc_sample, train_len);
    DataLoader val_loader = create_dataloader_v1(enc_sample + train_len, val_len);

    printf("Train tokens: %d (%d windows)\n", train_len, train_loader.inputs.shape[0]);
    printf("Val tokens: %d (%d windows)\n", val_len, val_loader.inputs.shape[0]);

    GPT_CONFIG cfg;
    cfg.vocab_size = vocab_size;
    cfg.context_length = CONTEXT_SIZE;
    cfg.emb_dim = EMBEDDIGS_OUTPUT_DIM;
    cfg.n_heads = N_HEADS;
    cfg.n_layers = N_LAYERS;
    cfg.drop_rate = 0.0f;
    cfg.qkv_bias = false;

    GPTModel model;
    GPTModel_init(&model, &cfg);

    ParamList params = collect_params(&model, &cfg);
    alloc_grads(&params);

    AdamW opt = adamw_init(&params);
    int batch_size = 2;
    int num_epochs = 30;
    float lr = 1e-3f, b1 = 0.9f, b2 = 0.999f, eps = 1e-8f, wd = 0.1f;
    int num_windows = train_loader.inputs.shape[0];

    printf("=== training: %d epochs, batch=%d, lr=%.0e ===\n", num_epochs, batch_size, lr);
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f; int nb = 0;
        for (int start = 0; start + batch_size <= num_windows; start += batch_size) {
            Tensor bi = tensor_slice_rows(&train_loader.inputs, start, batch_size);
            Tensor bt = tensor_slice_rows(&train_loader.targets, start, batch_size);

            zero_grads(&params);
            float loss = train_step(&model, &cfg, bi, bt);
            adamw_step(&params, &opt, lr, b1, b2, eps, wd);
            epoch_loss += loss; nb++;

            tensor_free(&bi);
            tensor_free(&bt);
        }
        float vl = calc_loss_loader(&model, &cfg, &val_loader, batch_size);
        printf("epoch %2d | train %.4f | val %.4f\n", epoch + 1, epoch_loss / nb, vl);
    }

    float train_loss = calc_loss_loader(&model, &cfg, &train_loader, batch_size);
    float val_loss = calc_loss_loader(&model, &cfg, &val_loader, batch_size);
    printf("\nFinal  train loss: %.4f   val loss: %.4f\n", train_loss, val_loss);

    {
        const char *seed = "I HAD always thought";
        int slen;
        int *senc = encode((char *)seed, merges, num_merges, &slen);
        Tensor idx = tensor_create((int[]){1, slen}, 2);
        for (int i = 0; i < slen; i++) tensor_set(&idx, (int[]){0, i}, senc[i]);
        Tensor out = generate_text(&model, &cfg, idx, 30);
        int *ids = (int *)malloc(out.shape[1] * sizeof(int));
        for (int i = 0; i < out.shape[1]; i++) ids[i] = (int)tensor_get(&out, (int[]){0, i}, 2);
        char *dec = decode(ids, out.shape[1], vocab);
        printf("sample: %s\n", dec);
        tensor_free(&out); free(ids); free(dec); free(senc);
    }

    save_weights("weights.bin", &params);

    adamw_free(&opt);
    free(params.params);

    GPTModel_free(&model, &cfg);

    Vocab *vc, *vtmp;
    HASH_ITER(hh, vocab, vc, vtmp) {
        HASH_DEL(vocab, vc);
        free(vc->bytes);
        free(vc);
    }

    free(enc_sample);
    free(merges);
    free(text);
}

int main(int argc, char *argv[]) {
    srand((unsigned int)time(NULL));

    /*
        The count, term by term (config: vocab=2048, ctx=64, emb=128, heads=8, layers=4)

        Embeddings:
        - tok_emb = [vocab, emb] = 2048 × 128 = 262,144
        - pos_emb = [ctx, emb] = 64 × 128 = 8,192

        One transformer block:
        - MHA: W_query, W_key, W_value, each [emb, emb] = 128 × 128 = 16,384. Three of them -> 3 × 16,384 = 49,152
        - FFN:
        - lin1 [emb, 4·emb] = 128 × 512 = 65,536
        - b1 [4·emb] = 512
        - lin2 [4·emb, emb] = 512 × 128 = 65,536
        - b2 [emb] = 128
        - FFN subtotal = 131,712
        - ln1: scale [emb] + shift [emb] = 128 + 128 = 256
        - ln2: scale + shift = 256 (and ln1 = 256)
        - Block total = 49,152 + 131,712 + 256 + 256 = 181,376

        Times n_layers = 4 -> 725,504

        Final layer norm: scale + shift = 128 + 128 = 256

        Output head: out_head = [emb, vocab] = 128 × 2048 = 262,144

        Grand total:
        tok_emb        262,144
        pos_emb          8,192
        4 × block      725,504
        final_norm         256
        out_head       262,144
        ----------------------
        TOTAL        1,258,240   ≈ 1.26M parameters
    */

    if (strcmp(argv[1], "train") == 0) {
        if (argc != 3) {
            fprintf(stderr, "Usage: %s train <merges.bin>\n", argv[0]);
            return 1;
        }
        train(argv[2]);
    } else if (strcmp(argv[1], "run") == 0) {
        if (argc != 4) {
            fprintf(stderr, "Usage: %s run <merges.bin> \"<text>\"\n", argv[0]);
            return 1;
        }
        run(argv[2], argv[3]);
    } else {
        fprintf(stderr, "Unknown command: %s\n", argv[1]);
        return 1;
    }

    return 0;
}