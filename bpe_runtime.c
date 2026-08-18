// Shared BPE runtime: linked into both the trainer (bpe.c) and the LLM (main.c).
// File format for merges.bin:
// [4 bytes] magic "BPE1"
// [int32] num_merges
// [int32 x 2] (p0, p1) per merge, in learned order

#include "bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *load_corpus(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "load_corpus: cannot open %s\n", path); exit(1); }

    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *text = malloc(fsize + 1);
    fread(text, 1, fsize, f);
    text[fsize] = '\0';
    fclose(f);
    return text;
}

Merge *load_merges(const char *path, int *out_num_merges) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "load_merges: cannot open %s\n", path); exit(1); }

    char magic[4];
    fread(magic, 1, 4, f);
    if (memcmp(magic, "BPE1", 4) != 0) {
        fprintf(stderr, "load_merges: bad magic in %s\n", path);
        exit(1);
    }

    int n;
    fread(&n, sizeof(int), 1, f);

    Merge *merges = (Merge *)malloc(n * sizeof(Merge));
    for (int i = 0; i < n; i++) {
        fread(&merges[i].p0, sizeof(int), 1, f);
        fread(&merges[i].p1, sizeof(int), 1, f);
        merges[i].new_id = 256 + i;  // derived, not stored
    }
    fclose(f);

    *out_num_merges = n;
    return merges;
}

void save_merges(const char *path, const Merge *merges, int num_merges) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "save_merges: cannot open %s\n", path); exit(1); }

    fwrite("BPE1", 1, 4, f);
    fwrite(&num_merges, sizeof(int), 1, f);
    for (int i = 0; i < num_merges; i++) {
        fwrite(&merges[i].p0, sizeof(int), 1, f);
        fwrite(&merges[i].p1, sizeof(int), 1, f);
    }
    fclose(f);
}

int *merge(const int *ids, Pair *pair, int num_tokens, int new_id, int *out_len) {
    int *newids = (int *)malloc(num_tokens * sizeof(int));
    int n = 0;
    int i = 0;
    while (i < num_tokens) {
        if (i < num_tokens - 1 && ids[i] == pair->key[0] && ids[i+1] == pair->key[1]) {
            newids[n++] = new_id;
            i += 2;
        } else {
            newids[n++] = ids[i];
            i += 1;
        }
    }
    *out_len = n;
    return newids;
}

Vocab *build_vocab(Merge *merges, int num_merges) {
    Vocab *vocab = NULL;

    // base bytes: vocab[0..255] = single-byte values
    for (int idx = 0; idx < 256; idx++) {
        Vocab *e = malloc(sizeof(Vocab));
        e->id = idx;
        e->bytes = malloc(1);
        e->bytes[0] = (unsigned char)idx;
        e->len = 1;
        HASH_ADD_INT(vocab, id, e);
    }

    // learned ids: replay merges, vocab[new_id] = vocab[p0] ++ vocab[p1]
    for (int i = 0; i < num_merges; i++) {
        int p0 = merges[i].p0;
        int p1 = merges[i].p1;
        int new_id = merges[i].new_id;

        Vocab *v0, *v1;
        HASH_FIND_INT(vocab, &p0, v0);
        HASH_FIND_INT(vocab, &p1, v1);

        Vocab *e = malloc(sizeof(Vocab));
        e->id = new_id;
        e->len = v0->len + v1->len;
        e->bytes = malloc(e->len);
        memcpy(e->bytes, v0->bytes, v0->len);
        memcpy(e->bytes + v0->len, v1->bytes, v1->len);
        HASH_ADD_INT(vocab, id, e);
    }

    return vocab;
}

int *encode(const char *text, const Merge *merges, int num_merges, int *out_len) {
    int len = strlen(text);
    int *tokens = malloc(len * sizeof(int));
    for (int i = 0; i < len; i++) tokens[i] = (unsigned char)text[i];

    while (1) {
        // apply the earliest-learned merge present in the stream
        int best_merge = -1;
        for (int i = 0; i < len - 1; i++) {
            for (int m = 0; m < num_merges; m++) {
                if (tokens[i] == merges[m].p0 && tokens[i+1] == merges[m].p1) {
                    if (best_merge == -1 || m < best_merge) best_merge = m;
                    break;
                }
            }
        }
        if (best_merge == -1) break;

        Pair p;
        p.key[0] = merges[best_merge].p0;
        p.key[1] = merges[best_merge].p1;
        int new_len;
        int *new_tokens = merge(tokens, &p, len, merges[best_merge].new_id, &new_len);
        free(tokens);
        tokens = new_tokens;
        len = new_len;
    }

    *out_len = len;
    return tokens;
}

char *decode(const int *ids, int ids_len, Vocab *vocab) {
    int total = 0;
    for (int i = 0; i < ids_len; i++) {
        Vocab *v;
        HASH_FIND_INT(vocab, &ids[i], v);
        total += v->len;
    }

    char *out = malloc(total + 1);
    int pos = 0;
    for (int i = 0; i < ids_len; i++) {
        Vocab *v;
        HASH_FIND_INT(vocab, &ids[i], v);
        memcpy(out + pos, v->bytes, v->len);
        pos += v->len;
    }
    out[total] = '\0';
    return out;
}