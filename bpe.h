#ifndef BPE_H
#define BPE_H

#include "uthash.h"

// GPT-2 / GPT-3 → vocab_size = 50,257
// GPT-4 (cl100k) → vocab_size = 100,277
// Llama 3 → vocab_size = 128,256
// Raschka's examples → vocab_size = 50,257 (tiktoken gpt-2)
// My learning model → vocab_size = 276
// My learning model plus → vocab_size = 2048
// #define VOCAB_SIZE 276 // 256 base bytes + 20 merges, this was trained only with the-verdict.txt
#define VOCAB_SIZE 2048 // 256 base bytes + 1792 merges (2048 - 256 = 1792), this was trained with tinyshakespeare.txt 55x more than the verdict. (it's what Karpathy's char-rnn/nanoGPT use),

typedef struct {
    int key[2]; // pair of token ids
    int value; // num of repeated occurrences
    UT_hash_handle hh;
} Pair;

typedef struct {
    int p0;
    int p1;
    int new_id;
} Merge;

typedef struct {
    int id; // key (0..vocab_size-1)
    unsigned char *bytes; // raw bytes this id expands to
    int len; // length of bytes
    UT_hash_handle hh;
} Vocab;

char *load_corpus(const char *path);
Merge *load_merges(const char *path, int *out_num_merges);
void save_merges(const char *path, const Merge *merges, int num_merges);

int *merge(const int *ids, Pair *pair, int num_tokens, int new_id, int *out_len);
Vocab *build_vocab(Merge *merges, int num_merges);
int *encode(const char *text, const Merge *merges, int num_merges, int *out_len);
char *decode(const int *ids, int ids_len, Vocab *vocab);

#endif