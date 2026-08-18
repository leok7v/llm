// BPE trainer: reads a corpus, learns `VOCAB_SIZE - 256` merges, writes them to disk.
// Compile: gcc bpe.c bpe_runtime.c -o train_bpe
// Run: ./train_bpe <corpus.txt> <out.bin>

/*
    Note: in a real foundational model (GPT-2/3/4, Llama), the
    tokenizer pipeline is: (1) load corpus, (2) regex pre-tokenize
    (split on word/number/punct boundaries so BPE can't merge
    across them), (3) convert to bytes, (4) BPE train, (5) build
    vocab, (6) add specials (<|endoftext|>, etc.), (7) serialize
    vocab+merges to disk. At inference: load them, encode/decode.
*/

#include "bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <corpus.txt> <out.bin>\n", argv[0]);
        return 1;
    }

    char *corpus = load_corpus(argv[1]);
    int num_chars = strlen(corpus);

    int *tokens = (int *)malloc(num_chars * sizeof(int));
    for (int i = 0; i < num_chars; i++) tokens[i] = (unsigned char)corpus[i];

    int *ids = tokens;
    int ids_len = num_chars;

    Merge merges[VOCAB_SIZE - 256];
    int num_merges = 0;

    for (int new_id = 256; new_id < VOCAB_SIZE; new_id++) {
        Pair *map = NULL;

        // count adjacent-pair frequencies
        for (int i = 0; i < ids_len - 1; i++) {
            int pair[2] = { ids[i], ids[i+1] };
            Pair *e;
            HASH_FIND(hh, map, pair, sizeof(pair), e);
            if (e) {
                e->value += 1;
            } else {
                e = calloc(1, sizeof(Pair));
                memcpy(e->key, pair, sizeof(pair));
                e->value = 1;
                HASH_ADD(hh, map, key, sizeof(e->key), e);
            }
        }

        // pick the most frequent pair
        Pair *cur, *tmp, *best = NULL;
        HASH_ITER(hh, map, cur, tmp) {
            if (!best || cur->value > best->value) best = cur;
        }
        if (!best) break;

        // replace it with the new id
        int n2 = 0;
        int *newids = merge(ids, best, ids_len, new_id, &n2);
        merges[num_merges++] = (Merge){best->key[0], best->key[1], new_id};

        HASH_ITER(hh, map, cur, tmp) { HASH_DEL(map, cur); free(cur); }

        if (ids != tokens) free(ids);
        ids = newids;
        ids_len = n2;
    }

    save_merges(argv[2], merges, num_merges);

    printf("\n--- BPE training done ---\n");
    printf("initial tokens length: %d\n", num_chars);
    printf("final ids length: %d\n", ids_len);
    printf("compression ratio: %.2fx\n", (double)num_chars / (double)ids_len);
    printf("merges saved to: %s (%d merges)\n", argv[2], num_merges);

    if (ids != tokens) free(ids);
    free(tokens);
    free(corpus);

    return 0;
}
