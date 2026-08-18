# llm

A GPT-style language model written from scratch in C. No PyTorch, no autograd
library, a tensor struct over a flat `float *`, hand-derived backward passes,
and `-lm`. ~1.26M parameters, trained on Tiny Shakespeare.

Full writeup: **[Building an LLM from scratch in C](https://blog.bernardoolisan.com/posts/building-an-llm-from-scratch-in-c)**

## Build

```sh
gcc bpe.c bpe_runtime.c -o train_bpe
gcc llm.c model.c train.c tensor.c utils.c bpe_runtime.c -o llm -lm
```

## Run

The tokenizer comes first, the model can't do anything without `merges.bin`.

```sh
# 1. learn the BPE merges (vocab 2048 = 256 base bytes + 1792 merges)
./train_bpe tinyshakespeare.txt merges.bin

# 2. train (writes weights.bin)
./llm train merges.bin

# 3. generate
./llm run merges.bin "I HAD always thought"
```

`run` loads `weights.bin` if it's there, otherwise it generates from a random
init, which is exactly as useless as it sounds.

On a slow machine, swap `tinyshakespeare.txt` (~1.1MB) for `the-verdict.txt`
(~20KB) in `train()` in [llm.c](llm.c).

## Layout

| file | what's in it |
|---|---|
| [tensor.c](tensor.c) | flat array + shape + strides. matmul, transpose, permute, cat, slicing. |
| [bpe.c](bpe.c) / [bpe_runtime.c](bpe_runtime.c) | BPE trainer; encode/decode/vocab shared with the model. |
| [model.c](model.c) | forward pass — attention, feed-forward, layer norm, GELU, generation. |
| [train.c](train.c) | backward pass, dataloader, loss, AdamW, gradient checking, weight I/O. |
| [llm.c](llm.c) | entry point: `train` and `run`. |

Config lives in [model.h](model.h): 4 layers, 8 heads, 128 embedding dim,
64-token context. The parameter count is derived term by term in the comment
at the top of [llm.c](llm.c).

`merges.bin` and `weights.bin` are gitignored, both regenerate from the
sources and datasets here, in that order.
