#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>
#include <Metal/Metal.h>
#include <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

enum {
    VOCAB   = 2048,
    MERGES  = VOCAB - 256,
    CTX     = 64,
    EMB     = 128,
    HID     = 4 * EMB,
    HEADS   = 8,
    HEAD    = EMB / HEADS,
    LAYERS  = 4,
    FIELDS  = 11,
    SLOTS   = 5 + FIELDS * LAYERS,
    PARAMS  = 2 * VOCAB * EMB + CTX * EMB + 2 * EMB +
              LAYERS * (3 * EMB * EMB + 2 * EMB * HID + HID + 5 * EMB),
    SCRATCH = 2 * VOCAB + LAYERS * (16 * EMB + 4 * HID + 2 * HEADS * CTX)
};

#define TILE 8
#define MSL_TEXT(...) #__VA_ARGS__
#define MSL(...) MSL_TEXT(__VA_ARGS__)
#define GEMM_ARGS struct gemm_args {                                        \
    uint a;                                                                 \
    uint b;                                                                 \
    uint c;                                                                 \
    uint lda;                                                               \
    uint ldb;                                                               \
    uint ldc;                                                               \
    uint ta;                                                                \
    uint tb;                                                                \
    uint m;                                                                 \
    uint n;                                                                 \
    uint k;                                                                 \
};

GEMM_ARGS

static const char * gemm_source = MSL(using namespace metal; GEMM_ARGS
    kernel void gemm(device float * heap [[buffer(0)]],
                     constant gemm_args & g [[buffer(1)]],
                     uint2 at [[thread_position_in_grid]]) {
        uint ars = g.ta ? 1 : g.lda;
        uint acs = g.ta ? g.lda : 1;
        uint brs = g.tb ? 1 : g.ldb;
        uint i = at.y * TILE;
        uint4 j = at.x * 4 + uint4(0, 1, 2, 3);
        uint4 bj = g.b + min(j, g.n - 1) * (g.tb ? g.ldb : 1);
        float4 sum[TILE] = {};
        for (uint p = 0; p < g.k; p++) {
            uint4 bp = bj + p * brs;
            float4 b = float4(heap[bp.x], heap[bp.y], heap[bp.z], heap[bp.w]);
            for (uint r = 0; r < TILE; r++) {
                sum[r] += heap[g.a + min(i + r, g.m - 1) * ars + p * acs] * b;
            }
        }
        for (uint r = 0; r < TILE * 4; r++) {
            uint row = i + r / 4;
            uint column = j[r % 4];
            if (row < g.m && column < g.n) {
                heap[g.c + row * g.ldc + column] += sum[r / 4][r % 4];
            }
        }
    });

struct backend {
    const char * name;
    void (* fn)(const struct gemm_args * g);
    int32_t min;
};

struct merge {
    int32_t p0;
    int32_t p1;
};

struct tokenizer {
    int32_t count;
    struct merge merges[MERGES];
};

struct affine {
    float * scale;
    float * shift;
};

struct layer {
    struct affine ln1;
    float * wq;
    float * wk;
    float * wv;
    struct affine ln2;
    float * lin1;
    float * b1;
    float * lin2;
    float * b2;
};

union weights {
    struct {
        float * tok;
        float * pos;
        struct layer layers[LAYERS];
        struct affine lnf;
        float * head;
    };
    float * slot[SLOTS];
};

struct norm {
    float * y;
    float * xhat;
    float * r;
};

struct layer_cache {
    struct norm n1;
    float * q;
    float * k;
    float * v;
    float * att;
    struct norm n2;
    float * pre;
    float * post;
};

struct cache {
    struct layer_cache layers[LAYERS];
    struct norm nf;
};

_Static_assert(sizeof(union weights) == SLOTS * sizeof(float *), "slots");

static int32_t args_count;
static char ** args;
static float * heap;
static size_t heap_size;
static size_t heap_used;
static int32_t gpu_share = 30;
static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static id<MTLBuffer> heap_buffer;
static id<MTLComputePipelineState> pipeline;

static void must(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "slm: %s failed\n", what);
        exit(1);
    }
}

static const char * option(const char * key, const char * fallback) {
    const char * value = fallback;
    const size_t n = strlen(key);
    for (int32_t i = 1; i < args_count; i++) {
        if (strncmp(args[i], key, n) == 0 && args[i][n] == '=') {
            value = args[i] + n + 1;
        }
    }
    return value;
}

static int32_t option_int(const char * key, int32_t fallback) {
    const char * value = option(key, NULL);
    return value != NULL ? atoi(value) : fallback;
}

static double seconds(void) {
    struct timespec t = { 0 };
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int32_t min_of(int32_t a, int32_t b) {
    return a < b ? a : b;
}

static float uniform(float scale) {
    return ((float)rand() / (float)RAND_MAX) * 2.0f * scale - scale;
}

static void heap_open(int32_t tokens) {
    const size_t page = (size_t)getpagesize();
    const size_t floats = 4 * (size_t)PARAMS + (size_t)tokens * SCRATCH;
    const size_t bytes = (floats * sizeof(float) + page - 1) / page * page;
    void * p = NULL;
    must(posix_memalign(&p, page, bytes) == 0, "heap");
    heap = p;
    heap_size = bytes / sizeof(float);
}

static float * take(size_t n) {
    assert(heap_used + n <= heap_size);
    float * p = heap + heap_used;
    heap_used += n;
    memset(p, 0, n * sizeof(float));
    return p;
}

static void gemm_cpu(const struct gemm_args * g) {
    const uint ars = g->ta ? 1 : g->lda;
    const uint acs = g->ta ? g->lda : 1;
    const uint brs = g->tb ? 1 : g->ldb;
    const uint bcs = g->tb ? g->ldb : 1;
    const float * a = heap + g->a;
    const float * b = heap + g->b;
    float * c = heap + g->c;
    for (uint i = 0; i < g->m; i++) {
        for (uint p = 0; p < g->k; p++) {
            const float v = a[i * ars + p * acs];
            for (uint j = 0; j < g->n; j++) {
                c[i * g->ldc + j] += v * b[p * brs + j * bcs];
            }
        }
    }
}

static void gemm_blas(const struct gemm_args * g) {
    cblas_sgemm(CblasRowMajor, g->ta ? CblasTrans : CblasNoTrans,
                g->tb ? CblasTrans : CblasNoTrans, g->m, g->n, g->k, 1.0f,
                heap + g->a, g->lda, heap + g->b, g->ldb, 1.0f,
                heap + g->c, g->ldc);
}

static void gpu_open(void) {
    NSError * error = nil;
    device = MTLCreateSystemDefaultDevice();
    must(device != nil, "Metal device");
    queue = [device newCommandQueue];
    heap_buffer = [device newBufferWithBytesNoCopy:heap
                                            length:heap_size * sizeof(float)
                                           options:MTLResourceStorageModeShared
                                       deallocator:nil];
    must(heap_buffer != nil, "Metal heap buffer");
    id<MTLLibrary> library = [device newLibraryWithSource:@(gemm_source)
                                                  options:nil
                                                    error:&error];
    must(library != nil, error.localizedDescription.UTF8String);
    id<MTLFunction> function = [library newFunctionWithName:@"gemm"];
    pipeline = [device newComputePipelineStateWithFunction:function
                                                     error:&error];
    must(pipeline != nil, error.localizedDescription.UTF8String);
    gpu_share = option_int("share", gpu_share);
    must(0 <= gpu_share && gpu_share <= 100, "share");
}

static void gpu_wait(id<MTLCommandBuffer> commands) {
    [commands waitUntilCompleted];
    must(commands.error == nil, "GPU command buffer");
}

static id<MTLCommandBuffer> metal_submit(const struct gemm_args * g) {
    id<MTLCommandBuffer> commands = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commands computeCommandEncoder];
    const NSUInteger w = pipeline.threadExecutionWidth;
    const NSUInteger h = pipeline.maxTotalThreadsPerThreadgroup / w;
    const MTLSize groups = {
        ((g->n + 3) / 4 + w - 1) / w, ((g->m + TILE - 1) / TILE + h - 1) / h, 1
    };
    const MTLSize group = { w, h, 1 };
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:heap_buffer offset:0 atIndex:0];
    [encoder setBytes:g length:sizeof(*g) atIndex:1];
    [encoder dispatchThreadgroups:groups threadsPerThreadgroup:group];
    [encoder endEncoding];
    [commands commit];
    return commands;
}

static void gemm_metal(const struct gemm_args * g) {
    @autoreleasepool {
        gpu_wait(metal_submit(g));
    }
}

static void gemm_both(const struct gemm_args * g) {
    const uint rows = g->m * gpu_share / 100 / TILE * TILE;
    struct gemm_args top = *g;
    struct gemm_args rest = *g;
    top.m = rows;
    rest.m = g->m - rows;
    rest.a = g->a + rows * (g->ta ? 1 : g->lda);
    rest.c = g->c + rows * g->ldc;
    if (rows == 0) {
        gemm_blas(g);
    } else {
        @autoreleasepool {
            id<MTLCommandBuffer> commands = metal_submit(&top);
            gemm_blas(&rest);
            gpu_wait(commands);
        }
    }
}

static MPSMatrix * mps_matrix(uint at, uint rows, uint columns, uint ld) {
    MPSMatrixDescriptor * descriptor =
        [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                              columns:columns
                                             rowBytes:ld * sizeof(float)
                                             dataType:MPSDataTypeFloat32];
    return [[MPSMatrix alloc] initWithBuffer:heap_buffer
                                      offset:at * sizeof(float)
                                  descriptor:descriptor];
}

static void gemm_mps(const struct gemm_args * g) {
    @autoreleasepool {
        MPSMatrix * a = mps_matrix(g->a, g->ta ? g->k : g->m,
                                   g->ta ? g->m : g->k, g->lda);
        MPSMatrix * b = mps_matrix(g->b, g->tb ? g->n : g->k,
                                   g->tb ? g->k : g->n, g->ldb);
        MPSMatrix * c = mps_matrix(g->c, g->m, g->n, g->ldc);
        MPSMatrixMultiplication * kernel =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                              transposeLeft:g->ta
                                             transposeRight:g->tb
                                                 resultRows:g->m
                                              resultColumns:g->n
                                            interiorColumns:g->k
                                                      alpha:1.0
                                                       beta:1.0];
        id<MTLCommandBuffer> commands = [queue commandBuffer];
        [kernel encodeToCommandBuffer:commands
                           leftMatrix:a
                          rightMatrix:b
                         resultMatrix:c];
        [commands commit];
        gpu_wait(commands);
    }
}

static const struct backend backends[] = {
    { "cpu",   gemm_cpu,   0 },
    { "blas",  gemm_blas,  0 },
    { "metal", gemm_metal, 1 << 22 },
    { "mps",   gemm_mps,   1 << 22 },
    { "both",  gemm_both,  1 << 22 }
};

enum { BACKENDS = sizeof(backends) / sizeof(backends[0]) };

static struct backend backend;

static void backend_select(void) {
    const char * name = option("gemm", "blas");
    int32_t found = BACKENDS;
    for (int32_t i = 0; i < BACKENDS; i++) {
        if (strcmp(backends[i].name, name) == 0) {
            found = i;
        }
    }
    must(found < BACKENDS, name);
    backend = backends[found];
    backend.min = option_int("min", backend.min);
    if (backends[found].min > 0) {
        gpu_open();
    }
}

static void gemm(bool ta, bool tb, int32_t m, int32_t n, int32_t k,
                 const float * a, int32_t lda, const float * b, int32_t ldb,
                 float * c, int32_t ldc) {
    const struct gemm_args g = {
        (uint)(a - heap), (uint)(b - heap), (uint)(c - heap),
        lda, ldb, ldc, ta, tb, m, n, k
    };
    if ((int64_t)m * n * k < backend.min) {
        gemm_blas(&g);
    } else {
        backend.fn(&g);
    }
}

static void linear(const float * x, const float * w, float * y,
                   int32_t n, int32_t in, int32_t out) {
    gemm(false, false, n, out, in, x, in, w, out, y, out);
}

static void linear_backward(const float * x, const float * w,
                            const float * dy, float * dw, float * dx,
                            int32_t n, int32_t in, int32_t out) {
    gemm(true, false, in, out, n, x, in, dy, out, dw, out);
    gemm(false, true, n, in, out, dy, out, w, out, dx, in);
}

static void add_rows(float * y, const float * row, int32_t n, int32_t d) {
    for (int32_t i = 0; i < n * d; i++) {
        y[i] += row[i % d];
    }
}

static void sum_rows(const float * y, float * row, int32_t n, int32_t d) {
    for (int32_t i = 0; i < n * d; i++) {
        row[i % d] += y[i];
    }
}

static void softmax(float * x, int32_t n) {
    float max = x[0];
    for (int32_t i = 1; i < n; i++) {
        max = x[i] > max ? x[i] : max;
    }
    float sum = 0.0f;
    for (int32_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - max);
        sum += x[i];
    }
    for (int32_t i = 0; i < n; i++) {
        x[i] /= sum;
    }
}

static float gelu(float x) {
    const float c = sqrtf(2.0f / (float)M_PI);
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

static float gelu_prime(float x) {
    const float c = sqrtf(2.0f / (float)M_PI);
    const float t = tanhf(c * (x + 0.044715f * x * x * x));
    const float slope = c * (1.0f + 3.0f * 0.044715f * x * x);
    return 0.5f * (1.0f + t) + 0.5f * x * (1.0f - t * t) * slope;
}

static int32_t slot_size(int32_t i) {
    static const int32_t fields[FIELDS] = {
        EMB, EMB, EMB * EMB, EMB * EMB, EMB * EMB,
        EMB, EMB, EMB * HID, HID, HID * EMB, EMB
    };
    int32_t size = EMB;
    if (i == 0 || i == SLOTS - 1) {
        size = VOCAB * EMB;
    } else if (i == 1) {
        size = CTX * EMB;
    } else if (i < SLOTS - 3) {
        size = fields[(i - 2) % FIELDS];
    }
    return size;
}

static union weights weights_take(void) {
    union weights w = { 0 };
    for (int32_t i = 0; i < SLOTS; i++) {
        w.slot[i] = take(slot_size(i));
    }
    assert(w.head + VOCAB * EMB == w.tok + PARAMS);
    return w;
}

static void fill(float * p, int32_t n, float value) {
    for (int32_t i = 0; i < n; i++) {
        p[i] = value;
    }
}

static void randomize(float * p, int32_t n, float scale) {
    for (int32_t i = 0; i < n; i++) {
        p[i] = uniform(scale);
    }
}

static union weights weights_new(void) {
    union weights w = weights_take();
    for (int32_t i = 0; i < SLOTS; i++) {
        randomize(w.slot[i], slot_size(i) > HID ? slot_size(i) : 0, 0.02f);
    }
    for (int32_t i = 0; i < LAYERS; i++) {
        fill(w.layers[i].ln1.scale, EMB, 1.0f);
        fill(w.layers[i].ln2.scale, EMB, 1.0f);
    }
    fill(w.lnf.scale, EMB, 1.0f);
    return w;
}

static bool weights_load(union weights * w, const char * path) {
    FILE * f = fopen(path, "rb");
    const bool ok = f != NULL &&
                    fread(w->tok, sizeof(float), PARAMS, f) == PARAMS;
    if (f != NULL) {
        fclose(f);
    }
    return ok;
}

static void weights_save(const union weights * w, const char * path) {
    FILE * f = fopen(path, "wb");
    must(f != NULL, path);
    must(fwrite(w->tok, sizeof(float), PARAMS, f) == PARAMS, path);
    fclose(f);
}

static struct norm norm_forward(const struct affine * a, const float * x,
                                int32_t n) {
    const struct norm c = { take(n * EMB), take(n * EMB), take(n) };
    for (int32_t i = 0; i < n; i++) {
        const float * row = x + i * EMB;
        float mean = 0.0f;
        for (int32_t j = 0; j < EMB; j++) {
            mean += row[j];
        }
        mean /= EMB;
        float var = 0.0f;
        for (int32_t j = 0; j < EMB; j++) {
            var += (row[j] - mean) * (row[j] - mean);
        }
        c.r[i] = 1.0f / sqrtf(var / EMB + 1e-5f);
        for (int32_t j = 0; j < EMB; j++) {
            const float xhat = (row[j] - mean) * c.r[i];
            c.xhat[i * EMB + j] = xhat;
            c.y[i * EMB + j] = a->scale[j] * xhat + a->shift[j];
        }
    }
    return c;
}

static void norm_backward(const struct affine * a, struct affine * d,
                          const struct norm * c, const float * dy,
                          float * dx, int32_t n) {
    for (int32_t i = 0; i < n; i++) {
        const float * g = dy + i * EMB;
        const float * xhat = c->xhat + i * EMB;
        float s1 = 0.0f;
        float s2 = 0.0f;
        for (int32_t j = 0; j < EMB; j++) {
            d->shift[j] += g[j];
            d->scale[j] += g[j] * xhat[j];
            s1 += g[j] * a->scale[j];
            s2 += g[j] * a->scale[j] * xhat[j];
        }
        for (int32_t j = 0; j < EMB; j++) {
            dx[i * EMB + j] += c->r[i] * (g[j] * a->scale[j] - s1 / EMB -
                                          xhat[j] * s2 / EMB);
        }
    }
}

static void causal_softmax(float * att, int32_t t) {
    const float scale = 1.0f / sqrtf((float)HEAD);
    for (int32_t q = 0; q < t; q++) {
        float * row = att + q * t;
        for (int32_t k = 0; k <= q; k++) {
            row[k] *= scale;
        }
        softmax(row, q + 1);
        fill(row + q + 1, t - q - 1, 0.0f);
    }
}

static void causal_softmax_backward(const float * att, float * d, int32_t t) {
    const float scale = 1.0f / sqrtf((float)HEAD);
    for (int32_t q = 0; q < t; q++) {
        const float * p = att + q * t;
        float * g = d + q * t;
        float dot = 0.0f;
        for (int32_t k = 0; k < t; k++) {
            dot += p[k] * g[k];
        }
        for (int32_t k = 0; k < t; k++) {
            g[k] = p[k] * (g[k] - dot) * scale;
        }
    }
}

static int32_t head_at(int32_t i, int32_t t) {
    return (i / HEADS) * t * EMB + (i % HEADS) * HEAD;
}

static void attention(const struct layer * w, struct layer_cache * c,
                      float * x, int32_t b, int32_t t) {
    const int32_t n = b * t;
    c->q = take(n * EMB);
    c->k = take(n * EMB);
    c->v = take(n * EMB);
    c->att = take(b * HEADS * t * t);
    linear(c->n1.y, w->wq, c->q, n, EMB, EMB);
    linear(c->n1.y, w->wk, c->k, n, EMB, EMB);
    linear(c->n1.y, w->wv, c->v, n, EMB, EMB);
    for (int32_t i = 0; i < b * HEADS; i++) {
        const int32_t at = head_at(i, t);
        float * att = c->att + i * t * t;
        gemm(false, true, t, t, HEAD, c->q + at, EMB, c->k + at, EMB, att, t);
        causal_softmax(att, t);
        gemm(false, false, t, HEAD, t, att, t, c->v + at, EMB, x + at, EMB);
    }
}

static float * attention_backward(const struct layer * w, struct layer * d,
                                  const struct layer_cache * c,
                                  const float * g, int32_t b, int32_t t) {
    const int32_t n = b * t;
    float * dq = take(n * EMB);
    float * dk = take(n * EMB);
    float * dv = take(n * EMB);
    float * datt = take(b * HEADS * t * t);
    float * dx = take(n * EMB);
    for (int32_t i = 0; i < b * HEADS; i++) {
        const int32_t at = head_at(i, t);
        const float * att = c->att + i * t * t;
        float * ds = datt + i * t * t;
        gemm(false, true, t, t, HEAD, g + at, EMB, c->v + at, EMB, ds, t);
        gemm(true, false, t, HEAD, t, att, t, g + at, EMB, dv + at, EMB);
        causal_softmax_backward(att, ds, t);
        gemm(false, false, t, HEAD, t, ds, t, c->k + at, EMB, dq + at, EMB);
        gemm(true, false, t, HEAD, t, ds, t, c->q + at, EMB, dk + at, EMB);
    }
    linear_backward(c->n1.y, w->wq, dq, d->wq, dx, n, EMB, EMB);
    linear_backward(c->n1.y, w->wk, dk, d->wk, dx, n, EMB, EMB);
    linear_backward(c->n1.y, w->wv, dv, d->wv, dx, n, EMB, EMB);
    return dx;
}

static void feed_forward(const struct layer * w, struct layer_cache * c,
                         float * x, int32_t n) {
    c->pre = take(n * HID);
    c->post = take(n * HID);
    add_rows(c->pre, w->b1, n, HID);
    linear(c->n2.y, w->lin1, c->pre, n, EMB, HID);
    for (int32_t i = 0; i < n * HID; i++) {
        c->post[i] = gelu(c->pre[i]);
    }
    add_rows(x, w->b2, n, EMB);
    linear(c->post, w->lin2, x, n, HID, EMB);
}

static float * feed_backward(const struct layer * w, struct layer * d,
                             const struct layer_cache * c, const float * g,
                             int32_t n) {
    float * dh = take(n * HID);
    float * dx = take(n * EMB);
    sum_rows(g, d->b2, n, EMB);
    linear_backward(c->post, w->lin2, g, d->lin2, dh, n, HID, EMB);
    for (int32_t i = 0; i < n * HID; i++) {
        dh[i] *= gelu_prime(c->pre[i]);
    }
    sum_rows(dh, d->b1, n, HID);
    linear_backward(c->n2.y, w->lin1, dh, d->lin1, dx, n, EMB, HID);
    return dx;
}

static float * forward(const union weights * w, struct cache * c,
                       const int32_t * ids, int32_t b, int32_t t) {
    const int32_t n = b * t;
    float * x = take(n * EMB);
    float * logits = take(n * VOCAB);
    for (int32_t i = 0; i < n * EMB; i++) {
        x[i] = w->tok[ids[i / EMB] * EMB + i % EMB] + w->pos[i % (t * EMB)];
    }
    for (int32_t i = 0; i < LAYERS; i++) {
        c->layers[i].n1 = norm_forward(&w->layers[i].ln1, x, n);
        attention(&w->layers[i], &c->layers[i], x, b, t);
        c->layers[i].n2 = norm_forward(&w->layers[i].ln2, x, n);
        feed_forward(&w->layers[i], &c->layers[i], x, n);
    }
    c->nf = norm_forward(&w->lnf, x, n);
    linear(c->nf.y, w->head, logits, n, EMB, VOCAB);
    return logits;
}

static double cross_entropy(float * logits, const int32_t * targets,
                            int32_t n) {
    double sum = 0.0;
    for (int32_t i = 0; i < n; i++) {
        softmax(logits + i * VOCAB, VOCAB);
        sum += log((double)logits[i * VOCAB + targets[i]]);
    }
    return -sum / n;
}

static void backward(const union weights * w, union weights * d,
                     const struct cache * c, float * probs,
                     const int32_t * ids, int32_t b, int32_t t) {
    const int32_t n = b * t;
    float * dnf = take(n * EMB);
    float * g = take(n * EMB);
    for (int32_t i = 0; i < n; i++) {
        probs[i * VOCAB + ids[i + 1]] -= 1.0f;
    }
    for (int32_t i = 0; i < n * VOCAB; i++) {
        probs[i] /= (float)n;
    }
    linear_backward(c->nf.y, w->head, probs, d->head, dnf, n, EMB, VOCAB);
    norm_backward(&w->lnf, &d->lnf, &c->nf, dnf, g, n);
    for (int32_t i = LAYERS - 1; i >= 0; i--) {
        const struct layer * wi = &w->layers[i];
        const struct layer_cache * ci = &c->layers[i];
        struct layer * di = &d->layers[i];
        const float * dn2 = feed_backward(wi, di, ci, g, n);
        norm_backward(&wi->ln2, &di->ln2, &ci->n2, dn2, g, n);
        const float * dn1 = attention_backward(wi, di, ci, g, b, t);
        norm_backward(&wi->ln1, &di->ln1, &ci->n1, dn1, g, n);
    }
    for (int32_t i = 0; i < n * EMB; i++) {
        d->tok[ids[i / EMB] * EMB + i % EMB] += g[i];
        d->pos[i % (t * EMB)] += g[i];
    }
}

static double step(const union weights * w, union weights * d,
                   const int32_t * ids, int32_t b, int32_t t) {
    const size_t mark = heap_used;
    struct cache c = { 0 };
    float * logits = forward(w, &c, ids, b, t);
    const double loss = cross_entropy(logits, ids + 1, b * t);
    if (d != NULL) {
        memset(d->tok, 0, PARAMS * sizeof(float));
        backward(w, d, &c, logits, ids, b, t);
    }
    heap_used = mark;
    return loss;
}

static int32_t windows_of(int32_t n) {
    return (n - 1 - CTX) / CTX + 1;
}

static double evaluate(const union weights * w, const int32_t * ids,
                       int32_t n, int32_t batch) {
    const int32_t windows = windows_of(n);
    double sum = 0.0;
    for (int32_t i = 0; i < windows; i += batch) {
        sum += step(w, NULL, ids + i * CTX, min_of(batch, windows - i), CTX);
    }
    return sum / ((windows + batch - 1) / batch);
}

static void adamw(float * w, const float * g, float * m, float * v,
                  int32_t t) {
    const float lr = 1e-3f;
    const float b1 = 0.9f;
    const float b2 = 0.999f;
    const float c1 = 1.0f - powf(b1, (float)t);
    const float c2 = 1.0f - powf(b2, (float)t);
    for (int32_t i = 0; i < PARAMS; i++) {
        m[i] = b1 * m[i] + (1.0f - b1) * g[i];
        v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
        w[i] -= lr * (m[i] / c1 / (sqrtf(v[i] / c2) + 1e-8f) + 0.1f * w[i]);
    }
}

static char * file_read(const char * path, int32_t * size) {
    FILE * f = fopen(path, "rb");
    must(f != NULL, path);
    fseek(f, 0, SEEK_END);
    const size_t n = (size_t)ftell(f);
    rewind(f);
    char * text = malloc(n + 1);
    must(text != NULL && fread(text, 1, n, f) == n, path);
    text[n] = 0;
    fclose(f);
    *size = (int32_t)n;
    return text;
}

static void tokenizer_load(struct tokenizer * tk, const char * path) {
    char magic[4] = { 0 };
    FILE * f = fopen(path, "rb");
    memset(tk, 0, sizeof(*tk));
    must(f != NULL, path);
    must(fread(magic, 1, 4, f) == 4 && memcmp(magic, "BPE1", 4) == 0, path);
    must(fread(&tk->count, sizeof(int32_t), 1, f) == 1, path);
    must(0 <= tk->count && tk->count <= MERGES, path);
    const size_t n = (size_t)tk->count;
    must(fread(tk->merges, sizeof(struct merge), n, f) == n, path);
    for (int32_t i = 0; i < tk->count; i++) {
        const struct merge m = tk->merges[i];
        must(0 <= m.p0 && m.p0 < 256 + i && 0 <= m.p1 && m.p1 < 256 + i, path);
    }
    fclose(f);
}

static void tokenizer_save(const struct tokenizer * tk, const char * path) {
    const size_t n = (size_t)tk->count;
    FILE * f = fopen(path, "wb");
    must(f != NULL, path);
    must(fwrite("BPE1", 1, 4, f) == 4, path);
    must(fwrite(&tk->count, sizeof(int32_t), 1, f) == 1, path);
    must(fwrite(tk->merges, sizeof(struct merge), n, f) == n, path);
    fclose(f);
}

static int32_t * bytes_to_ids(const char * text, int32_t n, int32_t room) {
    int32_t * ids = malloc((size_t)(n + room) * sizeof(int32_t));
    must(ids != NULL, "ids");
    for (int32_t i = 0; i < n; i++) {
        ids[i] = (uint8_t)text[i];
    }
    return ids;
}

static int32_t merge_pair(int32_t * ids, int32_t n, struct merge m,
                          int32_t id) {
    int32_t k = 0;
    int32_t i = 0;
    while (i < n) {
        const bool hit = i + 1 < n && ids[i] == m.p0 && ids[i + 1] == m.p1;
        ids[k] = hit ? id : ids[i];
        k++;
        i += hit ? 2 : 1;
    }
    return k;
}

static int32_t pair_best(const int32_t * ids, int32_t n, int32_t * counts) {
    int32_t best = 0;
    int32_t max = 0;
    for (int32_t i = 0; i + 1 < n; i++) {
        counts[ids[i] * VOCAB + ids[i + 1]]++;
    }
    for (int32_t i = 0; i + 1 < n; i++) {
        int32_t * count = &counts[ids[i] * VOCAB + ids[i + 1]];
        best = *count > max ? i : best;
        max = *count > max ? *count : max;
        *count = 0;
    }
    return best;
}

static int32_t encode(const struct tokenizer * tk, int32_t * ids, int32_t n) {
    for (int32_t i = 0; i < tk->count; i++) {
        n = merge_pair(ids, n, tk->merges[i], 256 + i);
    }
    return n;
}

static void token_print(const struct tokenizer * tk, int32_t id) {
    if (id < 256) {
        putchar(id);
    } else {
        token_print(tk, tk->merges[id - 256].p0);
        token_print(tk, tk->merges[id - 256].p1);
    }
}

static int32_t next_token(const union weights * w, const int32_t * ids,
                          int32_t n) {
    const size_t mark = heap_used;
    const int32_t t = min_of(n, CTX);
    struct cache c = { 0 };
    const float * last = forward(w, &c, ids + n - t, 1, t) + (t - 1) * VOCAB;
    int32_t best = 0;
    for (int32_t i = 1; i < VOCAB; i++) {
        best = last[i] > last[best] ? i : best;
    }
    heap_used = mark;
    return best;
}

static void generate(const union weights * w, const struct tokenizer * tk,
                     const char * prompt, int32_t more) {
    const int32_t bytes = (int32_t)strlen(prompt);
    must(bytes > 0 && more >= 0, "prompt or tokens");
    int32_t * ids = bytes_to_ids(prompt, bytes, more);
    int32_t n = encode(tk, ids, bytes);
    for (int32_t end = n + more; n < end; n++) {
        ids[n] = next_token(w, ids, n);
    }
    for (int32_t i = 0; i < n; i++) {
        token_print(tk, ids[i]);
    }
    putchar('\n');
    free(ids);
}

static int32_t bpe(void) {
    const char * path = args[3];
    int32_t bytes = 0;
    char * text = file_read(args[2], &bytes);
    int32_t * ids = bytes_to_ids(text, bytes, 0);
    int32_t * counts = calloc(VOCAB * VOCAB, sizeof(int32_t));
    must(counts != NULL, "counts");
    struct tokenizer tk = { 0 };
    int32_t n = bytes;
    while (tk.count < MERGES && n > 1) {
        const int32_t i = pair_best(ids, n, counts);
        const struct merge m = { ids[i], ids[i + 1] };
        tk.merges[tk.count] = m;
        n = merge_pair(ids, n, m, 256 + tk.count);
        tk.count++;
    }
    tokenizer_save(&tk, path);
    printf("%d bytes -> %d tokens, %d merges -> %s\n", bytes, n, tk.count,
           path);
    free(counts);
    free(ids);
    free(text);
    return 0;
}

static int32_t train(void) {
    const int32_t batch = option_int("batch", 2);
    const int32_t epochs = option_int("epochs", 30);
    const int32_t patience = option_int("patience", 1);
    struct tokenizer tk = { 0 };
    int32_t bytes = 0;
    must(batch > 0 && epochs > 0 && patience > 0, "batch, epochs, patience");
    tokenizer_load(&tk, args[2]);
    char * text = file_read(args[3], &bytes);
    int32_t * ids = bytes_to_ids(text, bytes, 0);
    const int32_t n = encode(&tk, ids, bytes);
    const int32_t fit = (int32_t)((float)n * 0.9f);
    const int32_t steps = windows_of(fit) / batch;
    must(steps > 0 && n - fit > CTX, "corpus size");
    heap_open(batch * CTX);
    backend_select();
    union weights w = weights_new();
    union weights d = weights_take();
    float * m = take(PARAMS);
    float * v = take(PARAMS);
    printf("%d tokens, %d steps of batch %d, gemm %s\n", n, steps, batch,
           backend.name);
    double best = INFINITY;
    int32_t stale = 0;
    for (int32_t epoch = 0; epoch < epochs && stale < patience; epoch++) {
        const double start = seconds();
        double sum = 0.0;
        for (int32_t i = 0; i < steps; i++) {
            sum += step(&w, &d, ids + i * batch * CTX, batch, CTX);
            adamw(w.tok, d.tok, m, v, epoch * steps + i + 1);
        }
        const double val = evaluate(&w, ids + fit, n - fit, batch);
        stale = val < best ? 0 : stale + 1;
        if (stale == 0) {
            best = val;
            weights_save(&w, "weights.bin");
        }
        printf("epoch %2d | train %.4f | val %.4f | %.1fs%s\n", epoch + 1,
               sum / steps, val, seconds() - start,
               stale == 0 ? " | saved" : "");
    }
    must(weights_load(&w, "weights.bin"), "weights.bin");
    printf("best train %.4f | val %.4f\n", evaluate(&w, ids, fit, batch),
           evaluate(&w, ids + fit, n - fit, batch));
    generate(&w, &tk, "I HAD always thought", 30);
    free(ids);
    free(text);
    return 0;
}

static int32_t run(void) {
    struct tokenizer tk = { 0 };
    tokenizer_load(&tk, args[2]);
    heap_open(CTX);
    backend_select();
    union weights w = weights_new();
    if (!weights_load(&w, "weights.bin")) {
        printf("no weights.bin, random weights\n");
    }
    generate(&w, &tk, args[3], option_int("tokens", 10));
    return 0;
}

static int32_t argmax_abs(const float * p, int32_t n) {
    int32_t best = 0;
    for (int32_t i = 1; i < n; i++) {
        best = fabsf(p[i]) > fabsf(p[best]) ? i : best;
    }
    return best;
}

static double numeric(const union weights * w, float * p,
                      const int32_t * ids, int32_t b) {
    const float eps = 1e-2f;
    const float saved = *p;
    *p = saved + eps;
    const double up = step(w, NULL, ids, b, CTX);
    *p = saved - eps;
    const double down = step(w, NULL, ids, b, CTX);
    *p = saved;
    return (up - down) / (2.0 * eps);
}

static int32_t check(void) {
    enum { B = 2 };
    heap_open(B * CTX);
    backend_select();
    union weights w = weights_new();
    union weights d = weights_take();
    int32_t ids[B * CTX + 1] = { 0 };
    for (int32_t i = 0; i <= B * CTX; i++) {
        ids[i] = rand() % VOCAB;
    }
    step(&w, &d, ids, B, CTX);
    int32_t faults = 0;
    for (int32_t i = 0; i < SLOTS; i++) {
        const int32_t at = argmax_abs(d.slot[i], slot_size(i));
        const double g = d.slot[i][at];
        const double ng = numeric(&w, &w.slot[i][at], ids, B);
        const double rel = fabs(g - ng) / (fabs(g) + fabs(ng) + 1e-12);
        const bool fault = fabs(g - ng) > 2e-2 * (fabs(g) + fabs(ng)) + 2e-5;
        faults += fault;
        printf("slot %2d | g %+.3e | numeric %+.3e | rel %.1e | %s\n", i,
               g, ng, rel, fault ? "FAULT" : "ok");
    }
    printf("check: %d fault(s) over %d slots, gemm %s\n", faults, SLOTS,
           backend.name);
    return faults;
}

static void bench_gemm(int32_t m, int32_t n, int32_t k, int32_t rounds) {
    const size_t mark = heap_used;
    float * a = take(m * k);
    float * b = take(k * n);
    float * c = take(m * n);
    float * expected = take(m * n);
    double best[BACKENDS] = { 0 };
    float diff[BACKENDS] = { 0 };
    randomize(a, m * k, 1.0f);
    randomize(b, k * n, 1.0f);
    backend = backends[0];
    gemm(false, false, m, n, k, a, k, b, n, expected, n);
    for (int32_t i = 0; i < rounds * BACKENDS; i++) {
        const int32_t at = i % BACKENDS;
        backend = backends[at];
        backend.min = 0;
        memset(c, 0, m * n * sizeof(float));
        const double start = seconds();
        gemm(false, false, m, n, k, a, k, b, n, c, n);
        const double spent = seconds() - start;
        best[at] = i < 2 * BACKENDS || spent < best[at] ? spent : best[at];
        for (int32_t j = 0; j < m * n; j++) {
            diff[at] = fmaxf(diff[at], fabsf(c[j] - expected[j]));
        }
    }
    for (int32_t i = 0; i < BACKENDS; i++) {
        printf("%5d x %4d x %4d | %-5s | %9.3f ms | %6.2f GFLOPS | "
               "diff %.1e\n", m, n, k, backends[i].name, best[i] * 1e3,
               2e-9 * m * n * k / best[i], diff[i]);
    }
    heap_used = mark;
}

static void bench_step(int32_t batch, int32_t rounds) {
    int32_t * ids = calloc((size_t)batch * CTX + 1, sizeof(int32_t));
    must(ids != NULL, "ids");
    double best[BACKENDS] = { 0 };
    union weights w = weights_new();
    union weights d = weights_take();
    for (int32_t i = 0; i < rounds * BACKENDS; i++) {
        const int32_t at = i % BACKENDS;
        backend = backends[at];
        backend.min = option_int("min", backend.min);
        const double start = seconds();
        step(&w, &d, ids, batch, CTX);
        const double spent = seconds() - start;
        best[at] = i < 2 * BACKENDS || spent < best[at] ? spent : best[at];
    }
    for (int32_t i = 0; i < BACKENDS; i++) {
        printf("step batch %d | %-5s | %9.1f ms\n", batch, backends[i].name,
               best[i] * 1e3);
    }
    free(ids);
}

static int32_t bench(void) {
    const int32_t batch = option_int("batch", 2);
    const int32_t rounds = option_int("rounds", 6);
    const int32_t n = batch * CTX;
    must(batch > 0 && rounds > 1, "batch or rounds");
    heap_open(n);
    gpu_open();
    bench_gemm(CTX, CTX, HEAD, rounds);
    bench_gemm(n, EMB, EMB, rounds);
    bench_gemm(n, HID, EMB, rounds);
    bench_gemm(n, EMB, HID, rounds);
    bench_gemm(n, VOCAB, EMB, rounds);
    bench_step(batch, rounds);
    return 0;
}

static int32_t usage(void) {
    fprintf(stderr, "usage: slm bpe <corpus> <merges.bin>\n"
                    "       slm train <merges.bin> <corpus>\n"
                    "       slm run <merges.bin> <prompt>\n"
                    "       slm check | bench\n"
                    "options: gemm=cpu|blas|metal|mps|both batch=2 epochs=30\n"
                    "         seed=N tokens=10 min=N share=30 rounds=6\n"
                    "         patience=1\n");
    return 1;
}

static const struct command {
    const char * name;
    int32_t needs;
    int32_t (* fn)(void);
} commands[] = {
    { "bpe",   4, bpe },
    { "train", 4, train },
    { "run",   4, run },
    { "check", 2, check },
    { "bench", 2, bench },
    { "",      0, usage }
};

enum { COMMANDS = sizeof(commands) / sizeof(commands[0]) };

int main(int argc, char ** argv) {
    const char * name = argc > 1 ? argv[1] : "";
    int32_t found = COMMANDS - 1;
    args_count = argc;
    args = argv;
    srand((unsigned)option_int("seed", (int32_t)time(NULL)));
    for (int32_t i = 0; i < COMMANDS - 1; i++) {
        if (strcmp(commands[i].name, name) == 0 && argc >= commands[i].needs) {
            found = i;
        }
    }
    return commands[found].fn() > 0;
}
