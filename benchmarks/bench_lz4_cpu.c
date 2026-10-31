/*
 * bench_lz4_cpu.c -- CPU reference benchmark for the same block API the NPU
 * kernels implement.
 *
 *   usage: bench_lz4_cpu <file> <codec-label> [iters=10] [warmup=3] [blockKB=64] [csv]
 *
 * Design notes (fairness):
 *   - block API only: LZ4_compress_default / LZ4_decompress_safe, 64 KiB blocks
 *     (identical to the NPU CHUNK), no frame format, no checksums, single thread.
 *   - the compressed size is the sum of per-block compressed sizes, exactly like
 *     the NPU benchmark (no container header on either side).
 *   - tiny files are repeated inside one timed sample so that a 1-byte input does
 *     not turn into a timer-resolution measurement; `reps` is printed.
 *   - bit-exact round-trip verification; any mismatch => non-zero exit.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#include "lz4.h"

static double now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

static int cmp_double(const void* a, const void* b)
{
    double x = *(const double*)a, y = *(const double*)b;
    return (x < y) ? -1 : (x > y) ? 1 : 0;
}

typedef struct { double mean, med, mn, mx, sd; } Stat5;

static Stat5 stat5(double* v, int n)
{
    Stat5 r = {0, 0, 0, 0, 0};
    if (n <= 0) return r;
    qsort(v, (size_t)n, sizeof(double), cmp_double);
    double s = 0;
    for (int i = 0; i < n; i++) s += v[i];
    r.mean = s / n;
    r.mn = v[0];
    r.mx = v[n - 1];
    r.med = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    double var = 0;
    for (int i = 0; i < n; i++) var += (v[i] - r.mean) * (v[i] - r.mean);
    r.sd = (n > 1) ? sqrt(var / (n - 1)) : 0.0;
    return r;
}

static int read_file(const char* path, unsigned char** buf, size_t* len)
{
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[IO] open failed: %s\n", path); return 0; }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 0; }
    long n = ftell(f);
    if (n < 0) { fclose(f); return 0; }
    rewind(f);
    *len = (size_t)n;
    *buf = (unsigned char*)malloc(*len ? *len : 1);
    if (!*buf) { fclose(f); return 0; }
    if (*len && fread(*buf, 1, *len, f) != *len) { fclose(f); free(*buf); return 0; }
    fclose(f);
    return 1;
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        fprintf(stderr,
            "usage: %s <file> <codec-label> [iters=10] [warmup=3] [blockKB=64] [csv]\n",
            argv[0]);
        return 2;
    }
    const char* path  = argv[1];
    const char* label = argv[2];
    const int iters   = (argc > 3) ? atoi(argv[3]) : 10;
    const int warmup  = (argc > 4) ? atoi(argv[4]) : 3;
    const int blockKB = (argc > 5) ? atoi(argv[5]) : 64;
    const int csv     = (argc > 6) && (strcmp(argv[6], "csv") == 0);

    if (iters <= 0 || blockKB <= 0) { fprintf(stderr, "bad args\n"); return 2; }

    unsigned char* in = NULL;
    size_t len = 0;
    if (!read_file(path, &in, &len)) return 3;

    const size_t blockSize = (size_t)blockKB * 1024u;
    const size_t nblocks = (len + blockSize - 1u) / blockSize;
    const size_t nb = nblocks ? nblocks : 1u;
    const int bound = LZ4_compressBound((int)blockSize);

    unsigned char* comp = (unsigned char*)malloc((size_t)bound * nb);
    unsigned char* out  = (unsigned char*)malloc(len ? len : 1);
    int* csize = (int*)calloc(nb, sizeof(int));
    if (!comp || !out || !csize) { fprintf(stderr, "[MEM] oom\n"); return 3; }

    /* one full round-trip first: correctness + warm caches */
    size_t totalComp = 0;
    for (size_t b = 0; b < nb; b++) {
        size_t off = b * blockSize;
        int l = (int)((len - off < blockSize) ? (len - off) : blockSize);
        int c = LZ4_compress_default((const char*)in + off, (char*)comp + (size_t)b * bound, l, bound);
        if (c <= 0) { fprintf(stderr, "[LZ4] compress failed at block %zu\n", b); return 4; }
        csize[b] = c;
        totalComp += (size_t)c;
    }
    size_t produced = 0;
    for (size_t b = 0; b < nb; b++) {
        size_t off = b * blockSize;
        int l = (int)((len - off < blockSize) ? (len - off) : blockSize);
        int d = LZ4_decompress_safe((const char*)comp + (size_t)b * bound,
                                    (char*)out + off, csize[b], l);
        if (d != l) { fprintf(stderr, "[LZ4] decompress failed at block %zu (d=%d l=%d)\n", b, d, l); return 4; }
        produced += (size_t)d;
    }
    if (produced != len || (len && memcmp(in, out, len) != 0)) {
        fprintf(stderr, "[FAIL] round-trip mismatch (%zu vs %zu)\n", produced, len);
        return 1;
    }

    /* repeat tiny inputs inside one sample so the timer is not the signal */
    int reps = (int)((1u << 20) / (len ? len : 1));
    if (reps < 1) reps = 1;
    if (reps > 4096) reps = 4096;

    for (int w = 0; w < warmup; w++) {
        for (int r = 0; r < reps; r++) {
            for (size_t b = 0; b < nb; b++) {
                size_t off = b * blockSize;
                int l = (int)((len - off < blockSize) ? (len - off) : blockSize);
                LZ4_compress_default((const char*)in + off, (char*)comp + (size_t)b * bound, l, bound);
            }
        }
    }

    double* cs = (double*)malloc(sizeof(double) * (size_t)iters);
    double* ds = (double*)malloc(sizeof(double) * (size_t)iters);
    if (!cs || !ds) { fprintf(stderr, "[MEM] oom\n"); return 3; }

    for (int i = 0; i < iters; i++) {
        double t0 = now_ms();
        for (int r = 0; r < reps; r++) {
            for (size_t b = 0; b < nb; b++) {
                size_t off = b * blockSize;
                int l = (int)((len - off < blockSize) ? (len - off) : blockSize);
                LZ4_compress_default((const char*)in + off, (char*)comp + (size_t)b * bound, l, bound);
            }
        }
        cs[i] = (now_ms() - t0) / (double)reps;
    }
    for (int i = 0; i < iters; i++) {
        double t0 = now_ms();
        for (int r = 0; r < reps; r++) {
            for (size_t b = 0; b < nb; b++) {
                size_t off = b * blockSize;
                int l = (int)((len - off < blockSize) ? (len - off) : blockSize);
                LZ4_decompress_safe((const char*)comp + (size_t)b * bound,
                                    (char*)out + off, csize[b], l);
            }
        }
        ds[i] = (now_ms() - t0) / (double)reps;
    }

    Stat5 c = stat5(cs, iters), d = stat5(ds, iters);
    double ratio = len ? (double)totalComp / (double)len : 0.0;
    const char* base = strrchr(path, '/');
    base = base ? base + 1 : path;

    if (csv) {
        printf("%s,%s,1,%zu,%zu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,,OK\n",
               base, label, len, totalComp, ratio,
               c.mean, c.med, c.mn, c.mx, c.sd,
               d.mean, d.med, d.mn, d.mx, d.sd);
    } else {
        printf("%-30s %-10s %9zu %8.4f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f OK  (block=%zuB reps=%d)\n",
               base, label, len, ratio,
               c.med, c.mn, c.mx, c.sd, d.med, d.mn, d.sd, blockSize, reps);
    }

    free(cs); free(ds); free(csize); free(comp); free(out); free(in);
    return 0;
}
