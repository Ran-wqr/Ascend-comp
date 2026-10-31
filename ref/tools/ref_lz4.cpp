// Reference LZ4 raw-block codec built on the official lz4 library.
// blocks file format: repeated [uint32 LE compressedLen][compressed bytes]
#include "lz4.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>

static std::vector<uint8_t> readAll(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "open fail %s\n", p); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(n);
    if (n) { size_t r = fread(v.data(), 1, n, f); (void)r; }
    fclose(f);
    return v;
}
static void putU32(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back(x & 0xff); v.push_back((x >> 8) & 0xff);
    v.push_back((x >> 16) & 0xff); v.push_back((x >> 24) & 0xff);
}

int main(int argc, char** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s compress|decompress <in> <out> [chunk]\n", argv[0]); return 2; }
    std::string mode = argv[1];
    std::vector<uint8_t> in = readAll(argv[2]);
    std::vector<uint8_t> out;

    if (mode == "compress") {
        uint32_t chunk = (argc >= 5) ? (uint32_t)atoi(argv[4]) : 65536u;
        size_t pos = 0;
        while (pos < in.size() || in.empty()) {
            size_t len = std::min((size_t)chunk, in.size() - pos);
            int bound = LZ4_compressBound((int)len);
            std::vector<char> dst(bound);
            int cn = LZ4_compress_default((const char*)in.data() + pos, dst.data(), (int)len, bound);
            if (cn <= 0) { fprintf(stderr, "lz4 compress failed\n"); return 3; }
            putU32(out, (uint32_t)cn);
            out.insert(out.end(), dst.begin(), dst.begin() + cn);
            pos += len;
            if (in.empty() || pos >= in.size()) break;
        }
    } else if (mode == "decompress") {
        size_t pos = 0;
        while (pos + 4 <= in.size()) {
            uint32_t cn = (uint32_t)in[pos] | ((uint32_t)in[pos+1] << 8) |
                          ((uint32_t)in[pos+2] << 16) | ((uint32_t)in[pos+3] << 24);
            pos += 4;
            std::vector<char> dst(64 * 1024 + 64);
            int dn = LZ4_decompress_safe((const char*)in.data() + pos, dst.data(), (int)cn, (int)dst.size());
            if (dn < 0) { fprintf(stderr, "lz4 decompress failed at pos=%zu cn=%u dn=%d\n", pos, cn, dn); return 4; }
            out.insert(out.end(), dst.begin(), dst.begin() + dn);
            pos += cn;
        }
    } else { fprintf(stderr, "bad mode\n"); return 2; }

    FILE* f = fopen(argv[3], "wb");
    if (!f) { fprintf(stderr, "open out fail\n"); return 2; }
    if (!out.empty()) { size_t w = fwrite(out.data(), 1, out.size(), f); (void)w; }
    fclose(f);
    fprintf(stdout, "[ref_lz4] %s -> %zu bytes\n", mode.c_str(), out.size());
    return 0;
}
