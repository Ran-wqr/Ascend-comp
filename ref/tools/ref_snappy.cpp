// Reference Snappy block codec built on the official google/snappy library.
// blocks file format: repeated [uint32 LE compressedLen][compressed bytes]
#include "snappy.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
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
        do {
            size_t len = std::min((size_t)chunk, in.size() - pos);
            size_t bound = snappy::MaxCompressedLength(len);
            std::vector<char> dst(bound + 16);
            size_t cn = 0;
            snappy::RawCompress((const char*)in.data() + pos, len, dst.data(), &cn);
            putU32(out, (uint32_t)cn);
            out.insert(out.end(), (uint8_t*)dst.data(), (uint8_t*)dst.data() + cn);
            pos += len;
        } while (!in.empty() && pos < in.size());
    } else if (mode == "decompress") {
        size_t pos = 0;
        while (pos + 4 <= in.size()) {
            uint32_t cn = (uint32_t)in[pos] | ((uint32_t)in[pos+1] << 8) |
                          ((uint32_t)in[pos+2] << 16) | ((uint32_t)in[pos+3] << 24);
            pos += 4;
            size_t ulen = 0;
            if (!snappy::GetUncompressedLength((const char*)in.data() + pos, cn, &ulen)) {
                fprintf(stderr, "snappy len fail\n"); return 4;
            }
            std::vector<char> dst(ulen ? ulen : 1);
            if (!snappy::RawUncompress((const char*)in.data() + pos, cn, dst.data())) {
                fprintf(stderr, "snappy decompress failed cn=%u\n", cn); return 4;
            }
            out.insert(out.end(), (uint8_t*)dst.data(), (uint8_t*)dst.data() + ulen);
            pos += cn;
        }
    } else { fprintf(stderr, "bad mode\n"); return 2; }

    FILE* f = fopen(argv[3], "wb");
    if (!f) { fprintf(stderr, "open out fail\n"); return 2; }
    if (!out.empty()) { size_t w = fwrite(out.data(), 1, out.size(), f); (void)w; }
    fclose(f);
    fprintf(stdout, "[ref_snappy] %s -> %zu bytes\n", mode.c_str(), out.size());
    return 0;
}
