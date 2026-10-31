/*
 * common_ascend.h
 * Host-side helpers shared by the LZ4 / Snappy demos, tests and benchmarks.
 *
 * Error handling policy:
 *   - every ACL call must be checked; a failure prints API name + code + site
 *     and must make the program exit non-zero.
 *   - ACL_CHECK* return an error code from the enclosing function.
 *   - CHECK_* report the same information but only set a flag (for helpers
 *     that must release resources before returning).
 */
#ifndef NVCOMP_ASCEND_COMMON_H
#define NVCOMP_ASCEND_COMMON_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <sys/stat.h>

#include "acl/acl.h"

/* ---- file IO ---- */

static inline bool read_file(const std::string& path, std::vector<uint8_t>& out)
{
    struct stat st;
    if (stat(path.c_str(), &st) != 0) {
        fprintf(stderr, "[IO ERROR] stat failed: %s\n", path.c_str());
        return false;
    }
    if (!S_ISREG(st.st_mode)) {
        fprintf(stderr, "[IO ERROR] not a regular file: %s\n", path.c_str());
        return false;
    }
    if (st.st_size < 0) {
        fprintf(stderr, "[IO ERROR] negative size: %s\n", path.c_str());
        return false;
    }
    out.clear();
    out.resize((size_t)st.st_size);
    if (out.empty()) return true;                      /* empty file is legal */
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "[IO ERROR] open failed: %s\n", path.c_str());
        return false;
    }
    f.read((char*)out.data(), (std::streamsize)out.size());
    if (!f || (size_t)f.gcount() != out.size()) {       /* truncated read */
        fprintf(stderr, "[IO ERROR] short read: %s (got %lld of %zu)\n",
                path.c_str(), (long long)f.gcount(), out.size());
        out.clear();
        return false;
    }
    return true;
}

static inline bool write_file(const std::string& path, const void* data, size_t n)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        fprintf(stderr, "[IO ERROR] open for write failed: %s\n", path.c_str());
        return false;
    }
    if (n) f.write((const char*)data, (std::streamsize)n);
    if (!f) {
        fprintf(stderr, "[IO ERROR] short write: %s\n", path.c_str());
        return false;
    }
    return true;
}

/* ---- ACL error helpers ---- */

static inline void acl_report(const char* api, aclError e, const char* file, int line)
{
    fprintf(stderr, "[ACL ERROR] %s:%d  %s -> %d\n", file, line, api, (int)e);
}

/* return <err> from the enclosing function */
#define ACL_CHECK(expr)                                                          \
    do {                                                                         \
        aclError __e = (expr);                                                   \
        if (__e != ACL_SUCCESS) {                                                \
            acl_report(#expr, __e, __FILE__, __LINE__);                          \
            return 3;                                                            \
        }                                                                        \
    } while (0)

/* same, but returns a caller-provided value */
#define ACL_CHECK_RET(expr, ret)                                                 \
    do {                                                                         \
        aclError __e = (expr);                                                   \
        if (__e != ACL_SUCCESS) {                                                \
            acl_report(#expr, __e, __FILE__, __LINE__);                          \
            return (ret);                                                        \
        }                                                                        \
    } while (0)

/* set a bool flag, keep going so the caller can free resources first */
#define ACL_CHECK_FLAG(expr, flag)                                               \
    do {                                                                         \
        aclError __e = (expr);                                                   \
        if (__e != ACL_SUCCESS) {                                                \
            acl_report(#expr, __e, __FILE__, __LINE__);                          \
            (flag) = false;                                                      \
        }                                                                        \
    } while (0)

/* free without clobbering an earlier error: only report */
#define ACL_FREE(ptr)                                                            \
    do {                                                                         \
        if ((ptr) != nullptr) {                                                  \
            aclError __e = aclrtFree(ptr);                                       \
            if (__e != ACL_SUCCESS) acl_report("aclrtFree", __e, __FILE__, __LINE__); \
            (ptr) = nullptr;                                                     \
        }                                                                        \
    } while (0)

#endif // NVCOMP_ASCEND_COMMON_H
