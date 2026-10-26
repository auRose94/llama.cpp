// k9.cpp — K9 (ninths-grid) stream decoding, vendored from
// base9-quantization/experiments/{rans_fast.c,rans_fast.py,k9.py} and adapted
// to write llama.cpp materialized tensors. Semantics must stay bit-exact with
// the Python reference; see k9.h for the file/KV layout.

#include "k9.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// ggml-impl.h is not on the llama include path: use the public fp16 API and a
// local copy of the q8_0 block layout (validated against ggml-common.h).
#ifndef QK8_0
#define QK8_0 32
#endif
typedef struct {
    ggml_fp16_t d;         // delta
    int8_t      qs[QK8_0]; // quants
} block_q8_0;
static_assert(sizeof(block_q8_0) == sizeof(ggml_fp16_t) + QK8_0, "wrong q8_0 block size/padding");

// Local IQ4_NL copies (ggml-common.h/ggml-quants.c are not on this include
// path) — must stay byte-compatible with upstream: 16-entry codebook,
// 18-byte blocks (f16 scale + 16 packed nibble bytes), qs[j] = L[j] |
// (L[j+16] << 4).
#ifndef QK4_NL
#define QK4_NL 32
#endif
typedef struct {
    ggml_fp16_t d;        // delta
    uint8_t     qs[QK4_NL / 2];
} block_iq4_nl;
static const int8_t k9_iq4nl_values[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};
static_assert(sizeof(block_iq4_nl) == sizeof(ggml_fp16_t) + QK4_NL / 2, "wrong iq4_nl block size/padding");

namespace {

// ---------------------------------------------------------------------------
// rANS (Duda, byte renorm, M = 1<<12, 32-bit state in [2^24, 2^32))
// Mirrors experiments/rans_fast.c k9_rans_decode byte-for-byte, with a
// resumable state so streams can be decoded row-chunk by row-chunk.
// The final 4 bytes of the stream hold the initial state big-endian; renorm
// bytes are consumed backwards from slen-5, so the full stream must be
// contiguous in memory.

constexpr uint32_t K9_SCALE_BITS = 12; // M = 4096; every real .k9 stream uses this

uint16_t rd_u16(const uint8_t * p) {
    return (uint16_t) (p[0] | (p[1] << 8));
}

uint32_t rd_u32(const uint8_t * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

float rd_f32(const uint8_t * p) {
    uint32_t u = rd_u32(p);
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

struct k9_rans_decoder {
    const uint8_t * stream = nullptr;
    size_t         slen    = 0;
    long           pos     = 0;
    uint64_t       x       = 0;
    const uint64_t * freq  = nullptr;   // [n_syms]
    const uint64_t * cumul = nullptr;   // [n_syms], cumul[0] == 0, no trailing M
    const uint32_t * inv   = nullptr;   // [1<<K9_SCALE_BITS]
    uint32_t       mask    = (1u << K9_SCALE_BITS) - 1;

    void init(const uint8_t * stream_, size_t slen_,
              const uint64_t * freq_, const uint64_t * cumul_, const uint32_t * inv_) {
        if (slen_ < 5) {
            throw std::runtime_error("k9 rANS stream shorter than 5 bytes");
        }
        stream = stream_;
        slen   = slen_;
        freq   = freq_;
        cumul  = cumul_;
        inv    = inv_;
        x   = ((uint64_t) stream[slen - 4] << 24)
            | ((uint64_t) stream[slen - 3] << 16)
            | ((uint64_t) stream[slen - 2] << 8)
            |  (uint64_t) stream[slen - 1];
        pos = (long) slen - 5;
    }

    void decode(uint8_t * out, size_t n) {
        for (size_t i = 0; i < n; i++) {
            const uint32_t s = inv[x & mask];
            x = freq[s] * (x >> K9_SCALE_BITS) + (x & mask) - cumul[s];
            while (x < ((uint64_t) 1 << 24)) {
                x = (x << 8) | stream[pos--];
            }
            out[i] = (uint8_t) s;
        }
    }
};

struct rans_tables {
    std::vector<uint64_t> freq, cumul; // cumul excludes the trailing M
    std::vector<uint32_t> inv;         // [M]

    void build(const uint8_t * f_u16le, size_t n_syms) {
        freq.resize(n_syms);
        cumul.assign(n_syms, 0);
        uint64_t total = 0;
        for (size_t s = 0; s < n_syms; s++) {
            freq[s]  = rd_u16(f_u16le + 2 * s);
            cumul[s] = total;
            total   += freq[s];
        }
        if (total != ((uint64_t) 1 << K9_SCALE_BITS)) {
            throw std::runtime_error("k9 freq table does not sum to M = 4096");
        }
        inv.assign((size_t) 1 << K9_SCALE_BITS, 0);
        for (size_t s = 0; s < n_syms; s++) {
            for (uint64_t j = cumul[s]; j < cumul[s] + freq[s]; j++) {
                inv[j] = (uint32_t) s;
            }
        }
    }
};

} // anonymous namespace

bool k9_parse_directory(const uint8_t * data, size_t n_bytes,
                        std::map<std::string, k9_tensor_meta> & out, std::string & err) {
    size_t pos = 0;
    if (n_bytes < 2) {
        err = "k9.directory too short";
        return false;
    }
    const uint16_t n = rd_u16(data + pos); pos += 2;
    for (uint16_t i = 0; i < n; i++) {
        if (n_bytes - pos < 2) {
            err = "k9.directory truncated (name header)";
            return false;
        }
        const uint16_t name_len = rd_u16(data + pos); pos += 2;
        if (n_bytes - pos < name_len + 1 + 4*4) {
            err = "k9.directory truncated (record body)";
            return false;
        }
        k9_tensor_meta m;
        m.name.assign((const char *) data + pos, name_len); pos += name_len;
        const uint8_t rank = data[pos];
        pos += 1;
        if (rank != 2 || n_bytes - pos < 4*rank + 4 + 4*4) {
            err = "k9.directory: unsupported rank or truncated record";
            return false;
        }
        m.shape[0] = rd_u32(data + pos);
        m.shape[1] = rd_u32(data + pos + 4);
        pos += 4 * rank;
        m.k          = data[pos + 0];
        m.group      = data[pos + 1];
        m.scale_mode = data[pos + 2];
        m.perm_flag  = data[pos + 3];
        pos += 4;
        m.tbl  = rd_u32(data + pos + 0);
        m.dig  = rd_u32(data + pos + 4);
        m.sca  = rd_u32(data + pos + 8);
        m.plen = rd_u32(data + pos + 12);
        pos   += 16;
        if (m.group == 0 || m.k == 0 || m.k > 255) {
            err = "k9.directory: unsupported k/group for " + m.name;
            return false;
        }
        out[m.name] = std::move(m);
    }
    if (pos != n_bytes) {
        err = "k9.directory trailing bytes";
        return false;
    }
    return true;
}

bool k9_parse_directory2(const uint8_t * data, size_t n_bytes,
                         std::map<std::string, k9_tensor_meta> & out, std::string & err) {
    size_t pos = 0;
    if (n_bytes < 4) {
        err = "k9.directory2 truncated (header)";
        return false;
    }
    if (rd_u16(data) != 2) {
        err = "k9.directory2: unsupported stream version";
        return false;
    }
    const uint16_t n = rd_u16(data + pos + 2); pos += 4;
    for (uint16_t i = 0; i < n; i++) {
        if (n_bytes - pos < 2) {
            err = "k9.directory2 truncated (name header)";
            return false;
        }
        const uint16_t name_len = rd_u16(data + pos); pos += 2;
        if (n_bytes - pos < name_len + 1 + 4*4) {
            err = "k9.directory2 truncated (record body)";
            return false;
        }
        k9_tensor_meta m;
        m.name.assign((const char *) data + pos, name_len); pos += name_len;
        const uint8_t rank = data[pos];
        pos += 1;
        if (rank != 2 || n_bytes - pos < 4*rank + 5 + 4*4) {
            err = "k9.directory2: unsupported rank or truncated record";
            return false;
        }
        m.shape[0] = rd_u32(data + pos);
        m.shape[1] = rd_u32(data + pos + 4);
        pos += 4 * rank;
        m.k          = data[pos + 0];
        m.group      = data[pos + 1];
        m.scale_mode = data[pos + 2];
        m.perm_flag  = data[pos + 3];
        m.lanes      = data[pos + 4];
        pos += 5;
        if (m.group == 0 || m.k == 0 || m.k > 255 || m.lanes == 0 || m.lanes > 64) {
            err = "k9.directory2: unsupported k/group/lanes for " + m.name;
            return false;
        }
        m.tbl  = rd_u32(data + pos + 0);
        m.dig  = rd_u32(data + pos + 4);
        m.sca  = rd_u32(data + pos + 8);
        m.plen = rd_u32(data + pos + 12);
        pos   += 16;
        out[m.name] = std::move(m);
    }
    if (pos != n_bytes) {
        err = "k9.directory2 trailing bytes";
        return false;
    }
    return true;
}

bool k9_decode_digits(const uint8_t * blob, size_t blob_len, const k9_tensor_meta & meta,
                      size_t n, uint8_t * out, std::string & err) {
    try {
        if (blob_len != (size_t) meta.tbl + meta.dig + meta.sca + meta.plen) {
            err = "k9 blob length mismatch for " + meta.name;
            return false;
        }
        if (meta.lanes > 1) {
            // v2 lane-interleaved stream (docs/11): u32 lane_off[L+1] then L
            // single-lane sub-streams, each row-aligned over the shared table.
            const uint64_t r = meta.shape[0], ccols = meta.shape[1];
            if (n != r * ccols) {
                err = "k9 v2 digit count mismatch for " + meta.name;
                return false;
            }
            const size_t otab = 4ull * (meta.lanes + 1);
            if (meta.dig < otab) {
                err = "k9 v2 digit area truncated for " + meta.name;
                return false;
            }
            const uint32_t * ot = (const uint32_t *) (blob + meta.tbl);
            for (uint32_t i = 0; i < (uint32_t) meta.lanes; i++) {
                if (ot[i] > ot[i + 1]) {
                    err = "k9 v2 lane offsets not ascending for " + meta.name;
                    return false;
                }
            }
            if (ot[meta.lanes] != (uint32_t) (meta.dig - otab)) {
                err = "k9 v2 lane offset tail mismatch for " + meta.name;
                return false;
            }
            rans_tables t;
            t.build(blob, meta.k);
            const uint8_t * sub = blob + meta.tbl + otab;
            const uint64_t rows_per = (r + meta.lanes - 1) / meta.lanes;
            for (uint32_t li = 0; li < (uint32_t) meta.lanes; li++) {
                const uint64_t a = std::min<uint64_t>(r, (uint64_t) li * rows_per);
                const uint64_t b = std::min<uint64_t>(r, (uint64_t)(li + 1) * rows_per);
                if (a >= b) {
                    break;
                }
                k9_rans_decoder dec;
                dec.init(sub + ot[li], ot[li + 1] - ot[li],
                         t.freq.data(), t.cumul.data(), t.inv.data());
                constexpr size_t CHUNK = 1 << 16;
                const size_t n_lane = (size_t)(b - a) * (size_t) ccols;
                size_t done = 0;
                while (done < n_lane) {
                    const size_t take = std::min(CHUNK, n_lane - done);
                    dec.decode(out + (size_t) a * ccols + done, take);
                    done += take;
                }
            }
            return true;
        }
        rans_tables t;
        t.build(blob, meta.k); // freq table lives at blob[0..tbl], k u16 entries
        k9_rans_decoder dec;
        dec.init(blob + meta.tbl, meta.dig, t.freq.data(), t.cumul.data(), t.inv.data());
        // Decode in chunks so scratch memory stays bounded for huge tensors.
        constexpr size_t CHUNK = 1 << 16;
        size_t done = 0;
        while (done < n) {
            const size_t take = std::min(CHUNK, n - done);
            dec.decode(out + done, take);
            done += take;
        }
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

bool k9_decode_scales(const uint8_t * blob, size_t blob_len, const k9_tensor_meta & meta,
                      size_t n, float * out, std::string & err) {
    try {
        const uint8_t * sc = blob + meta.tbl + meta.dig;
        if (meta.scale_mode == 0) {
            if (meta.sca < 4 * n) {
                err = "k9 fp32 scales truncated for " + meta.name;
                return false;
            }
            for (size_t i = 0; i < n; i++) {
                out[i] = rd_f32(sc + 4 * i);
            }
            return true;
        }
        if (meta.scale_mode == 1) {
            if (meta.sca < 2 * n) {
                err = "k9 fp16 scales truncated for " + meta.name;
                return false;
            }
            for (size_t i = 0; i < n; i++) {
                out[i] = ggml_fp16_to_fp32(*(const ggml_fp16_t *) (sc + 2 * i));
            }
            return true;
        }
        if (meta.scale_mode == 4) {
            if (meta.sca < 10) {
                err = "k9 ent8 scales truncated (header) for " + meta.name;
                return false;
            }
            const float lo = rd_f32(sc);
            float hi = rd_f32(sc + 4);
            (void) rd_u16(sc + 8); // table_len, always 256
            if (hi <= lo) {
                hi = lo + 1e-30f;
            }
            static constexpr size_t SC_HDR = 10;
            static constexpr size_t SC_TAB = 256;
            if (meta.sca < (uint32_t) (SC_HDR + 2 * SC_TAB)) {
                err = "k9 ent8 scales truncated (table) for " + meta.name;
                return false;
            }
            rans_tables t;
            t.build(sc + SC_HDR, SC_TAB);
            k9_rans_decoder dec;
            dec.init(sc + SC_HDR + 2 * SC_TAB, meta.sca - SC_HDR - 2 * SC_TAB,
                     t.freq.data(), t.cumul.data(), t.inv.data());
            constexpr size_t CHUNK = 1 << 16;
            std::vector<uint8_t> codes(CHUNK);
            size_t done = 0;
            while (done < n) {
                const size_t take = std::min(CHUNK, n - done);
                dec.decode(codes.data(), take);
                for (size_t i = 0; i < take; i++) {
                    // float64 exponent, then cast — matches k9.py exactly
                    out[done + i] = (float) exp2((double) lo + ((double) hi - (double) lo) * codes[i] / 255.0);
                }
                done += take;
            }
            return true;
        }
        err = "unsupported k9 scale mode " + std::to_string(meta.scale_mode);
        return false;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}

bool k9_dequantize(const uint8_t * blob, size_t blob_len, const k9_tensor_meta & meta,
                   const int64_t ne[2], ggml_type dst_type, void * dest, std::string & err) {
    try {
        const uint32_t k = meta.k;
        const uint64_t H = (k - 1) / 2;
        const int64_t  c = ne[0]; // stored columns (groups axis)
        const int64_t  r = ne[1]; // stored rows
        const int64_t  ng = c / meta.group;

        if ((uint32_t) ne[0] != meta.shape[1] || (uint32_t) ne[1] != meta.shape[0]) {
            err = "k9 directory shape does not match the materialized shape for " + meta.name;
            return false;
        }

        if (dst_type != GGML_TYPE_Q8_0 && dst_type != GGML_TYPE_F16 &&
                dst_type != GGML_TYPE_IQ4_NL) {
            err = "k9 materialization supports Q8_0/F16/IQ4_NL only";
            return false;
        }
        if (meta.perm_flag && dst_type == GGML_TYPE_Q8_0) {
            // permuted scales are scattered w.r.t. destination 32-blocks
            err = "k9 permuted tensor needs F16 materialization";
            return false;
        }
        if (blob_len != (size_t) meta.tbl + meta.dig + meta.sca + meta.plen) {
            err = "k9 blob length mismatch for " + meta.name;
            return false;
        }
        if (meta.plen && meta.plen != (uint32_t) (4 * c)) {
            err = "k9 permutation must be full stored-column width";
            return false;
        }

        rans_tables dt;
        dt.build(blob, meta.k);
        k9_rans_decoder ddec;
        ddec.init(blob + meta.tbl, meta.dig, dt.freq.data(), dt.cumul.data(), dt.inv.data());

        // scales, stored row-major over (r, ng)
        std::vector<float> ms(r * ng);
        if (!k9_decode_scales(blob, blob_len, meta, r * ng, ms.data(), err)) {
            return false;
        }

        // permutation (stored column -> original column): perm_col[j] = perm[j]
        std::vector<uint32_t> perm_col;
        if (meta.perm_flag) {
            perm_col.resize(c);
            const uint8_t * pd = blob + meta.tbl + meta.dig + meta.sca;
            for (int64_t j = 0; j < c; j++) {
                const uint32_t orig = rd_u32(pd + 4 * j);
                if (orig >= (uint32_t) c) {
                    err = "k9 permutation index out of range";
                    return false;
                }
                perm_col[j] = orig;
            }
        }

        const size_t n_blocks_per_row = (size_t) (c / 32);
        auto pack_row = [&](const uint8_t * row_digits, int64_t i1) {
            if (dst_type == GGML_TYPE_F16) {
                ggml_fp16_t * dst = (ggml_fp16_t *) dest + i1 * c;
                if (meta.perm_flag) {
                    for (int64_t j = 0; j < ng; j++) {
                        const float m = ms[i1 * ng + j];
                        for (int64_t t = 0; t < meta.group; t++) {
                            const float v = (m * ((float) row_digits[j * meta.group + t] - (float) H)) / (float) H;
                            dst[perm_col[j * meta.group + t]] = ggml_fp32_to_fp16(v);
                        }
                    }
                } else {
                    for (int64_t j = 0; j < ng; j++) {
                        const float m = ms[i1 * ng + j];
                        for (int64_t t = 0; t < meta.group; t++) {
                            const float v = (m * ((float) row_digits[j * meta.group + t] - (float) H)) / (float) H;
                            dst[j * meta.group + t] = ggml_fp32_to_fp16(v);
                        }
                    }
                }
            } else if (dst_type == GGML_TYPE_IQ4_NL) {
                // Non-exact rematerialization (docs/11 §8): one RTN pass onto
                // the 16-value IQ4_NL codebook, per 32-weight block, with a
                // deliberately simple deterministic scale d = amax/113 (code
                // 15) — the C and Python gate implementations must stay
                // byte-identical, so no scale-search here. k99 aux tensors
                // never land here; the loader routes them to Q8_0.
                block_iq4_nl * blk = (block_iq4_nl *) dest;
                float w[32];
                for (int64_t jg = 0; jg < ng; jg++) {
                    const float m = ms[i1 * ng + jg];
                    for (int64_t sb = 0; sb < 2; sb++) {
                        float amax = 0.f;
                        for (int64_t t = 0; t < 32; t++) {
                            const float v = (m * ((float) row_digits[jg * meta.group + sb * 32 + t] - (float) H)) / (float) H;
                            w[t] = v;
                            const float av = v < 0 ? -v : v;
                            if (av > amax) {
                                amax = av;
                            }
                        }
                        block_iq4_nl * b = blk + i1 * n_blocks_per_row + 2 * jg + sb;
                        if (amax < 1e-8f) {
                            b->d = ggml_fp32_to_fp16(0.f);
                            memset(b->qs, 0x88, sizeof(b->qs)); // code idx 8 pairs, d=0
                            continue;
                        }
                        const float id = 1.0f / (amax / 113.0f);
                        b->d = ggml_fp32_to_fp16(amax / 113.0f);
                        uint8_t L[32];
                        for (int64_t t = 0; t < 32; t++) {
                            const float target = id * w[t];
                            uint8_t best = 0;
                            float bestdiff = 1e30f;
                            for (int e = 0; e < 16; e++) {
                                float diff = target - (float) k9_iq4nl_values[e];
                                if (diff < 0) {
                                    diff = -diff;
                                }
                                if (diff < bestdiff) {
                                    bestdiff = diff;
                                    best = (uint8_t) e;
                                }
                            }
                            L[t] = best;
                        }
                        for (int t = 0; t < 16; t++) {
                            b->qs[t] = L[t] | (uint8_t)(L[16 + t] << 4);
                        }
                    }
                }
            } else {
                // Q8_0: blocks are 32 columns of this row; two blocks share a
                // 64-group, so scale d = fp16(m/H) and qs = digit - H exactly.
                block_q8_0 * blk = (block_q8_0 *) dest;
                for (int64_t j = 0; j < ng; j++) {
                    const float m = ms[i1 * ng + j];
                    const ggml_fp16_t d = ggml_fp32_to_fp16(m / (float) H);
                    block_q8_0 * b0 = blk + i1 * n_blocks_per_row + 2 * j + 0;
                    block_q8_0 * b1 = blk + i1 * n_blocks_per_row + 2 * j + 1;
                    b0->d = d;
                    b1->d = d;
                    for (int64_t t = 0; t < meta.group / 2; t++) {
                        b0->qs[t] = (int8_t) row_digits[j * meta.group + t] - (int8_t) H;
                        b1->qs[t] = (int8_t) row_digits[j * meta.group + meta.group / 2 + t] - (int8_t) H;
                    }
                }
            }
        };

        unsigned hw = std::thread::hardware_concurrency();
        if (const char * envp = std::getenv("K9_DECODE_THREADS")) {
            hw = (unsigned) std::max(1, atoi(envp));
        } else if (hw == 0) {
            hw = 1;
        }

        if (meta.lanes > 1) {
            // v2 (docs/11): L independent rANS lanes over row-aligned chunks.
            // Each worker decodes its own sub-stream (one lane == one state,
            // all lanes share the frequency table) and packs its rows — no
            // cross-lane synchronization, output bit-identical to v1.
            const size_t otab = 4ull * (meta.lanes + 1);
            if (meta.dig < otab) {
                throw std::runtime_error("k9 v2 digit area truncated for " + meta.name);
            }
            const uint32_t * ot = (const uint32_t *) (blob + meta.tbl);
            for (uint32_t i = 0; i < (uint32_t) meta.lanes; i++) {
                if (ot[i] > ot[i + 1]) {
                    throw std::runtime_error("k9 v2 lane offsets not ascending for " + meta.name);
                }
            }
            if (ot[meta.lanes] != (uint32_t) (meta.dig - otab)) {
                throw std::runtime_error("k9 v2 lane offset tail mismatch for " + meta.name);
            }
            const uint8_t * sub = blob + meta.tbl + otab;
            const int64_t rows_per = ((int64_t) r + (int64_t) meta.lanes - 1) / (int64_t) meta.lanes;
            const unsigned nthr = std::min<unsigned>(hw, (unsigned) meta.lanes);
            std::atomic<uint32_t> next_lane{0};
            auto lane_worker = [&]() {
                for (;;) {
                    const uint32_t li = next_lane.fetch_add(1);
                    if (li >= (uint32_t) meta.lanes) {
                        break;
                    }
                    const int64_t a = std::min<int64_t>((int64_t) r, (int64_t) li * rows_per);
                    const int64_t b = std::min<int64_t>((int64_t) r, (int64_t)(li + 1) * rows_per);
                    if (a >= b) {
                        continue;
                    }
                    std::vector<uint8_t> lane_digits((size_t)(b - a) * (size_t) c);
                    k9_rans_decoder dl;
                    dl.init(sub + ot[li], ot[li + 1] - ot[li],
                            dt.freq.data(), dt.cumul.data(), dt.inv.data());
                    constexpr size_t CH = 1 << 24;
                    for (size_t done = 0; done < lane_digits.size(); ) {
                        const size_t take = std::min<size_t>(CH, lane_digits.size() - done);
                        dl.decode(lane_digits.data() + done, take);
                        done += take;
                    }
                    for (int64_t i1 = a; i1 < b; i1++) {
                        pack_row(lane_digits.data() + (i1 - (size_t) a) * (size_t) c, i1);
                    }
                }
            };
            std::vector<std::thread> pool;
            for (unsigned t = 0; t < nthr; t++) {
                pool.emplace_back(lane_worker);
            }
            for (auto & th : pool) {
                th.join();
            }
            return true;
        }

        // v1: the digit stream is a single sequential (backward-renorm) rANS
        // lane, so it cannot be split across threads without changing the
        // on-disk format; decode it once (bit-identical to the original
        // row-chunked calls) and pack rows in parallel.
        std::vector<uint8_t> digits((size_t) r * c);
        for (size_t done = 0; done < (size_t) r * c; ) {
            const size_t take = std::min<uint64_t>((uint64_t) 1 << 24, (size_t) r * c - done);
            ddec.decode(digits.data() + done, take);
            done += take;
        }

        unsigned n_threads = hw;
        if ((int64_t) n_threads > r / 2) {
            n_threads = (unsigned) std::max<int64_t>(1, r / 2);
        }
        if (n_threads <= 1) {
            for (int64_t i1 = 0; i1 < r; i1++) {
                pack_row(digits.data() + i1 * c, i1);
            }
        } else {
            const int64_t rows_per_thread = (r + (int64_t) n_threads - 1) / (int64_t) n_threads;
            std::vector<std::thread> pool;
            for (unsigned t = 0; t < n_threads; t++) {
                pool.emplace_back([&, t] {
                    const int64_t b = std::min<int64_t>((int64_t) t * rows_per_thread, r);
                    const int64_t e = std::min<int64_t>(b + rows_per_thread, r);
                    for (int64_t i1 = b; i1 < e; i1++) {
                        pack_row(digits.data() + i1 * c, i1);
                    }
                });
            }
            for (auto & th : pool) {
                th.join();
            }
        }
        return true;
    } catch (const std::exception & e) {
        err = e.what();
        return false;
    }
}