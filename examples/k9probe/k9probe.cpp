// k9probe — dump K9 stream tensors from a GGUF for bit-exactness gates
// against base9-quantization's Python reference (experiments/k9.py).
//
//   k9probe <model.gguf> <tensor-name> <digits|scales|weights> <out.bin>
//
// digits  : raw decoded digits, r*c bytes, stored-frame row-major order
// scales  : decoded scales, f32 LE, r*(c/group) values, stored-frame order
// weights : the materialized tensor exactly as the llama.cpp loader produces
//           it (raw Q8_0 or F16 bytes)

#include "k9.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gguf.h"

int main(int argc, char ** argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s <model.gguf> <tensor-name> <digits|scales|weights> <out.bin>\n", argv[0]);
        return 1;
    }
    const std::string fname = argv[1];
    const std::string tname = argv[2];
    const std::string mode  = argv[3];
    const std::string outp  = argv[4];

    std::FILE * fh = fopen(fname.c_str(), "rb");
    if (!fh) {
        fprintf(stderr, "cannot open %s\n", fname.c_str());
        return 1;
    }

    gguf_init_params params = {
        /*.no_alloc = */ true,
        /*.ctx      = */ nullptr,
    };
    gguf_context * meta = gguf_init_from_file(fname.c_str(), params);
    if (!meta) {
        fprintf(stderr, "gguf_init_from_file failed for %s\n", fname.c_str());
        return 1;
    }

    // ninths.directory2 (v2) primary, then ninths.directory (k9.directory =
    // legacy alias — see llama-model-loader.cpp)
    int kid = gguf_find_key(meta, "ninths.directory2");
    const bool k9_v2 = kid != -1;
    if (kid == -1) {
        kid = gguf_find_key(meta, "ninths.directory");
    }
    if (kid == -1) {
        kid = gguf_find_key(meta, "k9.directory");
    }
    if (kid == -1) {
        fprintf(stderr, "no ninths.directory[2]/k9.directory KV in %s\n", fname.c_str());
        return 1;
    }
    std::map<std::string, k9_tensor_meta> k9_meta;
    std::string err;
    const bool parsed = k9_v2
        ? k9_parse_directory2((const uint8_t *) gguf_get_arr_data(meta, kid),
                              gguf_get_arr_n(meta, kid), k9_meta, err)
        : k9_parse_directory((const uint8_t *) gguf_get_arr_data(meta, kid),
                             gguf_get_arr_n(meta, kid), k9_meta, err);
    if (!parsed) {
        fprintf(stderr, "k9_parse_directory failed: %s\n", err.c_str());
        return 1;
    }
    auto it = k9_meta.find(tname);
    if (it == k9_meta.end()) {
        fprintf(stderr, "tensor %s not in k9.directory\n", tname.c_str());
        return 1;
    }
    const k9_tensor_meta & m = it->second;

    const int tid = gguf_find_tensor(meta, tname.c_str());
    if (tid == -1) {
        fprintf(stderr, "tensor %s not in GGUF\n", tname.c_str());
        return 1;
    }
    if (gguf_get_tensor_type(meta, tid) != GGML_TYPE_K9) {
        fprintf(stderr, "%s is not a K9 tensor\n", tname.c_str());
        return 1;
    }
    const size_t blob_len = gguf_get_tensor_size(meta, tid);
    const size_t blob_off = gguf_get_data_offset(meta) + gguf_get_tensor_offset(meta, tid);
    std::vector<uint8_t> blob(blob_len);
    if (fseek(fh, (long) blob_off, SEEK_SET) || fread(blob.data(), 1, blob_len, fh) != blob_len) {
        fprintf(stderr, "failed to read blob for %s\n", tname.c_str());
        return 1;
    }
    fclose(fh);

    std::vector<uint8_t> out;

    if (mode == "digits") {
        const size_t n = m.shape[0] * m.shape[1];
        out.resize(n);
        if (!k9_decode_digits(blob.data(), blob_len, m, n, (uint8_t *) out.data(), err)) {
            fprintf(stderr, "digit decode failed: %s\n", err.c_str());
            return 1;
        }
        printf("digits: %zu bytes (k=%u group=%u scale_mode=%u perm=%u)\n", n, m.k, m.group, m.scale_mode, m.perm_flag);
    } else if (mode == "scales") {
        const size_t n = m.shape[0] * (m.shape[1] / m.group);
        out.resize(n * sizeof(float));
        if (!k9_decode_scales(blob.data(), blob_len, m, n, (float *) out.data(), err)) {
            fprintf(stderr, "scale decode failed: %s\n", err.c_str());
            return 1;
        }
        printf("scales: %zu floats (mode=%u)\n", n, m.scale_mode);
    } else if (mode == "debugrow0") {
        // instrumented: trace the dequantize path for stored row 0
        const int64_t ne[2] = { (int64_t) m.shape[1], (int64_t) m.shape[0] };
        std::vector<uint8_t> digits(ne[0] * ne[1]);
        if (!k9_decode_digits(blob.data(), blob_len, m, ne[0] * ne[1],
                              (uint8_t *) digits.data(), err)) {
            fprintf(stderr, "digit decode failed: %s\n", err.c_str());
            return 1;
        }
        std::vector<float> scales(ne[0] * (ne[1] / (int64_t) m.group));
        if (!k9_decode_scales(blob.data(), blob_len, m, ne[0] * (ne[1] / (int64_t) m.group),
                              (float *) scales.data(), err)) {
            fprintf(stderr, "scale decode failed: %s\n", err.c_str());
            return 1;
        }
        const int64_t c = ne[0], r = ne[1], ng = c / m.group, H = (m.k - 1) / 2;
        printf("c=%lld r=%lld ng=%lld H=%llu\n", (long long) c, (long long) r,
               (long long) ng, (unsigned long long) H);
        printf("row0 scales[0..3]: %.6f %.6f %.6f %.6f\n", scales[0], scales[1], scales[2], scales[3]);
        printf("row0 digits[0..7]:");
        for (int t = 0; t < 8; t++) printf(" %d", digits[t]);
        printf("\n");
        // full dequantize through the real code path, then report row 0
        const ggml_type dst = m.perm_flag ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
        std::vector<uint8_t> dest((size_t) ggml_row_size(dst, ne[0]) * ne[1]);
        if (!k9_dequantize(blob.data(), blob_len, m, ne, dst, dest.data(), err)) {
            fprintf(stderr, "dequantize failed: %s\n", err.c_str());
            return 1;
        }
        const ggml_fp16_t * df = (const ggml_fp16_t *) dest.data();
        printf("dest(c=%lld) values[0..7] (f16, original col order):", (long long) c);
        for (int t = 0; t < 8; t++) printf(" %.5f", ggml_fp16_to_fp32(df[t]));
        printf("\n");
        return 0;
    } else if (mode == "weights") {
        const int64_t ne[2] = { (int64_t) m.shape[1], (int64_t) m.shape[0] }; // GGUF: (c, r)
        const ggml_type dst = m.perm_flag ? GGML_TYPE_F16 : GGML_TYPE_Q8_0;
        const size_t nbytes = (size_t) ggml_row_size(dst, ne[0]) * ne[1];
        out.resize(nbytes);
        if (!k9_dequantize(blob.data(), blob_len, m, ne, dst, out.data(), err)) {
            fprintf(stderr, "dequantize failed: %s\n", err.c_str());
            return 1;
        }
        printf("weights: %zu bytes as %s\n", nbytes, ggml_type_name(dst));
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 1;
    }

    FILE * ofh = fopen(outp.c_str(), "wb");
    if (!ofh || fwrite(out.data(), 1, out.size(), ofh) != out.size()) {
        fprintf(stderr, "failed to write %s\n", outp.c_str());
        return 1;
    }
    fclose(ofh);
    return 0;
}