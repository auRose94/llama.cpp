#pragma once
// K9 (ninths-grid) stream decoding for llama.cpp.
//
// The GGUF file stores K9 weight tensors as a single variable-length compressed
// blob under the loader-only sentinel type GGML_TYPE_K9 (blck_size 1,
// type_size 1, so ggml_nbytes() == ne[0] == blob length). Per-tensor metadata
// comes from the KV key "k9.directory", a little-endian byte array laid out
// like the .k9 container directory (experiments/k9.py in base9-quantization,
// which is the normative on-disk format; ignore docs/04 §5.2 which is stale):
//
//   u16 n_tensors, then per record:
//     u16 name_len, name (utf8),
//     u8 rank, u32 dims[rank]        (stored-frame shape: rows out/vocab, cols in/embd)
//     u8 k, u8 group, u8 scale_mode, u8 perm_flag,
//     u32 tbl, u32 dig, u32 sca, u32 plen
//
// The blob is identical to the .k9 blob area for the tensor:
//   [ freq table: k * u16 LE, sum == M = 4096 ]
//   [ digit rANS stream: dig bytes ]
//   [ scales blob: sca bytes (mode-dependent) ]
//   [ perm: plen bytes (u32 LE, full stored-column width, GPTQ act-order) ]
//
// The stored frame follows the HF/torch module weight layout (rows = out
// features / vocab, columns = in features / embd, digits row-major, one scale
// per (row, 64-column group)); the GGUF materialized layout is the transpose,
// which the decoder below handles. Dequantization (normative, k9.py):
//
//     W[row, col] = m[row, col/64] * (d - H) / H,   H = (k-1) // 2
//
// with all tensor arithmetic in float32 and the ent8 scale exponent computed
// in float64.

#include <stddef.h>
#include <stdint.h>

#include <map>
#include <string>

#include <ggml.h>

struct k9_tensor_meta {
    std::string name;
    uint32_t    shape[2] = { 0, 0 }; // stored-frame shape (rows, cols); rank == 2
    uint32_t k          = 0; // odd grid size: 9, 15, 27, 63, 99 ...
    uint32_t group      = 0; // scale group along the stored column axis (64)
    uint32_t scale_mode = 0; // 0 fp32, 1 fp16, 4 ent8
    uint32_t perm_flag  = 0;
    uint8_t  lanes      = 1; // v2: 1 = single-lane (v1 layout), >1 = lane-interleaved
    uint32_t tbl        = 0; // freq table bytes
    uint32_t dig        = 0; // digit rANS stream bytes
    uint32_t sca        = 0; // scales blob bytes
    uint32_t plen       = 0; // permutation bytes
};

// Parse the "k9.directory" KV payload (raw bytes) into a per-name map.
// Returns false (and may throw std::runtime_error messages into err) on
// malformed input.
bool k9_parse_directory(const uint8_t * data, size_t n_bytes,
                        std::map<std::string, k9_tensor_meta> & out, std::string & err);

// v2 (docs/11, lane-interleaved streams): leading u16 version == 2, then the
// v1 record layout with one extra u8 lanes field per record.
bool k9_parse_directory2(const uint8_t * data, size_t n_bytes,
                         std::map<std::string, k9_tensor_meta> & out, std::string & err);

// Decode one K9 blob into dequantized weights, writing ggml_nbytes-ordered raw
// data for a 2-D materialized tensor of type dst_type with shape ne[2 dims].
// ne is the GGUF (transposed) shape: ne[0] == stored columns, ne[1] == stored
// rows. dst_type must be GGML_TYPE_Q8_0 (perm_flag must be 0) or GGML_TYPE_F16.
// Returns false on malformed input/unsupported mode.
bool k9_dequantize(const uint8_t * blob, size_t blob_len, const k9_tensor_meta & meta,
                   const int64_t ne[2], ggml_type dst_type, void * dest, std::string & err);

// Exposed for probes/tests: decode the digit rANS stream for a tensor in
// stored order into out (r*c bytes).
bool k9_decode_digits(const uint8_t * blob, size_t blob_len, const k9_tensor_meta & meta,
                      size_t n, uint8_t * out, std::string & err);

// Exposed for probes/tests: decode scales into f32 (n values).
bool k9_decode_scales(const uint8_t * blob, size_t blob_len, const k9_tensor_meta & meta,
                      size_t n, float * out, std::string & err);