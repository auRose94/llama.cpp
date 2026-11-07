#pragma once
// v3 residency (docs/13 §2.6, gate 3): tensor-granular lazy materialization
// for K9 stream tensors.
//
// K9_RESIDENT=1 switches the loader from decode-on-load to registered lazy
// materialization: K9 tensors register their (mmap-retained) compressed blob
// and are decoded into their already-assigned model-buffer memory on first
// graph use. A virtual budget (K9_BUDGET_GIB) marks LRU tensors absent when
// exceeded, forcing re-decode on next use — this exercises the eviction /
// re-materialization machinery. Decoding is k9_dequantize (byte-exact), so a
// fully-resident run and a constrained-budget run produce identical weights
// (docs/13 §2.4 gate 3: token-identical under eviction).
//
// v1 scope and honest limits: tensors are pre-assigned to full-size model
// buffers at creation, so the virtual budget exercises re-decode correctness
// but does NOT yet free memory. True memory relief lands with the
// expert-granular MoE rung (page == expert, docs/13 §2.6) or a dedicated
// residency buffer type. CPU backend only (blob pointers come from the
// loader's mmap mappings, which llama_model retains); without mmap the
// loader falls back to eager decode-on-load. One registry per process.
//
// Env:
//   K9_RESIDENT=1      enable lazy materialization (default off = eager)
//   K9_BUDGET_GIB=f    virtual residency budget in GiB; 0 or unset = full
//                      (materialize-all, no eviction)

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>

#include "ggml.h"

#include "k9.h"

struct ggml_cgraph;

class k9_residency {
public:
    static bool active();
    static void disable();   // fall back to eager decode-on-load for the rest of this load
    static k9_residency & get();

    // Drop all state (call before a fresh model load in the same process).
    void reset();

    // Called from the loader's K9 branch for tensors destined to host
    // buffers. blob must point into memory that outlives the model.
    void register_tensor(const std::string & name,
                         const uint8_t * blob, size_t blob_len,
                         const k9_tensor_meta & meta,
                         const int64_t f_ne[2], ggml_type dst,
                         struct ggml_tensor * cur, size_t nbytes);

    // Decode-on-demand + LRU eviction for every registered tensor referenced
    // by the graph. Cheap no-op when nothing is registered. Called from
    // llama_context::graph_compute before the scheduler runs the graph.
    // Matching is BY NAME: the graph holds leaf tensors that may be
    // different ggml_tensor objects than the loader's list, but names are
    // stable and unique; entries rebind to the graph's object at match time.
    void ensure_graph(const struct ggml_cgraph * gf);

private:
    k9_residency() = default;

    struct entry {
        const uint8_t * blob     = nullptr;
        size_t          blob_len = 0;
        k9_tensor_meta  meta;               // copy
        int64_t         f_ne[2] = { 0, 0 }; // flattened stored frame (cols, rows)
        ggml_type       dst     = GGML_TYPE_F32;
        ggml_tensor   * cur     = nullptr;  // rebound to the graph's object per ensure
        size_t          nbytes  = 0;
        bool            materialized = false;
        uint64_t        stamp   = 0;
        bool            in_graph = false;   // scratch for the current ensure pass
        int64_t         last_use = -1;      // highest node index using this tensor
    };

    void materialize_entry(entry & e, int64_t node_idx);
    void evict_until(size_t need, int64_t before_idx);

    // transparent comparator: graph lookups pass string_view without allocs
    std::map<std::string, entry, std::less<>> by_name;
    size_t total_bytes = 0;
    size_t max_bytes   = 0;
    size_t materialized_bytes = 0;

    uint64_t clock = 0;
    size_t budget = 0;   // bytes; decided on first materialize (0 = not yet)
    size_t n_hits = 0, n_mat = 0, n_evict = 0;
};
