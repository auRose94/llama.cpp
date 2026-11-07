#pragma once
// v3 residency (docs/13 §2.6, gates 3+): tensor-granular lazy
// materialization for K9 stream tensors.
//
// M2.5 (real-budget arena): K9_RESIDENT=1 homes every K9 tensor into a
// BUDGET-SIZED arena at load — all tensors get overlapping dummy data
// pointers at a zeroed poison slab so the scheduler never re-homes them
// into its transient compute buffers (the v1 hard-won invariant). At each
// graph step, ensure_graph re-points a tensor's data into a real arena
// slot after decoding it there, and LRU eviction frees slots (re-pointing
// victims at the poison slab). Memory usage is now bounded by
// K9_BUDGET_GIB instead of the full materialized size.
//
// Decoding is k9_dequantize (byte-exact), so a fully-resident run and a
// constrained-budget run produce identical weights (docs/13 §2.4 gate 3:
// token-identical under eviction).
//
// Env:
//   K9_RESIDENT=1      enable lazy materialization (default off = eager)
//   K9_BUDGET_GIB=f    arena budget in GiB; 0 or unset = full residency
//                      (arena = sum of K9 tensors, no eviction — v1 size)
//
// CPU backend (the soak/rung target). The arena is allocated from the CPU
// buffer type; the A9/HIP port (unified-memory buft) follows.

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "k9.h"

struct ggml_cgraph;

class k9_residency {
public:
    static bool active();
    static void disable();   // fall back to eager decode-on-load for the rest of this load
    static k9_residency & get();

    // Drop all state (call before a fresh model load in the same process).
    void reset();

    // Allocate the arena. Call once from the loader before the first
    // register/home (it knows the K9 totals). budget 0 = full residency.
    void prepare_arena(size_t total_bytes, size_t max_tensor_bytes);

    // Dummy-home a tensor into the arena (overlapping at the poison slab)
    // so the scheduler sees it buffered. Returns false if the tensor was
    // already homed elsewhere (caller should register it as external).
    bool home_dummy(struct ggml_tensor * cur);

    // Called from the loader's K9 branch. blob must point into memory that
    // outlives the model.
    void register_tensor(const std::string & name,
                         const uint8_t * blob, size_t blob_len,
                         const k9_tensor_meta & meta,
                         const int64_t f_ne[2], ggml_type dst,
                         struct ggml_tensor * cur, size_t nbytes,
                         bool in_arena);

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
        bool            in_arena  = true;   // false = externally homed (in-place)
        uint64_t        stamp   = 0;
        bool            in_graph = false;   // scratch for the current ensure pass
        int64_t         last_use = -1;      // highest node index using this tensor

        // materialized state is DERIVED: an arena tensor is resident iff
        // cur->data points outside the poison slab. (A plain bool field was
        // observed going stale vs cur->data during bring-up — the tensor's
        // own pointer is the single source of truth.)
        bool resident() const {
            return cur != nullptr && cur->data != nullptr &&
                   cur->data != k9_residency::poison_base();
        }
    };

    void materialize_entry(entry & e, int64_t node_idx);
    void evict_until(size_t need, int64_t before_idx);
    size_t carve(size_t nbytes);

public:
    // poison-slab identity check for entry::resident() (single definition point)
    static const void * poison_base();
private:

    // transparent comparator: graph lookups pass string_view without allocs
    std::map<std::string, entry, std::less<>> by_name;
    size_t total_bytes = 0;
    size_t max_bytes   = 0;
    size_t materialized_bytes = 0;

    ggml_backend_buffer_t arena = nullptr;
    uint8_t * arena_base = nullptr;
    size_t    arena_size = 0;   // total budget bytes
    size_t    slot_start = 0;   // first usable offset (after the poison slab)
    size_t    arena_high = 0;   // bump frontier; recycled ranges in free_ranges
    std::vector<std::pair<size_t, size_t>> free_ranges; // (offset, size), 64-aligned

    uint64_t clock = 0;
    size_t budget = 0;   // bytes; decided at prepare_arena
    size_t n_hits = 0, n_mat = 0, n_evict = 0;
};
