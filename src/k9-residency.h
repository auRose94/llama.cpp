#pragma once
// v3 residency (docs/13 §2.6): lazy materialization of K9 stream tensors.
//
// M3 (real memory relief): ONE MAP_NORESERVE region backs every K9 tensor
// at a fixed offset (virtual cost only — anonymous zero pages are free).
// Pages are the container's v3 lanes (row-aligned, independently decodable
// via k9_decode_page). Non-MoE tensors fill wholly at the first graph use;
// MoE expert tensors are HOOK-managed: the CPU mul_mat_id op calls
// k9_res_moe_ensure at execution time, when the router ids are actually
// known, and only the selected experts' lanes get decoded. Eviction is
// madvise(MADV_DONTNEED) — the kernel reclaims the pages, so RSS is the
// honest measure. K9_BUDGET_GIB caps decoded expert-page bytes (0 =
// unlimited); non-MoE tensors are the irreducible working set and always
// resident.
//
// The hook is execution-safe BY CONSTRUCTION (decode happens inside the op,
// before the kernel reads) — dissolving the M2.5 finding that dense
// pre-execution fills cannot evict. CPU backend today; the A9/HIP port
// reuses the same region over unified memory.
//
// Env:
//   K9_RESIDENT=1      enable (default off = eager decode-on-load)
//   K9_BUDGET_GIB=f    expert-page budget in GiB (0/unset = unlimited)

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

#include "k9.h"

struct ggml_cgraph;

class k9_residency {
public:
    static bool active();
    static void disable();   // fall back to eager decode-on-load for the rest of this load
    // Clear ALL state including a sticky disable() from a previous load
    // (e.g. the fit pass, which loads with LOAD_MODE_NONE). Call at the
    // start of every real model load.
    static void fresh_load();
    static k9_residency & get();

    // Drop all state (call before a fresh model load in the same process).
    void reset();

    // Map the NORESERVE region. Call once from the loader before any homing.
    void prepare_region(size_t total_bytes);

    // Fixed home for a tensor at `offset` in the region. Returns false when
    // the tensor was homed elsewhere (caller registers it as external).
    bool home_fixed(struct ggml_tensor * cur, size_t offset);

    // Called from the loader's K9 branch. blob must outlive the model.
    // moe = rank-3 expert tensor (ne[2] > 1) → hook-managed pages.
    void register_tensor(const std::string & name,
                         const uint8_t * blob, size_t blob_len,
                         const k9_tensor_meta & meta,
                         const int64_t f_ne[2], ggml_type dst,
                         struct ggml_tensor * cur, size_t nbytes,
                         size_t region_off, bool in_arena, bool moe);

    // Pre-execution fill for non-MoE tensors (MoE tensors are hook-managed).
    void ensure_graph(const struct ggml_cgraph * gf);

    // M3 hook: called from ggml-cpu's mul_mat_id (thread 0).
    bool moe_ensure(const struct ggml_tensor * weights, const struct ggml_tensor * ids);

    // M3 hook, HIP flavor: ids already copied to host by the caller.
    void moe_ensure_ids(const struct ggml_tensor * weights,
                        const int32_t * idata, int64_t n_used, int64_t n_tok);

    // M3 hook, MoE-cache flavor: decode the lanes covering the given expert
    // ids of one managed tensor (called before the cache uploads bytes).
    void moe_ensure_experts(const struct ggml_tensor * weights,
                            const int32_t * experts, size_t n);

    // registry membership check for the C managed-check (no decode)
    bool is_managed(std::string_view name) const {
        auto it = by_name.find(name);
        return it != by_name.end() && it->second.in_arena && it->second.moe;
    }
    size_t registry_size() const { return by_name.size(); }
    // the region's buffer (for scheduler-level checks); null before prepare_region
    ggml_backend_buffer_t region_buffer() const { return buf; }
    // k9 debug: FNV(4096) of an expert's bytes in the region (no decode)
    unsigned long long debug_expert_fnv(std::string_view name, int32_t expert) const;
    bool region_ready() const { return region != nullptr; }

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
        size_t          region_off = 0;     // fixed home in the region
        bool            in_arena = true;    // false = externally homed
        bool            moe     = false;    // rank-3 expert tensor → hook-managed
        uint32_t        n_exp   = 0;        // experts along the outer axis (moe)
        size_t          rows    = 0;        // stored rows (f_ne[1])
        uint32_t        lanes   = 1;        // page count
        uint32_t        rows_per = 0;       // stored rows per page
        size_t          row_bytes = 0;      // materialized bytes per stored row
        // INLINE page state: lanes <= 255 (u8 field), so 4 u64 of bitmap +
        // 255 stamps fit without heap. Heap-resident vectors for these were
        // observed being wiped between hook calls during bring-up (the
        // byte counter in the same entry survived) — inline storage is
        // immune to whatever recycles those blocks.
        std::array<uint64_t, 4>  page_dec{};   // decoded bitmap (lanes bits)
        std::array<uint32_t, 255> page_stamp{}; // LRU per page
        float         * ms      = nullptr;  // cached decoded scales (r*ng floats)
        uint64_t        stamp   = 0;        // entry-level LRU (non-MoE)
        bool            in_graph = false;   // scratch for the current ensure pass
        int64_t         last_use = -1;
    };

    bool ensure_page(entry & e, uint32_t page);
    size_t page_off(const entry & e, uint32_t p) const;
    size_t page_bytes(const entry & e, uint32_t p) const;

    // transparent comparator: graph lookups pass string_view without allocs
    std::map<std::string, entry, std::less<>> by_name;
    size_t total_bytes = 0;

    void * region = nullptr;             // MAP_NORESERVE anonymous mapping
    size_t region_size = 0;
    ggml_backend_buffer_t buf = nullptr; // cpu buffer over the region
    size_t budget = 0;                   // expert-page decoded-bytes cap (0 = unlimited)
    size_t moe_decoded = 0;              // decoded expert-page bytes
    uint64_t clock = 0;                  // page LRU clock
    size_t n_mat = 0, n_evict = 0;
};
