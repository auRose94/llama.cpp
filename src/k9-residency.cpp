// v3 residency implementation — see k9-residency.h (docs/13 §2.6).
// M3: MAP_NORESERVE region + lane pages + mul_mat_id hook. Eviction is
// madvise(MADV_DONTNEED) — the kernel reclaims, so RSS is honest.
#include "k9-residency.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <unistd.h>

#include <sys/mman.h>

// HIP API calls live behind the C hooks in ggml-cuda.cu (k9_res_hip_*); this
// TU stays HIP-header-free (libllama has no ROCm include path)

#include "ggml-backend.h"
#include "k9-moe-hook.h"
#include "llama-impl.h"

namespace {

bool env_flag(const char * name) {
    const char * v = getenv(name);
    return v && v[0] == '1';
}

double env_float(const char * name) {
    const char * v = getenv(name);
    return v ? atof(v) : 0.0;
}

// MADV_DONTNEED with page-aligned ABSOLUTE bounds — raw unaligned calls
// return EINVAL silently and nothing is ever dropped (the bring-up gotcha:
// blob pointers start at arbitrary offsets, so aligning an offset from an
// unaligned base is not enough). inward=true rounds the range IN (region
// pages: neighbor tensors share 4K pages, so a rounded-out drop would zero
// a live neighbor); inward=false rounds OUT (file-backed blob pages are
// safe to over-drop — they re-fault from the file).
static void k9_madvise_drop(void * ptr, size_t len, bool inward) {
    static const size_t PS = sysconf(_SC_PAGESIZE);
    uintptr_t a = (uintptr_t) ptr;
    uintptr_t z = a + len;
    if (inward) {
        const uintptr_t na = (a + PS - 1) / PS * PS;
        const uintptr_t nz = z / PS * PS;
        if (na >= nz) {
            return;
        }
        a = na;
        z = nz;
    } else {
        a = a / PS * PS;
        z = (z + PS - 1) / PS * PS;
    }
    if (madvise((void *) a, z - a, MADV_DONTNEED) != 0) {
        static bool warned = false;
        if (!warned) {
            warned = true;
            LLAMA_LOG_WARN("k9 residency: madvise(MADV_DONTNEED) failed: %s (RSS will track the whole blob)\n",
                           strerror(errno));
        }
    }
}

bool g_enabled_init = false;
bool g_enabled      = false;
bool g_killed       = false;

} // namespace

bool k9_residency::active() {
    if (!g_enabled_init) {
        g_enabled      = env_flag("K9_RESIDENT");
        g_enabled_init = true;
    }
    return g_enabled && !g_killed;
}

void k9_residency::disable() {
    g_killed = true;
}

void k9_residency::fresh_load() {
    g_killed = false; // a previous load's disable (fit pass: LOAD_MODE_NONE)
                      // must not poison this load
    get().reset();
}

k9_residency & k9_residency::get() {
    static k9_residency inst;
    return inst;
}

void k9_residency::reset() {
    if (buf) {
        ggml_backend_buffer_free(buf);
        buf = nullptr;
    }
    if (region) {
        munmap(region, region_size);
        region = nullptr;
    }
    by_name.clear();
    total_bytes = 0;
    region_size = 0;
    moe_decoded = 0;
    clock = 0;
    budget = 0;
    n_mat = n_evict = 0;
}

void k9_residency::prepare_region(size_t total) {
    if (region) {
        return; // idempotent (multi-ctx loads)
    }
    const double gib = env_float("K9_BUDGET_GIB");
    budget = gib > 0.0 ? (size_t)(gib * 1073741824.0) : 0; // 0 = unlimited
    region_size = total;
#ifdef MAP_NORESERVE
    region = mmap(nullptr, total, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
#else
    region = mmap(nullptr, total, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
    if (region == MAP_FAILED) {
        region = nullptr;
        throw std::runtime_error(format(
            "k9 residency: failed to map the %.2f GiB NORESERVE region",
            total / 1073741824.0));
    }
    // M3-HIP finding: a HIP-registered region is NOT viable — kernels
    // overread tile-aligned neighbor pages (HMM zero-fault), and once the
    // GPU has mapped an undecoded page it never sees the host's later
    // decode writes (stale IOMMU state). The correct HIP architecture is
    // the MoE cache: the region stays CPU-resident (this CPU buffer), the
    // MoE cache uploads selected experts to device banks per token
    // (llama-moe-cache.cpp + k9_res_moe_ensure_experts), and the GPU
    // matmuls read the banks. Enabled with --moe-cache-mib.
    buf = ggml_backend_cpu_buffer_from_ptr(region, total);
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    LLAMA_LOG_WARN("k9 residency: region %.2f GiB mapped (NORESERVE, CPU buffer)%s\n",
                   total / 1073741824.0,
                   budget ? format(", expert-page budget %.2f GiB", budget / 1073741824.0).c_str() : "");
}

bool k9_residency::home_fixed(struct ggml_tensor * cur, size_t offset) {
    if (cur->data != nullptr) {
        return false; // real data elsewhere — caller decodes eagerly
    }
    // the tensor may carry a DUMMY buffer (create_buffers marks K9-resident
    // tensors this way so the scheduler won't allocate them) — clearing it
    // is safe: the region buffer replaces it as the tensor's home
    cur->buffer = nullptr;
    ggml_backend_tensor_alloc(buf, cur, (uint8_t *) region + offset);
    return true;
}

void k9_residency::register_tensor(const std::string & name,
                                   const uint8_t * blob, size_t blob_len,
                                   const k9_tensor_meta & meta,
                                   const int64_t f_ne[2], ggml_type dst,
                                   struct ggml_tensor * cur, size_t nbytes,
                                   size_t region_off, bool in_arena, bool moe) {
    auto & e = by_name[name];
    if (e.blob != nullptr) {
        LLAMA_LOG_ERROR("k9 residency: DUPLICATE REGISTER for %s (previous blob=%p)\n",
                        name.c_str(), (void *) e.blob);
    }
    e.blob = blob;
    e.blob_len = blob_len;
    e.meta = meta;
    e.f_ne[0] = f_ne[0];
    e.f_ne[1] = f_ne[1];
    e.dst = dst;
    e.cur = cur;
    e.nbytes = nbytes;
    e.region_off = region_off;
    e.in_arena = in_arena;
    e.moe = moe && in_arena; // external MoE tensors decode eagerly in the loader
    e.n_exp = (uint32_t)(e.moe ? cur->ne[2] : 0);
    e.lanes = meta.lanes ? meta.lanes : 1;
    e.rows = (size_t) f_ne[1];
    e.row_bytes = e.rows ? nbytes / e.rows : 0;
    e.rows_per = (uint32_t) ((e.rows + e.lanes - 1) / e.lanes);
    if (e.lanes > 255) {
        throw std::runtime_error(format(
            "k9 residency: %s has %u lanes — inline page state caps at 255 (re-export with a larger --page-mib)",
            name.c_str(), e.lanes));
    }
    e.page_dec.fill(0);
    e.page_stamp.fill(0);
    e.ms = nullptr;
    e.stamp = 0;
    e.in_graph = false;
    e.last_use = -1;
    total_bytes += nbytes;
    LLAMA_LOG_WARN("k9 residency: registered %s (%.2f MiB, %u pages%s) — %zu tensors\n",
                   name.c_str(), nbytes / 1048576.0, e.lanes,
                   e.moe ? ", MoE" : "", by_name.size());
}

size_t k9_residency::page_off(const entry & e, uint32_t p) const {
    return e.region_off + (size_t) p * e.rows_per * e.row_bytes;
}

size_t k9_residency::page_bytes(const entry & e, uint32_t p) const {
    const size_t a = (size_t) p * e.rows_per;
    const size_t b = std::min<size_t>(e.rows, a + e.rows_per);
    return (b - a) * e.row_bytes;
}

bool k9_residency::ensure_page(entry & e, uint32_t p) {
    if ((e.page_dec[p >> 6] >> (p & 63)) & 1) {
        e.page_stamp[p] = (uint32_t) ++clock;
        return true;
    }
    const size_t sz = page_bytes(e, p);
    if (e.ms == nullptr) {
        // cached scales: decoded once (the ent8 stream is whole-tensor)
        const int64_t ng = e.f_ne[0] / (int64_t) e.meta.group;
        e.ms = (float *) malloc(sizeof(float) * e.rows * ng);
        std::string err;
        if (!k9_decode_scales(e.blob, e.blob_len, e.meta, e.rows * ng, e.ms, err)) {
            throw std::runtime_error(format("k9 residency: scale decode of %s failed: %s",
                                            e.meta.name.c_str(), err.c_str()));
        }
    }
    std::string err;
    // dest = the tensor's region base: k9_decode_page packs the page's rows
    // at their global materialized positions (verified byte-equal in v2)
    if (!k9_decode_page(e.blob, e.blob_len, e.meta, e.f_ne, e.dst, p,
                        e.rows_per, e.ms, (uint8_t *) region + e.region_off, err)) {
        throw std::runtime_error(format("k9 residency: page decode of %s failed: %s",
                                        e.meta.name.c_str(), err.c_str()));
    }
    e.page_dec[p >> 6] |= (uint64_t)(1ull << (p & 63));
    if (env_flag("K9_RESIDENT_DEBUG")) {
        uint64_t h = 1469598103934665603ull;
        const unsigned char * pp = (const unsigned char *) region + page_off(e, p);
        for (size_t b = 0; b < 4096 && b < sz; b++) {
            h = (h ^ pp[b]) * 1099511628211ull;
        }
        LLAMA_LOG_WARN("k9 residency: decoded %s page %u fnv=%llx (moedec=%.1f MiB)\n",
                       e.meta.name.c_str(), p, (unsigned long long) h, moe_decoded / 1048576.0);
    }
    e.page_stamp[p] = (uint32_t) ++clock;
    moe_decoded += sz;
    n_mat++;
    return true;
}

bool k9_residency::moe_ensure(const struct ggml_tensor * weights, const struct ggml_tensor * ids) {
    const char * nm = ggml_get_name(weights);
    if (!nm || !nm[0]) {
        return false;
    }
    auto it = by_name.find(std::string_view(nm));
    if (it == by_name.end() || !it->second.in_arena || !it->second.moe) {
        return false;
    }
    if (ids->type != GGML_TYPE_I32 || ids->data == nullptr) {
        return false;
    }
    moe_ensure_ids(weights, (const int32_t *) ids->data, ids->ne[0], ids->ne[1]);
    return true;
}

void k9_residency::moe_ensure_ids(const struct ggml_tensor * weights,
                                  const int32_t * idata, int64_t n_used, int64_t n_tok) {
    const char * nm = ggml_get_name(weights);
    if (!nm || !nm[0] || !idata) {
        return;
    }
    auto it = by_name.find(std::string_view(nm));
    if (it == by_name.end() || !it->second.in_arena || !it->second.moe) {
        return;
    }
    entry & e = it->second;
    if (env_flag("K9_RESIDENT_DEBUG")) {
        // pointer audit: does the graph's tensor still point at its
        // registered region home? (a mismatch means the kernel reads bytes
        // the hook never wrote)
        const uint8_t * reg = (const uint8_t *) region + e.region_off;
        const uint8_t * gph = (const uint8_t *) weights->data;
        uint64_t hg = 1469598103934665603ull;
        for (size_t b = 0; b < 4096 && b < e.nbytes; b++) {
            hg = (hg ^ gph[b]) * 1099511628211ull;
        }
        uint64_t hr = 1469598103934665603ull;
        for (size_t b = 0; b < 4096 && b < e.nbytes; b++) {
            hr = (hr ^ reg[b]) * 1099511628211ull;
        }
        static int ptr_dbg = 0;
        if (ptr_dbg < 8) {
            ptr_dbg++;
            LLAMA_LOG_ERROR("k9 residency: ptr %s reg=%p gph=%p delta=%lld reg_fnv=%llx gph_fnv=%llx buf=%s buft=%s\n",
                           e.meta.name.c_str(), (void *) reg, (void *) gph,
                           (long long) (gph - reg), (unsigned long long) hr, (unsigned long long) hg,
                           weights->buffer ? ggml_backend_buffer_name(weights->buffer) : "(null)",
                           weights->buffer ? ggml_backend_buft_name(ggml_backend_buffer_get_type(weights->buffer)) : "(null)");
        }
    }
    if (env_flag("K9_RESIDENT_DEBUG")) {
        uint32_t bits = 0;
        for (uint32_t p = 0; p < e.lanes; p++) {
            bits += (uint32_t)((e.page_dec[p >> 6] >> (p & 63)) & 1);
        }
        LLAMA_LOG_WARN("k9 residency: hook-enter %s [&e=%p] bits=%u/%u moe_dec=%.1f MiB\n",
                       e.meta.name.c_str(), (void *) &e, bits, e.lanes,
                       moe_decoded / 1048576.0);
    }

    const uint32_t n_exp = e.n_exp;
    if (n_exp == 0 || e.rows % n_exp != 0) {
        throw std::runtime_error(format(
            "k9 residency: MoE tensor %s rows (%zu) not divisible by experts (%u)",
            e.meta.name.c_str(), e.rows, n_exp));
    }
    const size_t expert_rows = e.rows / n_exp;

    // RESERVATION semantics: collect the WHOLE call's lane set first, evict
    // only non-set pages until the set fits the budget, THEN decode. A
    // per-page check lets the call evict its own earlier pages (each new
    // lane eats the previous ones) and the kernel reads zero pages.
    std::vector<std::pair<entry *, uint32_t>> want;
    auto want_has = [&](const entry * en, uint32_t p) {
        for (const auto & w : want) {
            if (w.first == en && w.second == p) {
                return true;
            }
        }
        return false;
    };
    std::vector<uint64_t> seen((n_exp + 63) / 64, 0);
    size_t need = 0;      // bytes to decode for this call
    size_t in_set_resident = 0; // bytes of the set already decoded
    for (int64_t j = 0; j < n_used * n_tok; j++) {
        const int32_t id = idata[j];
        if (id < 0 || (uint32_t) id >= n_exp || ((seen[id >> 6] >> (id & 63)) & 1)) {
            continue;
        }
        seen[id >> 6] |= (uint64_t)(1ull << (id & 63));
        const size_t a = (size_t) id * expert_rows;
        const size_t b = a + expert_rows;
        for (uint32_t p = (uint32_t)(a / e.rows_per);
             p <= (uint32_t)((b - 1) / e.rows_per) && p < e.lanes; p++) {
            if (want_has(&e, p)) {
                continue;
            }
            want.emplace_back(&e, p);
            const size_t sz = page_bytes(e, p);
            if ((e.page_dec[p >> 6] >> (p & 63)) & 1) {
                in_set_resident += sz;
            } else {
                need += sz;
            }
        }
    }

    // evict LRU pages OUTSIDE the set until the call fits
    while (budget != 0 && moe_decoded - in_set_resident + need > budget) {
        entry * ve = nullptr;
        uint32_t vp = 0;
        for (auto & [en2, en] : by_name) {
            if (!en.moe) {
                continue;
            }
            for (uint32_t p = 0; p < en.lanes; p++) {
                const bool dec = (en.page_dec[p >> 6] >> (p & 63)) & 1;
                if (dec && !want_has(&en, p) &&
                        (!ve || en.page_stamp[p] < ve->page_stamp[vp])) {
                    ve = &en;
                    vp = p;
                }
            }
        }
        if (!ve) {
            // accounting audit: bitmap-derived decoded bytes vs the counter
            size_t bitmap_bytes = 0;
            int audit_n = 0;
            for (auto & [en2, en] : by_name) {
                if (!en.moe) {
                    continue;
                }
                size_t entry_bits = 0;
                for (uint32_t p = 0; p < en.lanes; p++) {
                    if ((en.page_dec[p >> 6] >> (p & 63)) & 1) {
                        entry_bits++;
                        bitmap_bytes += page_bytes(en, p);
                    }
                }
                if (audit_n < 3) {
                    LLAMA_LOG_ERROR("k9 residency:   audit %s lanes=%u rows=%zu row_bytes=%zu bits=%zu first=%lx,%lx page0=%zu\n",
                                    en2.c_str(), en.lanes, en.rows, en.row_bytes,
                                    entry_bits,
                                    en.lanes > 0 ? en.page_dec[0] : 0ul,
                                    en.lanes > 1 ? en.page_dec[1] : 0ul,
                                    en.lanes > 0 ? page_bytes(en, 0) : 0ul);
                    audit_n++;
                }
            }
            LLAMA_LOG_ERROR("k9 residency: AUDIT moe_decoded=%.2f MiB bitmap=%.2f MiB drift=%.2f MiB (in_set=%.2f MiB, need=%.2f MiB)\n",
                            moe_decoded / 1048576.0, bitmap_bytes / 1048576.0,
                            ((double) moe_decoded - (double) bitmap_bytes) / 1048576.0,
                            in_set_resident / 1048576.0, need / 1048576.0);
            throw std::runtime_error(format(
                "k9 residency: expert-page budget %.2f GiB smaller than one op's "
                "selected-expert set (%.2f MiB) and nothing outside it to evict — raise K9_BUDGET_GIB",
                budget / 1073741824.0, need / 1048576.0));
        }
        const size_t off = page_off(*ve, vp);
        const size_t sz  = page_bytes(*ve, vp);
        // inward rounding: adjacent region slots share 4K pages
        k9_madvise_drop((uint8_t *) region + off, sz, true); // back to zero pages
        ve->page_dec[vp >> 6] &= ~(uint64_t)(1ull << (vp & 63));
        ve->page_stamp[vp] = 0;
        moe_decoded -= std::min(sz, moe_decoded);
        n_evict++;
        if (n_evict <= 8 || n_evict % 512 == 0) {
            LLAMA_LOG_WARN("k9 residency: evicted %s page %u (%.2f MiB) — evictions %zu\n",
                           ve->meta.name.c_str(), vp, sz / 1048576.0, n_evict);
        }
    }

    // decode the set (all fits now; per-page stamps refresh in order)
    for (const auto & w : want) {
        ensure_page(*w.first, w.second);
    }
    // drop the blob's file pages for the decoded lanes: the digit streams
    // re-fault from the file on the next decode of that range, so peak RSS
    // tracks the expert budget instead of the whole blob mapping
    {
        const size_t otab = 4ull * (e.lanes + 1);
        const uint32_t * ot = (const uint32_t *) (e.blob + e.meta.tbl);
        for (const auto & w : want) {
            if (w.first != &e) {
                continue;
            }
            const uint32_t p = w.second;
            const size_t off = e.meta.tbl + otab + ot[p];
            const size_t len = ot[p + 1] - ot[p];
            // outward rounding: file-backed pages are safe to over-drop
            k9_madvise_drop((uint8_t *) e.blob + off, len, false);
        }
    }
}

void k9_residency::moe_ensure_experts(const struct ggml_tensor * weights,
                                      const int32_t * experts, size_t n) {
    const char * nm = ggml_get_name(weights);
    if (!nm || !nm[0]) {
        return;
    }
    auto it = by_name.find(std::string_view(nm));
    if (it == by_name.end() || !it->second.in_arena || !it->second.moe) {
        return;
    }
    // reuse the reservation pipeline: the expert list is the id payload
    moe_ensure_ids(weights, experts, (int64_t) n, 1);
}

unsigned long long k9_residency::debug_expert_fnv(std::string_view name, int32_t expert) const {
    auto it = by_name.find(name);
    if (it == by_name.end() || region == nullptr) {
        return 0;
    }
    const entry & e = it->second;
    if (expert >= 0 && (e.n_exp == 0 || (uint32_t) expert >= e.n_exp)) {
        return 0;
    }
    const size_t expert_rows = e.n_exp > 0 ? e.rows / e.n_exp : e.rows;
    const size_t off = expert >= 0 ? (size_t) expert * expert_rows * e.row_bytes : 0;
    const size_t len = expert >= 0 ? expert_rows * e.row_bytes : e.nbytes;
    const unsigned char * p = (const unsigned char *) region + e.region_off + off;
    unsigned long long h = 1469598103934665603ull;
    for (size_t b = 0; b < 4096 && b < len; b++) {
        h = (h ^ p[b]) * 1099511628211ull;
    }
    return h;
}

void k9_residency::ensure_graph(const struct ggml_cgraph * gf) {
    if (by_name.empty() || gf == nullptr) {
        return;
    }
    static bool traced = false;
    for (auto & [nm, e] : by_name) {
        e.in_graph = false;
    }
    const int n = ggml_graph_n_nodes(const_cast<struct ggml_cgraph *>(gf));

    // mark membership (weights are graph LEAFS — walk node srcs, depth 2;
    // match by name; rebind to the graph's object, preferring non-views)
    size_t matched = 0, distinct = 0;
    std::function<void(ggml_tensor *, int)> visit = [&](ggml_tensor * t, int depth) {
        if (!t) {
            return;
        }
        const char * nm = ggml_get_name(t);
        if (nm && nm[0]) {
            auto it = by_name.find(std::string_view(nm));
            if (it != by_name.end()) {
                entry & e = it->second;
                if (!e.in_graph) {
                    distinct++;
                }
                e.in_graph = true;
                if (e.cur == nullptr || (e.cur->view_src && !t->view_src)) {
                    e.cur = t;
                }
                matched++;
            }
        }
        if (depth > 0) {
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                if (!t->src[s]) {
                    break;
                }
                visit(t->src[s], depth - 1);
            }
        }
    };
    for (int i = 0; i < n; i++) {
        ggml_tensor * node = ggml_graph_node(const_cast<struct ggml_cgraph *>(gf), i);
        if (node) {
            visit(node, 2);
        }
    }
    if (!traced) {
        traced = true;
        LLAMA_LOG_WARN("k9 residency: first ensure_graph: %d nodes, %zu registered, "
                       "%zu matched, %zu distinct\n",
                       n, by_name.size(), matched, distinct);
        if (env_flag("K9_RESIDENT_DEBUG")) {
            for (auto & [nm, e] : by_name) {
                if (!e.in_graph) {
                    LLAMA_LOG_WARN("k9 residency: NOT in graph: %s (moe=%d)\n",
                                   nm.c_str(), (int) e.moe);
                }
            }
        }
    }

    // fill non-MoE tensors wholly (first use, then resident forever); MoE
    // expert tensors are hook-managed at mul_mat_id execution time
    for (auto & [nm, e] : by_name) {
        if (!e.in_graph || e.moe || !e.in_arena) {
            continue;
        }
        if (e.stamp != 0) {
            e.stamp = ++clock;
            continue;
        }
        std::string err;
        if (!k9_dequantize(e.blob, e.blob_len, e.meta, e.f_ne, e.dst,
                           (uint8_t *) region + e.region_off, err)) {
            throw std::runtime_error(format("k9 residency: decode of %s failed: %s",
                                            e.meta.name.c_str(), err.c_str()));
        }
        if (env_flag("K9_RESIDENT_DEBUG")) {
            uint64_t h = 1469598103934665603ull;
            const unsigned char * pp = (const unsigned char *) region + e.region_off;
            for (size_t b = 0; b < 4096 && b < e.nbytes; b++) {
                h = (h ^ pp[b]) * 1099511628211ull;
            }
            LLAMA_LOG_WARN("k9 residency: filled %s fnv=%llx (off=%.2f MiB, %.2f MiB) buf=%s\n",
                           e.meta.name.c_str(), (unsigned long long) h,
                           e.region_off / 1048576.0, e.nbytes / 1048576.0,
                           e.cur && e.cur->buffer ? ggml_backend_buffer_name(e.cur->buffer) : "(null)");
        }
        e.stamp = ++clock;
        n_mat++;
    }
}

// C hook (see ggml/include/k9-moe-hook.h), called from ggml-cpu's mul_mat_id.
// MUST NOT let exceptions cross the C frames of ggml-cpu.c ("terminate called
// recursively") — operational impossibilities abort loudly instead.
bool k9_res_moe_is_managed(const struct ggml_tensor * weights) {
    if (!k9_residency::active()) {
        return false;
    }
    const char * nm = ggml_get_name(weights);
    if (!nm || !nm[0]) {
        return false;
    }
    const bool r = k9_residency::get().is_managed(std::string_view(nm));
    static int dbg = 0;
    if (dbg < 6) {
        dbg++;
        LLAMA_LOG_ERROR("k9 debug: is_managed(%s) = %d (registry %zu)\n", nm, (int) r,
                        k9_residency::get().registry_size());
    }
    return r;
}

bool k9_res_moe_ensure(const struct ggml_tensor * weights, const struct ggml_tensor * ids) {
    if (!k9_residency::active()) {
        return false;
    }
    try {
        return k9_residency::get().moe_ensure(weights, ids);
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("k9 residency: moe_ensure failed: %s\n", e.what());
    } catch (...) {
        LLAMA_LOG_ERROR("k9 residency: moe_ensure failed (unknown exception)\n");
    }
    abort();
}

// C hook, MoE-cache flavor: decode the lanes covering the given expert ids
// of one managed tensor (the cache upload calls this before copying bytes
// host→device). Same no-exceptions contract.
void k9_res_moe_ensure_experts(const struct ggml_tensor * weights, const int32_t * experts, size_t n) {
    if (!k9_residency::active() || n == 0) {
        return;
    }
    const char * nm = ggml_get_name(weights);
    if (!nm || !nm[0]) {
        return;
    }
    try {
        k9_residency::get().moe_ensure_experts(weights, experts, n);
        return;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("k9 residency: moe_ensure_experts failed: %s\n", e.what());
    } catch (...) {
        LLAMA_LOG_ERROR("k9 residency: moe_ensure_experts failed (unknown exception)\n");
    }
    abort();
}

// C hook, HIP flavor (ggml-cuda.cu's mul_mat_id): ids already copied to
// host after the stream sync. Same no-exceptions contract.
void k9_res_moe_ensure_ids(const struct ggml_tensor * weights, const int32_t * ids, int64_t n_used, int64_t n_tok) {
    if (!k9_residency::active()) {
        LLAMA_LOG_ERROR("k9 residency: HIP moe hook fired with K9_RESIDENT off\n");
        abort();
    }
    try {
        k9_residency::get().moe_ensure_ids(weights, ids, n_used, n_tok);
        return;
    } catch (const std::exception & e) {
        LLAMA_LOG_ERROR("k9 residency: moe_ensure_ids failed: %s\n", e.what());
    } catch (...) {
        LLAMA_LOG_ERROR("k9 residency: moe_ensure_ids failed (unknown exception)\n");
    }
    abort();
}

bool k9_res_moe_is_managed_loose(const char * name) {
    if (!k9_residency::active() || !name || !name[0]) {
        return false;
    }
    std::string nm = name;
    const size_t p1 = nm.find('#');
    const size_t p2 = nm.rfind('#');
    if (p1 != std::string::npos && p2 != p1) {
        nm = nm.substr(p1 + 1, p2 - p1 - 1);
    }
    return k9_residency::get().is_managed(nm);
}

bool k9_residency_is_active(void) {
    return k9_residency::active();
}

unsigned long long k9_res_debug_expert_fnv(const char * name, int32_t expert) {
    if (!k9_residency::active() || !name) {
        return 0;
    }
    return k9_residency::get().debug_expert_fnv(name, expert);
}
