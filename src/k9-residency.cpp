// v3 residency implementation — see k9-residency.h (docs/13 §2.6, gates 3+).
// M2.5: real-budget arena — K9 tensors dummy-home at load (poison slab),
// ensure_graph re-points data into decoded slots, LRU eviction frees them.
#include "k9-residency.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>

#include "llama-impl.h"

namespace {

constexpr size_t K9_SLOT_ALIGN  = 64;
constexpr size_t K9_POISON_SIZE = 1 << 20; // zeroed slab; evicted/dummy tensors point here

size_t round_up(size_t v, size_t a) {
    return (v + a - 1) / a * a;
}

bool env_flag(const char * name) {
    const char * v = getenv(name);
    return v && v[0] == '1';
}

double env_float(const char * name) {
    const char * v = getenv(name);
    return v ? atof(v) : 0.0;
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

k9_residency & k9_residency::get() {
    static k9_residency inst;
    return inst;
}

const void * k9_residency::poison_base() {
    return get().arena_base;
}

void k9_residency::reset() {
    if (arena) {
        ggml_backend_buffer_free(arena);
        arena = nullptr;
    }
    by_name.clear();
    free_ranges.clear();
    arena_base = nullptr;
    arena_size = 0;
    slot_start = 0;
    arena_high = 0;
    total_bytes = 0;
    max_bytes = 0;
    materialized_bytes = 0;
    clock = 0;
    budget = 0;
    n_hits = n_mat = n_evict = 0;
}

void k9_residency::prepare_arena(size_t total, size_t max_tensor) {
    if (arena) {
        return; // already prepared
    }
    const double gib = env_float("K9_BUDGET_GIB");
    // full-residency mode still needs the poison slab + per-tensor rounding
    // slack on top of the raw byte sum, or the frontier wedges at the last
    // tensor
    budget = gib > 0.0 ? (size_t)(gib * 1073741824.0)
                       : total + K9_POISON_SIZE + ((size_t) 4 << 20);
    const size_t need_min = K9_POISON_SIZE + round_up(max_tensor, K9_SLOT_ALIGN);
    if (budget < need_min) {
        throw std::runtime_error(format(
            "k9 residency: K9_BUDGET_GIB=%.2f too small — need at least the largest tensor (%.2f GiB) + poison slab",
            gib, (double) need_min / 1073741824.0));
    }
    arena = ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), budget);
    ggml_backend_buffer_set_usage(arena, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    arena_base = (uint8_t *) ggml_backend_buffer_get_base(arena);
    arena_size = budget;
    slot_start = K9_POISON_SIZE;
    arena_high = slot_start;
    memset(arena_base, 0, K9_POISON_SIZE); // poison slab reads as zeros
    LLAMA_LOG_WARN("k9 residency: arena %.2f GiB for %.2f GiB of K9 tensors%s\n",
                   budget / 1073741824.0, total / 1073741824.0,
                   budget < total ? " (paged)" : "");
}

bool k9_residency::home_dummy(struct ggml_tensor * cur) {
    if (cur->data != nullptr || cur->buffer != nullptr) {
        return false; // homed elsewhere — caller registers as external
    }
    // Overlapping dummy assignment: every K9 tensor points at the poison
    // slab until its first materialization re-points it. Individually each
    // tensor fits (prepare_arena enforces budget >= largest tensor), and
    // ggml_backend_tensor_alloc does no cumulative tracking, so the shared
    // base address is legal. This must happen BEFORE the scheduler's first
    // reserve, or unhomed tensors get transient compute-buffer homes whose
    // fills the ops never see (v1 gotcha).
    ggml_backend_tensor_alloc(arena, cur, arena_base);
    return true;
}

void k9_residency::register_tensor(const std::string & name,
                                   const uint8_t * blob, size_t blob_len,
                                   const k9_tensor_meta & meta,
                                   const int64_t f_ne[2], ggml_type dst,
                                   struct ggml_tensor * cur, size_t nbytes,
                                   bool in_arena) {
    auto & e = by_name[name];
    e.blob = blob;
    e.blob_len = blob_len;
    e.meta = meta;
    e.f_ne[0] = f_ne[0];
    e.f_ne[1] = f_ne[1];
    e.dst = dst;
    e.cur = cur;
    e.nbytes = nbytes;
    e.in_arena = in_arena;
    e.stamp = 0;
    e.in_graph = false;
    e.last_use = -1;
    total_bytes += nbytes;
    max_bytes = std::max(max_bytes, nbytes);
    LLAMA_LOG_WARN("k9 residency: registered %s (%.2f MiB, %u pages)%s — %zu tensors, %.2f GiB total\n",
                   name.c_str(), nbytes / 1048576.0, meta.lanes,
                   in_arena ? "" : " [external]",
                   by_name.size(), total_bytes / 1073741824.0);
}

size_t k9_residency::carve(size_t nbytes) {
    const size_t want = round_up(nbytes, K9_SLOT_ALIGN);
    // first fit over recycled ranges
    for (auto it = free_ranges.begin(); it != free_ranges.end(); ++it) {
        if (it->second >= want) {
            const size_t off = it->first;
            if (it->second > want) {
                *it = { it->first + want, it->second - want };
            } else {
                free_ranges.erase(it);
            }
            return off;
        }
    }
    // then the bump frontier (evict_until guarantees room in bytes; guard
    // the physical frontier against fragmentation edge cases)
    if (arena_high + want > arena_size) {
        throw std::runtime_error(format(
            "k9 residency: arena frontier exhausted (%.2f GiB arena, need %.2f MiB) — raise K9_BUDGET_GIB",
            arena_size / 1073741824.0, want / 1048576.0));
    }
    const size_t off = arena_high;
    arena_high += want;
    return off;
}

void k9_residency::evict_until(size_t need, int64_t before_idx) {
    // PHYSICAL condition: the arena holds the poison slab + rounded slots,
    // so a byte-sum check against budget can leave eviction asleep while
    // the frontier hits the wall. Evict while physical free space cannot
    // hold the rounded request.
    const size_t want = round_up(need, K9_SLOT_ALIGN);
    // CONTIGUITY-aware condition: scattered crumbs can sum past the request
    // while no single span fits it. Evict until the largest contiguous span
    // (bump room or a free range) can hold `want`.
    auto max_contig = [&]() {
        size_t m = arena_size - arena_high;
        for (const auto & r : free_ranges) {
            m = std::max(m, r.second);
        }
        return m;
    };
    while (max_contig() < want) {
        // EXECUTION-SAFE victims ONLY: last_use > before_idx — such a tensor
        // is re-decoded at its next-use node later in THIS ensure pass, so
        // every execution read sees decoded bytes. A tensor with
        // last_use <= before_idx would be read by the graph's execution
        // AFTER this ensure with no re-fill in between (fills all happen
        // pre-execution) — freeing it corrupts the run. For DENSE models
        // every tensor is used every token, so a sub-working-set budget
        // cannot ever evict: the error below is the honest answer, and real
        // memory relief for MoE lands at the expert-granular rung (M3),
        // where the graph only touches selected experts per token.
        entry * victim = nullptr;
        for (auto & [nm, e] : by_name) {
            if (e.in_arena && e.resident() && e.last_use > before_idx &&
                    (!victim || e.stamp < victim->stamp)) {
                victim = &e;
            }
        }
        if (!victim) {
            throw std::runtime_error(format(
                "k9 residency: arena budget %.2f GiB below this graph's working set "
                "(need %.2f MiB contiguous; dense models need the full model as budget — "
                "sub-working-set paging arrives with expert-granular M3). Raise K9_BUDGET_GIB.",
                budget / 1073741824.0, need / 1048576.0));
        }
        // free the slot FIRST (read the offset while data still points at
        // it), then poison — swapping this order once cost a debugging hour
        if (env_flag("K9_RESIDENT_DEBUG")) {
            LLAMA_LOG_WARN("k9 residency: victim %s [&e=%p] data=%p last_use=%lld by node %lld\n",
                           victim->meta.name.c_str(), (void *) victim,
                           victim->cur->data, (long long) victim->last_use,
                           (long long) before_idx);
        }
        const size_t off = (size_t)((uint8_t *) victim->cur->data - arena_base);
        const size_t sz  = round_up(victim->nbytes, K9_SLOT_ALIGN);
        bool merged = false;
        for (auto & r : free_ranges) {
            if (r.first + r.second == off) {
                r.second += sz;
                merged = true;
                break;
            }
            if (off + sz == r.first) {
                r.first = off;
                r.second += sz;
                merged = true;
                break;
            }
        }
        if (!merged) {
            free_ranges.emplace_back(off, sz);
        }
        if (env_flag("K9_RESIDENT_DEBUG")) {
            LLAMA_LOG_WARN("k9 residency: evicted %s slot@%.2f MiB by node %lld (victim last_use=%lld)\n",
                           victim->meta.name.c_str(), off / 1048576.0,
                           (long long) before_idx, (long long) victim->last_use);
        }
        victim->cur->data = arena_base; // poison slab: reads give zeros, never stale garbage
        victim->stamp = 0;
        materialized_bytes -= victim->nbytes;
        n_evict++;
        if (n_evict <= 8 || n_evict % 512 == 0) {
            LLAMA_LOG_WARN("k9 residency: evicted %s (evictions %zu)\n",
                           victim->meta.name.c_str(), n_evict);
        }
    }
}

void k9_residency::materialize_entry(entry & e, int64_t node_idx) {
    if (!e.in_arena) {
        if (e.stamp == 0) {
            // externally homed: decode in place once (stamp doubles as the
            // filled-flag for externals), never evicted
            std::string err;
            if (!k9_dequantize(e.blob, e.blob_len, e.meta, e.f_ne, e.dst, e.cur->data, err)) {
                throw std::runtime_error(format("k9 residency: decode of %s failed: %s",
                                                e.meta.name.c_str(), err.c_str()));
            }
            e.stamp = ++clock;
            materialized_bytes += e.nbytes;
            n_mat++;
        }
        return;
    }
    evict_until(e.nbytes, node_idx);
    const size_t off = carve(e.nbytes);
    uint8_t * slot = arena_base + off;
    std::string err;
    if (!k9_dequantize(e.blob, e.blob_len, e.meta, e.f_ne, e.dst, slot, err)) {
        throw std::runtime_error(format("k9 residency: decode of %s failed: %s",
                                        e.meta.name.c_str(), err.c_str()));
    }
    e.cur->data = slot;
    e.stamp = ++clock;
    materialized_bytes += e.nbytes;
    n_mat++;
    if (env_flag("K9_RESIDENT_DEBUG") && n_mat <= 24) {
        LLAMA_LOG_WARN("k9 residency: carve #%zu %s (%.2f MiB) at node %lld — high=%.1f MiB mat=%.1f MiB\n",
                       n_mat, e.meta.name.c_str(), e.nbytes / 1048576.0,
                       (long long) node_idx, (arena_high - slot_start) / 1048576.0,
                       materialized_bytes / 1048576.0);
    }
}

void k9_residency::ensure_graph(const struct ggml_cgraph * gf) {
    if (by_name.empty() || gf == nullptr) {
        return;
    }
    static bool traced = false;
    for (auto & [nm, e] : by_name) {
        e.in_graph = false;
        e.last_use = -1;
    }
    const int n = ggml_graph_n_nodes(const_cast<struct ggml_cgraph *>(gf));

    // Weights are graph LEAFS (srcs of ops), not nodes, and ggml_cgraph is
    // opaque with no leaf accessors — so walk node srcs (depth 2 covers
    // op -> view -> weight chains). Match by name (stable, unique) and
    // rebind the entry to the graph's tensor object, preferring non-view
    // tensors (a view's data is a subrange — the full-tensor decode must
    // land in the owning tensor).
    size_t matched = 0, distinct = 0, n_view = 0;
    std::function<void(ggml_tensor *, int, int)> visit = [&](ggml_tensor * t, int depth, int idx) {
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
                e.last_use = std::max<int64_t>(e.last_use, idx);
                if (e.cur == nullptr || (e.cur->view_src && !t->view_src)) {
                    e.cur = t;
                }
                if (t->view_src) {
                    n_view++;
                }
                matched++;
            }
        }
        if (depth > 0) {
            for (int s = 0; s < GGML_MAX_SRC; s++) {
                if (!t->src[s]) {
                    break;
                }
                visit(t->src[s], depth - 1, idx);
            }
        }
    };

    for (int i = 0; i < n; i++) {
        ggml_tensor * node = ggml_graph_node(const_cast<struct ggml_cgraph *>(gf), i);
        if (node) {
            visit(node, 2, i);
        }
    }
    if (!traced) {
        traced = true;
        LLAMA_LOG_WARN("k9 residency: first ensure_graph: %d nodes, %zu registered, "
                       "%zu matched, %zu distinct, %zu view-bound\n",
                       n, by_name.size(), matched, distinct, n_view);
    }

    // materialize misses in node order; LRU-hit the rest. Eviction may only
    // drop tensors whose last use is not the current node.
    for (int i = 0; i < n; i++) {
        ggml_tensor * node = ggml_graph_node(const_cast<struct ggml_cgraph *>(gf), i);
        if (!node) {
            continue;
        }
        std::function<void(ggml_tensor *, int)> fill = [&](ggml_tensor * t, int depth) {
            if (!t) {
                return;
            }
            const char * nm = ggml_get_name(t);
            if (nm && nm[0]) {
                auto it = by_name.find(std::string_view(nm));
                if (it != by_name.end()) {
                    entry & e = it->second;
                    if (e.in_arena && e.stamp != 0 && !e.resident()) {
                        // resident-state is derived; an arena tensor with a
                        // fill stamp but poisoned data was invalidated out-
                        // side the evict path — log it and self-heal by
                        // re-decoding below
                        LLAMA_LOG_WARN("k9 residency: invalidated %s at node %lld — re-decoding\n",
                                       nm, (long long) i);
                    }
                    if (e.resident() || (!e.in_arena && e.stamp != 0)) {
                        e.stamp = ++clock;
                        n_hits++;
                    } else {
                        materialize_entry(e, i);
                    }
                }
            }
            if (depth > 0) {
                for (int s = 0; s < GGML_MAX_SRC; s++) {
                    if (!t->src[s]) {
                        break;
                    }
                    fill(t->src[s], depth - 1);
                }
            }
        };
        fill(node, 2);
    }
}
