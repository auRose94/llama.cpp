// v3 residency implementation — see k9-residency.h (docs/13 §2.6, gate 3).
#include "k9-residency.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>

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

void k9_residency::reset() {
    by_name.clear();
    total_bytes = 0;
    max_bytes = 0;
    materialized_bytes = 0;
    clock = 0;
    budget = 0;
    n_hits = n_mat = n_evict = 0;
}

void k9_residency::register_tensor(const std::string & name,
                                   const uint8_t * blob, size_t blob_len,
                                   const k9_tensor_meta & meta,
                                   const int64_t f_ne[2], ggml_type dst,
                                   struct ggml_tensor * cur, size_t nbytes) {
    auto & e = by_name[name];
    e.blob = blob;
    e.blob_len = blob_len;
    e.meta = meta;
    e.f_ne[0] = f_ne[0];
    e.f_ne[1] = f_ne[1];
    e.dst = dst;
    e.cur = cur;
    e.nbytes = nbytes;
    e.materialized = false;
    e.stamp = 0;
    e.in_graph = false;
    total_bytes += nbytes;
    max_bytes = std::max(max_bytes, nbytes);
    LLAMA_LOG_WARN("k9 residency: registered %s (%.2f MiB, %u pages) — %zu tensors, %.2f GiB total\n",
                   name.c_str(), nbytes / 1048576.0, meta.lanes, by_name.size(),
                   total_bytes / 1073741824.0);
}

void k9_residency::evict_until(size_t need, int64_t before_idx) {
    while (materialized_bytes + need > budget) {
        // victim: any materialized tensor NOT used by the current node.
        // last_use > before_idx: pass-2 re-decodes it when its node arrives.
        // last_use <  before_idx: not needed again this token; the next
        //   token's pass re-fills it at its node.
        // last_use == before_idx: pinned — it is (or may be) an src of the
        //   current node, so dropping it could corrupt this node's compute.
        // Oldest stamp first. Dense models under a sub-full budget =
        // worst-case thrash — exactly the correctness path docs/13 gate 3
        // exercises.
        entry * victim = nullptr;
        for (auto & [nm, e] : by_name) {
            if (e.materialized && e.last_use != before_idx &&
                    (!victim || e.stamp < victim->stamp)) {
                victim = &e;
            }
        }
        if (!victim) {
            if (env_flag("K9_RESIDENT_DEBUG")) {
                size_t held = 0, held_pinned = 0;
                for (auto & [nm, e] : by_name) {
                    if (e.materialized) {
                        held += e.nbytes;
                        if (e.last_use <= before_idx) {
                            held_pinned += e.nbytes;
                            LLAMA_LOG_WARN("k9 residency:   pinned %s (%.2f MiB, last_use=%lld)\n",
                                           nm.c_str(), e.nbytes / 1048576.0, (long long) e.last_use);
                        }
                    }
                }
                LLAMA_LOG_WARN("k9 residency: materialized=%.2f MiB (pinned %.2f MiB) at node %lld\n",
                               held / 1048576.0, held_pinned / 1048576.0, (long long) before_idx);
            }
            throw std::runtime_error(format(
                "k9 residency: virtual budget %.2f GiB exhausted (need %.2f MiB, no evictable tensor) — raise K9_BUDGET_GIB",
                budget / 1073741824.0, need / 1048576.0));
        }
        // v1: memory stays allocated in the model buffer (in-place
        // materialization); the tensor is marked absent and re-decoded when
        // its node arrives. True memory relief lands with the
        // expert-granular rung.
        victim->materialized = false;
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
    if (budget == 0) {
        const double gib = env_float("K9_BUDGET_GIB");
        budget = gib > 0.0 ? (size_t)(gib * 1073741824.0) : total_bytes;
        if (gib > 0.0 && budget < max_bytes) {
            throw std::runtime_error(format(
                "k9 residency: K9_BUDGET_GIB=%.2f smaller than the largest tensor (%.2f GiB)",
                gib, max_bytes / 1073741824.0));
        }
        LLAMA_LOG_WARN("k9 residency: budget %.2f GiB for %.2f GiB of K9 tensors (in-place materialization)\n",
                       budget / 1073741824.0, total_bytes / 1073741824.0);
    }
    evict_until(e.nbytes, node_idx);

    if (e.cur == nullptr || e.cur->data == nullptr) {
        throw std::runtime_error(format(
            "k9 residency: %s has no assigned buffer to materialize into", e.meta.name.c_str()));
    }
    std::string err;
    if (!k9_dequantize(e.blob, e.blob_len, e.meta, e.f_ne, e.dst, e.cur->data, err)) {
        throw std::runtime_error(format("k9 residency: decode of %s failed: %s",
                                        e.meta.name.c_str(), err.c_str()));
    }
    if (env_flag("K9_RESIDENT_DEBUG")) {
        const size_t nchk = std::min<size_t>(e.nbytes, 1 << 20);
        uint64_t h = 1469598103934665603ull;
        const uint8_t * p = (const uint8_t *) e.cur->data;
        for (size_t i = 0; i < nchk; i++) {
            h = (h ^ p[i]) * 1099511628211ull;
        }
        LLAMA_LOG_WARN("k9 residency: materialized %s: cur=%p data=%p view_src=%p nbytes=%zu fnv=%llx\n",
                       e.meta.name.c_str(), (void *) e.cur, e.cur->data,
                       (void *) e.cur->view_src, e.nbytes, (unsigned long long) h);
    }
    e.materialized = true;
    e.stamp = ++clock;
    materialized_bytes += e.nbytes;
    n_mat++;
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
    // drop tensors whose last use is behind the current node.
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
                    if (e.materialized) {
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
