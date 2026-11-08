#pragma once
// K9 M3 hook (docs/13 §2.6): expert-granular page decode for paged MoE
// weights, called from the CPU mul_mat_id op where the router ids are
// actually known (they are a runtime tensor — pre-execution fills cannot
// know them). C-compatible: ggml-cpu.c compiles as C.
#include <stdbool.h>

struct ggml_tensor;

#ifdef __cplusplus
extern "C" {
#endif

// Registry-membership check ONLY (no decode, thread-safe to call from every
// worker thread): true when `weights` is a K9 paged MoE tensor this build
// manages. The caller runs the decode on thread 0 (k9_res_moe_ensure) and
// barriers all threads before the kernel.
bool k9_res_moe_is_managed(const struct ggml_tensor * weights);

// Decode the lanes covering the experts selected by `ids` for the K9 paged
// tensor `weights`, evicting LRU pages over the expert budget. MUST be
// called by exactly one thread (params->ith == 0) — concurrent calls race
// the registry state. Returns true when `weights` is registry-managed.
bool k9_res_moe_ensure(const struct ggml_tensor * weights,
                       const struct ggml_tensor * ids);

#ifdef __cplusplus
}
#endif
