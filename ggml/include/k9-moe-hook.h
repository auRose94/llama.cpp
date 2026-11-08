#pragma once
// K9 M3 hook (docs/13 §2.6): expert-granular page decode for paged MoE
// weights. CPU: called from the mul_mat_id op (ggml-cpu.c) where the router
// ids are actually known. HIP: called from ggml_cuda_mul_mat_id with the
// ids copied out after a stream sync (they are a runtime device tensor —
// pre-execution fills cannot know them). C-compatible: ggml-cpu.c compiles
// as C.
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registry-membership check ONLY (no decode, thread-safe to call from every
// worker thread): true when `weights` is a K9 paged MoE tensor this build
// manages. The caller runs the decode on thread 0 (k9_res_moe_ensure) and
// barriers all threads before the kernel.
bool k9_res_moe_is_managed(const struct ggml_tensor * weights);

// Decode the lanes covering the experts selected by `ids` for the K9 paged
// tensor `weights`, evicting LRU pages over the expert budget. Returns true
// when `weights` is registry-managed (the caller must then run a thread
// barrier before the kernel, since thread 0 did the decoding work).
bool k9_res_moe_ensure(const struct ggml_tensor * weights,
                       const struct ggml_tensor * ids);

// HIP path: ids already copied to host (caller syncs the stream first).
// Aborts loudly on operational impossibility — never returns false for a
// managed tensor that simply cannot fit the budget.
void k9_res_moe_ensure_ids(const struct ggml_tensor * weights,
                           const int32_t * ids, int64_t n_used, int64_t n_tok);

// HIP region registration: makes a host mapping device-readable (HMM on
// APUs — the probe in docs/13 §2.6). CPU builds: returns false.
bool k9_res_hip_register(void * ptr, size_t size);
void k9_res_hip_unregister(void * ptr);
int k9_res_hip_primary_device(void);

// Buffer over the registered region with the HIP buffer type (so the
// scheduler routes K9-tensor ops to the GPU) — kernels read the host
// pointers directly through HMM. The region's memory stays owned by the
// residency registry (the buffer's free is a no-op).
ggml_backend_buffer_t k9_res_hip_region_buffer(void * host_ptr, size_t size, int device);

#ifdef __cplusplus
}
#endif
