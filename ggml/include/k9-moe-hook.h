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

// Decode the lanes covering the given expert ids of a K9 paged MoE tensor
// into its residency region (no-op for non-managed tensors). Used by the
// MoE cache upload path before it copies expert bytes host→device.
void k9_res_moe_ensure_experts(const struct ggml_tensor * weights,
                               const int32_t * experts, size_t n);

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


// Managed check that tolerates scheduler copy-name decorations
// ("ROCm0#blk.0.ffn_gate_exps.weight#0"). The decorated copies hide the K9
// identity from the plain name lookup, which wrongly re-enables fusion for
// paged weights (docs/13 §2.6).
bool k9_res_moe_is_managed_loose(const char * name);

// The residency region pointer/size (null/0 before prepare_region, or
// when residency is off). HIP-side code uses these to pin the region once
// (hipHostRegister) so kernel reads of decoded pages are coherent.
void * k9_res_region_pointer(void);
size_t k9_res_region_size(void);

// True when K9 residency is active (K9_RESIDENT=1 and a region is mapped).
// Graph-capture decisions must consult this: replayed graphs skip the
// host-side decode hooks.
bool k9_residency_is_active(void);

// k9 debug: FNV(4096) of `expert`'s bytes of the managed tensor `name` in
// the residency region (decode-on-demand NOT triggered). Returns 0 when
// unknown.
unsigned long long k9_res_debug_expert_fnv(const char * name, int32_t expert);

#ifdef __cplusplus
}
#endif
