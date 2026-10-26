#pragma once

#include "ggml-common.h"
#include "convert.cuh"

static __device__ __forceinline__ int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

static __device__ void quantize_f32_q4_0_block(const float * __restrict__ x, block_q4_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -8;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK4_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK4_0/2 + j]*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 8.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 8.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q4_1_block(const float * __restrict__ x, block_q4_1 * __restrict__ y) {
    float vmin = FLT_MAX;
    float vmax = -FLT_MAX;

    for (int j = 0; j < QK4_1; ++j) {
        const float v = x[j];
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
    }

    const float d  = (vmax - vmin) / ((1 << 4) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = vmin;

    for (int j = 0; j < QK4_1/2; ++j) {
        const float x0 = (x[0       + j] - vmin)*id;
        const float x1 = (x[QK4_1/2 + j] - vmin)*id;

        const uint8_t xi0 = min(15, (int8_t)(x0 + 0.5f));
        const uint8_t xi1 = min(15, (int8_t)(x1 + 0.5f));

        y->qs[j]  = xi0;
        y->qs[j] |= xi1 << 4;
    }
}

static __device__ void quantize_f32_q5_0_block(const float * __restrict__ x, block_q5_0 * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK5_0; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    const float d  = vmax / -16;
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        const float x0 = x[0       + j]*id;
        const float x1 = x[QK5_0/2 + j]*id;

        const uint8_t xi0 = min(31, (int8_t)(x0 + 16.5f));
        const uint8_t xi1 = min(31, (int8_t)(x1 + 16.5f));

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_0/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q5_1_block(const float * __restrict__ x, block_q5_1 * __restrict__ y) {
    float min = x[0];
    float max = x[0];

    for (int j = 1; j < QK5_1; ++j) {
        const float v = x[j];
        min = v < min ? v : min;
        max = v > max ? v : max;
    }

    const float d  = (max - min) / 31;
    const float id = d ? 1.0f/d : 0.0f;

    y->dm.x = d;
    y->dm.y = min;

    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        const float x0 = (x[0       + j] - min)*id;
        const float x1 = (x[QK5_1/2 + j] - min)*id;

        const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
        const uint8_t xi1 = (uint8_t)(x1 + 0.5f);

        y->qs[j]  = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= ((xi1 & 0x10u) >> 4) << (j + QK5_1/2);
    }
    memcpy(y->qh, &qh, sizeof(qh));
}

static __device__ void quantize_f32_q8_0_block(const float * __restrict__ x, block_q8_0 * __restrict__ y) {
    float amax = 0.0f; // absolute max

    for (int j = 0; j < QK8_0; j++) {
        const float v = x[j];
        amax = fmaxf(amax, fabsf(v));
    }

    const float d = amax / ((1 << 7) - 1);
    const float id = d ? 1.0f/d : 0.0f;

    y->d = d;

    for (int j = 0; j < QK8_0; ++j) {
        const float x0 = x[j]*id;
        y->qs[j] = roundf(x0);
    }
}

static __device__ void quantize_f32_iq4_nl_block(const float * __restrict__ x, block_iq4_nl * __restrict__ y) {
    float amax = 0.0f;
    float vmax = 0.0f;

    for (int j = 0; j < QK4_NL; ++j) {
        const float v = x[j];
        if (amax < fabsf(v)) {
            amax = fabsf(v);
            vmax = v;
        }
    }

    float d = vmax / kvalues_iq4nl[0];
    const float id = d ? 1.0f/d : 0.0f;

    float sumqx = 0, sumq2 = 0;
    for (int j = 0; j < QK4_NL/2; ++j) {
        const float x0 = x[0        + j]*id;
        const float x1 = x[QK4_NL/2 + j]*id;
        const uint8_t xi0 = best_index_int8(16, kvalues_iq4nl, x0);
        const uint8_t xi1 = best_index_int8(16, kvalues_iq4nl, x1);
        y->qs[j] = xi0 | (xi1 << 4);
        const float v0 = kvalues_iq4nl[xi0];
        const float v1 = kvalues_iq4nl[xi1];
        const float w0 = x[0        + j]*x[0        + j];
        const float w1 = x[QK4_NL/2 + j]*x[QK4_NL/2 + j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2 + j];
        sumq2 += w0*v0*v0 + w1*v1*v1;
    }

    y->d = sumq2 > 0 ? sumqx/sumq2 : d;
}

// Wrapper functions for cpy.cu compatibility
static __device__ void cpy_blck_f32_q4_0(const char * cxi, char * cdsti) {
    quantize_f32_q4_0_block((const float *)cxi, (block_q4_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q4_1(const char * cxi, char * cdsti) {
    quantize_f32_q4_1_block((const float *)cxi, (block_q4_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_0(const char * cxi, char * cdsti) {
    quantize_f32_q5_0_block((const float *)cxi, (block_q5_0 *)cdsti);
}

static __device__ void cpy_blck_f32_q5_1(const char * cxi, char * cdsti) {
    quantize_f32_q5_1_block((const float *)cxi, (block_q5_1 *)cdsti);
}

static __device__ void cpy_blck_f32_q8_0(const char * cxi, char * cdsti) {
    quantize_f32_q8_0_block((const float *)cxi, (block_q8_0 *)cdsti);
}

// ninths KV layouts: per-64 fp16 scale, pre-biased codes (docs/08 grid).

static __device__ void quantize_f32_k9_4kv_block(const float * __restrict__ x, block_k9_4kv * __restrict__ y) {
    float amax = 0.0f;
    for (int j = 0; j < QK9_GROUP; ++j) {
        amax = fmaxf(amax, fabsf(x[j]));
    }
    y->ds = fmaxf(amax / 7.0f, 1e-4f);          // k15 grid
    const float d = (float) y->ds;
    for (int j = 0; j < QK9_GROUP/2; ++j) {
        int c0 = (int) roundf(x[2*j    ] / d);
        int c1 = (int) roundf(x[2*j + 1] / d);
        c0 = (int) fminf(fmaxf(c0, -7), 7);
        c1 = (int) fminf(fmaxf(c1, -7), 7);
        y->qs[j] = (uint8_t) (((c0 + 8) & 0x0F) | (((c1 + 8) & 0x0F) << 4));
    }
}

static __device__ void quantize_f32_k9_6kv_block(const float * __restrict__ x, block_k9_6kv * __restrict__ y) {
    float amax = 0.0f;
    for (int j = 0; j < QK9_GROUP; ++j) {
        amax = fmaxf(amax, fabsf(x[j]));
    }
    y->ds = fmaxf(amax / 31.0f, 1e-4f);         // k63 grid
    const float d = (float) y->ds;
    for (int j = 0; j < 16; ++j) {
        y->hh[j] = 0;
    }
    for (int j = 0; j < QK9_GROUP; ++j) {
        int c = (int) roundf(x[j] / d);
        c = (int) fminf(fmaxf(c, -31), 31) + 32;    // u6 = code + 32
        const uint8_t lb = (uint8_t) (c & 0x0F);
        const uint8_t h  = (uint8_t) ((c >> 4) & 0x03);
        const uint8_t lb_old = y->qs[j >> 1];
        y->qs[j >> 1] = (j & 1) ? (uint8_t) ((lb_old & 0x0F) | (lb << 4))
                                : (uint8_t) ((lb_old & 0xF0) | lb);
        y->hh[j >> 2] |= (uint8_t) (h << ((j & 3) * 2));
    }
}

static __device__ void cpy_blck_f32_k9_4kv(const char * cxi, char * cdsti) {
    quantize_f32_k9_4kv_block((const float *)cxi, (block_k9_4kv *)cdsti);
}

static __device__ void cpy_blck_f32_k9_6kv(const char * cxi, char * cdsti) {
    quantize_f32_k9_6kv_block((const float *)cxi, (block_k9_6kv *)cdsti);
}

static __device__ void cpy_blck_k9_4kv_f32(const char * cxi, char * cdsti) {
    const block_k9_4kv * x = (const block_k9_4kv *) cxi;
    float * cdstf = (float *) cdsti;
    const float d = (float) x->ds;
    for (int j = 0; j < QK9_GROUP/2; ++j) {
        const uint8_t byte = x->qs[j];
        *(cdstf + 2*j)     = ((int)(byte & 0x0F) - 8) * d;
        *(cdstf + 2*j + 1) = ((int)(byte >>   4) - 8) * d;
    }
}

static __device__ void cpy_blck_k9_6kv_f32(const char * cxi, char * cdsti) {
    const block_k9_6kv * x = (const block_k9_6kv *) cxi;
    float * cdstf = (float *) cdsti;
    const float d = (float) x->ds;
    for (int j = 0; j < QK9_GROUP; ++j) {
        const uint8_t lb = x->qs[j >> 1];
        const int nib = (j & 1) ? (lb >> 4) : (lb & 0x0F);
        const int h = (x->hh[j >> 2] >> ((j & 3) * 2)) & 0x03;
        *(cdstf + j) = (float)((int)((nib | (h << 4)) - 32)) * d;
    }
}

static __device__ void cpy_blck_f32_iq4_nl(const char * cxi, char * cdsti) {
    quantize_f32_iq4_nl_block((const float *)cxi, (block_iq4_nl *)cdsti);
}

template<typename src_t, typename dst_t>
static __device__ void cpy_1_scalar(const char * cxi, char * cdsti) {
    *(dst_t *) cdsti = ggml_cuda_cast<dst_t>(*(const src_t *) cxi);
}
