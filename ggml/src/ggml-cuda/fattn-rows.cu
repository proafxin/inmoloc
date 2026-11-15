#include "fattn-rows.cuh"
#include "convert.cuh"
#include "dequantize.cuh"

// One block per (query, head): its warps take the rows of the range in turn, each with its own online softmax, and
// combine them at the end. A lane holds the dimensions lane, lane + WARP_SIZE, ... of a row.

static constexpr int fattn_rows_nwarps = 4;
static constexpr int fattn_rows_nreg   = FATTN_ROWS_D_MAX / WARP_SIZE;

// value i of a row of the cache; the quantized types are blocks of 32 values, the 4 and 5 bit ones hold value j and
// j + 16 of a block in one byte
template <ggml_type type>
static __device__ __forceinline__ float fattn_rows_load(const char * row, const int i) {
    if constexpr (type == GGML_TYPE_F16) {
        return __half2float(((const half *) row)[i]);
    } else if constexpr (type == GGML_TYPE_BF16) {
        return ggml_cuda_cast<float>(((const nv_bfloat16 *) row)[i]);
    } else if constexpr (type == GGML_TYPE_F32) {
        return ((const float *) row)[i];
    } else {
        const int64_t ib = i / 32;
        const int     j  = i % 32;
        if constexpr (type == GGML_TYPE_Q8_0) {
            const block_q8_0 * b = (const block_q8_0 *) row + ib;
            return __half2float(b->d) * b->qs[j];
        } else if constexpr (type == GGML_TYPE_IQ4_NL) {
            const block_iq4_nl * b = (const block_iq4_nl *) row + ib;
            const int q = j < 16 ? b->qs[j] & 0xF : b->qs[j - 16] >> 4;
            return __half2float(b->d) * kvalues_iq4nl[q];
        } else {
            float2 v;
            if constexpr (type == GGML_TYPE_Q4_0) {
                dequantize_q4_0(row, ib, j & 15, v);
            } else if constexpr (type == GGML_TYPE_Q4_1) {
                dequantize_q4_1(row, ib, j & 15, v);
            } else if constexpr (type == GGML_TYPE_Q5_0) {
                dequantize_q5_0(row, ib, j & 15, v);
            } else {
                static_assert(type == GGML_TYPE_Q5_1, "unsupported type");
                dequantize_q5_1(row, ib, j & 15, v);
            }
            return j < 16 ? v.x : v.y;
        }
    }
}

template <ggml_type type_K, ggml_type type_V>
static __global__ void flash_attn_ext_rows(
        const char    * __restrict__ Q,
        const char    * __restrict__ K,
        const char    * __restrict__ V,
        const int32_t * __restrict__ kv_idx,
        const int32_t * __restrict__ q_rng,
        const float   * __restrict__ sinks,
        const float   * __restrict__ rel_bias,
        float         * __restrict__ dst,
        const int DK, const int DV, const int n_head, const int gqa_ratio,
        const int64_t nbq1, const int64_t nbq2, const int64_t nbk1, const int64_t nbk2, const int64_t nbv1, const int64_t nbv2,
        const int64_t list_stride, const int64_t pos_plane, const int n_rel,
        const float scale, const float logit_softcap, const float max_bias, const float m0, const float m1, const uint32_t n_head_log2) {
    const int iq   = blockIdx.x;
    const int h    = blockIdx.y;
    const int warp = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;

    __shared__ float q_s[FATTN_ROWS_D_MAX + 64];
    __shared__ float acc_s[fattn_rows_nwarps][FATTN_ROWS_D_MAX];
    __shared__ float m_s[fattn_rows_nwarps];
    __shared__ float sum_s[fattn_rows_nwarps];

    // the query, scaled: with a softcap the scale applies before it, see ggml_flash_attn_ext
    const float * q = (const float *) (Q + iq*nbq1 + h*nbq2);
    const float scale_q = logit_softcap != 0.0f ? scale/logit_softcap : scale;
    for (int i = threadIdx.x; i < DK; i += blockDim.x) {
        q_s[i] = q[i]*scale_q;
    }
    __syncthreads();

    const int32_t * rng  = q_rng + 3*iq;
    const int32_t * list = kv_idx + int64_t(rng[0])*list_stride;
    const int32_t * pos  = pos_plane != 0 ? list + pos_plane : nullptr;
    const int lo = rng[1];
    const int hi = rng[2];

    // ALiBi and the relative bias by distance read the distance to the last row of the range
    const int   pos_ref = pos && hi > lo ? pos[hi - 1] : 0;
    const float slope   = max_bias > 0.0f ? get_alibi_slope(max_bias, h, n_head_log2, m0, m1) : 0.0f;

    const char * K_h = K + (h / gqa_ratio)*nbk2;
    const char * V_h = V + (h / gqa_ratio)*nbv2;

    float acc[fattn_rows_nreg];
#pragma unroll
    for (int r = 0; r < fattn_rows_nreg; ++r) {
        acc[r] = 0.0f;
    }
    float M   = -INFINITY;
    float sum = 0.0f;

    for (int k = lo + warp; k < hi; k += fattn_rows_nwarps) {
        const int32_t row = list[k];
        if (row < 0) {
            continue; // padding of a list of rows
        }

        const char * K_row = K_h + int64_t(row)*nbk1;
        float s = 0.0f;
        for (int i = lane; i < DK; i += WARP_SIZE) {
            s += q_s[i]*fattn_rows_load<type_K>(K_row, i);
        }
        s = warp_reduce_sum(s);

        if (logit_softcap != 0.0f) {
            s = logit_softcap*tanhf(s);
        }
        if (pos) {
            const int dist = pos_ref - pos[k];
            s -= slope*float(dist);
            if (rel_bias) {
                s += rel_bias[int64_t(h)*n_rel + min(max(dist, 0), n_rel - 1)];
            }
        }

        const float M_new = fmaxf(M, s);
        const float ms    = expf(M - M_new);
        const float vs    = expf(s - M_new);
        M   = M_new;
        sum = sum*ms + vs;

        const char * V_row = V_h + int64_t(row)*nbv1;
#pragma unroll
        for (int r = 0; r < fattn_rows_nreg; ++r) {
            const int i = lane + r*WARP_SIZE;
            if (i < DV) {
                acc[r] = acc[r]*ms + vs*fattn_rows_load<type_V>(V_row, i);
            }
        }
    }

#pragma unroll
    for (int r = 0; r < fattn_rows_nreg; ++r) {
        const int i = lane + r*WARP_SIZE;
        if (i < DV) {
            acc_s[warp][i] = acc[r];
        }
    }
    if (lane == 0) {
        m_s[warp]   = M;
        sum_s[warp] = sum;
    }
    __syncthreads();

    // the warps, and the sink, a logit of no row
    float M_all = sinks ? sinks[h] : -INFINITY;
#pragma unroll
    for (int w = 0; w < fattn_rows_nwarps; ++w) {
        M_all = fmaxf(M_all, m_s[w]);
    }

    float scale_w[fattn_rows_nwarps];
    float sum_all = sinks && M_all != -INFINITY ? expf(sinks[h] - M_all) : 0.0f;
#pragma unroll
    for (int w = 0; w < fattn_rows_nwarps; ++w) {
        scale_w[w] = m_s[w] == -INFINITY ? 0.0f : expf(m_s[w] - M_all);
        sum_all   += sum_s[w]*scale_w[w];
    }

    // a range with no rows attends to nothing
    const float inv = sum_all > 0.0f ? 1.0f/sum_all : 0.0f;

    float * dst_q = dst + (int64_t(iq)*n_head + h)*DV;
    for (int i = threadIdx.x; i < DV; i += blockDim.x) {
        float v = 0.0f;
#pragma unroll
        for (int w = 0; w < fattn_rows_nwarps; ++w) {
            v += acc_s[w][i]*scale_w[w];
        }
        dst_q[i] = v*inv;
    }
}

// the types of a KV cache; a quantized row is whole blocks of 32 values
static bool fattn_rows_type_ok(const ggml_type type, const int64_t n) {
    switch (type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F32:
            return true;
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_IQ4_NL:
            return n % 32 == 0;
        default:
            return false;
    }
}

bool ggml_cuda_flash_attn_ext_rows_supported(const ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    // per query ranges only, without a mask
    if (dst->src[6] == nullptr || dst->src[3] != nullptr || Q->type != GGML_TYPE_F32 || Q->ne[3] != 1) {
        return false;
    }
    if (dst->src[5]->ne[2] > 2) {
        return false;
    }
    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) dst->op_params + 1, sizeof(float));
    if ((max_bias != 0.0f || dst->src[7] != nullptr) && dst->src[5]->ne[2] != 2) {
        return false;
    }
    return fattn_rows_type_ok(K->type, K->ne[0]) && fattn_rows_type_ok(V->type, V->ne[0]) &&
        K->ne[0] <= FATTN_ROWS_D_MAX && V->ne[0] <= FATTN_ROWS_D_MAX && Q->ne[2] % K->ne[2] == 0;
}

template <ggml_type type_K, ggml_type type_V>
static void fattn_rows_launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q      = dst->src[0];
    const ggml_tensor * K      = dst->src[1];
    const ggml_tensor * V      = dst->src[2];
    const ggml_tensor * sinks  = dst->src[4];
    const ggml_tensor * kv_idx = dst->src[5];
    const ggml_tensor * q_rng  = dst->src[6];
    const ggml_tensor * rel    = dst->src[7];

    float scale         = 1.0f;
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale,         (const float *) dst->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float    m0          = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float    m1          = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    const dim3 blocks(Q->ne[1], n_head, 1);
    const dim3 threads(fattn_rows_nwarps*WARP_SIZE, 1, 1);

    flash_attn_ext_rows<type_K, type_V><<<blocks, threads, 0, ctx.stream()>>>(
        (const char *) Q->data, (const char *) K->data, (const char *) V->data,
        (const int32_t *) kv_idx->data, (const int32_t *) q_rng->data,
        sinks ? (const float *) sinks->data : nullptr, rel ? (const float *) rel->data : nullptr, (float *) dst->data,
        K->ne[0], V->ne[0], n_head, Q->ne[2] / K->ne[2],
        Q->nb[1], Q->nb[2], K->nb[1], K->nb[2], V->nb[1], V->nb[2],
        kv_idx->nb[1] / sizeof(int32_t), kv_idx->ne[2] == 2 ? int64_t(kv_idx->nb[2] / sizeof(int32_t)) : 0,
        rel ? int(rel->ne[0]) : 0,
        scale, logit_softcap, max_bias, m0, m1, n_head_log2);
    CUDA_CHECK(cudaGetLastError());
}

#define FATTN_ROWS_TYPES(X) X(F16) X(BF16) X(F32) X(Q8_0) X(Q4_0) X(Q4_1) X(Q5_0) X(Q5_1) X(IQ4_NL)

template <ggml_type type_K>
static void fattn_rows_switch_v(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    switch (dst->src[2]->type) {
#define FATTN_ROWS_CASE_V(T) case GGML_TYPE_##T: fattn_rows_launch<type_K, GGML_TYPE_##T>(ctx, dst); break;
        FATTN_ROWS_TYPES(FATTN_ROWS_CASE_V)
#undef FATTN_ROWS_CASE_V
        default: GGML_ABORT("fatal error");
    }
}

void ggml_cuda_flash_attn_ext_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_flash_attn_ext_rows_supported(dst));

    switch (dst->src[1]->type) {
#define FATTN_ROWS_CASE_K(T) case GGML_TYPE_##T: fattn_rows_switch_v<GGML_TYPE_##T>(ctx, dst); break;
        FATTN_ROWS_TYPES(FATTN_ROWS_CASE_K)
#undef FATTN_ROWS_CASE_K
        default: GGML_ABORT("fatal error");
    }
}

#undef FATTN_ROWS_TYPES
