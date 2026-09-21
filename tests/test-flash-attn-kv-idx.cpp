// Flash attention with index lists (ggml_flash_attn_ext_set_kv_idx) must match flash attention with the equivalent
// dense mask. The K/V rows are visited in a different order, so the online softmax accumulates in a different order:
// the results are compared with a small relative tolerance.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

struct test_case {
    std::string name;
    int64_t   head_dim;
    int64_t   n_head;
    int64_t   n_head_kv;
    int64_t   n_kv;      // K/V rows
    int64_t   n_group;   // sequences
    int64_t   n_q;       // query rows
    ggml_type kv_type;
    int32_t   n_swa;     // 0 = full causal
    bool      causal;
    bool      sinks;
    float     softcap;
};

// a sequence: some of the K/V rows, with increasing positions
struct group {
    std::vector<int32_t> cells; // sorted by position
    std::vector<int32_t> pos;
};

static bool run(const test_case & tc, std::mt19937 & rng) {
    // assign every K/V row to one group, in a shuffled order, so the groups interleave in the K/V rows
    std::vector<int32_t> order(tc.n_kv);
    for (int32_t i = 0; i < (int32_t) tc.n_kv; ++i) {
        order[i] = i;
    }
    std::shuffle(order.begin(), order.end(), rng);

    std::vector<group> groups(tc.n_group);
    for (int64_t i = 0; i < tc.n_kv; ++i) {
        auto & g = groups[i % tc.n_group];
        g.cells.push_back(order[i]);
        g.pos.push_back((int32_t) g.pos.size());
    }

    int64_t n_kv_max = 0;
    for (const auto & g : groups) {
        n_kv_max = std::max<int64_t>(n_kv_max, g.cells.size());
    }

    // per query: group, position -> [lo, hi) in the group's list
    struct query { int32_t g, lo, hi; };
    std::vector<query> queries(tc.n_q);
    for (auto & q : queries) {
        q.g = std::uniform_int_distribution<int32_t>(0, (int32_t) tc.n_group - 1)(rng);
        const int32_t n = (int32_t) groups[q.g].cells.size();
        const int32_t p = std::uniform_int_distribution<int32_t>(0, n - 1)(rng);
        if (!tc.causal) {
            q.lo = 0;
            q.hi = n;
        } else {
            q.hi = p + 1;
            q.lo = tc.n_swa > 0 ? std::max(0, p + 1 - tc.n_swa) : 0;
        }
    }

    const size_t mem = 64*1024*1024 + 4*tc.n_head*tc.head_dim*(tc.n_q + 2*tc.n_kv)*sizeof(float) + 4*tc.n_kv*tc.n_q*sizeof(float);
    ggml_init_params ip = { mem, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, tc.head_dim, tc.n_q,  tc.n_head,    1);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, tc.kv_type,    tc.head_dim, tc.n_kv, tc.n_head_kv, 1);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, tc.kv_type,    tc.head_dim, tc.n_kv, tc.n_head_kv, 1);

    std::normal_distribution<float> nd(0.0f, 1.0f);
    auto fill = [&](ggml_tensor * t) {
        std::vector<float> data(ggml_nelements(t));
        for (auto & x : data) {
            x = nd(rng);
        }
        if (t->type == GGML_TYPE_F32) {
            memcpy(t->data, data.data(), ggml_nbytes(t));
        } else {
            const int64_t n_per_row = t->ne[0];
            ggml_quantize_chunk(t->type, data.data(), t->data, 0, ggml_nelements(t)/n_per_row, n_per_row, nullptr);
        }
    };
    fill(q);
    fill(k);
    fill(v);

    ggml_tensor * mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, tc.n_kv, tc.n_q, 1, 1);
    {
        const ggml_fp16_t ninf = ggml_fp32_to_fp16(-std::numeric_limits<float>::infinity());
        const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
        auto * data = (ggml_fp16_t *) mask->data;
        for (int64_t i = 0; i < ggml_nelements(mask); ++i) {
            data[i] = ninf;
        }
        for (int64_t iq = 0; iq < tc.n_q; ++iq) {
            const auto & qq = queries[iq];
            for (int32_t j = qq.lo; j < qq.hi; ++j) {
                data[iq*tc.n_kv + groups[qq.g].cells[j]] = zero;
            }
        }
    }

    ggml_tensor * kv_idx = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_kv_max, tc.n_group);
    ggml_tensor * q_rng  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, tc.n_q);
    {
        auto * data = (int32_t *) kv_idx->data;
        std::fill(data, data + ggml_nelements(kv_idx), 0);
        for (int64_t g = 0; g < tc.n_group; ++g) {
            std::copy(groups[g].cells.begin(), groups[g].cells.end(), data + g*n_kv_max);
        }
        auto * r = (int32_t *) q_rng->data;
        for (int64_t iq = 0; iq < tc.n_q; ++iq) {
            r[3*iq + 0] = queries[iq].g;
            r[3*iq + 1] = queries[iq].lo;
            r[3*iq + 2] = queries[iq].hi;
        }
    }

    ggml_tensor * sinks = nullptr;
    if (tc.sinks) {
        sinks = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, tc.n_head);
        fill(sinks);
    }

    const float scale = 1.0f/std::sqrt((float) tc.head_dim);

    // the reference uses an F32 V: the CPU accumulates an F16 V in F16, which depends on the order of the rows
    ggml_tensor * v_ref = v->type == GGML_TYPE_F16 ? ggml_cast(ctx, v, GGML_TYPE_F32) : v;

    ggml_tensor * out_mask = ggml_flash_attn_ext(ctx, q, k, v_ref, mask, scale, 0.0f, tc.softcap);
    ggml_flash_attn_ext_add_sinks(out_mask, sinks);
    ggml_prec_set_acc(out_mask, GGML_PREC_F32);

    ggml_tensor * out_idx = ggml_flash_attn_ext(ctx, q, k, v, nullptr, scale, 0.0f, tc.softcap);
    ggml_flash_attn_ext_add_sinks(out_idx, sinks);
    ggml_prec_set_acc(out_idx, GGML_PREC_F32);
    ggml_flash_attn_ext_set_kv_idx(out_idx, kv_idx, q_rng);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out_mask);
    ggml_build_forward_expand(gf, out_idx);

    if (ggml_graph_compute_with_ctx(ctx, gf, 4) != GGML_STATUS_SUCCESS) {
        printf("%-40s compute failed\n", tc.name.c_str());
        ggml_free(ctx);
        return false;
    }

    const auto * a = (const float *) out_mask->data;
    const auto * b = (const float *) out_idx->data;

    double max_rel = 0.0;
    for (int64_t i = 0; i < ggml_nelements(out_mask); ++i) {
        const double d = std::fabs((double) a[i] - b[i]) / std::max(1e-3, std::fabs((double) a[i]));
        max_rel = std::max(max_rel, d);
    }

    const bool ok = max_rel < 1e-4;
    printf("%-40s max rel diff = %.2e %s\n", tc.name.c_str(), max_rel, ok ? "OK" : "FAIL");

    ggml_free(ctx);
    return ok;
}

int main() {
    std::mt19937 rng(42);

    // ggml_backend_dev_supports_op must accept the indexed op on the CPU
    {
        ggml_init_params ip = { 16*1024*1024, nullptr, false };
        ggml_context * ctx = ggml_init(ip);
        ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 64, 4, 8, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, 64, 16, 2, 1);
        ggml_tensor * out = ggml_flash_attn_ext(ctx, q, k, k, nullptr, 1.0f, 0.0f, 0.0f);
        ggml_flash_attn_ext_set_kv_idx(out, ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 16, 1), ggml_new_tensor_2d(ctx, GGML_TYPE_I32, 3, 4));
        ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        const bool supported = cpu && ggml_backend_dev_supports_op(cpu, out);
        printf("%-40s %s\n", "cpu supports_op", supported ? "OK" : "FAIL");
        ggml_free(ctx);
        if (!supported) {
            return 1;
        }
    }

    const std::vector<test_case> cases = {
        { "f16 causal",                 64,  8, 2, 257, 3,  33, GGML_TYPE_F16,  0, true,  false, 0.0f  },
        { "f16 causal decode",          128, 8, 8, 511, 8,  8,  GGML_TYPE_F16,  0, true,  false, 0.0f  },
        { "f32 causal",                 64,  4, 4, 129, 2,  17, GGML_TYPE_F32,  0, true,  false, 0.0f  },
        { "q8_0 causal",                64,  8, 2, 257, 3,  33, GGML_TYPE_Q8_0, 0, true,  false, 0.0f  },
        { "f16 sliding window",         64,  8, 4, 300, 4,  40, GGML_TYPE_F16,  16, true, false, 0.0f  },
        { "f16 non-causal",             64,  8, 4, 120, 2,  24, GGML_TYPE_F16,  0, false, false, 0.0f  },
        { "f16 sinks",                  64,  8, 2, 200, 3,  20, GGML_TYPE_F16,  0, true,  true,  0.0f  },
        { "f16 softcap",                64,  8, 2, 200, 3,  20, GGML_TYPE_F16,  0, true,  false, 30.0f },
        { "f16 many rows (tiled range)", 128, 8, 2, 1024, 4, 300, GGML_TYPE_F16, 0, true, false, 0.0f  },
        { "f16 single query, long kv",  128, 8, 2, 2048, 2, 1,  GGML_TYPE_F16,  0, true,  false, 0.0f  },
    };

    bool all_ok = true;
    for (const auto & tc : cases) {
        all_ok = run(tc, rng) && all_ok;
    }

    printf("%s\n", all_ok ? "ALL OK" : "SOME FAILED");
    return all_ok ? 0 : 1;
}
