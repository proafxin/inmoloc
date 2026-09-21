// Rolling back a gated delta-net state by replaying tokens must give the same state as the snapshot the op
// writes for that position (see ggml_gated_delta_net). This is what lets a speculative rollback cache the
// scan inputs of the last tokens instead of one full state per draft position.
//
// Checks, for every rollback distance:
//   replay   - scanning the first k tokens from the initial state equals the op's snapshot k tokens back
//   padding  - padding a short replay with neutral tokens (g = 0, beta = 0) leaves the state unchanged,
//              which is what a batch replaying different numbers of tokens per sequence needs

#include "ggml.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

struct shape {
    int64_t S;        // head dim (S_k == S_v)
    int64_t H_k;      // key heads
    int64_t H_v;      // value heads (multiple of H_k)
    int64_t n_tokens;
    int64_t n_seqs;
    bool    gate_per_dim; // KDA-style gate [S, H_v] instead of [1, H_v]
};

struct inputs {
    std::vector<float> q, k, v, g, b, s0;
};

static inputs make_inputs(const shape & sh, std::mt19937 & rng) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::uniform_real_distribution<float> ud(0.0f, 1.0f);

    inputs in;
    in.q.resize(sh.S*sh.H_k*sh.n_tokens*sh.n_seqs);
    in.k.resize(in.q.size());
    in.v.resize(sh.S*sh.H_v*sh.n_tokens*sh.n_seqs);
    in.g.resize((sh.gate_per_dim ? sh.S : 1)*sh.H_v*sh.n_tokens*sh.n_seqs);
    in.b.resize(sh.H_v*sh.n_tokens*sh.n_seqs);
    in.s0.resize(sh.S*sh.S*sh.H_v*sh.n_seqs);

    for (auto & x : in.q)  { x = nd(rng); }
    for (auto & x : in.k)  { x = nd(rng); }
    for (auto & x : in.v)  { x = nd(rng); }
    for (auto & x : in.s0) { x = nd(rng); }
    for (auto & x : in.g)  { x = -ud(rng); }       // log-decay, in (-1, 0]
    for (auto & x : in.b)  { x = ud(rng); }        // update strength, in [0, 1)

    return in;
}

// runs the op over the first n_run tokens of each sequence, padded to n_pad tokens with neutral tokens
// returns the K state snapshots, most recent first
static std::vector<float> run(const shape & sh, const inputs & in, int64_t n_run, int64_t n_pad, int64_t K) {
    const int64_t n_tok = n_pad;
    const int64_t g0    = sh.gate_per_dim ? sh.S : 1;

    ggml_init_params ip = { (size_t) 512*1024*1024, nullptr, false };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, sh.S, sh.H_k, n_tok, sh.n_seqs);
    ggml_tensor * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, sh.S, sh.H_k, n_tok, sh.n_seqs);
    ggml_tensor * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, sh.S, sh.H_v, n_tok, sh.n_seqs);
    ggml_tensor * g = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, g0,   sh.H_v, n_tok, sh.n_seqs);
    ggml_tensor * b = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1,    sh.H_v, n_tok, sh.n_seqs);
    ggml_tensor * s = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, sh.S, sh.S,   sh.H_v, sh.n_seqs);

    memcpy(s->data, in.s0.data(), in.s0.size()*sizeof(float));

    // copy the first n_run tokens of each sequence, leave the padding neutral: g = 0 (decay 1), beta = 0
    auto copy_tokens = [&](ggml_tensor * dst, const std::vector<float> & src, int64_t per_token, float pad) {
        auto * d = (float *) dst->data;
        for (int64_t is = 0; is < sh.n_seqs; ++is) {
            for (int64_t t = 0; t < n_tok; ++t) {
                float * dt = d + (is*n_tok + t)*per_token;
                if (t < n_run) {
                    const float * st = src.data() + (is*sh.n_tokens + t)*per_token;
                    memcpy(dt, st, per_token*sizeof(float));
                } else {
                    for (int64_t i = 0; i < per_token; ++i) {
                        dt[i] = pad;
                    }
                }
            }
        }
    };

    copy_tokens(q, in.q, sh.S*sh.H_k, 0.0f);
    copy_tokens(k, in.k, sh.S*sh.H_k, 0.0f);
    copy_tokens(v, in.v, sh.S*sh.H_v, 0.0f);
    copy_tokens(g, in.g, g0*sh.H_v,   0.0f); // exp(0) = 1 -> state kept
    copy_tokens(b, in.b, sh.H_v,      0.0f); // no update

    ggml_tensor * out = ggml_gated_delta_net(ctx, q, k, v, g, b, s, K);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    if (ggml_graph_compute_with_ctx(ctx, gf, 4) != GGML_STATUS_SUCCESS) {
        printf("compute failed\n");
        exit(1);
    }

    const int64_t n_scores = sh.S*sh.H_v*n_tok*sh.n_seqs;
    const int64_t n_state  = sh.S*sh.S*sh.H_v*sh.n_seqs;

    const float * src = (const float *) out->data + n_scores;
    std::vector<float> res(src, src + K*n_state);

    ggml_free(ctx);

    return res;
}

static double max_rel(const float * a, const float * b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; ++i) {
        m = std::max(m, std::fabs((double) a[i] - b[i]) / std::max(1e-3, std::fabs((double) a[i])));
    }
    return m;
}

int main() {
    std::mt19937 rng(1234);

    const std::vector<shape> shapes = {
        { 16, 2, 4,  4, 2, false },
        { 32, 4, 8,  5, 3, false },
        { 16, 2, 2,  3, 1, true  },
        { 64, 2, 4,  4, 2, false },
    };

    bool all_ok = true;

    for (const auto & sh : shapes) {
        const inputs in = make_inputs(sh, rng);

        const int64_t n_state = sh.S*sh.S*sh.H_v*sh.n_seqs;

        // the op's own snapshots: slot 0 = after all tokens, slot s = s tokens back
        const std::vector<float> snaps = run(sh, in, sh.n_tokens, sh.n_tokens, sh.n_tokens);

        for (int64_t k = 1; k <= sh.n_tokens; ++k) {
            const int64_t slot = sh.n_tokens - k; // state after k tokens

            const std::vector<float> replay = run(sh, in, k, k, 1);
            const std::vector<float> padded = run(sh, in, k, sh.n_tokens, 1);

            const double d_replay = max_rel(snaps.data() + slot*n_state, replay.data(), (size_t) n_state);
            const double d_padded = max_rel(snaps.data() + slot*n_state, padded.data(), (size_t) n_state);

            const bool ok = d_replay < 1e-5 && d_padded < 1e-5;
            all_ok = all_ok && ok;

            printf("S=%2lld H_k=%lld H_v=%lld tokens=%lld seqs=%lld gate=%s | replay %lld tokens: snapshot %.2e, padded %.2e %s\n",
                    (long long) sh.S, (long long) sh.H_k, (long long) sh.H_v, (long long) sh.n_tokens,
                    (long long) sh.n_seqs, sh.gate_per_dim ? "per-dim" : "scalar",
                    (long long) k, d_replay, d_padded, ok ? "OK" : "FAIL");
        }
    }

    printf("%s\n", all_ok ? "ALL OK" : "SOME FAILED");

    return all_ok ? 0 : 1;
}
