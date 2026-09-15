// Tests for a recurrent-state cell pool smaller than n_seq_max (n_rs_cells < n_seq_max).
//
//  1. no stale state: a cell freed by one seq and reused by another starts from a zeroed state
//  2. bookkeeping: interleaved seqs over many steps in a small pool match a context whose pool is full size
//  3. refusal: a batch that cannot get a cell is refused cleanly and leaves every other seq intact
//  4. rollback: MTP-style partial rollbacks while seqs join, leave and get their cells swapped
//  5. nextn row order: the per-token hidden states MTP reads come back in batch order when the
//     batch is split into ubatches out of order
//
// Each test runs with the cache buffers poisoned, so any path that reads a cell without zeroing it
// produces wrong logits instead of accidentally reading zeros.

#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"

#include "../src/llama-ext.h"
#include "../src/llama-io.h"
#include "../src/llama-memory.h"

#include <algorithm>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>
#include <string>
#include <vector>

struct cache_buffer_collector : llama_io_write_i {
    std::set<ggml_backend_buffer_t> buffers;
    size_t size = 0;

    void write(const void *, size_t n) override {
        size += n;
    }

    void write_tensor(ggml_tensor * tensor, size_t, size_t n) override {
        buffers.insert(tensor->buffer);
        size += n;
    }

    size_t n_bytes() override {
        return size;
    }
};

static llama_token tok(uint32_t salt, llama_pos pos, int n_vocab) {
    return (llama_token) ((7*(uint32_t) pos + 31*salt + 1) % (uint32_t) n_vocab);
}

static bool decode_prompt(llama_context * ctx, llama_seq_id seq, uint32_t salt, llama_pos n, int n_vocab) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (llama_pos pos = 0; pos < n; ++pos) {
        common_batch_add(batch, tok(salt, pos, n_vocab), pos, { seq }, pos + 1 == n);
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

static llama_context * make_ctx(const common_params & params, llama_model * model, uint32_t n_seq_max, uint32_t n_rs_cells, uint8_t fill, uint32_t n_rs_seq = 0) {
    auto cparams = common_context_params_to_llama(params);
    cparams.n_seq_max  = n_seq_max;
    cparams.n_rs_cells = n_rs_cells;
    cparams.n_rs_seq   = n_rs_seq;
    cparams.n_ctx      = 512;
    cparams.n_batch    = 256;
    cparams.n_ubatch   = 256;
    cparams.kv_unified = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr || fill == 0) {
        return ctx;
    }

    // allocate the cache buffers, then poison them
    if (!decode_prompt(ctx, 0, 0, 8, llama_vocab_n_tokens(llama_model_get_vocab(model)))) {
        llama_free(ctx);
        return nullptr;
    }
    llama_synchronize(ctx);
    cache_buffer_collector collector;
    llama_get_memory(ctx)->state_write(collector);
    llama_memory_clear(llama_get_memory(ctx), true);
    if (collector.buffers.empty()) {
        fprintf(stderr, "%s : no cache buffers found\n", __func__);
        llama_free(ctx);
        return nullptr;
    }
    for (auto * buffer : collector.buffers) {
        ggml_backend_buffer_clear(buffer, fill);
    }
    return ctx;
}

static float max_logit_diff(llama_context * a, int32_t ia, llama_context * b, int32_t ib, int n_vocab) {
    const float * la = llama_get_logits_ith(a, ia);
    const float * lb = llama_get_logits_ith(b, ib);
    if (la == nullptr || lb == nullptr) {
        return std::numeric_limits<float>::infinity();
    }
    float diff = 0.0f;
    for (int t = 0; t < n_vocab; ++t) {
        if (!std::isfinite(la[t]) || !std::isfinite(lb[t])) {
            return std::numeric_limits<float>::infinity();
        }
        diff = std::max(diff, std::fabs(la[t] - lb[t]));
    }
    return diff;
}

// stale state produces order-1 differences, so this only has to absorb backend noise
constexpr float eps = 1e-4f;

static bool test_no_stale_state(const common_params & params, llama_model * model, int n_vocab, uint8_t fill) {
    // a single cell, so seq 3 must reuse exactly the cell seq 0 used
    llama_context * ctx_dirty = make_ctx(params, model, 4, 1, fill);
    llama_context * ctx_clean = make_ctx(params, model, 4, 1, fill);
    if (ctx_dirty == nullptr || ctx_clean == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        llama_free(ctx_dirty);
        llama_free(ctx_clean);
        return false;
    }

    bool ok = decode_prompt(ctx_dirty, 0, 11, 24, n_vocab);
    ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_dirty), 0, -1, -1);
    ok = ok && decode_prompt(ctx_dirty, 3, 23, 17, n_vocab);
    ok = ok && decode_prompt(ctx_clean, 3, 23, 17, n_vocab);

    const float diff = ok ? max_logit_diff(ctx_dirty, -1, ctx_clean, -1, n_vocab) : std::numeric_limits<float>::infinity();

    llama_free(ctx_dirty);
    llama_free(ctx_clean);

    if (!ok || diff > eps) {
        fprintf(stderr, "%s : FAILED (fill 0x%02x): reused cell leaked state, max logit diff %g\n", __func__, fill, (double) diff);
        return false;
    }
    fprintf(stderr, "%s : ok (fill 0x%02x), max logit diff %g\n", __func__, fill, (double) diff);
    return true;
}

static bool test_interleaved_small_pool(const common_params & params, llama_model * model, int n_vocab, uint8_t fill) {
    llama_context * ctx_pool = make_ctx(params, model, 4, 2, fill);
    llama_context * ctx_full = make_ctx(params, model, 4, 4, fill);
    if (ctx_pool == nullptr || ctx_full == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        llama_free(ctx_pool);
        llama_free(ctx_full);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx_pool);
        llama_free(ctx_full);
    };

    // seq 2 is >= the pool size, which the old cells[seq_id] indexing could not address
    bool ok = true;
    for (llama_context * ctx : { ctx_pool, ctx_full }) {
        ok = ok && decode_prompt(ctx, 0, 5, 19, n_vocab);
        ok = ok && decode_prompt(ctx, 2, 7, 13, n_vocab);
    }
    if (!ok) {
        fprintf(stderr, "%s : prompt decode failed\n", __func__);
        cleanup();
        return false;
    }

    float diff_max = 0.0f;

    // step both seqs together in one batch; the small pool is full, so every step reorders and dry-runs
    const auto step = [&](llama_seq_id sa, uint32_t salt_a, llama_pos pa, llama_seq_id sb, uint32_t salt_b, llama_pos pb) {
        for (llama_context * ctx : { ctx_pool, ctx_full }) {
            llama_batch batch = llama_batch_init(2, 0, 1);
            common_batch_add(batch, tok(salt_a, pa, n_vocab), pa, { sa }, true);
            common_batch_add(batch, tok(salt_b, pb, n_vocab), pb, { sb }, true);
            ok = ok && llama_decode(ctx, batch) == 0;
            llama_batch_free(batch);
        }
        if (ok) {
            diff_max = std::max(diff_max, max_logit_diff(ctx_pool, 0, ctx_full, 0, n_vocab));
            diff_max = std::max(diff_max, max_logit_diff(ctx_pool, 1, ctx_full, 1, n_vocab));
        }
    };

    for (llama_pos i = 0; i < 16 && ok; ++i) {
        step(0, 5, 19 + i, 2, 7, 13 + i);
    }

    // free seq 0 and start seq 3 in its cell, then keep stepping seq 2 alongside it
    for (llama_context * ctx : { ctx_pool, ctx_full }) {
        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx), 0, -1, -1);
        ok = ok && decode_prompt(ctx, 3, 9, 11, n_vocab);
    }
    if (ok) {
        diff_max = std::max(diff_max, max_logit_diff(ctx_pool, -1, ctx_full, -1, n_vocab));
    }

    for (llama_pos i = 0; i < 16 && ok; ++i) {
        step(2, 7, 29 + i, 3, 9, 11 + i);
    }

    cleanup();

    if (!ok || diff_max > eps) {
        fprintf(stderr, "%s : FAILED (fill 0x%02x): decode ok = %d, max logit diff %g\n", __func__, fill, ok, (double) diff_max);
        return false;
    }
    fprintf(stderr, "%s : ok (fill 0x%02x), max logit diff %g\n", __func__, fill, (double) diff_max);
    return true;
}

static bool test_refusal_leaves_cache_intact(const common_params & params, llama_model * model, int n_vocab, uint8_t fill) {
    llama_context * ctx = make_ctx(params, model, 4, 2, fill);
    llama_context * ref = make_ctx(params, model, 4, 2, fill);
    if (ctx == nullptr || ref == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        llama_free(ctx);
        llama_free(ref);
        return false;
    }

    const auto cleanup = [&]() {
        llama_free(ctx);
        llama_free(ref);
    };

    bool ok = true;
    for (llama_context * c : { ctx, ref }) {
        ok = ok && decode_prompt(c, 0, 3, 15, n_vocab);
        ok = ok && decode_prompt(c, 1, 4, 21, n_vocab);
    }
    if (!ok) {
        fprintf(stderr, "%s : prompt decode failed\n", __func__);
        cleanup();
        return false;
    }

    // both cells are owned: a third seq must be refused, not crash
    if (decode_prompt(ctx, 2, 6, 9, n_vocab)) {
        fprintf(stderr, "%s : FAILED: decode for a third seq succeeded with only 2 cells\n", __func__);
        cleanup();
        return false;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (llama_memory_seq_pos_max(mem, 2) != -1 || llama_memory_seq_pos_max(mem, 0) != 14 || llama_memory_seq_pos_max(mem, 1) != 20) {
        fprintf(stderr, "%s : FAILED: refusal changed the cache (pos_max seq0 %d, seq1 %d, seq2 %d)\n", __func__,
                llama_memory_seq_pos_max(mem, 0), llama_memory_seq_pos_max(mem, 1), llama_memory_seq_pos_max(mem, 2));
        cleanup();
        return false;
    }

    // the surviving seqs must continue exactly as in a context that never saw the refused batch
    for (llama_context * c : { ctx, ref }) {
        llama_batch batch = llama_batch_init(2, 0, 1);
        common_batch_add(batch, tok(3, 15, n_vocab), 15, { 0 }, true);
        common_batch_add(batch, tok(4, 21, n_vocab), 21, { 1 }, true);
        ok = ok && llama_decode(c, batch) == 0;
        llama_batch_free(batch);
    }
    const float diff = ok
        ? std::max(max_logit_diff(ctx, 0, ref, 0, n_vocab), max_logit_diff(ctx, 1, ref, 1, n_vocab))
        : std::numeric_limits<float>::infinity();

    // once a cell is free, the refused seq must go through
    ok = ok && llama_memory_seq_rm(mem, 1, -1, -1);
    const bool admitted = ok && decode_prompt(ctx, 2, 6, 9, n_vocab);

    cleanup();

    if (!ok || diff > eps || !admitted) {
        fprintf(stderr, "%s : FAILED (fill 0x%02x): decode ok = %d, max logit diff %g, admitted after free = %d\n",
                __func__, fill, ok, (double) diff, admitted);
        return false;
    }
    fprintf(stderr, "%s : ok (fill 0x%02x), max logit diff %g\n", __func__, fill, (double) diff);
    return true;
}

// one seq's history: its prompt, then per step the 3 decoded positions, how many drafts were kept, and the logits
struct step_rec {
    llama_pos          pos;
    uint32_t           accepted; // drafts kept out of 2
    std::vector<float> logits;   // 3*n_vocab
};

struct seq_trace {
    llama_seq_id          id;
    uint32_t              salt;
    llama_pos             n_prompt;
    llama_pos             pos = 0;
    std::vector<step_rec> steps;
};

static bool decode_group(llama_context * ctx, const std::vector<seq_trace *> & group, int n_vocab) {
    llama_batch batch = llama_batch_init(3*(int32_t) group.size(), 0, 1);
    for (const auto * t : group) {
        for (llama_pos j = 0; j < 3; ++j) {
            common_batch_add(batch, tok(t->salt, t->pos + j, n_vocab), t->pos + j, { t->id }, true);
        }
    }
    const bool ok = llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);
    return ok;
}

// MTP verifies a token plus 2 drafts in one decode, then rolls back the rejected drafts.
// Seqs join and leave a 2-cell pool so their cells get swapped, and seq 3 carries a pending
// rollback across a change of batch composition. Compared against a full-size pool with the same
// batch shapes, and against each seq replayed alone, which catches a wrong state even if the
// original swap path shares the bug.
// with_rollback = false keeps every draft, so the same schedule runs without rollbacks: its solo
// diff is the numeric drift from batch shapes alone, the baseline for reading the rollback diff
static bool test_rollback_under_reorder(const common_params & params, llama_model * model, int n_vocab, uint8_t fill, bool with_rollback) {
    constexpr uint32_t n_rs_seq = 2;

    llama_context * ctx_pool = make_ctx(params, model, 4, 2, fill, n_rs_seq);
    llama_context * ctx_full = make_ctx(params, model, 4, 4, fill, n_rs_seq);
    if (ctx_pool == nullptr || ctx_full == nullptr) {
        fprintf(stderr, "%s : failed to init contexts\n", __func__);
        llama_free(ctx_pool);
        llama_free(ctx_full);
        return false;
    }

    if (llama_n_rs_seq(ctx_pool) < n_rs_seq) {
        fprintf(stderr, "%s : skipping, model does not support recurrent rollback\n", __func__);
        llama_free(ctx_pool);
        llama_free(ctx_full);
        return true;
    }

    seq_trace s0 { 0,  5, 19, 0, {} };
    seq_trace s2 { 2,  7, 13, 0, {} };
    seq_trace s3 { 3,  9, 11, 0, {} };
    seq_trace s1 { 1, 11, 17, 0, {} };

    bool  ok        = true;
    float diff_full = 0.0f;

    const auto prompt = [&](seq_trace & t) {
        ok = ok && decode_prompt(ctx_pool, t.id, t.salt, t.n_prompt, n_vocab);
        ok = ok && decode_prompt(ctx_full, t.id, t.salt, t.n_prompt, n_vocab);
        if (ok) {
            diff_full = std::max(diff_full, max_logit_diff(ctx_pool, -1, ctx_full, -1, n_vocab));
        }
        t.pos = t.n_prompt;
    };

    const auto remove = [&](seq_trace & t) {
        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_pool), t.id, -1, -1);
        ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_full), t.id, -1, -1);
    };

    const auto run_steps = [&](const std::vector<seq_trace *> & group, int n_steps) {
        for (int k = 0; k < n_steps && ok; ++k) {
            ok = decode_group(ctx_pool, group, n_vocab) && decode_group(ctx_full, group, n_vocab);
            if (!ok) {
                return;
            }
            for (size_t g = 0; g < group.size() && ok; ++g) {
                seq_trace * t = group[g];

                step_rec rec;
                rec.pos      = t->pos;
                rec.accepted = with_rollback ? (uint32_t) (t->id*7 + t->steps.size()*5) % 3 : 2;
                rec.logits.resize(3*(size_t) n_vocab);

                for (int32_t j = 0; j < 3; ++j) {
                    const int32_t i = (int32_t) g*3 + j;
                    const float * l = llama_get_logits_ith(ctx_pool, i);
                    if (l == nullptr) {
                        ok = false;
                        return;
                    }
                    std::copy(l, l + n_vocab, rec.logits.begin() + j*n_vocab);
                    diff_full = std::max(diff_full, max_logit_diff(ctx_pool, i, ctx_full, i, n_vocab));
                }

                if (rec.accepted < 2) {
                    const llama_pos p0 = t->pos + (llama_pos) rec.accepted + 1;
                    ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_pool), t->id, p0, -1);
                    ok = ok && llama_memory_seq_rm(llama_get_memory(ctx_full), t->id, p0, -1);
                }

                t->pos += (llama_pos) rec.accepted + 1;
                t->steps.push_back(std::move(rec));
            }
        }
    };

    prompt(s0);
    prompt(s2);
    run_steps({ &s0, &s2 }, 8);

    remove(s0);
    prompt(s3);
    run_steps({ &s2, &s3 }, 8);

    remove(s2);
    prompt(s1);
    run_steps({ &s3, &s1 }, 8);

    llama_free(ctx_pool);
    llama_free(ctx_full);

    const char * mode = with_rollback ? "rollback" : "no-rollback";

    if (!ok || diff_full > eps) {
        fprintf(stderr, "%s : FAILED (%s, fill 0x%02x): decode/rollback ok = %d, pool vs full max logit diff %g\n",
                __func__, mode, fill, ok, (double) diff_full);
        return false;
    }

    // replay every seq alone: no other seq, no swaps, same decodes and rollbacks.
    // batch shapes differ from the interleaved run, so this is reported rather than asserted:
    // compare it against the no-rollback run of the same schedule
    float diff_solo = 0.0f;
    std::string per_seq;

    for (const seq_trace * t : { &s0, &s2, &s3, &s1 }) {
        float diff_seq = 0.0f;

        llama_context * solo = make_ctx(params, model, 1, 1, fill, n_rs_seq);
        if (solo == nullptr) {
            fprintf(stderr, "%s : failed to init solo context\n", __func__);
            return false;
        }

        bool ok_solo = decode_prompt(solo, 0, t->salt, t->n_prompt, n_vocab);
        for (const auto & rec : t->steps) {
            if (!ok_solo) {
                break;
            }

            llama_batch batch = llama_batch_init(3, 0, 1);
            for (llama_pos j = 0; j < 3; ++j) {
                common_batch_add(batch, tok(t->salt, rec.pos + j, n_vocab), rec.pos + j, { 0 }, true);
            }
            ok_solo = llama_decode(solo, batch) == 0;
            llama_batch_free(batch);

            for (int32_t j = 0; j < 3 && ok_solo; ++j) {
                const float * l = llama_get_logits_ith(solo, j);
                if (l == nullptr) {
                    ok_solo = false;
                    break;
                }
                for (int v = 0; v < n_vocab; ++v) {
                    const float a = l[v];
                    const float b = rec.logits[(size_t) j*n_vocab + v];
                    diff_seq = std::max(diff_seq, std::isfinite(a) && std::isfinite(b) ? std::fabs(a - b) : std::numeric_limits<float>::infinity());
                }
            }

            if (ok_solo && rec.accepted < 2) {
                ok_solo = llama_memory_seq_rm(llama_get_memory(solo), 0, rec.pos + (llama_pos) rec.accepted + 1, -1);
            }
        }

        llama_free(solo);

        if (!ok_solo) {
            fprintf(stderr, "%s : FAILED (%s, fill 0x%02x): solo replay of seq %d failed\n", __func__, mode, fill, t->id);
            return false;
        }

        diff_solo = std::max(diff_solo, diff_seq);
        per_seq += " seq" + std::to_string(t->id) + "=" + std::to_string(diff_seq);
    }

    fprintf(stderr, "%s : ok (%s, fill 0x%02x), pool vs full max logit diff %g | pool vs solo (report only) %g:%s\n",
            __func__, mode, fill, (double) diff_full, (double) diff_solo, per_seq.c_str());
    return true;
}

// The MTP draft reads one hidden row per batch token from the target (unmasked nextn embeddings).
// With rollback enabled, split_equal keeps the trailing 1 + n_rs_seq tokens of a seq together, so a
// batch of seq 0 (6 tokens, only the last one an output, like a prompt chunk) followed by seq 1
// (3 output tokens) is split as [0 0 0 1 1 1] [0 0 0]. Every row must still land at its batch index.
// Reference rows come from each seq decoded alone; a misplaced row is another token's hidden state,
// so each row is checked to be nearest to its own reference row.
static bool test_nextn_row_order(const common_params & params, llama_model * model, int n_vocab, uint8_t fill) {
    constexpr uint32_t n_rs_seq = 2;

    struct seq_part {
        llama_seq_id id;
        uint32_t     salt;
        llama_pos    n_prompt;
        llama_pos    n_new;
        bool         all_outputs;
    };

    const seq_part parts[] = {
        { 0, 5, 19, 6, false },
        { 1, 7, 13, 3, true  },
    };

    llama_context * ctx = make_ctx(params, model, 4, 2, fill, n_rs_seq);
    if (ctx == nullptr) {
        fprintf(stderr, "%s : failed to init context\n", __func__);
        return false;
    }
    if (llama_n_rs_seq(ctx) < n_rs_seq) {
        fprintf(stderr, "%s : skipping, model does not support recurrent rollback\n", __func__);
        llama_free(ctx);
        return true;
    }

    const int32_t n_embd = llama_model_n_embd_out(model);

    std::vector<float> ref;
    bool ok = true;

    for (const auto & p : parts) {
        llama_context * solo = make_ctx(params, model, 1, 1, fill, n_rs_seq);
        if (solo == nullptr) {
            fprintf(stderr, "%s : failed to init solo context\n", __func__);
            llama_free(ctx);
            return false;
        }
        llama_set_embeddings_nextn(solo, true, false);

        ok = ok && decode_prompt(solo, 0, p.salt, p.n_prompt, n_vocab);

        llama_batch batch = llama_batch_init(p.n_new, 0, 1);
        for (llama_pos j = 0; j < p.n_new; ++j) {
            common_batch_add(batch, tok(p.salt, p.n_prompt + j, n_vocab), p.n_prompt + j, { 0 }, p.all_outputs || j + 1 == p.n_new);
        }
        ok = ok && llama_decode(solo, batch) == 0;
        llama_batch_free(batch);

        const float * h = ok ? llama_get_embeddings_nextn(solo) : nullptr;
        ok = ok && h != nullptr;
        if (ok) {
            ref.insert(ref.end(), h, h + (size_t) p.n_new*n_embd);
        }

        llama_free(solo);
    }

    llama_set_embeddings_nextn(ctx, true, false);

    int32_t n_rows = 0;
    llama_batch batch = llama_batch_init(16, 0, 1);
    for (const auto & p : parts) {
        ok = ok && decode_prompt(ctx, p.id, p.salt, p.n_prompt, n_vocab);
    }
    for (const auto & p : parts) {
        for (llama_pos j = 0; j < p.n_new; ++j) {
            common_batch_add(batch, tok(p.salt, p.n_prompt + j, n_vocab), p.n_prompt + j, { p.id }, p.all_outputs || j + 1 == p.n_new);
        }
        n_rows += p.n_new;
    }
    ok = ok && llama_decode(ctx, batch) == 0;
    llama_batch_free(batch);

    const float * h_all = ok ? llama_get_embeddings_nextn(ctx) : nullptr;
    ok = ok && h_all != nullptr;

    const auto dist = [&](const float * a, const float * b) {
        double d = 0.0;
        for (int32_t k = 0; k < n_embd; ++k) {
            d += ((double) a[k] - b[k])*((double) a[k] - b[k]);
        }
        return std::sqrt(d);
    };

    std::string misplaced;
    double diff_max = 0.0;

    for (int32_t i = 0; i < n_rows && ok; ++i) {
        const float * row_all = h_all + (size_t) i*n_embd;
        const float * row_ith = llama_get_embeddings_nextn_ith(ctx, i);

        int32_t nearest = -1;
        double  d_min   = std::numeric_limits<double>::infinity();
        for (int32_t r = 0; r < n_rows; ++r) {
            const double d = dist(row_all, ref.data() + (size_t) r*n_embd);
            if (d < d_min) {
                d_min   = d;
                nearest = r;
            }
        }

        if (nearest != i || row_ith != row_all) {
            misplaced += " " + std::to_string(i) + "->" + std::to_string(nearest);
        } else {
            diff_max = std::max(diff_max, d_min);
        }
    }

    llama_free(ctx);

    if (!ok || !misplaced.empty()) {
        fprintf(stderr, "%s : FAILED (fill 0x%02x): decode ok = %d, misplaced rows (batch index->nearest reference):%s\n",
                __func__, fill, ok, misplaced.c_str());
        return false;
    }
    fprintf(stderr, "%s : ok (fill 0x%02x), max L2 distance to own reference row %g\n", __func__, fill, diff_max);
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.sampling.seed = 1234;
    params.n_predict = 1;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);
    llama_model * model = llama_init->model();
    if (model == nullptr) {
        fprintf(stderr, "%s : failed to init model\n", __func__);
        return 1;
    }

    if (!llama_model_is_recurrent(model) && !llama_model_is_hybrid(model)) {
        fprintf(stderr, "%s : skipping for non-recurrent model\n", __func__);
        return 0;
    }

    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    bool ok = true;
    for (uint8_t fill : { 0, 0x3e }) {
        ok = test_no_stale_state(params, model, n_vocab, fill) && ok;
        ok = test_interleaved_small_pool(params, model, n_vocab, fill) && ok;
        ok = test_refusal_leaves_cache_intact(params, model, n_vocab, fill) && ok;
        ok = test_rollback_under_reorder(params, model, n_vocab, fill, true) && ok;
        ok = test_rollback_under_reorder(params, model, n_vocab, fill, false) && ok;
        ok = test_nextn_row_order(params, model, n_vocab, fill) && ok;
    }

    fprintf(stderr, "%s : %s\n", __func__, ok ? "ALL PASSED" : "FAILED");
    return ok ? 0 : 1;
}
