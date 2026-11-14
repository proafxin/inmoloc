#include "models.h"
#include "llama-kv-cache-msa.h"
#include <cmath>
#include <vector>
#include <cstdint>

// MiniMax-M3: MiniMax-M2 style GQA (per-head QK-norm, partial rotary) with
// DeepSeek-V3 leading-dense + routed/shared experts (sigmoid gating, routed scaling),
// swigluoai activation, and MiniMax Sparse Attention (MSA). MTP is not in released model weights.
// MSA blocks are defined over token positions. The graph translates between position space (block
// selection) and cell space (K/V/indexer storage) via per-ubatch pos<->cell maps populated from llama_kv_cells

void llama_model_minimax_m3::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_LEADING_DENSE_BLOCK_COUNT,   hparams.n_layer_dense_lead, false);
    ml.get_key_or_arr(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, hparams.n_ff_exp_arr, hparams.n_layer_all);
    ml.get_key(LLM_KV_EXPERT_SHARED_COUNT,         hparams.n_expert_shared);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_SCALE,        hparams.expert_weights_scale, false);
    ml.get_key(LLM_KV_EXPERT_WEIGHTS_NORM,         hparams.expert_weights_norm, false);
    ml.get_key(LLM_KV_EXPERT_GATING_FUNC,          hparams.expert_gating_func);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_HEAD_COUNT,    hparams.indexer_n_head);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_KEY_LENGTH,    hparams.indexer_head_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_TOP_K,         hparams.indexer_top_k);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_BLOCK_SIZE,    hparams.indexer_block_size);
    ml.get_key(LLM_KV_ATTENTION_INDEXER_LOCAL_BLOCKS,  hparams.indexer_local_blocks);
    msa_p = { (int) hparams.indexer_block_size, (int) hparams.indexer_top_k, (int) hparams.indexer_local_blocks };

    GGML_ASSERT(hparams.indexer_block_size > 0); // avoid div by zero

    switch (hparams.n_layer()) {
        case 60: type = LLM_TYPE_428B_A23B; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_minimax_m3::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;
    const int64_t n_expert_shared = hparams.n_expert_shared;
    const int64_t n_ff_exp        = hparams.n_ff_exp();

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    // output
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, 0);

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), { n_embd_head_k * n_head, n_embd }, 0);

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);
        // per-head QK-norm: a single head_dim vector applied to every head
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);

        if (i < (int) hparams.n_layer_dense_lead) {
            // leading dense layers
            layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd,   n_ff}, 0);
            layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {  n_ff, n_embd}, 0);
            layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd,   n_ff}, 0);
        } else {
            // routed experts
            layer.ffn_gate_inp    = create_tensor(tn(LLM_TENSOR_FFN_GATE_INP,    "weight", i), {n_embd, n_expert}, 0);
            layer.ffn_exp_probs_b = create_tensor(tn(LLM_TENSOR_FFN_EXP_PROBS_B, "bias",   i), {n_expert}, 0);
            layer.ffn_gate_exps   = create_tensor(tn(LLM_TENSOR_FFN_GATE_EXPS,   "weight", i), {n_embd, n_ff_exp, n_expert}, 0);
            layer.ffn_down_exps   = create_tensor(tn(LLM_TENSOR_FFN_DOWN_EXPS,   "weight", i), {n_ff_exp, n_embd, n_expert}, 0);
            layer.ffn_up_exps     = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,     "weight", i), {n_embd, n_ff_exp, n_expert}, 0);

            // shared expert
            layer.ffn_gate_shexp = create_tensor(tn(LLM_TENSOR_FFN_GATE_SHEXP, "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);
            layer.ffn_down_shexp = create_tensor(tn(LLM_TENSOR_FFN_DOWN_SHEXP, "weight", i), {        n_ff_exp * n_expert_shared, n_embd}, 0);
            layer.ffn_up_shexp   = create_tensor(tn(LLM_TENSOR_FFN_UP_SHEXP,   "weight", i), {n_embd, n_ff_exp * n_expert_shared}, 0);

            // indexer
            layer.index_q_proj = create_tensor(tn(LLM_TENSOR_INDEXER_Q_PROJ, "weight", i), {n_embd, hparams.indexer_n_head * hparams.indexer_head_size}, 0);
            layer.index_k_proj = create_tensor(tn(LLM_TENSOR_INDEXER_K_PROJ, "weight", i), {n_embd, hparams.indexer_head_size}, 0);
            layer.index_q_norm = create_tensor(tn(LLM_TENSOR_INDEXER_Q_NORM, "weight", i), {hparams.indexer_head_size}, 0);
            layer.index_k_norm = create_tensor(tn(LLM_TENSOR_INDEXER_K_NORM, "weight", i), {hparams.indexer_head_size}, 0);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_minimax_m3::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

class llm_graph_input_msa : public llm_graph_input_i {
public:
    llm_graph_input_msa(const llama_kv_cache_msa_context * mctx, int blk, int local) :
        mctx(mctx), blk(blk), local(local) {}

    void set_input(const llama_ubatch * ubatch) override {
        mctx->set_input_box(box_rows, box_win, box_off, bias, box_q_base, ubatch, blk, local);
    }

    // valid as long as the list and the windows keep their size
    bool can_reuse(const llm_graph_params & params) override {
        const auto * mctx_new = static_cast<const llama_kv_cache_msa_context *>(params.mctx);

        this->mctx = mctx_new;

        int64_t n_list_new = 0;
        int64_t n_win_new  = 0;
        mctx_new->box_layout(params.ubatch, blk, n_list_new, n_win_new);

        return box_rows->ne[0] == n_list_new && n_win == n_win_new && box_win->ne[1] == params.ubatch.n_tokens;
    }

    ggml_tensor * bias       = nullptr; // F32 [nblk, n_tokens] local-force bias (position blocks)
    ggml_tensor * box_rows   = nullptr; // I32 [n_list, 1]      the elastic memory, see set_input_box()
    ggml_tensor * box_win    = nullptr; // I32 [3, n_tokens]
    ggml_tensor * box_off    = nullptr; // I32 [1, 1, n_tokens]
    ggml_tensor * box_q_base = nullptr; // I32 [2, n_tokens]
    int64_t       n_win      = 0;       //                      the slots of a window

    const llama_kv_cache_msa_context * mctx;

    int blk;
    int local;
};

ggml_tensor * llama_model_minimax_m3::graph::build_attn_msa_box(
        ggml_tensor * q_cur,
        ggml_tensor * k,
        ggml_tensor * v,
        ggml_tensor * iq,
        ggml_tensor * ik,
        ggml_tensor * rows,
        ggml_tensor * win,
        ggml_tensor * off,
        ggml_tensor * bias,
        ggml_tensor * q_base,
        int64_t n_win, int64_t blk, int64_t K, float kq_scale, int il) const {
    const int64_t D      = q_cur->ne[0];
    const int64_t T      = q_cur->ne[2];
    const int64_t Hd     = iq->ne[1];
    const int64_t Gp     = q_cur->ne[1]/Hd;
    const int64_t n_list = rows->ne[0];
    const int64_t nblk   = n_win/blk;
    const int64_t L      = blk*K;     // the entries of a list
    const int64_t Q      = Hd*T;      // the lists

    GGML_ASSERT(nblk == bias->ne[0] && n_win == nblk*blk && n_list % blk == 0);

    // each head of each token scores the window of its sequence: the slots it does not see are -INFINITY
    ggml_tensor * sc = ggml_lightning_indexer_heads(ctx0, iq, ik, rows, win, n_win);   // [n_win, Hd, T]
    cb(sc, "msa_scores", il);

    ggml_tensor * bs = ggml_pool_2d(ctx0, sc, GGML_OP_POOL_MAX, blk, 1, blk, 1, 0, 0);
    cb(bs, "msa_bs", il);

    // the best K blocks, the blocks around the token first, as blocks of the list
    ggml_tensor * idx = ggml_argsort_top_k_off(ctx0, ggml_add(ctx0, bs, ggml_reshape_3d(ctx0, bias, nblk, 1, T)), K, off);   // [K, Hd, T]
    idx = ggml_cont(ctx0, idx);

    // their rows: the list as rows of one block each
    ggml_tensor * list = ggml_get_rows(ctx0, ggml_reshape_2d(ctx0, rows, blk, n_list/blk), ggml_reshape_1d(ctx0, idx, K*Q));   // [blk, K*Q]
    list = ggml_reshape_3d(ctx0, list, 1, L, Q);

    // what each token sees of them, from the scores of the window: the blocks of the window are those of the list after
    // the block the window starts at, exact in F32, see llama_kv_cache_msa_context::set_input_box()
    ggml_tensor * rel = ggml_sub(ctx0, ggml_cast(ctx0, idx, GGML_TYPE_F32), ggml_cast(ctx0, off, GGML_TYPE_F32));
    rel = ggml_cast(ctx0, ggml_reshape_2d(ctx0, rel, K, Q), GGML_TYPE_I32);

    ggml_tensor * seen = ggml_get_rows(ctx0, ggml_reshape_3d(ctx0, sc, blk, nblk, Q), rel);   // [blk, K, Q]
    seen = ggml_step(ctx0, ggml_scale_bias(ctx0, ggml_reshape_2d(ctx0, seen, L, Q), 1.0f, 1e30f));    // 1 seen, 0 not

    // a list reads a range: the entries a token sees first, as many as there are
    ggml_tensor * order = ggml_cont(ctx0, ggml_argsort_top_k(ctx0, seen, L));   // [L, Q]

    list = ggml_get_rows(ctx0, list, order);   // [1, L, Q]
    list = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, list, L, Hd, T), 0, 2, 1, 3));   // [L, T, Hd]

    ggml_tensor * n_seen = ggml_cast(ctx0, ggml_sum_rows(ctx0, seen), GGML_TYPE_I32);
    n_seen = ggml_cont(ctx0, ggml_permute(ctx0, ggml_reshape_3d(ctx0, n_seen, 1, Hd, T), 0, 2, 1, 3));   // [1, T, Hd]

    // each head group reads its K/V head of the rows, in place
    ggml_tensor * cur = nullptr;

    for (int64_t h = 0; h < Hd; ++h) {
        ggml_tensor * q_h = ggml_view_3d(ctx0, q_cur, D, Gp, T, q_cur->nb[1], q_cur->nb[2], h*Gp*q_cur->nb[1]);
        ggml_tensor * k_h = ggml_view_4d(ctx0, k, D, 1, k->ne[2], 1, k->nb[1], k->nb[2], k->nb[3], h*k->nb[1]);
        ggml_tensor * v_h = ggml_view_4d(ctx0, v, D, 1, v->ne[2], 1, v->nb[1], v->nb[2], v->nb[3], h*v->nb[1]);

        ggml_tensor * list_h = ggml_view_2d(ctx0, list, L, T, list->nb[1], h*list->nb[2]);
        ggml_tensor * rng_h  = ggml_concat(ctx0, q_base, ggml_view_2d(ctx0, n_seen, 1, T, n_seen->nb[1], h*n_seen->nb[2]), 0);

        ggml_tensor * out = build_attn_mha(q_h, k_h, v_h, nullptr, nullptr, nullptr, nullptr, 0, kq_scale, il, list_h, rng_h);

        cur = cur ? ggml_concat(ctx0, cur, out, 0) : out;
    }

    return cur;
}

llama_model_minimax_m3::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();
    const auto & mm = static_cast<const llama_model_minimax_m3 &>(model);

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    // partial rotary: head_dim != n_rot, so don't assert n_embd_head == n_rot

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    // MSA reads the list of the rows of the sequence in the elastic memory: block selection is anchored to its slots
    auto * inp_attn = build_attn_inp_kv_msa();

    // hoisted per-graph MSA state (shared by every sparse layer)
    llm_graph_input_msa * msa = nullptr;
    int64_t nblk = 0;
    const int     blk = mm.msa_p.blk;
    const int64_t Hd  = hparams.indexer_n_head;   // one indexer head per GQA group

    {
        const auto * mctx_msa = static_cast<const llama_kv_cache_msa_context *>(mctx);

        // a list of the slots of the sequences and the window of each token in it, padded to whole blocks
        int64_t n_list = 0;
        int64_t n_win  = 0;
        mctx_msa->box_layout(ubatch, blk, n_list, n_win);
        nblk = n_win / blk;

        auto inp = std::make_unique<llm_graph_input_msa>(mctx_msa, blk, mm.msa_p.local);

        inp->bias = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, nblk, n_tokens);
        ggml_set_input(inp->bias);

        inp->n_win = n_win;

        inp->box_rows = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, n_list, 1);
        ggml_set_input(inp->box_rows);

        inp->box_win = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 3, n_tokens);
        ggml_set_input(inp->box_win);

        inp->box_off = ggml_new_tensor_3d(ctx0, GGML_TYPE_I32, 1, 1, n_tokens);
        ggml_set_input(inp->box_off);

        inp->box_q_base = ggml_new_tensor_2d(ctx0, GGML_TYPE_I32, 2, n_tokens);
        ggml_set_input(inp->box_q_base);

        msa = (llm_graph_input_msa *) res->add_input(std::move(inp));
    }

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        // self-attention
        {
            cur = build_norm(inpL, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
            cb(cur, "attn_norm", il);

            auto [Qcur, Kcur, Vcur] = build_qkv(model.layers[il], cur,
                    n_embd_head, n_head, n_head_kv, il);

            // per-head QK RMSNorm (weights already include Gemma's +1)
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            cb(Qcur, "Qcur_normed", il);
            Kcur = build_norm(Kcur, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            cb(Kcur, "Kcur_normed", il);

            // partial rotary: only the first n_rot dims are rotated
            Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

            cb(Qcur, "Qcur", il);
            cb(Kcur, "Kcur", il);
            cb(Vcur, "Vcur", il);

            const bool is_sparse = il >= (int) hparams.n_layer_dense_lead;

            if (!is_sparse) {
                cur = build_attn(inp_attn, model.layers[il].wo, NULL, model.layers[il].wo_s,
                        Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                        1.0f/sqrtf(float(n_embd_head)), il);
            } else {
                const int64_t n_idx_dim = hparams.indexer_head_size;   // 128

                // Index Branch, project, norm, partial RoPE, cache
                ggml_tensor * iq = build_lora_mm(model.layers[il].index_q_proj, cur);
                ggml_tensor * ik = build_lora_mm(model.layers[il].index_k_proj, cur);
                iq = ggml_reshape_3d(ctx0, iq, n_idx_dim, Hd, n_tokens);
                ik = ggml_reshape_3d(ctx0, ik, n_idx_dim, 1,  n_tokens);
                iq = build_norm(iq, model.layers[il].index_q_norm, NULL, LLM_NORM_RMS, il);  // +1 baked
                ik = build_norm(ik, model.layers[il].index_k_norm, NULL, LLM_NORM_RMS, il);
                iq = ggml_rope_ext(ctx0, iq, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                   freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);
                ik = ggml_rope_ext(ctx0, ik, inp_pos, nullptr, n_rot, rope_type, n_ctx_orig,
                                   freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow);

                const auto * mctx_msa_l = static_cast<const llama_kv_cache_msa_context *>(mctx);
                const auto * mctx_cur = mctx_msa_l->get_base();
                const auto * mctx_idx = mctx_msa_l->get_idx();
                ggml_build_forward_expand(gf, mctx_idx->cpy_k(ctx0, ik, inp_attn->get_k_idxs_idx(), il));
                ggml_tensor * ik_kv = mctx_idx->get_k_all(ctx0, il);

                if (inp_attn->self_k_rot) {
                    Qcur = llama_mul_mat_hadamard(ctx0, Qcur, inp_attn->self_k_rot);
                    Kcur = llama_mul_mat_hadamard(ctx0, Kcur, inp_attn->self_k_rot);
                }
                if (inp_attn->self_v_rot) {
                    Vcur = llama_mul_mat_hadamard(ctx0, Vcur, inp_attn->self_v_rot);
                }

                // Main branch: store K/V, take cache views
                ggml_build_forward_expand(gf, Qcur);
                ggml_build_forward_expand(gf, Kcur);
                ggml_build_forward_expand(gf, Vcur);
                ggml_build_forward_expand(gf, mctx_cur->cpy_k(ctx0, Kcur, inp_attn->get_k_idxs(), il));
                ggml_build_forward_expand(gf, mctx_cur->cpy_v(ctx0, Vcur, inp_attn->get_v_idxs(), il));
                ggml_tensor * k = mctx_cur->get_k_all(ctx0, il);
                ggml_tensor * v = mctx_cur->get_v_all(ctx0, il);

                GGML_ASSERT(k->ne[1] == Hd && "MSA: one indexer head per GQA group");
                const int K = mm.msa_p.topk_blocks < (int) nblk ? mm.msa_p.topk_blocks : (int) nblk;

                const float kq_scale = 1.0f/sqrtf(float(n_embd_head));

                cur = build_attn_msa_box(Qcur, k, v, iq, ik_kv, msa->box_rows, msa->box_win, msa->box_off, msa->bias, msa->box_q_base,
                        msa->n_win, blk, K, kq_scale, il);
                if (inp_attn->self_v_rot) {
                    cur = llama_mul_mat_hadamard(ctx0, cur, inp_attn->self_v_rot);
                }
                cb(cur, "kqv_out", il);
                if (model.layers[il].wo) {
                    cur = build_lora_mm(model.layers[il].wo, cur, model.layers[il].wo_s);
                }
            }
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        cur = build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        if ((uint32_t) il < hparams.n_layer_dense_lead) {
            // leading dense FFN (swigluoai)
            cur = build_ffn(cur,
                    model.layers[il].ffn_up,   NULL, NULL,
                    model.layers[il].ffn_gate, NULL, NULL,
                    model.layers[il].ffn_down, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU_OAI_MOE, LLM_FFN_PAR, il);
            cb(cur, "ffn_out", il);
        } else {
            // routed experts (swigluoai MoE)
            ggml_tensor * moe_out = build_moe_ffn(cur,
                    model.layers[il].ffn_gate_inp,
                    model.layers[il].ffn_up_exps,
                    model.layers[il].ffn_gate_exps,
                    model.layers[il].ffn_down_exps,
                    model.layers[il].ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SWIGLU_OAI_MOE, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il);
            cb(moe_out, "ffn_moe_out", il);

            // shared expert (swigluoai)
            ggml_tensor * ffn_shexp = build_ffn(cur,
                    model.layers[il].ffn_up_shexp,   NULL, NULL,
                    model.layers[il].ffn_gate_shexp, NULL, NULL,
                    model.layers[il].ffn_down_shexp, NULL, NULL,
                    NULL,
                    LLM_FFN_SWIGLU_OAI_MOE, LLM_FFN_PAR, il);
            cb(ffn_shexp, "ffn_shexp", il);

            cur = ggml_add(ctx0, moe_out, ffn_shexp);
            cb(cur, "ffn_out", il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        // input for next layer
        inpL = cur;
    }

    cur = inpL;

    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head
    cur = build_lora_mm(model.output, cur, model.output_s);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
