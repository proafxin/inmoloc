#include "llama-memory-hybrid.h"

#include "llama-box.h"
#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

#include <limits>
#include <map>

// whether the device of the box runs the ops that read it: the list of the rows of a prompt (with the prefix of it each
// query sees), the ranges of the cells of the other tokens and the scans of the states in place; the context only
// describes the ops, nothing is allocated
static bool llama_box_device_supported(const llama_hparams & hparams, ggml_backend_buffer_type_t buft, const llama_box::attn_layer & attn, bool recr) {
    ggml_init_params params = {
        /*.mem_size   =*/ 32*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        return false;
    }

    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr) {
        return false;
    }

    const int64_t hs    = hparams.n_embd_head_k(attn.il);
    const int64_t nh    = hparams.n_head_kv(attn.il);
    const int64_t n_q   = hparams.n_head(attn.il);
    const int64_t nb    = 32;
    const int64_t n_kv  = 1024;
    const int64_t n_idx = 256;

    const auto attention = [&]() {
        ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, hs, nb, n_q, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), attn.type_k, hs, n_kv, nh, 1);
        ggml_tensor * v = ggml_new_tensor_4d(ctx.get(), attn.type_v, hs, n_kv, nh, 1);

        ggml_tensor * res = ggml_flash_attn_ext(ctx.get(), q, k, v, nullptr, 1.0f, 0.0f, 0.0f);
        ggml_prec_set_acc(res, GGML_PREC_F32);
        return res;
    };

    ggml_tensor * fa_rows = attention();
    ggml_flash_attn_ext_set_kv_rows_hi(fa_rows,
            ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_idx), ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, nb));

    ggml_tensor * fa_idx = attention();
    ggml_flash_attn_ext_set_kv_idx(fa_idx,
            ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_idx, 1), ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 3, nb));

    if (!ggml_backend_dev_supports_op(dev, fa_rows) || !ggml_backend_dev_supports_op(dev, fa_idx)) {
        return false;
    }

    if (!recr) {
        return true;
    }

    const int64_t S = hparams.ssm_d_state;
    const int64_t H = hparams.ssm_d_inner/hparams.ssm_d_state;
    const int64_t G = hparams.ssm_n_group;

    ggml_tensor * gdn = ggml_gated_delta_net_rows(ctx.get(),
            ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, G, 1, 1),
            ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, G, 1, 1),
            ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, S, H, 1, 1),
            ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, 1, 1),
            ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, 1, H, 1, 1),
            ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, S*S*H, 1),
            ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1),
            ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, 1),
            nullptr);

    return ggml_backend_dev_supports_op(dev, gdn);
}

// the elastic memory of the model, see llama_box, or nullptr when the model or the parameters cannot use it
static std::unique_ptr<llama_box> llama_memory_hybrid_make_box(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   rs_replay_req,
                     bool   offload,
                     bool   unified,
                 uint32_t   n_swa,
                   size_t   box_bytes,
                     bool   box_draft,
                ggml_type   box_draft_type_k,
                ggml_type   box_draft_type_v,
    const llama_memory_i::layer_filter_cb & filter_attn,
    const llama_memory_i::layer_filter_cb & filter_recr) {
    const auto & hparams = model.hparams;

    const char * why = nullptr;

    // the graphs of the delta-net models read and write their states through the rows of the box
    const bool arch_ok = model.arch == LLM_ARCH_QWEN3NEXT || model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN35MOE;

    // the state of a seq has one scan row: no rollback, or a rollback that replays (see llama_memory_recurrent::rs_replay)
    const bool rs_replay = rs_replay_req && n_rs_seq > 0 && llm_arch_supports_rs_replay(model.arch) && hparams.n_embd_gdn_inp() > 0;

    if (!arch_ok) {
        why = "the model";
    } else if (v_trans) {
        why = "attention without flash attention";
    } else if (!unified) {
        why = "a KV cache per sequence";
    } else if (n_swa > 0) {
        why = "sliding window attention";
    } else if (n_rs_seq > 0 && !rs_replay) {
        why = "recurrent rollback by snapshots (--rs-rollback replay is supported)";
    } else if (hparams.ple_conv_state() > 0) {
        why = "the PLE conv state";
    }

    std::vector<llama_box::attn_layer> attn;
    std::vector<llama_box::recr_layer> recr;

    llama_box::attn_layer draft = {};
    bool has_draft = false;

    ggml_backend_buffer_type_t buft = nullptr;

    const auto layer_buft = [&](uint32_t il) {
        return offload ? ggml_backend_dev_buffer_type(model.dev_layer(il)) : ggml_backend_cpu_buffer_type();
    };

    if (!why) {
        for (uint32_t il = 0; il < hparams.n_layer_all && !why; ++il) {
            if (!hparams.has_kv(il) || (filter_attn && !filter_attn(il))) {
                continue;
            }
            if (buft && layer_buft(il) != buft) {
                why = "layers on more than one device";
            }
            buft = layer_buft(il);
            attn.push_back({ (int32_t) il, type_k, type_v, hparams.n_embd_k_gqa(il), hparams.n_embd_v_gqa(il), !hparams.is_mla() });
        }

        for (uint32_t il = 0; il < hparams.n_layer() && !why; ++il) {
            if (filter_recr && !filter_recr(il)) {
                continue;
            }
            if (buft && layer_buft(il) != buft) {
                why = "layers on more than one device";
            }
            buft = layer_buft(il);
            const uint32_t n_embd_x = rs_replay ? hparams.n_embd_gdn_inp()*(1 + n_rs_seq) : 0;
            recr.push_back({ (int32_t) il, type_r, type_s, hparams.n_embd_r(), hparams.n_embd_s(), n_embd_x, 1 + n_rs_seq });
        }

        // the one layer of an MTP head keeps its rows in the blocks, next to the layers of the same tokens
        if (box_draft && !why && hparams.n_layer_all == hparams.n_layer() + 1 && layer_buft(hparams.n_layer()) == buft) {
            const uint32_t il = hparams.n_layer();
            if (hparams.has_kv(il)) {
                draft = { (int32_t) il, box_draft_type_k, box_draft_type_v, hparams.n_embd_k_gqa(il), hparams.n_embd_v_gqa(il), !hparams.is_mla() };
                has_draft = true;
            }
        }

        if (!why && !llama_box::supported(attn, recr, has_draft ? &draft : nullptr)) {
            why = "K and V rows of different sizes";
        }

        if (!why && !llama_box_device_supported(hparams, buft, attn[0], !recr.empty())) {
            why = "the device";
        }
    }

    if (why) {
        LLAMA_LOG_INFO("%s: the elastic memory is off: %s does not support it\n", __func__, why);
        return nullptr;
    }

    // the memory of box_bytes bytes, or of kv_size cells and rs_size states, elastic
    auto box = std::make_unique<llama_box>(std::move(attn), std::move(recr), has_draft ? &draft : nullptr, box_bytes, kv_size, rs_size, n_seq_max);
    box->alloc(buft, hparams.no_alloc);

    return box;
}

//
// llama_memory_hybrid
//

llama_memory_hybrid::llama_memory_hybrid(
        const llama_model & model,
                            /* attn */
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   v_trans,
                 uint32_t   kv_size,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
                            /* recurrent */
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                            /* common */
                 uint32_t   n_seq_max,
                 uint32_t   n_rs_seq,
                     bool   rs_replay,
                     bool   offload,
                     bool   unified,
                            /* layer filters */
    const layer_filter_cb & filter_attn,
    const layer_filter_cb & filter_recr,
                     bool   use_box,
                   size_t   box_bytes,
                     bool   box_draft,
                ggml_type   box_draft_type_k,
                ggml_type   box_draft_type_v) :
    hparams(model.hparams),
    box(use_box ? llama_memory_hybrid_make_box(model, type_k, type_v, v_trans, kv_size, type_r, type_s, rs_size, n_seq_max, n_rs_seq,
                rs_replay, offload, unified, n_swa, box_bytes, box_draft, box_draft_type_k, box_draft_type_v,
                filter_attn == nullptr ? [&](int32_t il) { return !model.hparams.is_recr(il); } : filter_attn,
                filter_recr == nullptr ? [&](int32_t il) { return  model.hparams.is_recr(il); } : filter_recr) : nullptr),
    mem_attn(new llama_kv_cache(
        model,
        model.hparams,
        type_k,
        type_v,
        v_trans,
        offload,
        unified,
        box ? box->get_n_cells() : kv_size,
        n_seq_max,
        n_pad,
        n_swa,
        swa_type,
        nullptr,
        filter_attn == nullptr ?
            [&](int32_t il) { return !hparams.is_recr(il); }
            : filter_attn,
        nullptr,
        nullptr,
        "",
        box.get()
    )),
    mem_recr(new llama_memory_recurrent(
        model,
        type_r,
        type_s,
        offload,
        box ? std::max<uint32_t>(1, box->get_max_states()) : rs_size,
        n_seq_max,
        n_rs_seq,
        rs_replay,
        filter_recr == nullptr ?
            [&](int32_t il) { return hparams.is_recr(il); }
            : filter_recr,
        box.get()
    )) {
    // an admitted request keeps the state its admission took, see llama_box::reserve()
    if (box) {
        mem_recr->box_keep = [this](llama_seq_id seq_id) {
            return box->seq_admitted(seq_id);
        };
    }
}

llama_memory_hybrid::~llama_memory_hybrid() = default;

const llama_box * llama_memory_hybrid::get_box() const {
    return box.get();
}

bool llama_memory_hybrid::box_prepare(const std::vector<llama_ubatch> & ubatches) {
    if (!mem_recr->box_prepare(ubatches)) {
        return false;
    }

    // the new tokens of each seq, which take the empty cells of its blocks, then blocks from the free stack
    std::map<llama_seq_id, uint32_t> n_new;
    for (const auto & ubatch : ubatches) {
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            n_new[ubatch.seq_id[i][0]]++;
        }
    }

    for (const auto & [seq_id, n] : n_new) {
        while (mem_attn->box_seq_free(seq_id) < n) {
            if (!box->block_take(seq_id)) {
                LLAMA_LOG_WARN("%s: no free block in the box for the tokens of seq %d\n", __func__, seq_id);
                return false;
            }
        }
    }

    return true;
}

void llama_memory_hybrid::box_seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // all positions of a seq: its blocks go back to the box, except those holding cells of other seqs
    const bool all = p0 <= 0 && (p1 < 0 || p1 == std::numeric_limits<llama_pos>::max());

    if (all) {
        if (seq_id >= 0) {
            box->seq_release(seq_id);
        } else {
            for (llama_seq_id s = 0; s < (llama_seq_id) mem_attn->get_n_seq_max(); ++s) {
                box->seq_release(s);
            }
        }
    }

    // blocks held for other seqs may be empty now
    box->reclaim();
}

bool llama_memory_hybrid::box_reserve(llama_seq_id seq_id, uint32_t n_input, uint32_t n_output) {
    GGML_ASSERT(box);

    std::function<int32_t()> state_alloc = nullptr;
    if (mem_recr->seq_tails[seq_id] < 0) {
        state_alloc = [&]() -> int32_t {
            return mem_recr->box_state_take(seq_id) ? mem_recr->seq_tails[seq_id] : -1;
        };
    }

    return box->reserve(seq_id, n_input, n_output, state_alloc);
}

void llama_memory_hybrid::box_reserve_drop(llama_seq_id seq_id) {
    GGML_ASSERT(box);

    box->reserve_drop(seq_id);
}

llama_memory_context_ptr llama_memory_hybrid::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        // follow the recurrent pattern for creating the ubatch splits
        std::vector<llama_ubatch> ubatches;

        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // Use non-sequential split when KV cache is unified (needed for hellaswag/winogrande/multiple-choice)
                const bool unified = (mem_attn->get_n_stream() == 1);

                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                const uint32_t n_rs_seq = mem_recr->n_rs_seq;

                ubatch = balloc.split_equal(n_ubatch, !unified, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
            }

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        // every ubatch is another pass over the weights: log how the batch was split
        if (ubatches.size() > 1) {
            std::string desc;
            for (const auto & ub : ubatches) {
                desc += format(" %ux%u", ub.n_seqs, ub.n_seq_tokens);
            }
            LLAMA_LOG_DEBUG("%s: %u tokens split into %zu ubatches (seqs x tokens):%s\n",
                    __func__, balloc.get_n_tokens(), ubatches.size(), desc.c_str());
        }

        // the box first gives the seqs their states and pages, which the caches then place the ubatches in
        if (box && !box_prepare(ubatches)) {
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the recurrent batches first
        if (!mem_recr->prepare(ubatches)) {
            // TODO: will the recurrent cache be in an undefined context at this point?
            LLAMA_LOG_ERROR("%s: failed to prepare recurrent ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        // prepare the attention cache
        auto heads_attn = mem_attn->prepare(ubatches);
        if (heads_attn.empty()) {
            LLAMA_LOG_ERROR("%s: failed to prepare attention ubatches\n", __func__);
            return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
        }

        return std::make_unique<llama_memory_hybrid_context>(
                this, std::move(heads_attn), std::move(ubatches));
    } while(false);

    return std::make_unique<llama_memory_hybrid_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_hybrid::init_full() {
    return std::make_unique<llama_memory_hybrid_context>(this);
}

llama_memory_context_ptr llama_memory_hybrid::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_memory_hybrid_context>(this, lctx, optimize);
}

bool llama_memory_hybrid::get_can_shift() const {
    // Shifting is trivially supported for recurrent
    return mem_attn->get_can_shift();
}

void llama_memory_hybrid::clear(bool data) {
    mem_attn->clear(data);
    mem_recr->clear(data);

    if (box) {
        box->clear();
    }
}

bool llama_memory_hybrid::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // Try removing from the recurrent cache first since it may fail. If it does
    // fail, the cache will not have been mutated.
    if (!mem_recr->seq_rm(seq_id, p0, p1)) {
        return false;
    }

    const bool res = mem_attn->seq_rm(seq_id, p0, p1);

    if (box) {
        box_seq_rm(seq_id, p0, p1);
    }

    return res;
}

void llama_memory_hybrid::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    mem_attn->seq_cp(seq_id_src, seq_id_dst, p0, p1);
    mem_recr->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_hybrid::seq_keep(llama_seq_id seq_id) {
    mem_attn->seq_keep(seq_id);
    mem_recr->seq_keep(seq_id);

    if (box) {
        for (llama_seq_id s = 0; s < (llama_seq_id) mem_attn->get_n_seq_max(); ++s) {
            if (s != seq_id) {
                box->seq_release(s);
            }
        }
    }
}

void llama_memory_hybrid::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    mem_attn->seq_add(seq_id, p0, p1, shift);
    mem_recr->seq_add(seq_id, p0, p1, shift);
}

void llama_memory_hybrid::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    mem_attn->seq_div(seq_id, p0, p1, d);
    mem_recr->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_hybrid::seq_pos_min(llama_seq_id seq_id) const {
    // the min of the total cache is the max of the two caches' min values
    return std::max(mem_attn->seq_pos_min(seq_id), mem_recr->seq_pos_min(seq_id));
}

llama_pos llama_memory_hybrid::seq_pos_max(llama_seq_id seq_id) const {
    // the max of the total cache is the min of the two caches' max values
    return std::min(mem_attn->seq_pos_max(seq_id), mem_recr->seq_pos_max(seq_id));
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_hybrid::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> mb = mem_attn->memory_breakdown();
    for (const auto & buft_size : mem_recr->memory_breakdown()) {
        mb[buft_size.first] += buft_size.second;
    }
    if (box) {
        mb[ggml_backend_buffer_get_type(box->get_buffer())] += box->get_total_bytes();
    }
    return mb;
}

void llama_memory_hybrid::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    if ((flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0) {
        mem_attn->state_write(io, seq_id, flags);
    }
    mem_recr->state_write(io, seq_id, flags);
}

void llama_memory_hybrid::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    const bool read_attn = (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY) == 0;

    if (read_attn) {
        mem_attn->state_read(io, seq_id, flags);
    }

    try {
        mem_recr->state_read(io, seq_id, flags);
    } catch (...) {
        // the attention part is already restored - undo it
        if (read_attn) {
            mem_attn->state_clear(seq_id);
        }

        throw;
    }
}

llama_kv_cache * llama_memory_hybrid::get_mem_attn() const {
    return mem_attn.get();
}

llama_memory_recurrent * llama_memory_hybrid::get_mem_recr() const {
    return mem_recr.get();
}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_status status) : status(status) {}

llama_memory_hybrid_context::llama_memory_hybrid_context(llama_memory_hybrid * mem) :
    ctx_attn(mem->get_mem_attn()->init_full()),
    ctx_recr(mem->get_mem_recr()->init_full()),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
        llama_memory_hybrid * mem,
              llama_context * lctx,
                       bool   optimize) :
    ctx_attn(mem->get_mem_attn()->init_update(lctx, optimize)),
    ctx_recr(mem->get_mem_recr()->init_update(lctx, optimize)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

llama_memory_hybrid_context::llama_memory_hybrid_context(
              llama_memory_hybrid * mem,
                  slot_info_vec_t   sinfos_attn,
        std::vector<llama_ubatch>   ubatches) :
    ubatches(std::move(ubatches)),
    // note: here we copy the ubatches. not sure if this is ideal
    ctx_attn(new llama_kv_cache_context(mem->get_mem_attn(), std::move(sinfos_attn), this->ubatches)),
    ctx_recr(new llama_memory_recurrent_context(mem->get_mem_recr(), this->ubatches)),
    status(llama_memory_status_combine(ctx_attn->get_status(), ctx_recr->get_status())) {
}

bool llama_memory_hybrid_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_attn->next();
    ctx_recr->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_hybrid_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_attn->apply();
    res = res & ctx_recr->apply();

    return res;
}

llama_memory_status llama_memory_hybrid_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_hybrid_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);
    return ubatches[i_next];
}

const llama_kv_cache_context * llama_memory_hybrid_context::get_attn() const {
    return static_cast<const llama_kv_cache_context *>(ctx_attn.get());
}

const llama_memory_recurrent_context * llama_memory_hybrid_context::get_recr() const {
    return static_cast<const llama_memory_recurrent_context *>(ctx_recr.get());
}
