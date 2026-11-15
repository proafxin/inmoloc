#include "llama-kv-cache-dsv4.h"

#include "ggml-backend.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>

static constexpr uint32_t DSV4_CSA_RATIO = 4;
static constexpr uint32_t DSV4_HCA_RATIO = 128;

static constexpr uint32_t DSV4_STATE_MAGIC         = 0x34565344; // DSV4
static constexpr uint32_t DSV4_STATE_VERSION       = 1;
static constexpr uint32_t DSV4_STATE_MODE_FULL     = 0;
static constexpr uint32_t DSV4_STATE_MODE_PARTIAL  = 1;

// whether a sequence has tokens of different sets of sequences in the batch: a token of several sequences is decoded
// once for all of them, and their tokens of their own come in another batch, see llama_kv_cache_dsv4::box_prepare()
static bool dsv4_batch_has_mixed(const llama_batch & batch) {
    if (!batch.n_seq_id) {
        return false;
    }

    std::map<llama_seq_id, std::vector<llama_seq_id>> seq_sets;

    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        // in order: the first one leads, see dsv4_build_comp_plan()
        const std::vector<llama_seq_id> set(batch.seq_id[i], batch.seq_id[i] + batch.n_seq_id[i]);

        for (const llama_seq_id seq_id : set) {
            const auto [it, added] = seq_sets.emplace(seq_id, set);
            if (!added && it->second != set) {
                return true;
            }
        }
    }

    return false;
}

// a token of several sequences: the first one (the lead) writes the compressed rows, which the others share, and each one
// keeps its state, see llama_kv_cache_dsv4::box_prepare()
static llama_kv_cache_dsv4_context::comp_plan dsv4_build_comp_plan(
        const llama_ubatch & ubatch,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t n_rs_seq,
        const std::vector<uint32_t> & rs_idx,
        const llama_kv_cache_dsv4 * kv,
        const llama_dsv4_comp_state * state,
        bool lists) {
    llama_kv_cache_dsv4_context::comp_plan plan;
    plan.n_visible.resize(ubatch.n_tokens);

    llama_box * box = kv->get_box();

    const llama_box::block_kind kind = ratio == DSV4_CSA_RATIO ? llama_box::BLOCK_CSA : llama_box::BLOCK_HCA;

    // the graph reads the state rows of the sequences of the ubatch into [state rows | scratch], which the source indices
    // below refer to, and the rows of the box are written directly
    std::vector<int32_t> seq_base(box->get_n_seq_max(), -1);
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        seq_base[ubatch.seq_id_unq[s]] = (int32_t) (s*state_size);
    }

    const auto write_row = [&](llama_seq_id seq_id, uint32_t r) -> int64_t {
        return box->comp_row(seq_id, kind, r);
    };

    const int64_t state_rows = (int64_t) state_size*ubatch.n_seqs_unq;

    struct persist_row {
        int32_t dst;
        int32_t src;
        llama_pos pos;
    };

    std::vector<persist_row> persist_rows;

    // For the overlap compressor, build_overlap_compressed_kv_from_state() consumes
    // state_read_idxs as two contiguous halves: the first ratio*n_blocks entries are
    // the "previous-window" gather indices for every block, followed by the
    // "current-window" indices for every block. Collect them separately here and
    // append cur after prev once the loop has visited all completed blocks
    std::vector<int32_t> overlap_prev_reads;
    std::vector<int32_t> overlap_cur_reads;

    std::map<std::pair<llama_seq_id, llama_pos>, int64_t> curr_token_idx_map;
    std::map<llama_seq_id, uint32_t> state_write_counts;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            curr_token_idx_map[std::make_pair(ubatch.seq_id[i][s], ubatch.pos[i])] = i;
        }
    }

    const auto state_source_idx = [&](llama_seq_id seq_id, llama_pos pos) -> int32_t {
        if (pos < 0) {
            // The overlap compressor needs a zero/-inf source for the first
            // block's previous half. The graph appends that row after the
            // current-ubatch scratch rows.
            return (int32_t) (state_rows + ubatch.n_tokens);
        }

        const auto key = std::make_pair(seq_id, pos);
        if (curr_token_idx_map.find(key) != curr_token_idx_map.end()) {
            return (int32_t) (state_rows + curr_token_idx_map.at(key));
        }

        return (int32_t) (seq_base[seq_id] + pos%state_size);
    };

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_pos pos = ubatch.pos[i];

        if (pos < 0) {
            continue;
        }

        plan.state_pos.push_back((int32_t) (pos%ratio));

        const int64_t n_visible = (int64_t) (pos + 1)/ratio;
        plan.n_visible[i] = (int32_t) n_visible;
        plan.n_kv = std::max(plan.n_kv, n_visible);

        for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id[i][s];
            const int32_t state_idx = (int32_t) (seq_base[seq_id] + pos%state_size);

            const auto it = std::find_if(persist_rows.begin(), persist_rows.end(),
                    [state_idx](const persist_row & row) {
                        return row.dst == state_idx;
                    });
            if (it == persist_rows.end()) {
                persist_rows.push_back({ state_idx, (int32_t) i, pos });
            } else if (pos > it->pos) {
                it->src = (int32_t) i;
                it->pos = pos;
            }

            // the others share the rows of the lead
            if (s > 0 || (pos + 1) % ratio != 0) {
                continue;
            }

            const llama_pos source_start = pos + 1 - ratio;

            plan.state_write_idxs.push_back(write_row(seq_id, pos/ratio));
            plan.state_write_pos.push_back((int32_t) source_start);
            ++state_write_counts[seq_id];

            if (overlap) {
                const llama_pos prev_start = source_start - ratio;

                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_prev_reads.push_back(state_source_idx(seq_id, prev_start + j));
                }
                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_cur_reads.push_back(state_source_idx(seq_id, source_start + j));
                }
            } else {
                for (uint32_t j = 0; j < ratio; ++j) {
                    plan.state_read_idxs.push_back(state_source_idx(seq_id, source_start + j));
                }
            }
        }
    }

    if (ratio == DSV4_CSA_RATIO && !plan.state_pos.empty()) {
        // Pad each sequence to the reserve plan's block count.
        const auto append_dummy_block = [&](llama_seq_id seq_id, uint32_t i) {
            const int32_t source_idx = state_source_idx(seq_id, ubatch.pos[i]);

            // the next row of the sequence, which no token sees yet
            plan.state_write_idxs.push_back(write_row(seq_id, (ubatch.pos[i] + 1)/ratio));
            plan.state_write_pos .push_back(0);

            if (overlap) {
                for (uint32_t j = 0; j < ratio; ++j) {
                    overlap_prev_reads.push_back(source_idx);
                    overlap_cur_reads .push_back(source_idx);
                }
            } else {
                for (uint32_t j = 0; j < ratio; ++j) {
                    plan.state_read_idxs.push_back(source_idx);
                }
            }
        };

        const uint32_t n_blocks = (std::max<uint32_t>(1, ubatch.n_seq_tokens) + ratio - 1)/ratio;

        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id_unq[s];

            // the last token of the seq: the row after it is the one no token sees yet, and no block of the ubatch
            // writes it (the row of the first token is the one its block writes)
            uint32_t i = ubatch.n_tokens;
            for (uint32_t j = 0; j < ubatch.n_tokens; ++j) {
                if (ubatch.pos[j] >= 0 && ubatch.seq_id[j][0] == seq_id && (i == ubatch.n_tokens || ubatch.pos[j] > ubatch.pos[i])) {
                    i = j;
                }
            }

            // the sharers write no rows
            if (i == ubatch.n_tokens) {
                continue;
            }

            const uint32_t n_writes = state_write_counts[seq_id];
            if (n_writes >= n_blocks) {
                continue;
            }
            if (n_writes + 1 != n_blocks) {
                throw std::runtime_error("DSV4 CSA sequence positions are not contiguous");
            }

            append_dummy_block(seq_id, i);
        }
    }

    if (ratio == DSV4_HCA_RATIO && !plan.state_pos.empty() && plan.state_write_idxs.empty()) {
        // Keep the compress/write ops in the graph when no HCA block completes in this ubatch: the dummy block writes the
        // next row of the sequence, which no token sees yet
        uint32_t i = 0;
        while (i < ubatch.n_tokens && ubatch.pos[i] < 0) {
            ++i;
        }
        assert(i < ubatch.n_tokens);

        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        const int32_t source_idx = state_source_idx(seq_id, ubatch.pos[i]);

        plan.state_write_idxs.push_back(write_row(seq_id, (ubatch.pos[i] + 1)/ratio));
        plan.state_write_pos .push_back(0);

        for (uint32_t j = 0; j < ratio; ++j) {
            plan.state_read_idxs.push_back(source_idx);
        }
    }

    if (overlap) {
        // [ all blocks' prev-window indices | all blocks' cur-window indices ]
        plan.state_read_idxs.reserve(overlap_prev_reads.size() + overlap_cur_reads.size());
        plan.state_read_idxs.insert(plan.state_read_idxs.end(),
                overlap_prev_reads.begin(), overlap_prev_reads.end());
        plan.state_read_idxs.insert(plan.state_read_idxs.end(),
                overlap_cur_reads.begin(), overlap_cur_reads.end());
    }

    {
        // the attentions read lists; the list of the indexer holds the rows of the sequences one after the other
        plan.n_kv = 0;

        if (lists) {
            std::vector<int32_t> seq_rows(box->get_n_seq_max(), 0);
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                seq_rows[ubatch.seq_id[i][0]] = std::max(seq_rows[ubatch.seq_id[i][0]], plan.n_visible[i]);
            }

            std::vector<int32_t> seq_start(box->get_n_seq_max(), 0);
            int64_t n_rows = 0;
            for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                seq_start[ubatch.seq_id_unq[s]] = (int32_t) n_rows;
                n_rows += seq_rows[ubatch.seq_id_unq[s]];
            }

            plan.n_kv = std::max<int64_t>(GGML_PAD(n_rows, 256u), 256);

            plan.n_skip.resize(ubatch.n_tokens);
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                plan.n_skip[i]     = seq_start[ubatch.seq_id[i][0]];
                plan.n_visible[i] += plan.n_skip[i];
            }
        }

        // the state rows of the sequences, from the plane of a pending rollback
        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id_unq[s];
            const int32_t st = kv->get_seq_state(seq_id);
            const uint32_t rollback = (size_t) seq_id < rs_idx.size() ? rs_idx[seq_id] : 0;

            for (uint32_t r = 0; r < state_size; ++r) {
                plan.state_restore_src_idxs.push_back(state->box_row(st, rollback, r));
            }
        }

        // the state after the ubatch: the last token of a row, or the row as it was
        std::vector<int32_t> persist_src(state_rows);
        for (int64_t r = 0; r < state_rows; ++r) {
            persist_src[r] = (int32_t) r;
        }
        for (const persist_row & row : persist_rows) {
            persist_src[row.dst] = (int32_t) (state_rows + row.src);
        }

        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id_unq[s];
            const int32_t st   = kv->get_seq_state(seq_id);
            const int64_t base = seq_base[seq_id];

            for (uint32_t r = 0; r < state_size; ++r) {
                plan.state_persist_src_idxs.push_back(persist_src[base + r]);
                plan.state_persist_dst_idxs.push_back(state->box_row(st, 0, r));
            }

            // the tokens of the sequence, those it shares included
            std::vector<uint32_t> token_idxs;
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                if (std::find(ubatch.seq_id[i], ubatch.seq_id[i] + ubatch.n_seq_id[i], seq_id) != ubatch.seq_id[i] + ubatch.n_seq_id[i]) {
                    token_idxs.push_back(i);
                }
            }

            // a rollback of d tokens finds the state before the last d tokens of the ubatch
            for (uint32_t d = 1; d <= n_rs_seq; ++d) {
                const uint32_t prefix = d <= token_idxs.size() ? (uint32_t) token_idxs.size() - d : 0;

                for (uint32_t r = 0; r < state_size; ++r) {
                    int32_t src = (int32_t) (base + r);

                    for (uint32_t j = 0; j < prefix; ++j) {
                        const uint32_t i_tok = token_idxs[j];
                        if (ubatch.pos[i_tok] >= 0 && (uint32_t) (ubatch.pos[i_tok]%state_size) == r) {
                            src = (int32_t) (state_rows + i_tok);
                        }
                    }

                    plan.state_snapshot_src_idxs.push_back(src);
                    plan.state_snapshot_dst_idxs.push_back(state->box_row(st, d, r));
                }
            }
        }

    }

    return plan;
}

static std::vector<llama_kv_cache_dsv4_context::comp_plan> dsv4_build_comp_plans(
        const std::vector<llama_ubatch> & ubatches,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t n_rs_seq,
        const std::vector<uint32_t> & rs_idx,
        const llama_kv_cache_dsv4 * kv,
        const llama_dsv4_comp_state * state,
        bool lists) {
    std::vector<llama_kv_cache_dsv4_context::comp_plan> plans;
    plans.reserve(ubatches.size());

    // the first ubatch touching a seq consumes its rollback restore
    std::vector<uint32_t> rs(rs_idx);
    for (const llama_ubatch & ubatch : ubatches) {
        plans.push_back(dsv4_build_comp_plan(ubatch, ratio, overlap, state_size, n_rs_seq, rs, kv, state, lists));

        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
            const llama_seq_id seq_id = ubatch.seq_id_unq[s];
            if (seq_id >= 0 && (size_t) seq_id < rs.size()) {
                rs[seq_id] = 0;
            }
        }
    }

    return plans;
}

static llama_kv_cache_dsv4_context::comp_plan dsv4_build_reserve_comp_plan(
        const llama_ubatch & ubatch,
        uint32_t ratio,
        bool overlap,
        uint32_t state_size,
        uint32_t kv_size,
        uint32_t n_rs_seq,
        bool lists) {
    llama_kv_cache_dsv4_context::comp_plan plan;
    plan.n_visible.resize(ubatch.n_tokens);
    plan.n_kv = lists ? kv_size : 0;

    if (ubatch.n_tokens == 0) {
        return plan;
    }

    // worst case over every seq split: sum of per-seq ceil(tokens/ratio) is at
    // most floor(n_tokens/ratio) + n_seqs
    const uint32_t n_seqs = std::max<uint32_t>(1, ubatch.n_seqs);
    const size_t n_blocks = (size_t) ubatch.n_tokens/ratio + n_seqs;

    // every token of a ubatch can be a sequence of its own
    const uint64_t state_rows = (uint64_t) state_size*ubatch.n_tokens;
    const size_t n_snapshot = (size_t) n_rs_seq*state_rows;

    plan.state_pos .resize(ubatch.n_tokens);
    plan.state_persist_src_idxs.resize(state_rows);
    plan.state_persist_dst_idxs.resize(state_rows);
    plan.state_restore_src_idxs.resize(state_rows);
    plan.state_snapshot_src_idxs.resize(n_snapshot);
    plan.state_snapshot_dst_idxs.resize(n_snapshot);
    plan.n_skip.resize(lists ? ubatch.n_tokens : 0);
    plan.state_read_idxs .resize((overlap ? 2u : 1u)*ratio*n_blocks);
    plan.state_write_idxs.resize(n_blocks);
    plan.state_write_pos .resize(n_blocks);

    return plan;
}

static void dsv4_make_k_only(llama_hparams & hparams) {
    // llama_kv_cache uses hparams.is_mla() to allocate K-only storage.
    hparams.n_embd_head_k_mla_impl = hparams.n_embd_head_k();
    hparams.n_embd_head_v_mla_impl = hparams.n_embd_head_k();
}

//
// llama_dsv4_comp_state
//

llama_dsv4_comp_state::llama_dsv4_comp_state(
        llama_box * box,
        uint32_t    ratio,
        uint32_t    state_size,
        uint32_t    n_rs_seq,
        uint32_t    piece0,
        const std::vector<uint32_t> & ils) :
    ratio(ratio),
    state_size(state_size),
    n_rs_seq(n_rs_seq),
    box(box),
    piece0(piece0),
    ils(ils) {
    for (size_t i = 0; i < ils.size(); ++i) {
        layer l = {};
        l.il    = ils[i];
        l.kv    = box->get_piece(piece0 + 2*(uint32_t) i);
        l.score = box->get_piece(piece0 + 2*(uint32_t) i + 1);

        map_layer_ids[ils[i]] = (int32_t) layers.size();
        layers.push_back(std::move(l));
    }
}

int32_t llama_dsv4_comp_state::box_row(int32_t s, uint32_t plane, uint32_t r) const {
    return (int32_t) box->p_row(s, piece0, plane*state_size + r);
}

ggml_tensor * llama_dsv4_comp_state::get_kv_all(int32_t il) const {
    return layers[map_layer_ids.at(il)].kv;
}

ggml_tensor * llama_dsv4_comp_state::get_score_all(int32_t il) const {
    return layers[map_layer_ids.at(il)].score;
}

ggml_tensor * llama_dsv4_comp_state::cpy_kv(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const {
    return ggml_set_rows(ctx, get_kv_all(il), cur, idxs);
}

ggml_tensor * llama_dsv4_comp_state::cpy_score(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const {
    return ggml_set_rows(ctx, get_score_all(il), cur, idxs);
}

//
// llama_kv_cache_dsv4_draft
//

llama_dsv4_draft_hparams::llama_dsv4_draft_hparams(const llama_hparams & hparams) : hparams_draft(hparams) {
    dsv4_make_k_only(hparams_draft);
}

llama_kv_cache_dsv4_draft::llama_kv_cache_dsv4_draft(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   offload,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch,
                 uint32_t   n_rs_seq,
    const layer_filter_cb & filter,
    const std::shared_ptr<llama_box> & box) :
    llama_dsv4_draft_hparams(model.hparams),
    llama_kv_cache_iswa(model, hparams_draft, type_k, type_v, offload, false, box->get_n_cells(), n_seq_max, n_ubatch, 1,
            nullptr, filter, nullptr, nullptr, nullptr, box) {
    // what a token of the next batch reads, and a rollback before it
    set_box_keep(hparams_draft.n_swa - 1 + n_rs_seq);
}

//
// llama_kv_cache_dsv4
//

llama_kv_cache_dsv4::llama_kv_cache_dsv4(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   offload,
                     bool   swa_full,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_ubatch,
                 uint32_t   n_pad,
                 uint32_t   n_rs_seq,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
                   size_t   box_bytes,
                     bool   box_deferred,
                 uint32_t   n_batch,
                     const llama_box_draft & box_draft) :
    hparams_raw(model.hparams),
    hparams_csa(model.hparams),
    hparams_hca(model.hparams),
    hparams_lid(model.hparams),
    n_seq_max(n_seq_max),
    n_rs_seq(n_rs_seq),
    rs_idx(n_seq_max, 0) {

    const layer_filter_cb filter_raw = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        // the MTP draft has a cache of its own: the box only holds the layers of the model
        return il < (int32_t) model.hparams.n_layer();
    };

    hparams_raw.n_layer_nextn = 0;
    hparams_csa.n_layer_nextn = 0;
    hparams_hca.n_layer_nextn = 0;
    hparams_lid.n_layer_nextn = 0;

    LLAMA_LOG_INFO("%s: creating DSV4 raw KV cache\n", __func__);

    dsv4_make_k_only(hparams_raw);

    const uint32_t n_layer = model.hparams.n_layer();

    const auto is_csa = [&](uint32_t il) { return model.hparams.dsv4_compress_ratios[il] == DSV4_CSA_RATIO; };
    const auto is_hca = [&](uint32_t il) { return model.hparams.dsv4_compress_ratios[il] == DSV4_HCA_RATIO; };

    std::vector<uint32_t> ils_csa;
    std::vector<uint32_t> ils_hca;
    for (uint32_t il = 0; il < n_layer; ++il) {
        if (filter && !filter(il)) {
            continue;
        }
        if (is_csa(il)) {
            ils_csa.push_back(il);
        } else if (is_hca(il)) {
            ils_hca.push_back(il);
        }
    }

    const uint32_t state_csa = 2*DSV4_CSA_RATIO;
    const uint32_t state_hca = DSV4_HCA_RATIO;

    {
        const uint32_t n_embd_head = model.hparams.n_embd_head_k();
        const uint32_t n_embd_lid  = model.hparams.indexer_head_size;

        std::vector<llama_box::attn_layer>  lid;
        std::vector<llama_box::state_piece> pieces;

        const auto add_pieces = [&](const std::vector<uint32_t> & ils, uint32_t n_embd, uint32_t size) {
            for (size_t i = 0; i < 2*ils.size(); ++i) {
                pieces.push_back({ GGML_TYPE_F32, n_embd, size*(1 + n_rs_seq) });
            }
        };

        add_pieces(ils_csa, 2*n_embd_head, state_csa);
        add_pieces(ils_hca,   n_embd_head, state_hca);
        add_pieces(ils_csa, 2*n_embd_lid,  state_csa);

        for (const uint32_t il : ils_csa) {
            lid.push_back({ (int32_t) il, type_k, type_k, n_embd_lid, n_embd_lid, false });
        }

        // the memory of n_seq_max sequences of kv_size tokens when no size is given: raw rows, compressed rows and states
        constexpr uint32_t n_block = llama_box::n_block_tokens;

        const uint32_t n_comp = kv_size/(DSV4_CSA_RATIO*n_block) + kv_size/(DSV4_HCA_RATIO*n_block) + 3;

        // the MTP layers have the K-only rows of the draft cache, see llama_kv_cache_dsv4_draft
        const llama_dsv4_draft_hparams hparams_draft(box_draft.hparams ? *box_draft.hparams : model.hparams);

        llama_box_draft draft = box_draft;
        draft.hparams = &hparams_draft.hparams_draft;

        box = llama_box_make(model, type_k, type_v, kv_size + n_seq_max*n_comp*n_block, n_seq_max, GGML_TYPE_F32, GGML_TYPE_F32, n_seq_max, 0, false, offload,
                box_bytes, box_deferred, draft,
                [&](int32_t il) { return filter_raw(il);

        // the draft is known by its model, not by the K-only copy of its hparams
        box->set_draft(box_draft.standalone ? box_draft.hparams : nullptr, box->get_draft_il_offset()); }, nullptr, {}, std::move(lid), std::move(pieces), &hparams_raw);

        kv_size   = box->get_n_cells();
        n_seq_max = box->get_n_seq_max();

        this->n_seq_max = n_seq_max;

        rs_idx.assign(n_seq_max, 0);
        seq_states.assign(n_seq_max, -1);

        // the window of raw rows of a sequence, whatever its length, and its compressed rows, each kind with the block the
        // next row goes to
        const uint32_t n_swa      = hparams_raw.n_swa;
        // a draft decodes a batch after its target, and holds its cells in the blocks until then
        const uint32_t n_raw_keep = n_swa + n_rs_seq + (box->has_draft() ? 2 : 1)*std::max(n_batch, n_ubatch);

        box->set_blocks_fn([=](uint32_t n) {
            constexpr uint32_t n_block = llama_box::n_block_tokens;

            const uint32_t n_raw = (std::min(n, n_raw_keep) + n_block - 1)/n_block + 1;
            const uint32_t n_csa = ils_csa.empty() ? 0 : n/(DSV4_CSA_RATIO*n_block) + 1;
            const uint32_t n_hca = ils_hca.empty() ? 0 : n/(DSV4_HCA_RATIO*n_block) + 1;

            return n_raw + n_csa + n_hca;
        });
    }

    kv_raw = std::make_unique<llama_kv_cache_iswa>(
            model, hparams_raw, type_k, type_v,
            offload, swa_full, kv_size, n_seq_max, n_ubatch, n_pad,
            nullptr, filter_raw, reuse, nullptr, box);

    dsv4_make_k_only(hparams_csa);
    dsv4_make_k_only(hparams_hca);

    std::fill(hparams_lid.n_head_kv_arr.begin(), hparams_lid.n_head_kv_arr.end(), 1);
    hparams_lid.n_embd_head_k_full = model.hparams.indexer_head_size;
    hparams_lid.n_embd_head_v_full = model.hparams.indexer_head_size;
    hparams_lid.n_embd_head_k_swa  = model.hparams.indexer_head_size;
    hparams_lid.n_embd_head_v_swa  = model.hparams.indexer_head_size;
    hparams_lid.rope_type          = LLAMA_ROPE_TYPE_NEOX;
    dsv4_make_k_only(hparams_lid);

    const layer_filter_cb filter_csa = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return model.hparams.dsv4_compress_ratios[il] == DSV4_CSA_RATIO;
    };

    const layer_filter_cb filter_hca = [&](int32_t il) {
        if (filter && !filter(il)) {
            return false;
        }

        return model.hparams.dsv4_compress_ratios[il] == DSV4_HCA_RATIO;
    };

    // the elastic memory holds the compressed rows, these caches of one cell keep the rotation matrices of the graph
    kv_csa = std::make_unique<llama_kv_cache>(
            model, hparams_csa, type_k, type_v,
            offload, 1, 1, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_csa, nullptr, nullptr);

    kv_hca = std::make_unique<llama_kv_cache>(
            model, hparams_hca, type_k, type_v,
            offload, 1, 1, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_hca, nullptr, nullptr);

    kv_lid = std::make_unique<llama_kv_cache>(
            model, hparams_lid, type_k, type_v,
            offload, 1, 1, n_pad,
            0, LLAMA_SWA_TYPE_NONE, nullptr, filter_csa, nullptr, nullptr);

    const uint32_t n_csa = (uint32_t) ils_csa.size();
    const uint32_t n_hca = (uint32_t) ils_hca.size();

    csa_state = std::make_unique<llama_dsv4_comp_state>(box.get(), DSV4_CSA_RATIO, state_csa, n_rs_seq, 0, ils_csa);
    hca_state = std::make_unique<llama_dsv4_comp_state>(box.get(), DSV4_HCA_RATIO, state_hca, n_rs_seq, 2*n_csa, ils_hca);
    lid_state = std::make_unique<llama_dsv4_comp_state>(box.get(), DSV4_CSA_RATIO, state_csa, n_rs_seq, 2*(n_csa + n_hca), ils_csa);
}

llama_memory_context_ptr llama_kv_cache_dsv4::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    if (dsv4_batch_has_mixed(balloc.get_batch())) {
        LLAMA_LOG_ERROR("%s: a DSV4 sequence has tokens it shares with other sequences and tokens of its own in one batch, "
                "decode them in separate batches\n", __func__);
        return std::make_unique<llama_kv_cache_dsv4_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    balloc.split_reset();

    std::vector<llama_ubatch> ubatches;
    while (true) {
        // [TAG_RECURRENT_ROLLBACK_SPLITS]
        auto ubatch = balloc.split_equal(n_ubatch, false, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
        if (ubatch.n_tokens == 0) {
            break;
        }
        ubatches.push_back(std::move(ubatch)); // NOLINT
    }

    if (balloc.get_n_used() < balloc.get_n_tokens()) {
        return std::make_unique<llama_kv_cache_dsv4_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    }

    // the blocks box_prepare() made own blocks of hold the rows before the batch: no graph copies them
    const auto failed = [this]() {
        box->comp_copy(take_block_copies());
        return std::make_unique<llama_kv_cache_dsv4_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
    };

    if (!box_prepare(ubatches)) {
        return failed();
    }

    auto sinfos_base = kv_raw->get_base()->prepare(ubatches);
    if (sinfos_base.empty()) {
        return failed();
    }

    auto sinfos_swa = kv_raw->get_swa()->prepare(ubatches);
    if (sinfos_swa.empty()) {
        return failed();
    }

    return std::make_unique<llama_kv_cache_dsv4_context>(
            this,
            std::move(sinfos_base),
            std::move(sinfos_swa),
            std::move(ubatches));
}

llama_memory_context_ptr llama_kv_cache_dsv4::init_full() {
    return std::make_unique<llama_kv_cache_dsv4_context>(this);
}

llama_memory_context_ptr llama_kv_cache_dsv4::init_update(llama_context * lctx, bool optimize) {
    return std::make_unique<llama_kv_cache_dsv4_context>(this, lctx, optimize);
}

bool llama_kv_cache_dsv4::get_can_shift() const {
    // Compressed row metadata uses block-derived positions. Keep shifting
    // disabled until DSV4 compressed-cache shift semantics are wired.
    return false;
}

void llama_kv_cache_dsv4::clear(bool data) {
    kv_raw->clear(data);

    std::fill(seq_states.begin(), seq_states.end(), -1);
    std::fill(rs_idx.begin(), rs_idx.end(), 0);
}

bool llama_kv_cache_dsv4::box_state_take(llama_seq_id seq_id) {
    if (seq_states[seq_id] < 0) {
        seq_states[seq_id] = box->state_alloc();

        LLAMA_LOG_DEBUG("%s: seq %d takes state %d, %u of %u blocks free\n", __func__, seq_id, seq_states[seq_id], box->n_free(),
                box->get_n_blocks());
    }

    return seq_states[seq_id] >= 0;
}

bool llama_kv_cache_dsv4::box_prepare(const std::vector<llama_ubatch> & ubatches) {
    constexpr uint32_t n_block = llama_box::n_block_tokens;

    // the leads, which write the compressed rows of their tokens, and the sequences that share them, see
    // dsv4_batch_has_mixed()
    std::map<llama_seq_id, std::pair<llama_pos, llama_pos>> seqs;
    std::map<llama_seq_id, llama_seq_id> sharers;

    for (const auto & ubatch : ubatches) {
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            const llama_seq_id seq_id = ubatch.seq_id[i][0];

            auto it = seqs.emplace(seq_id, std::make_pair(ubatch.pos[i], ubatch.pos[i])).first;

            it->second.first  = std::min(it->second.first,  ubatch.pos[i]);
            it->second.second = std::max(it->second.second, ubatch.pos[i]);

            for (int32_t s = 1; s < ubatch.n_seq_id[i]; ++s) {
                sharers[ubatch.seq_id[i][s]] = seq_id;
            }
        }
    }

    const bool has_csa = !csa_state->get_ils().empty();
    const bool has_hca = !hca_state->get_ils().empty();

    std::vector<llama_box::block_kind> kinds;
    if (has_csa) {
        kinds.push_back(llama_box::BLOCK_CSA);
    }
    if (has_hca) {
        kinds.push_back(llama_box::BLOCK_HCA);
    }

    block_copies.clear();

    // the sharers take the rows of their lead: their own go first, so the lead copies no block it shares with them only
    for (const auto & [seq_id, lead] : sharers) {
        for (const auto kind : kinds) {
            box->comp_clear(seq_id, kind);
        }
    }

    const auto take = [&](llama_seq_id seq_id, std::pair<llama_pos, llama_pos> range) {
        if (!box_state_take(seq_id)) {
            LLAMA_LOG_WARN("%s: no room in the box for the state of seq %d\n", __func__, seq_id);
            return false;
        }

        // the raw rows out of reach of the first token of the batch, and of a rollback before it, leave the window
        const llama_pos p_keep = range.first - (llama_pos) hparams_raw.n_swa + 1 - (llama_pos) n_rs_seq;
        if (p_keep > 0) {
            kv_raw->seq_rm(seq_id, -1, p_keep);
            box->seq_evict(seq_id);
        }

        return true;
    };

    for (const auto & [seq_id, range] : seqs) {
        if (!take(seq_id, range)) {
            return false;
        }

        // the rows of the batch and the next one, which stands for a row not written yet, see dsv4_build_comp_plan()
        const uint32_t n_next = (uint32_t) range.second + 1;

        for (const auto kind : kinds) {
            const uint32_t ratio = kind == llama_box::BLOCK_CSA ? DSV4_CSA_RATIO : DSV4_HCA_RATIO;

            // the blocks the batch writes, which other sequences still hold, become the lead's own: the graph of the
            // first ubatch copies them, see llama_kv_cache_dsv4_context::get_block_copies()
            if (!box->comp_own(seq_id, kind, ((uint32_t) range.first/ratio)/n_block, block_copies) ||
                !box->comp_take(seq_id, kind, n_next/(ratio*n_block) + 1)) {
                LLAMA_LOG_WARN("%s: no free block in the box for the compressed rows of seq %d\n", __func__, seq_id);
                return false;
            }
        }
    }

    for (const auto & [seq_id, lead] : sharers) {
        if (!take(seq_id, seqs.at(lead))) {
            return false;
        }

        for (const auto kind : kinds) {
            if (!box->comp_share(lead, seq_id, kind, box->comp_blocks(lead, kind))) {
                LLAMA_LOG_WARN("%s: no free block in the box for the compressed rows of seq %d\n", __func__, seq_id);
                return false;
            }
        }
    }

    return true;
}

bool llama_kv_cache_dsv4::box_reserve(llama_seq_id seq_id, uint32_t n_input, uint32_t n_output) {
    std::function<int32_t()> state_alloc;
    if (seq_states[seq_id] < 0) {
        state_alloc = [this, seq_id]() {
            return box_state_take(seq_id) ? seq_states[seq_id] : -1;
        };
    }

    return box->reserve(seq_id, n_input, n_output, state_alloc);
}

void llama_kv_cache_dsv4::box_reserve_drop(llama_seq_id seq_id) {
    box->reserve_drop(seq_id);
}

void llama_kv_cache_dsv4::box_finalized() {
    kv_raw->box_finalized();

    const uint32_t n = box->get_n_seq_max();

    n_seq_max = n;

    rs_idx.assign(n, 0);
    seq_states.assign(n, -1);
}

void llama_kv_cache_dsv4::box_seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst) {
    constexpr uint32_t n_block = llama_box::n_block_tokens;

    // the compressed rows of the source are a history both sequences have: the blocks are shared, except those holding
    // the rows the destination may still write, its next ones and those a rollback rewrites
    const llama_pos n_pos = kv_raw->seq_pos_max(seq_id_src) + 1;

    for (const auto kind : { llama_box::BLOCK_CSA, llama_box::BLOCK_HCA }) {
        const uint32_t ratio = kind == llama_box::BLOCK_CSA ? DSV4_CSA_RATIO : DSV4_HCA_RATIO;

        const uint32_t r_write = (uint32_t) std::max<llama_pos>(0, n_pos - (llama_pos) n_rs_seq)/ratio;

        if (!box->comp_share(seq_id_src, seq_id_dst, kind, r_write/n_block)) {
            throw std::runtime_error("no free block in the box to copy a DSV4 sequence");
        }
    }

    if (!box_state_take(seq_id_dst)) {
        throw std::runtime_error("no room in the box for the state of a DSV4 sequence");
    }

    box->state_copy(seq_states[seq_id_src], seq_states[seq_id_dst]);

    rs_idx[seq_id_dst] = rs_idx[seq_id_src];
}

bool llama_kv_cache_dsv4::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    if (p1 >= 0) {
        return false;
    }

    if (p0 > 0) {
        if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
            return false;
        }

        const llama_pos pos_max = kv_raw->seq_pos_max(seq_id);
        if (p0 > pos_max) {
            return kv_raw->seq_rm(seq_id, p0, -1);
        }

        if (n_rs_seq == 0) {
            return false;
        }

        const llama_pos rollback = pos_max - (p0 - 1);
        if (rollback < 1 || rollback > (llama_pos) n_rs_seq) {
            return false;
        }

        // pending rollback is single-use: stacked partial removals don't compose
        if (rs_idx[seq_id] != 0) {
            return false;
        }

        const bool res = kv_raw->seq_rm(seq_id, p0, p1);
        if (res) {
            rs_idx[seq_id] = (uint32_t) rollback;
        }

        return res;
    }

    const bool res = kv_raw->seq_rm(seq_id, p0, p1);

    if (res) {
        clear_compressed(seq_id);
    }

    return res;
}

void llama_kv_cache_dsv4::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    GGML_ASSERT(p0 <= 0 && p1 < 0 && "DSV4 only supports full sequence copies");

    kv_raw->seq_cp(seq_id_src, seq_id_dst, p0, p1);

    if (seq_id_src != seq_id_dst) {
        box_seq_cp(seq_id_src, seq_id_dst);
    }
}

void llama_kv_cache_dsv4::seq_keep(llama_seq_id seq_id) {
    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max);

    kv_raw->seq_keep(seq_id);

    for (llama_seq_id id = 0; id < (llama_seq_id) n_seq_max; ++id) {
        if (id != seq_id) {
            kv_raw->seq_rm(id, -1, -1);
            clear_compressed(id);
        }
    }
}

void llama_kv_cache_dsv4::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    kv_raw->seq_add(seq_id, p0, p1, shift);
}

void llama_kv_cache_dsv4::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    kv_raw->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_kv_cache_dsv4::seq_pos_min(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
        return -1;
    }

    // The raw SWA cache may contain a wider window, but the compressed DSV4
    // state cannot be rolled back within that window. Report only the current
    // boundary so server-context uses checkpoints for rollback.
    return kv_raw->seq_pos_max(seq_id);
}

llama_pos llama_kv_cache_dsv4::seq_pos_max(llama_seq_id seq_id) const {
    if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
        return -1;
    }

    return kv_raw->seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache_dsv4::memory_breakdown() const {
    // the box counts once, in the raw cache that holds it
    return kv_raw->memory_breakdown();
}

void llama_kv_cache_dsv4::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    const bool partial_only = flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;

    const uint32_t magic   = DSV4_STATE_MAGIC;
    const uint32_t version = DSV4_STATE_VERSION;
    const uint32_t mode    = partial_only ? DSV4_STATE_MODE_PARTIAL : DSV4_STATE_MODE_FULL;

    io.write(&magic,   sizeof(magic));
    io.write(&version, sizeof(version));
    io.write(&mode,    sizeof(mode));

    LLAMA_LOG_DEBUG("%s: seq %d: raw cells at byte %zu\n", __func__, seq_id, io.n_bytes());
    kv_raw->state_write(io, seq_id, flags);

    LLAMA_LOG_DEBUG("%s: seq %d: compressed rows and state at byte %zu\n", __func__, seq_id, io.n_bytes());
    box_state_write(io, seq_id, partial_only);
}

void llama_kv_cache_dsv4::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    uint32_t magic;
    uint32_t version;
    uint32_t mode = DSV4_STATE_MODE_FULL;

    io.read(&magic,   sizeof(magic));
    io.read(&version, sizeof(version));

    if (magic != DSV4_STATE_MAGIC) {
        throw std::runtime_error("DSV4 state magic mismatch");
    }
    if (version != DSV4_STATE_VERSION) {
        throw std::runtime_error("DSV4 state version mismatch");
    }

    io.read(&mode, sizeof(mode));
    if (mode != DSV4_STATE_MODE_FULL && mode != DSV4_STATE_MODE_PARTIAL) {
        throw std::runtime_error("DSV4 state mode mismatch");
    }

    const bool partial_only = mode == DSV4_STATE_MODE_PARTIAL;
    if (partial_only != !!(flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)) {
        throw std::runtime_error("DSV4 state flags mismatch");
    }

    kv_raw->state_read(io, seq_id, flags);

    box_state_read(io, seq_id, partial_only);
}

llama_kv_cache_iswa * llama_kv_cache_dsv4::get_raw() const {
    return kv_raw.get();
}

llama_kv_cache * llama_kv_cache_dsv4::get_csa() const {
    return kv_csa.get();
}

llama_kv_cache * llama_kv_cache_dsv4::get_hca() const {
    return kv_hca.get();
}

llama_kv_cache * llama_kv_cache_dsv4::get_lid() const {
    return kv_lid.get();
}

llama_dsv4_comp_state * llama_kv_cache_dsv4::get_csa_state() const {
    return csa_state.get();
}

llama_dsv4_comp_state * llama_kv_cache_dsv4::get_hca_state() const {
    return hca_state.get();
}

llama_dsv4_comp_state * llama_kv_cache_dsv4::get_lid_state() const {
    return lid_state.get();
}

uint32_t llama_kv_cache_dsv4::get_n_rs_seq() const {
    return n_rs_seq;
}

const std::vector<uint32_t> & llama_kv_cache_dsv4::get_rs_idx() const {
    return rs_idx;
}

void llama_kv_cache_dsv4::reset_rs_idx_for_ubatches(const std::vector<llama_ubatch> & ubatches) {
    if (n_rs_seq == 0) {
        return;
    }

    for (const llama_ubatch & ubatch : ubatches) {
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            for (int32_t s = 0; s < ubatch.n_seq_id[i]; ++s) {
                const llama_seq_id seq_id = ubatch.seq_id[i][s];
                if (seq_id >= 0 && (uint32_t) seq_id < n_seq_max) {
                    rs_idx[seq_id] = 0;
                }
            }
        }
    }
}

void llama_kv_cache_dsv4::box_state_write(llama_io_write_i & io, llama_seq_id seq_id, bool partial_only) const {
    GGML_ASSERT(seq_id >= 0 && "the box saves single sequences only");

    const int32_t st = seq_states[seq_id];

    io.write(&st, sizeof(st));
    if (st < 0) {
        return;
    }

    // the compressed rows only grow, a partial state leaves them in place
    for (const auto kind : { llama_box::BLOCK_CSA, llama_box::BLOCK_HCA }) {
        if (partial_only) {
            break;
        }

        const uint32_t n = box->comp_blocks(seq_id, kind);

        LLAMA_LOG_DEBUG("%s: seq %d: %u compressed blocks of kind %d at byte %zu\n", __func__, seq_id, n, (int) kind, io.n_bytes());
        io.write(&n, sizeof(n));

        std::vector<uint8_t> data(box->get_block_bytes());
        for (uint32_t i = 0; i < n; ++i) {
            box->comp_get(seq_id, kind, i, data.data());
            io.write(data.data(), data.size());
        }
    }

    std::vector<uint8_t> data(box->get_state_bytes());
    box->state_get(st, data.data());
    LLAMA_LOG_DEBUG("%s: seq %d: state %d of %zu bytes at byte %zu\n", __func__, seq_id, st, data.size(), io.n_bytes());
    io.write(data.data(), data.size());

    const uint32_t rs = rs_idx[seq_id];
    io.write(&rs, sizeof(rs));
}

void llama_kv_cache_dsv4::box_state_read(llama_io_read_i & io, llama_seq_id seq_id, bool partial_only) {
    GGML_ASSERT(seq_id >= 0 && "the box restores single sequences only");

    int32_t st;
    io.read(&st, sizeof(st));
    if (st < 0) {
        return;
    }

    if (!box_state_take(seq_id)) {
        throw std::runtime_error("no room in the box for the state of a DSV4 sequence");
    }

    for (const auto kind : { llama_box::BLOCK_CSA, llama_box::BLOCK_HCA }) {
        if (partial_only) {
            break;
        }

        uint32_t n;
        io.read(&n, sizeof(n));

        // the restored rows go to blocks of the seq alone: a block it shares holds the history of other seqs too
        box->comp_clear(seq_id, kind);

        if (!box->comp_take(seq_id, kind, n)) {
            throw std::runtime_error("no free block in the box for a DSV4 sequence");
        }

        std::vector<uint8_t> data(box->get_block_bytes());
        for (uint32_t i = 0; i < n; ++i) {
            io.read(data.data(), data.size());
            box->comp_set(seq_id, kind, i, data.data());
        }
    }

    std::vector<uint8_t> data(box->get_state_bytes());
    io.read(data.data(), data.size());
    box->state_set(seq_states[seq_id], data.data());

    uint32_t rs;
    io.read(&rs, sizeof(rs));
    rs_idx[seq_id] = rs;
}

void llama_kv_cache_dsv4::clear_compressed(llama_seq_id seq_id) {
    // the blocks went back with the raw cells; a request keeps what its admission took
    for (llama_seq_id s = seq_id >= 0 ? seq_id : 0; s < (seq_id >= 0 ? seq_id + 1 : (llama_seq_id) n_seq_max); ++s) {
        rs_idx[s] = 0;

        if (seq_states[s] >= 0 && !box->seq_admitted(s)) {
            box->state_free(seq_states[s]);
            seq_states[s] = -1;
        }
    }
}

//
// llama_kv_cache_dsv4_raw_context
//

llama_kv_cache_dsv4_raw_context::llama_kv_cache_dsv4_raw_context(llama_kv_cache_iswa * kv) :
    kv_swa(kv->get_swa()),
    ctx_base_mem(nullptr),
    ctx_swa_mem(nullptr),
    n_kv(kv_swa->get_size()),
    status(LLAMA_MEMORY_STATUS_SUCCESS) {
    // a dummy slot info, the graphs reserved at startup only need its shape
    sinfos.resize(1);
    sinfos[0].idxs.resize(1, 0);
}

llama_kv_cache_dsv4_raw_context::llama_kv_cache_dsv4_raw_context(
        llama_kv_cache_iswa * kv,
        llama_context * lctx,
        bool optimize) :
    kv_swa(kv->get_swa()),
    ctx_base_mem(kv->get_base()->init_update(lctx, optimize)),
    ctx_swa_mem(kv->get_swa()->init_update(lctx, optimize)),
    n_kv(kv_swa->get_size()),
    status(llama_memory_status_combine(ctx_base_mem->get_status(), ctx_swa_mem->get_status())) {
}

llama_kv_cache_dsv4_raw_context::llama_kv_cache_dsv4_raw_context(
        llama_kv_cache_iswa * kv,
        slot_info_vec_t sinfos_base,
        slot_info_vec_t sinfos_swa,
        std::vector<llama_ubatch> ubatches) :
    kv_swa(kv->get_swa()),
    sinfos(std::move(sinfos_swa)),
    ubatches(std::move(ubatches)),
    ctx_base_mem(std::make_unique<llama_kv_cache_context>(
                kv->get_base(), std::move(sinfos_base), this->ubatches)),
    ctx_swa_mem(nullptr),
    n_kv(kv_swa->get_size()),
    status(LLAMA_MEMORY_STATUS_SUCCESS) {
}

bool llama_kv_cache_dsv4_raw_context::next() {
    if (ubatches.empty()) {
        return true;
    }

    if (ctx_base_mem) {
        ctx_base_mem->next();
    }

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_dsv4_raw_context::apply() {
    bool res = true;

    if (ctx_base_mem) {
        res = res & ctx_base_mem->apply();
    }
    if (ctx_swa_mem) {
        res = res & ctx_swa_mem->apply();
    }
    if (!ubatches.empty()) {
        kv_swa->apply_ubatch(sinfos[i_next], ubatches[i_next]);
        n_kv = kv_swa->get_n_kv();
    }

    return res;
}

llama_memory_status llama_kv_cache_dsv4_raw_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_dsv4_raw_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

uint32_t llama_kv_cache_dsv4_raw_context::get_n_kv() const {
    return n_kv;
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv_swa->get_k(ctx, il, n_kv);
}

void llama_kv_cache_dsv4_raw_context::seq_rows(llama_seq_id seq_id, bool is_2d, std::vector<int32_t> & rows, std::vector<llama_pos> & pos) const {
    kv_swa->seq_cells_causal(seq_id, is_2d, rows, pos);

    for (auto & r : rows) {
        r = (int32_t) kv_swa->cell_row((uint32_t) r);
    }
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv_swa->cpy_k(ctx, k_cur, k_idxs, il);
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, ubatch.n_tokens);
    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache_dsv4_raw_context::build_input_k_rot(ggml_context * ctx) const {
    return kv_swa->build_input_k_rot(ctx);
}

void llama_kv_cache_dsv4_raw_context::set_input_k_idxs(ggml_tensor * dst) const {
    kv_swa->set_input_k_idxs(dst, &ubatches[i_next], sinfos[i_next]);
}

void llama_kv_cache_dsv4_raw_context::set_input_k_rot(ggml_tensor * dst) const {
    kv_swa->set_input_k_rot(dst);
}

//
// llama_kv_cache_dsv4_comp_context
//

llama_kv_cache_dsv4_comp_context::llama_kv_cache_dsv4_comp_context(llama_kv_cache * kv, llama_box * box, bool lid, uint32_t n_kv) :
    box(box), lid(lid), kv(kv), n_kv(n_kv) {
}

uint32_t llama_kv_cache_dsv4_comp_context::get_n_kv() const {
    return n_kv;
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::get_k(ggml_context * ctx, int32_t il) const {
    ggml_tensor * k = lid ? box->get_lid(il) : box->get_k(il);

    return ggml_view_4d(ctx, k, k->ne[0], 1, k->ne[1], 1, ggml_row_size(k->type, k->ne[0]), k->nb[1], k->nb[1]*k->ne[1], 0);
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    // [n_embd, 1, n_rows] -> [n_embd, n_rows]
    k_cur = ggml_view_2d(ctx, k_cur, k_cur->ne[0]*k_cur->ne[1], k_cur->ne[2], k_cur->nb[2], 0);

    return ggml_set_rows(ctx, lid ? box->get_lid(il) : box->get_k(il), k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache_dsv4_comp_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

void llama_kv_cache_dsv4_comp_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

//
// llama_kv_cache_dsv4_context
//

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(llama_memory_status status) : status(status) {}

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(
        llama_kv_cache_dsv4 * kv) :
    kv_dsv4(kv),
    kv_box(kv->get_box()),
    ctx_raw(std::make_unique<llama_kv_cache_dsv4_raw_context>(kv->get_raw())),
    ctx_csa_mem(kv->get_csa()->init_full()),
    ctx_hca_mem(kv->get_hca()->init_full()),
    ctx_lid_mem(kv->get_lid()->init_full()),
    ctx_csa(comp_context(kv, kv->get_csa(), false, DSV4_CSA_RATIO)),
    ctx_hca(comp_context(kv, kv->get_hca(), false, DSV4_HCA_RATIO)),
    ctx_lid(comp_context(kv, kv->get_lid(), true,  DSV4_CSA_RATIO)),
    csa_state(kv->get_csa_state()),
    hca_state(kv->get_hca_state()),
    lid_state(kv->get_lid_state()),
    reserve_plans(true),
    status(llama_memory_status_combine(
                llama_memory_status_combine(ctx_raw->get_status(), ctx_csa_mem->get_status()),
                llama_memory_status_combine(ctx_hca_mem->get_status(), ctx_lid_mem->get_status()))) {
}

// the compressed rows are in the box, the cache only holds the rotation matrix
std::unique_ptr<llama_kv_cache_dsv4_comp_context> llama_kv_cache_dsv4_context::comp_context(
        llama_kv_cache_dsv4 * kv, llama_kv_cache * kv_comp, bool lid, uint32_t ratio) {
    return std::make_unique<llama_kv_cache_dsv4_comp_context>(kv_comp, kv->get_box(), lid, kv->get_n_cells_max()/ratio);
}

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(
        llama_kv_cache_dsv4 * kv,
        llama_context * lctx,
        bool optimize) :
    ctx_raw(std::make_unique<llama_kv_cache_dsv4_raw_context>(kv->get_raw(), lctx, optimize)),
    ctx_csa_mem(kv->get_csa()->init_update(lctx, optimize)),
    ctx_hca_mem(kv->get_hca()->init_update(lctx, optimize)),
    ctx_lid_mem(kv->get_lid()->init_update(lctx, optimize)),
    csa_state(kv->get_csa_state()),
    hca_state(kv->get_hca_state()),
    lid_state(kv->get_lid_state()),
    status(llama_memory_status_combine(
                llama_memory_status_combine(ctx_raw->get_status(), ctx_csa_mem->get_status()),
                llama_memory_status_combine(ctx_hca_mem->get_status(), ctx_lid_mem->get_status()))) {
}

llama_kv_cache_dsv4_context::llama_kv_cache_dsv4_context(
        llama_kv_cache_dsv4 * kv,
        slot_info_vec_t sinfos_raw_base,
        slot_info_vec_t sinfos_raw_swa,
        std::vector<llama_ubatch> ubatches) :
    kv_dsv4(kv),
    kv_box(kv->get_box()),
    ubatches(std::move(ubatches)),
    plans_csa(dsv4_build_comp_plans(this->ubatches, DSV4_CSA_RATIO, true,
                kv->get_csa_state()->get_state_size(), kv->get_n_rs_seq(), kv->get_rs_idx(), kv, kv->get_csa_state(), false)),
    plans_hca(dsv4_build_comp_plans(this->ubatches, DSV4_HCA_RATIO, false,
                kv->get_hca_state()->get_state_size(), kv->get_n_rs_seq(), kv->get_rs_idx(), kv, kv->get_hca_state(), false)),
    plans_lid(dsv4_build_comp_plans(this->ubatches, DSV4_CSA_RATIO, true,
                kv->get_lid_state()->get_state_size(), kv->get_n_rs_seq(), kv->get_rs_idx(), kv, kv->get_lid_state(), true)),
    ctx_raw(std::make_unique<llama_kv_cache_dsv4_raw_context>(
                kv->get_raw(),
                std::move(sinfos_raw_base),
                std::move(sinfos_raw_swa),
                this->ubatches)),
    ctx_csa_mem(nullptr),
    ctx_hca_mem(nullptr),
    ctx_lid_mem(nullptr),
    ctx_csa(comp_context(kv, kv->get_csa(), false, DSV4_CSA_RATIO)),
    ctx_hca(comp_context(kv, kv->get_hca(), false, DSV4_HCA_RATIO)),
    ctx_lid(comp_context(kv, kv->get_lid(), true,  DSV4_CSA_RATIO)),
    csa_state(kv->get_csa_state()),
    hca_state(kv->get_hca_state()),
    lid_state(kv->get_lid_state()),
    status(ctx_raw->get_status()) {
    kv->reset_rs_idx_for_ubatches(this->ubatches);

    block_copies = kv->take_block_copies();
}

const std::vector<std::pair<uint32_t, uint32_t>> & llama_kv_cache_dsv4_context::get_block_copies() const {
    static const std::vector<std::pair<uint32_t, uint32_t>> none;

    return i_next == 0 ? block_copies : none;
}

llama_kv_cache_dsv4_context::~llama_kv_cache_dsv4_context() = default;

bool llama_kv_cache_dsv4_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    ctx_raw->next();

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_dsv4_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    bool res = true;

    res = res & ctx_raw->apply();

    if (ctx_csa_mem) {
        res = res & ctx_csa_mem->apply();
        res = res & ctx_hca_mem->apply();
        res = res & ctx_lid_mem->apply();
    }

    return res;
}

llama_memory_status llama_kv_cache_dsv4_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_dsv4_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

const llama_kv_cache_dsv4_raw_context * llama_kv_cache_dsv4_context::get_raw() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_raw.get();
}

const llama_kv_cache_dsv4_comp_context * llama_kv_cache_dsv4_context::get_csa() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_csa.get();
}

const llama_kv_cache_dsv4_comp_context * llama_kv_cache_dsv4_context::get_hca() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_hca.get();
}

const llama_kv_cache_dsv4_comp_context * llama_kv_cache_dsv4_context::get_lid() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ctx_lid.get();
}

const llama_dsv4_comp_state * llama_kv_cache_dsv4_context::get_csa_state() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return csa_state;
}

const llama_dsv4_comp_state * llama_kv_cache_dsv4_context::get_hca_state() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return hca_state;
}

const llama_dsv4_comp_state * llama_kv_cache_dsv4_context::get_lid_state() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return lid_state;
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_csa_plan() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    static const comp_plan empty;
    if (plans_csa.empty()) {
        return empty;
    }

    return plans_csa[i_next];
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_hca_plan() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    static const comp_plan empty;
    if (plans_hca.empty()) {
        return empty;
    }

    return plans_hca[i_next];
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_lid_plan() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    static const comp_plan empty;
    if (plans_lid.empty()) {
        return empty;
    }

    return plans_lid[i_next];
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_csa_plan(const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (!reserve_plans) {
        return get_csa_plan();
    }

    reserve_plan_csa = dsv4_build_reserve_comp_plan(
            ubatch, DSV4_CSA_RATIO, true,
            csa_state->get_state_size(), get_csa()->get_n_kv(), csa_state->get_n_rs_seq(), false);

    return reserve_plan_csa;
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_hca_plan(const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (!reserve_plans) {
        return get_hca_plan();
    }

    reserve_plan_hca = dsv4_build_reserve_comp_plan(
            ubatch, DSV4_HCA_RATIO, false,
            hca_state->get_state_size(), get_hca()->get_n_kv(), hca_state->get_n_rs_seq(), false);

    return reserve_plan_hca;
}

const llama_kv_cache_dsv4_context::comp_plan & llama_kv_cache_dsv4_context::get_lid_plan(const llama_ubatch & ubatch) const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (!reserve_plans) {
        return get_lid_plan();
    }

    reserve_plan_lid = dsv4_build_reserve_comp_plan(
            ubatch, DSV4_CSA_RATIO, true,
            lid_state->get_state_size(), get_lid()->get_n_kv(), lid_state->get_n_rs_seq(), true);

    return reserve_plan_lid;
}

uint32_t llama_kv_cache_dsv4_context::get_n_raw() const {
    return kv_dsv4->get_n_swa();
}

int64_t llama_kv_cache_dsv4_context::get_n_hca(const llama_ubatch & ubatch) const {
    if (reserve_plans) {
        return kv_dsv4->get_n_cells_max()/DSV4_HCA_RATIO;
    }

    int64_t n = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        n = std::max<int64_t>(n, (ubatch.pos[i] + 1)/DSV4_HCA_RATIO);
    }

    return std::max<int64_t>(GGML_PAD(n, 256u), 256);
}

int64_t llama_kv_cache_dsv4_context::get_n_win_lid(const llama_ubatch & ubatch) const {
    // a request holds at most the context window of the model
    const int64_t n_max = std::max<int64_t>(256, GGML_PAD((int64_t) kv_dsv4->get_n_cells_max()/DSV4_CSA_RATIO + 1, 256));

    if (reserve_plans) {
        return n_max;
    }

    const auto & plan = get_lid_plan(ubatch);

    int64_t n = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        n = std::max<int64_t>(n, plan.n_visible[i] - plan.n_skip[i]);
    }

    // the window of a token covers what it sees of its sequence
    GGML_ASSERT(n <= n_max);

    return std::min(n_max, std::max<int64_t>(256, GGML_PAD(n, 256)));
}

void llama_kv_cache_dsv4_context::set_input_box(ggml_tensor * raw, ggml_tensor * hca, ggml_tensor * lid, ggml_tensor * rng_raw, ggml_tensor * rng_hca, ggml_tensor * rng_csa,
        ggml_tensor * win_lid, uint32_t n_top, const llama_ubatch * ubatch) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(raw->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(hca->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(lid->buffer));

    const int64_t n_raw = raw->ne[0];
    const int64_t n_hca = hca->ne[0];
    const int64_t n_lid = lid->ne[0];

    const uint32_t n_swa = kv_dsv4->get_n_swa();

    auto * d_raw = (int32_t *) raw->data;
    auto * d_hca = (int32_t *) hca->data;
    auto * d_lid = (int32_t *) lid->data;
    auto * r_raw = (int32_t *) rng_raw->data;
    auto * r_hca = (int32_t *) rng_hca->data;
    auto * r_csa = (int32_t *) rng_csa->data;

    std::fill(d_raw, d_raw + n_raw*ubatch->n_tokens, 0);
    std::fill(d_hca, d_hca + n_hca*ubatch->n_tokens, 0);
    std::fill(d_lid, d_lid + n_lid, 0);

    std::map<llama_seq_id, std::pair<std::vector<int32_t>, std::vector<llama_pos>>> seqs;

    for (uint32_t s = 0; s < ubatch->n_seqs_unq; ++s) {
        auto & seq = seqs[ubatch->seq_id_unq[s]];
        ctx_raw->seq_rows(ubatch->seq_id_unq[s], ubatch->is_pos_2d(), seq.first, seq.second);
    }

    const auto & plan_lid = get_lid_plan(*ubatch);
    const auto & plan_csa = get_csa_plan(*ubatch);

    // the indexer list, each sequence as far as its last token of the ubatch sees
    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        for (int32_t k = plan_lid.n_skip[i]; k < plan_lid.n_visible[i]; ++k) {
            GGML_ASSERT(k < n_lid);
            d_lid[k] = (int32_t) kv_box->comp_row(seq_id, llama_box::BLOCK_CSA, (uint32_t) (k - plan_lid.n_skip[i]));
        }
    }

    for (uint32_t i = 0; i < ubatch->n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];
        const llama_pos    p      = ubatch->pos[i];

        const auto & rows = seqs.at(seq_id).first;
        const auto & pos  = seqs.at(seq_id).second;

        const int64_t hi = std::upper_bound(pos.begin(), pos.end(), p) - pos.begin();
        const int64_t lo = std::lower_bound(pos.begin(), pos.end(), p - (llama_pos) n_swa + 1) - pos.begin();

        const int64_t n_read = hi - lo;

        GGML_ASSERT(n_read <= n_raw);

        // at the end of the slots of the token, its compressed rows follow
        for (int64_t k = 0; k < n_read; ++k) {
            d_raw[i*n_raw + n_raw - n_read + k] = rows[lo + k];
        }

        const int32_t n_hca_i = (int32_t) ((p + 1)/DSV4_HCA_RATIO);

        GGML_ASSERT(n_hca_i <= n_hca);

        for (int32_t k = 0; k < n_hca_i; ++k) {
            d_hca[i*n_hca + k] = (int32_t) kv_box->comp_row(seq_id, llama_box::BLOCK_HCA, (uint32_t) k);
        }

        const int32_t n_csa_i = plan_csa.n_visible[i];

        // the indexer scores the CSA rows of the sequence the token sees: one range of lid
        if (win_lid) {
            auto * w_lid = (int32_t *) win_lid->data;

            w_lid[3*i + 0] = plan_lid.n_skip[i];
            w_lid[3*i + 1] = plan_lid.n_skip[i];
            w_lid[3*i + 2] = plan_lid.n_visible[i];
        }

        r_raw[3*i + 0] = (int32_t) i;
        r_raw[3*i + 1] = (int32_t) (n_raw - n_read);
        r_raw[3*i + 2] = (int32_t) n_raw;

        r_hca[3*i + 0] = (int32_t) i;
        r_hca[3*i + 1] = (int32_t) (n_raw - n_read);
        r_hca[3*i + 2] = (int32_t) n_raw + n_hca_i;

        r_csa[3*i + 0] = (int32_t) i;
        r_csa[3*i + 1] = (int32_t) (n_raw - n_read);
        r_csa[3*i + 2] = (int32_t) n_raw + std::min<int32_t>(n_csa_i, (int32_t) std::min<int64_t>(n_top, n_lid));
    }
}
