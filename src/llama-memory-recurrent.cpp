#include "llama-memory-recurrent.h"

#include "ggml-backend.h"
#include "llama-box.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-batch.h"
#include "llama-model.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>

//
// llama_memory_recurrent
//

llama_memory_recurrent::llama_memory_recurrent(
        const llama_model & model,
                 uint32_t   n_rs_seq,
                     bool   rs_replay_req,
    const layer_filter_cb & filter,
                llama_box * box_in,
    std::shared_ptr<llama_box> box_own_in) :
    box(box_own_in ? box_own_in.get() : box_in), box_own(std::move(box_own_in)), hparams(model.hparams) {
    // the sequences are what fits in the elastic memory, a cell of the box is a state id, see llama_box
    n_seq_max = box->get_n_seq_max();

    const uint32_t mem_size = box->get_max_states();
    const int32_t  n_layer  = hparams.n_layer();

    head = 0;
    size = mem_size;
    used = 0;

    this->n_rs_seq = n_rs_seq;
    rs_idx.assign(n_seq_max, 0);

    cells.clear();
    cells.resize(mem_size);

    seq_tails.assign(n_seq_max, -1);

    box_cow   .assign(n_seq_max, -1);
    box_x_cell.assign(n_seq_max, -1);
    box_fresh .assign(mem_size, 0);

    // an admitted request keeps the state its admission took, see llama_box::reserve()
    if (box_own) {
        box_keep = [this](llama_seq_id seq_id) {
            return this->box->seq_admitted(seq_id);
        };
    }

    // replaying the cached inputs of the last tokens instead of keeping a state snapshot per draft position needs
    // a model whose recurrent layers are gated delta-net layers, see llama_context_params::rs_rollback
    rs_replay = rs_replay_req && n_rs_seq > 0 && llm_arch_supports_rs_replay(model.arch) && hparams.n_embd_gdn_inp() > 0;

    if (rs_replay) {
        LLAMA_LOG_INFO("%s: the scan state rolls back by replaying cached inputs\n", __func__);
    }

    if (rs_replay) {
        x_cur .assign(n_seq_max, 0);
        n_pend.assign(n_seq_max, 0);
        n_rep .assign(n_seq_max, 0);
    }

    n_snap.assign(n_seq_max, 0);

    r_l.resize(n_layer);
    s_l.resize(n_layer);
    p_l.resize(n_layer);
    x_l.resize(n_layer);

    // views of the whole box, read and written through the rows of the states, see llama_box: the conv state keeps one
    // row per draft position; the scan state keeps a single row, which lags behind the cached tokens of the last
    // speculative step
    for (int i = 0; i < n_layer; i++) {
        if (filter && !filter(i)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: skipped\n", __func__, i);
            continue;
        }

        GGML_ASSERT(box->recr_index(i) >= 0);
        r_l[i] = box->get_r();
        s_l[i] = box->get_s();
        x_l[i] = rs_replay ? box->get_x() : nullptr;
        p_l[i] = box->has_c(box->recr_index(i)) ? box->get_c() : nullptr;
        GGML_ASSERT(r_l[i] && s_l[i] && (!rs_replay || x_l[i]));
    }

    LLAMA_LOG_INFO("%s: %6u states, %3d layers, %2u seqs, %2u rs_seq\n", __func__, mem_size, n_layer, n_seq_max, n_rs_seq);
}

void llama_memory_recurrent::clear(bool data) {
    GGML_UNUSED(data);

    for (int32_t i = 0; i < (int32_t) size; ++i) {
        if (!cells[i].seq_id.empty()) {
            box->state_free(i);
        }
        cells[i].pos = -1;
        cells[i].seq_id.clear();
        cells[i].src = -1;
    }

    std::fill(seq_tails.begin(), seq_tails.end(), -1);

    head = 0;
    used = 0;

    if (box_own) {
        box_own->clear();
    }

    std::fill(rs_idx.begin(), rs_idx.end(), 0);

    std::fill(box_cow   .begin(), box_cow   .end(), -1);
    std::fill(box_x_cell.begin(), box_x_cell.end(), -1);
    std::fill(box_fresh .begin(), box_fresh .end(), 0);
    box_free_next.clear();

    if (rs_replay) {
        std::fill(x_cur .begin(), x_cur .end(), 0);
        std::fill(n_pend.begin(), n_pend.end(), 0);
        std::fill(n_rep .begin(), n_rep .end(), 0);
    }

    std::fill(n_snap.begin(), n_snap.end(), 0);
}

bool llama_memory_recurrent::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    uint32_t new_head = size;

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // a negative seq_id means any seq, see the handling below
    if (seq_id >= 0 && (uint32_t) seq_id >= this->n_seq_max) {
        LLAMA_LOG_ERROR("%s: invalid seq_id (%d) - larger than n_seq_max (%d)\n", __func__, seq_id, this->n_seq_max);
        return false;
    }

    const bool rm_all = p0 == 0 && p1 == std::numeric_limits<llama_pos>::max();
    if (rm_all) {
        set_rs_idx(seq_id, 0);

        // pending tokens of the previous use of this seq would be replayed into the next one
        if (rs_replay) {
            if (seq_id < 0) {
                std::fill(n_pend.begin(), n_pend.end(), 0);
            } else if ((size_t) seq_id < n_pend.size()) {
                n_pend[seq_id] = 0;
            }
        }

        if (seq_id < 0) {
            std::fill(n_snap.begin(), n_snap.end(), 0);
        } else {
            n_snap[seq_id] = 0;
        }
    }

    // models like Mamba or RWKV can't have a state partially erased at the end
    // of the sequence because their state isn't preserved for previous tokens
    if (seq_id >= (int64_t) n_seq_max) {
        // could be fatal
        return false;
    }

    // an admitted request keeps the state its admission took (box_keep): removing all of its tokens makes it a new state
    // again, which its next step reads as zeros
    if (rm_all && seq_id >= 0 && seq_tails[seq_id] >= 0 && box_keep && box_keep(seq_id) &&
            cells[seq_tails[seq_id]].seq_id.size() == 1) {
        const int32_t tail = seq_tails[seq_id];
        if (cells[tail].pos >= 0) {
            used--;
        }
        cells[tail].pos  = -1;
        box_fresh[tail]  = 1;
        box_cow[seq_id]  = -1;
        box_x_cell[seq_id] = -1;
        return true;
    }

    // a state of the box taken before the seq had any token (pos -1) is outside every range of positions
    if (rm_all && seq_id >= 0 && seq_tails[seq_id] >= 0 && cells[seq_tails[seq_id]].pos < 0) {
        auto & cell = cells[seq_tails[seq_id]];
        cell.seq_id.erase(seq_id);
        if (cell.is_empty()) {
            box_cell_freed(seq_tails[seq_id]);
        }
        seq_tails[seq_id] = -1;
        box_cow[seq_id]   = -1;
    }
    if (0 <= seq_id) {
        int32_t & tail_id = seq_tails[seq_id];
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];

            // partial rollback via per-token snapshot index (bounded by n_rs_seq)
            if (0 < p0 && p0 <= cell.pos && p1 > cell.pos) {
                const llama_pos rollback = cell.pos - (p0 - 1);

                // with rs_replay the scan state does not contain the pending tokens yet, so removing some of them
                // only means replaying fewer; the conv state still rolls back to one of its snapshots, which is
                // single-use like in the snapshot mode
                if (rs_replay) {
                    const bool pending = rs_idx[seq_id] != 0;

                    if (!pending && rollback >= 1 && rollback <= (llama_pos) std::min(n_pend[seq_id], n_rs_seq)) {
                        n_pend[seq_id] -= (uint32_t) rollback;

                        set_rs_idx(seq_id, (uint32_t) rollback);

                        LLAMA_LOG_DEBUG("%s: RS rollback seq %d cell %d: pos %d -> %d, pending %u, x_cur %d\n",
                                __func__, seq_id, tail_id, cell.pos, p0 - 1, n_pend[seq_id], (int) x_cur[seq_id]);

                        cell.pos = p0 - 1;
                        return true;
                    }
                    return false;
                }

                // pending rollback is single-use; snapshot d is the state before the last d tokens of the last step,
                // which wrote n_snap of them: the one before the whole step is not kept
                const bool pending = rs_idx[seq_id] != 0;
                if (!pending && rollback >= 1 && rollback < (llama_pos) n_snap[seq_id]) {
                    set_rs_idx(seq_id, (uint32_t) rollback);
                    cell.pos = p0 - 1;
                    return true;
                }
                return false;
            }
            // invalidate tails which will be cleared
            if (p0 <= cell.pos && cell.pos < p1) {
                tail_id = -1;
            }
        }
    } else {
        // seq_id is negative, then the range should include everything or nothing
        if (p0 != p1 && (p0 != 0 || p1 != std::numeric_limits<llama_pos>::max())) {
            //printf("[DEBUG] inside `llama_memory_recurrent::seq_rm`: `seq_id` is negative, so returning false\n");
            return false;
        }

        // every cell is cleared below, so no seq keeps a tail
        if (p0 != p1) {
            std::fill(seq_tails.begin(), seq_tails.end(), -1);
        }
    }

    for (uint32_t i = 0; i < size; ++i) {
        // an empty cell with a position is a shared state the last batch read: box_prepare() frees it (box_free_next)
        if (cells[i].is_empty()) {
            continue;
        }
        if (cells[i].pos >= p0 && cells[i].pos < p1) {
            if (seq_id < 0) {
                cells[i].seq_id.clear();
            } else if (cells[i].has_seq_id(seq_id)) {
                cells[i].seq_id.erase(seq_id);
            } else {
                continue;
            }
            if (cells[i].is_empty()) {
                // keep count of the number of used cells
                if (cells[i].pos >= 0) {
                    used--;
                }
                cells[i].pos = -1;
                cells[i].src = -1;
                box_cell_freed(i);
                if (new_head == size) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }

    return true;
}

void llama_memory_recurrent::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    if (seq_id_src == seq_id_dst) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    if ((uint32_t) seq_id_dst < n_seq_max && (uint32_t) seq_id_src < n_seq_max) {
        int32_t & tail_src = seq_tails[seq_id_src];
        int32_t & tail_dst = seq_tails[seq_id_dst];
        if (tail_dst >= 0) {
            // clear destination seq_id if it wasn't empty
            auto & cell_dst = cells[tail_dst];

            cell_dst.seq_id.erase(seq_id_dst);
            if (cell_dst.seq_id.empty()) {
                if (cell_dst.pos >= 0) {
                    used -= 1;
                }
                cell_dst.pos = -1;
                cell_dst.src = -1;
                box_cell_freed(tail_dst);
            }
            tail_dst = -1;
        }
        if (tail_src >= 0) {
            auto & cell_src = cells[tail_src];

            cell_src.seq_id.insert(seq_id_dst);
            tail_dst = tail_src;
        }

        // a rollback of src that is still pending applies to the shared cell, which is read with the bookkeeping of
        // either seq, so dst takes it over; with rs_replay, so do the pending tokens the state lags behind
        if (n_rs_seq > 0) {
            set_rs_idx(seq_id_dst, tail_src >= 0 ? rs_idx[seq_id_src] : 0);
        }

        // dst reads the state of src, and its cached replay inputs, until its next step gives it a state of its own
        // (box_prepare())
        box_cow[seq_id_dst] = tail_src >= 0 ? box_cow[seq_id_src] : -1;
        box_x_cell[seq_id_dst] = tail_src >= 0 ? box_x_cell[seq_id_src] : -1;
        if (rs_replay) {
            x_cur [seq_id_dst] = x_cur[seq_id_src];
            n_pend[seq_id_dst] = tail_src >= 0 ? n_pend[seq_id_src] : 0;
        }
        n_snap[seq_id_dst] = tail_src >= 0 ? n_snap[seq_id_src] : 0;
    }
}

void llama_memory_recurrent::seq_keep(llama_seq_id seq_id) {
    uint32_t new_head = size;

    for (uint32_t i = 0; i < n_seq_max; ++i) {
        if ((llama_seq_id) i != seq_id) {
            seq_tails[i] = -1;
            box_cow[i]   = -1;
            n_snap[i]    = 0;

            if (rs_replay) {
                n_pend[i] = 0;
            }
        }
    }

    for (uint32_t i = 0; i < size; ++i) {
        if (!cells[i].has_seq_id(seq_id)) {
            if (!cells[i].seq_id.empty()) {
                box_cell_freed(i);
            }
            if (cells[i].pos >= 0) {
                used--;
            }

            cells[i].pos = -1;
            cells[i].src = -1;
            cells[i].seq_id.clear();

            if (new_head == size){
                new_head = i;
            }
        } else {
            cells[i].seq_id.clear();
            cells[i].seq_id.insert(seq_id);
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != size && new_head < head) {
        head = new_head;
    }
}

void llama_memory_recurrent::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    if (shift == 0) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be shifted
    if (0 <= seq_id && seq_id < (int64_t) n_seq_max) {
        const int32_t tail_id = seq_tails[seq_id];
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos += shift;
            }
        }
    }
}

void llama_memory_recurrent::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    if (d == 1) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over the cache.
    if (p0 == p1) {
        return;
    }

    // for Mamba-like or RWKV models, only the pos needs to be changed
    if (0 <= seq_id && seq_id < (int64_t) n_seq_max) {
        const int32_t tail_id = seq_tails[seq_id];
        if (tail_id >= 0) {
            auto & cell = cells[tail_id];
            if (cell.has_seq_id(seq_id) && p0 <= cell.pos && cell.pos < p1) {
                cell.pos /= d;
            }
        }
    }
}

llama_pos llama_memory_recurrent::seq_pos_min(llama_seq_id seq_id) const {
    llama_pos result = std::numeric_limits<llama_pos>::max();

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::min(result, cells[i].pos);
        }
    }

    if (result == std::numeric_limits<llama_pos>::max()) {
        result = -1;
    }

    return result;
}

llama_pos llama_memory_recurrent::seq_pos_max(llama_seq_id seq_id) const {
    llama_pos result = -1;

    for (uint32_t i = 0; i < size; ++i) {
        if (cells[i].has_seq_id(seq_id)) {
            result = std::max(result, cells[i].pos);
        }
    }

    return result;
}

uint32_t llama_memory_recurrent::n_free_cells() const {
    return (uint32_t) std::count_if(cells.begin(), cells.end(), [](const mem_cell & cell) { return cell.is_empty(); });
}

void llama_memory_recurrent::set_rs_idx(llama_seq_id seq_id, uint32_t idx) {
    if (seq_id < 0) {
        std::fill(rs_idx.begin(), rs_idx.end(), 0);
        return;
    }

    assert(n_seq_max == rs_idx.size());

    GGML_ASSERT((uint32_t) seq_id < n_seq_max);
    GGML_ASSERT(idx <= n_rs_seq);

    rs_idx[seq_id] = idx;
}

std::map<ggml_backend_buffer_type_t, size_t> llama_memory_recurrent::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    if (box_own) {
        ret[ggml_backend_buffer_get_type(box_own->get_buffer())] += box_own->get_total_bytes();
    }
    return ret;
}

bool llama_memory_recurrent::box_reserve(llama_seq_id seq_id, uint32_t n_input, uint32_t n_output) {
    GGML_ASSERT(box_own);

    std::function<int32_t()> state_alloc = nullptr;
    if (seq_tails[seq_id] < 0) {
        state_alloc = [&]() -> int32_t {
            return box_state_take(seq_id) ? seq_tails[seq_id] : -1;
        };
    }

    return box_own->reserve(seq_id, n_input, n_output, state_alloc);
}

void llama_memory_recurrent::box_reserve_drop(llama_seq_id seq_id) {
    GGML_ASSERT(box_own);

    box_own->reserve_drop(seq_id);
}

llama_memory_context_ptr llama_memory_recurrent::init_batch(llama_batch_allocr & balloc, uint32_t n_ubatch, bool embd_all) {
    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            llama_ubatch ubatch;

            if (embd_all) {
                // if all tokens are output, split by sequence
                ubatch = balloc.split_seq(n_ubatch);
            } else {
                // TODO: non-sequential equal split can be done if using unified KV cache
                //       for simplicity, we always use sequential equal split for now
                // [TAG_RECURRENT_ROLLBACK_SPLITS]
                // the trailing (1 + n_rs_seq) tokens of each seq must stay in the same ubatch
                //   so that the rollback snapshots remain valid
                ubatch = balloc.split_equal(n_ubatch, true, n_rs_seq > 0 ? n_rs_seq + 1 : 0);
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

        // the box first gives the seqs their states
        if (box_own && !box_prepare(ubatches)) {
            break;
        }

        if (!prepare(ubatches)) {
            break;
        }

        return std::make_unique<llama_memory_recurrent_context>(this, std::move(ubatches));
    } while (false);

    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_memory_recurrent::init_full() {
    return std::make_unique<llama_memory_recurrent_context>(this);
}

llama_memory_context_ptr llama_memory_recurrent::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(lctx);
    GGML_UNUSED(optimize);

    // the states stay in their cells of the box: nothing to update
    return std::make_unique<llama_memory_recurrent_context>(LLAMA_MEMORY_STATUS_NO_UPDATE);
}

bool llama_memory_recurrent::prepare(const std::vector<llama_ubatch> & ubatches) {
    // simply remember the full state because it is very small for this type of cache
    // TODO: optimize
    // seq_tails must be restored together with cells, otherwise tails point at cells that no longer hold their seq
    auto org_cells = cells;
    auto org_seq_tails = seq_tails;
    auto org_used = used;
    auto org_head = head;

    // the replay bookkeeping advances in find_slot() too, so this dry run must not leave it advanced
    auto org_x_cur    = x_cur;
    auto org_n_pend   = n_pend;
    auto org_n_rep    = n_rep;
    auto org_n_snap   = n_snap;

    auto org_box_cow    = box_cow;
    auto org_box_x_cell = box_x_cell;
    auto org_box_fresh  = box_fresh;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        if (!find_slot(ubatch)) {
            success = false;
            break;
        }
    }

    // restore the original state
    cells = std::move(org_cells);
    seq_tails = std::move(org_seq_tails);
    used = org_used;
    head = org_head;

    x_cur    = std::move(org_x_cur);
    n_pend   = std::move(org_n_pend);
    n_rep    = std::move(org_n_rep);
    n_snap   = std::move(org_n_snap);

    box_cow    = std::move(org_box_cow);
    box_x_cell = std::move(org_box_x_cell);
    box_fresh  = std::move(org_box_fresh);

    return success;
}

void llama_memory_recurrent::box_cell_freed(uint32_t i) {
    GGML_ASSERT(cells[i].is_empty());

    box->state_free(i);

    cells[i].pos = -1;
    cells[i].src = -1;
}

bool llama_memory_recurrent::box_state_take(llama_seq_id seq_id) {
    GGML_ASSERT(seq_tails[seq_id] < 0);

    const int32_t p = box->state_alloc();
    if (p < 0) {
        return false;
    }

    // only offsets change hands: the state starts at zero in the computation, its first step reads zeros instead of it
    // (see box_fresh) and writes it
    box_fresh[p] = 1;

    auto & cell = cells[p];
    cell.pos = -1;
    cell.src = p;
    cell.seq_id.clear();
    cell.seq_id.insert(seq_id);

    seq_tails[seq_id]  = p;
    box_cow[seq_id]    = -1;
    box_x_cell[seq_id] = -1;
    n_snap[seq_id]     = 0;

    if (rs_replay) {
        x_cur [seq_id] = 0;
        n_pend[seq_id] = 0;
    }

    return true;
}

bool llama_memory_recurrent::box_prepare(const std::vector<llama_ubatch> & ubatches) {
    // the shared states left by the previous batch, which its steps read: they are free from now on
    for (const int32_t c : box_free_next) {
        if (cells[c].is_empty()) {
            box_cell_freed(c);
        }
    }
    box_free_next.clear();

    // the states shared at the start of the batch: a step reads them in place, so none of its seqs writes them; each seq
    // of the batch that holds one takes a state of its own, read from the shared one (box_cow)
    std::vector<uint8_t> shared(size, 0);
    for (const auto & ubatch : ubatches) {
        for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
            const llama_seq_id lead = ubatch.seq_id[s*ubatch.n_seq_tokens][0];
            if (lead >= 0 && (uint32_t) lead < n_seq_max && seq_tails[lead] >= 0 && cells[seq_tails[lead]].seq_id.size() > 1) {
                shared[seq_tails[lead]] = 1;
            }
        }
    }

    for (const auto & ubatch : ubatches) {
        for (uint32_t s = 0; s < ubatch.n_seqs; ++s) {
            const uint32_t i = s*ubatch.n_seq_tokens;

            const llama_seq_id lead = ubatch.seq_id[i][0];
            if (lead < 0 || (uint32_t) lead >= n_seq_max) {
                LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%u Try using a bigger --parallel value\n", __func__, lead, n_seq_max);
                return false;
            }

            int32_t & tail = seq_tails[lead];

            if (tail < 0) {
                // a new seq: a state of its own, which starts at zero
                if (!box_state_take(lead)) {
                    LLAMA_LOG_WARN("%s: no state fits in the box for seq %d\n", __func__, lead);
                    return false;
                }
            } else if (shared[tail]) {
                // a shared state: a state of its own, which the next step writes after reading the shared one
                const int32_t p = box->state_alloc();
                if (p < 0) {
                    LLAMA_LOG_WARN("%s: no state fits in the box for seq %d\n", __func__, lead);
                    return false;
                }

                if (box_cow[lead] < 0) {
                    box_cow[lead] = tail;
                }

                box_fresh[p] = 0;

                cells[p].pos = cells[tail].pos;
                cells[p].src = p;
                cells[p].seq_id.clear();
                cells[p].seq_id.insert(lead);

                cells[tail].seq_id.erase(lead);
                if (cells[tail].is_empty()) {
                    // read by this batch: free after it, see box_free_next
                    box_free_next.push_back(tail);
                    used--;
                }
                tail = p;
                used++;
            }

            // the other seqs of the tokens take the state of the lead: their own states go back to the box
            for (int32_t j = 1; j < ubatch.n_seq_id[i]; ++j) {
                const llama_seq_id seq_id = ubatch.seq_id[i][j];
                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: seq_id=%d >= n_seq_max=%u Try using a bigger --parallel value\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                int32_t & tail_j = seq_tails[seq_id];
                if (tail_j == tail) {
                    continue;
                }
                if (tail_j >= 0) {
                    auto & cell = cells[tail_j];
                    cell.seq_id.erase(seq_id);
                    if (cell.is_empty()) {
                        if (cell.pos >= 0) {
                            used--;
                        }
                        box_cell_freed(tail_j);
                    }
                }

                cells[tail].seq_id.insert(seq_id);
                tail_j = tail;
                box_cow[seq_id] = -1;
            }
        }
    }

    return true;
}

// the states stay in their cells of the box, the step reads each seq from its cell (or the state it shares, box_cow, or
// zeros for a new state) and writes it to its cell, see llama_memory_recurrent_context::box_row()
bool llama_memory_recurrent::find_slot(const llama_ubatch & ubatch) {
    const uint32_t n_seq_tokens = ubatch.n_seq_tokens;
    const uint32_t n_seqs       = ubatch.n_seqs;

    GGML_ASSERT(ubatch.equal_seqs());

    box_lead  .resize(n_seqs);
    box_src   .resize(n_seqs);
    box_dst   .resize(n_seqs);
    box_x_src .resize(n_seqs);
    box_x_half.resize(n_seqs);

    for (uint32_t s = 0; s < n_seqs; ++s) {
        const uint32_t i = s*n_seq_tokens;

        const llama_seq_id lead = ubatch.seq_id[i][0];
        if (lead < 0 || (uint32_t) lead >= n_seq_max || seq_tails[lead] < 0) {
            // box_prepare() gives every seq a state before the cells are searched
            LLAMA_LOG_ERROR("%s: seq %d has no state in the box\n", __func__, lead);
            return false;
        }

        const int32_t cell_id = seq_tails[lead];
        auto & cell = cells[cell_id];

        const int32_t src = box_cow[lead] >= 0 ? box_cow[lead] : cell_id;

        box_lead[s] = lead;
        box_src [s] = box_fresh[src] ? -1 : src; // a new state reads zeros
        box_dst [s] = cell_id;

        const llama_pos last_pos = ubatch.pos[i + n_seq_tokens - 1];

        if (cell.pos >= 0 && last_pos != cell.pos + (llama_pos) n_seq_tokens) {
            LLAMA_LOG_WARN("%s: non-consecutive token position %d after %d for sequence %d with %u new tokens\n",
                __func__, last_pos, cell.pos, lead, n_seq_tokens);
        }
        if (cell.pos < 0) {
            used++;
        }
        cell.pos = last_pos;

        // the step writes the state of the seq into its cell, so it reads its own cell from then on
        box_cow[lead]      = -1;
        box_fresh[cell_id] = 0;

        if (rs_replay) {
            const bool keep = n_seq_tokens <= n_x();

            // the pending inputs are in the cell that cached them, in its current half; none read zeros
            box_x_src [s] = n_pend[lead] > 0 && box_x_cell[lead] >= 0 ? box_x_cell[lead] : -1;
            box_x_half[s] = x_cur[lead];

            n_rep[lead] = n_pend[lead];

            if (keep) {
                x_cur[lead] ^= 1;
            }
            n_pend[lead] = keep ? n_seq_tokens : 0;

            box_x_cell[lead] = cell_id;

            // the other seqs of the cell share its state and its cached inputs
            for (const llama_seq_id seq_id : cell.seq_id) {
                if (seq_id == lead) {
                    continue;
                }
                x_cur [seq_id] = x_cur[lead];
                n_pend[seq_id] = n_pend[lead];
                box_x_cell[seq_id] = cell_id;
            }
        } else {
            box_x_src [s] = -1;
            box_x_half[s] = 0;

            box_x_cell[lead] = cell_id;

            // the snapshots the step writes, see seq_rm(); the other seqs of the cell share them
            n_snap[lead] = std::min(n_seq_tokens, n_rs_seq + 1);
            for (const llama_seq_id seq_id : cell.seq_id) {
                n_snap[seq_id] = n_snap[lead];
            }
        }
    }

    head = 0;
    n    = n_seqs;

    return true;
}

bool llama_memory_recurrent::get_can_shift() const {
    // shifting the pos is trivial for recurrent models
    return true;
}

void llama_memory_recurrent::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    GGML_UNUSED(flags);

    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges; // ranges, from inclusive, to exclusive
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges_data; // logical source row ranges
    std::vector<std::pair<uint32_t, uint32_t>> cell_ranges_s;    // the same for the scan state, see rs_replay
    uint32_t cell_count = 0;

    // Count the number of cells with the specified seq_id
    // Find all the ranges of cells with this seq id (or all, when -1)
    uint32_t cell_range_begin = size;
    for (uint32_t i = 0; i < size; ++i) {
        const auto & cell = cells[i];
        // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
        if ((seq_id == -1 && !cell.is_empty()) || cell.has_seq_id(seq_id)) {
            ++cell_count;
            uint32_t rs_idx_cur = 0;

            if (n_rs_seq != 0) {
                if (seq_id != -1) {
                    GGML_ASSERT(seq_id >= 0 && (size_t) seq_id < rs_idx.size());
                    rs_idx_cur = rs_idx[seq_id];
                } else {
                    bool has_rs_idx = false;
                    for (const llama_seq_id cell_seq_id : cell.seq_id) {
                        GGML_ASSERT(cell_seq_id >= 0 && (size_t) cell_seq_id < rs_idx.size());

                        const uint32_t seq_rs_idx = rs_idx[cell_seq_id];
                        if (!has_rs_idx) {
                            rs_idx_cur = seq_rs_idx;
                            has_rs_idx = true;
                        } else if (rs_idx_cur != seq_rs_idx) {
                            GGML_ABORT("cannot write shared recurrent state with different rollback indices");
                        }
                    }
                }
            }

            const uint32_t src_id  = cell.src >= 0 ? (uint32_t) cell.src : i;
            const uint32_t cell_id = rs_idx_cur * size + src_id;
            if (cell_ranges_data.empty() || cell_ranges_data.back().second != cell_id) {
                cell_ranges_data.emplace_back(cell_id, cell_id + 1);
            } else {
                cell_ranges_data.back().second++;
            }

            // the scan state has a single row instead of snapshots
            const uint32_t cell_id_s = rs_replay ? src_id : cell_id;
            if (cell_ranges_s.empty() || cell_ranges_s.back().second != cell_id_s) {
                cell_ranges_s.emplace_back(cell_id_s, cell_id_s + 1);
            } else {
                cell_ranges_s.back().second++;
            }

            if (cell_range_begin == size) {
                cell_range_begin = i;
            }
        } else {
            if (cell_range_begin != size) {
                cell_ranges.emplace_back(cell_range_begin, i);
                cell_range_begin = size;
            }
        }
    }
    if (cell_range_begin != size) {
        cell_ranges.emplace_back(cell_range_begin, size);
    }

    if ((flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) && cell_ranges.size() > 1) {
        GGML_ABORT("cannot save/load multiple ranges of cells to/from device memory\n");
    }

    // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
    uint32_t cell_count_check = 0;
    for (const auto & range : cell_ranges) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    cell_count_check = 0;
    for (const auto & range : cell_ranges_data) {
        cell_count_check += range.second - range.first;
    }
    GGML_ASSERT(cell_count == cell_count_check);

    io.write(&cell_count, sizeof(cell_count));

    state_write_meta(io, cell_ranges, seq_id);
    state_write_data(io, cell_ranges_data, cell_ranges_s);

    if (rs_replay) {
        state_write_pend(io, seq_id);
    }
}

// the scan state lags behind the pending tokens of each seq, so their cached inputs are part of the state
void llama_memory_recurrent::state_write_pend(llama_io_write_i & io, llama_seq_id seq_id) const {
    std::vector<llama_seq_id> seqs;
    for (uint32_t s = 0; s < n_seq_max; ++s) {
        if ((seq_id == -1 && seq_tails[s] >= 0) || (llama_seq_id) s == seq_id) {
            seqs.push_back((llama_seq_id) s);
        }
    }

    const uint32_t n_seqs = seqs.size();
    io.write(&n_seqs, sizeof(n_seqs));

    for (const llama_seq_id s : seqs) {
        const uint32_t n = seq_tails[s] >= 0 ? n_pend[s] : 0;

        io.write(&s, sizeof(s));
        io.write(&n, sizeof(n));

        if (n == 0) {
            continue;
        }

        for (size_t il = 0; il < x_l.size(); ++il) {
            ggml_tensor * x = x_l[il];
            if (x == nullptr) {
                continue;
            }

            const size_t n_bytes = ggml_row_size(x->type, (int64_t) hparams.n_embd_gdn_inp()*n);

            // the cached inputs are in the state that cached them, see box_x_cell
            const size_t offs = (size_t) box->x_row(box_x_cell[s] >= 0 ? box_x_cell[s] : seq_tails[s], box->recr_index(il), x_cur[s])*llama_box::unit;

            io.write_tensor(x, offs, n_bytes);
        }
    }
}

bool llama_memory_recurrent::state_read_pend(llama_io_read_i & io, llama_seq_id dest_seq_id) {
    uint32_t n_seqs;
    io.read(&n_seqs, sizeof(n_seqs));

    if (dest_seq_id != -1 && n_seqs > 1) {
        LLAMA_LOG_ERROR("%s: pending inputs of %u seqs for a single seq\n", __func__, n_seqs);
        return false;
    }

    for (uint32_t k = 0; k < n_seqs; ++k) {
        llama_seq_id s;
        uint32_t     n;

        io.read(&s, sizeof(s));
        io.read(&n, sizeof(n));

        if (dest_seq_id != -1) {
            s = dest_seq_id;
        }

        if (s < 0 || (uint32_t) s >= n_seq_max || n > n_x()) {
            LLAMA_LOG_ERROR("%s: invalid pending inputs: seq %d, %u tokens\n", __func__, s, n);
            return false;
        }

        n_pend[s] = n;

        if (n == 0) {
            continue;
        }

        if (seq_tails[s] < 0) {
            LLAMA_LOG_ERROR("%s: pending inputs of seq %d without a state\n", __func__, s);
            return false;
        }

        std::vector<uint8_t> row_buf;

        for (size_t il = 0; il < x_l.size(); ++il) {
            ggml_tensor * x = x_l[il];
            if (x == nullptr) {
                continue;
            }

            const size_t row_size = ggml_row_size(x->type, x->ne[0]);
            const size_t n_bytes  = ggml_row_size(x->type, (int64_t) hparams.n_embd_gdn_inp()*n);

            // the whole row is written, the token slots after the restored ones as zeros, since a replay reads every
            // slot of the row (and masks the ones that are not pending)
            row_buf.assign(row_size, 0);
            io.read(row_buf.data(), n_bytes);

            const size_t offs = (size_t) box->x_row(seq_tails[s], box->recr_index(il), x_cur[s])*llama_box::unit;
            ggml_backend_tensor_set(x, row_buf.data(), offs, row_size);
        }

        box_x_cell[s] = seq_tails[s];
    }

    return true;
}

void llama_memory_recurrent::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(flags);

    uint32_t cell_count;
    io.read(&cell_count, sizeof(cell_count));

    bool res = true;

    // save the head of the restored cells - could be needed to clear the state
    // the head is valid only when state_read_meta() succeeded
    const bool meta_read = state_read_meta(io, cell_count, seq_id);

    res = res && meta_read;

    try {
        res = res && state_read_data(io, cell_count);

        if (rs_replay) {
            res = res && state_read_pend(io, seq_id);
        }
    } catch (...) {
        res = false;
    }

    if (!res) {
        state_clear(seq_id);
        throw std::runtime_error("failed to restore kv cache");
    }

    if (n_rs_seq != 0) {
        set_rs_idx(seq_id, 0);
    }

    // a loaded state has no snapshots of a step: it is rolled back to after its next one
    if (seq_id < 0) {
        std::fill(n_snap.begin(), n_snap.end(), 0);
    } else {
        n_snap[seq_id] = 0;
    }
}

void llama_memory_recurrent::state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id) const {
    for (const auto & range : cell_ranges) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            const auto & cell = cells[i];
            const llama_pos pos      = cell.pos;
            const uint32_t  n_seq_id = seq_id == -1 ? cell.seq_id.size() : 0;

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id) {
                for (auto seq_id : cell.seq_id) {
                    io.write(&seq_id, sizeof(seq_id));
                }
            }
        }
    }
}

void llama_memory_recurrent::state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges_s) const {
    const uint32_t s_trans = 0;
    const uint32_t n_layer = hparams.n_layer();

    io.write(&s_trans, sizeof(s_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the R tensors first, each row is a cell
    // Get whole range at a time
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
        if (r_l[il] == nullptr) continue;

        // Write R tensor type
        const int32_t r_type_i = (int32_t)r_l[il]->type;
        io.write(&r_type_i, sizeof(r_type_i));

        // Write row size of R tensor
        const uint64_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        io.write(&r_size_row, sizeof(r_size_row));

        // Write each logical cell row range. With pending recurrent rollback,
        // the logical current state may live in a rollback snapshot plane.
        // the rows of the box are not contiguous across cells: one row each, conv row of the rollback plane
        for (const auto & range : cell_ranges) {
            for (uint32_t id = range.first; id < range.second; ++id) {
                io.write_tensor(r_l[il], (size_t) box->r_row(id % size, box->recr_index(il), id / size)*llama_box::unit, r_size_row);
            }
        }

        // the PLE conv history is a second recurrent row, so it has to travel with the first
        if (p_l[il] != nullptr) {
            const uint64_t p_size_row = ggml_row_size(p_l[il]->type, hparams.ple_conv_state());
            io.write(&p_size_row, sizeof(p_size_row));

            for (const auto & range : cell_ranges) {
                for (uint32_t id = range.first; id < range.second; ++id) {
                    io.write_tensor(p_l[il], (size_t) box->c_row(id % size, box->recr_index(il), id / size)*llama_box::unit, p_size_row);
                }
            }
        }
    }

    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers (read_data will handle this by checking "r_l" and "s_l" for null)
        if (s_l[il] == nullptr) continue;

        // Write S tensor type
        const int32_t s_type_i = (int32_t)s_l[il]->type;
        io.write(&s_type_i, sizeof(s_type_i));

        // Write row size of S tensor
        const uint64_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
        io.write(&s_size_row, sizeof(s_size_row));

        // Write each logical cell row range. With pending recurrent rollback,
        // the logical current state may live in a rollback snapshot plane.
        for (const auto & range : cell_ranges_s) {
            for (uint32_t id = range.first; id < range.second; ++id) {
                io.write_tensor(s_l[il], (size_t) box->s_row(id % size, box->recr_index(il), id / size)*llama_box::unit, s_size_row);
            }
        }
    }
}

bool llama_memory_recurrent::state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id) {
    if (dest_seq_id != -1) {
        // single sequence
        if (cell_count > size) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        seq_rm(dest_seq_id, -1, -1);

        if (cell_count == 0) {
            return true;
        }

        // a state of its own for the seq, the data goes into its cell
        if (cell_count != 1) {
            LLAMA_LOG_ERROR("%s: the box restores one state per seq, not %u\n", __func__, cell_count);
            return false;
        }

        llama_pos pos;
        uint32_t  n_seq_id;

        io.read(&pos,      sizeof(pos));
        io.read(&n_seq_id, sizeof(n_seq_id));

        if (n_seq_id != 0) {
            LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
            return false;
        }

        // the data read below overwrites the rows the state is read from, the rest is written before it is read;
        // an admitted request kept the state its admission took (see box_keep)
        int32_t p = seq_tails[dest_seq_id];
        if (p < 0) {
            p = box->state_alloc();
            if (p < 0) {
                LLAMA_LOG_ERROR("%s: no state fits in the box for seq %d\n", __func__, dest_seq_id);
                return false;
            }
            used++;
        } else if (cells[p].pos < 0) {
            used++; // the kept state is not counted while it holds no token
        }

        box_fresh[p] = 0;

        auto & cell = cells[p];
        cell.pos = pos;
        cell.src = p;
        cell.seq_id.clear();
        cell.seq_id.insert(dest_seq_id);

        seq_tails[dest_seq_id]  = p;
        box_cow[dest_seq_id]    = -1;
        box_x_cell[dest_seq_id] = p;

        head = p;

        return true;
    }

    // the states of the box belong to the pages they were given
    LLAMA_LOG_ERROR("%s: the box restores single sequences only\n", __func__);
    return false;
}

bool llama_memory_recurrent::state_read_data(llama_io_read_i & io, uint32_t cell_count) {
    uint32_t s_trans;
    uint32_t n_layer;
    io.read(&s_trans, sizeof(s_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != hparams.n_layer()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, hparams.n_layer());
        return false;
    }
    if (cell_count > size) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, size);
        return false;
    }
    if (false != (bool) s_trans) {
        LLAMA_LOG_ERROR("%s: incompatible s transposition\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers
        if (r_l[il] == nullptr) continue;

        // Read type of key
        int32_t r_type_i_ref;
        io.read(&r_type_i_ref, sizeof(r_type_i_ref));
        const int32_t r_type_i = (int32_t) r_l[il]->type;
        if (r_type_i != r_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r type (%d != %d, layer %d)\n", __func__, r_type_i, r_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t r_size_row_ref;
        io.read(&r_size_row_ref, sizeof(r_size_row_ref));
        const size_t r_size_row = ggml_row_size(r_l[il]->type, hparams.n_embd_r());
        if (r_size_row != r_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched r row size (%zu != %zu, layer %d)\n", __func__, r_size_row, (size_t) r_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            // the box restores the one state of a seq, conv row 0
            const size_t offs = (size_t) box->r_row(head, box->recr_index(il), 0)*llama_box::unit;
            io.read_tensor(r_l[il], offs, cell_count * r_size_row);
        }

        if (p_l[il] != nullptr) {
            uint64_t p_size_row_ref;
            io.read(&p_size_row_ref, sizeof(p_size_row_ref));
            const size_t p_size_row = ggml_row_size(p_l[il]->type, hparams.ple_conv_state());
            if (p_size_row != p_size_row_ref) {
                LLAMA_LOG_ERROR("%s: mismatched ple row size (%zu != %zu, layer %d)\n", __func__, p_size_row, (size_t) p_size_row_ref, il);
                return false;
            }

            if (cell_count) {
                const size_t offs = (size_t) box->c_row(head, box->recr_index(il), 0)*llama_box::unit;
                io.read_tensor(p_l[il], offs, cell_count * p_size_row);
            }
        }
    }

    for (uint32_t il = 0; il < n_layer; ++il) {
        // skip null layers
        if (s_l[il] == nullptr) continue;

        // Read type of value
        int32_t s_type_i_ref;
        io.read(&s_type_i_ref, sizeof(s_type_i_ref));
        const int32_t s_type_i = (int32_t)s_l[il]->type;

        if (s_type_i != s_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched s type (%d != %d, layer %d)\n", __func__, s_type_i, s_type_i_ref, il);
            return false;
        }

        // Read row size of value
        uint64_t s_size_row_ref;
        io.read(&s_size_row_ref, sizeof(s_size_row_ref));
        const size_t s_size_row = ggml_row_size(s_l[il]->type, hparams.n_embd_s());
        if (s_size_row != s_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched s row size (%zu != %zu, layer %d)\n", __func__, s_size_row, (size_t) s_size_row_ref, il);
            return false;
        }

        if (cell_count) {
            const size_t offs = (size_t) box->s_row(head, box->recr_index(il))*llama_box::unit;
            io.read_tensor(s_l[il], offs, cell_count * s_size_row);
        }
    }

    return true;
}

void llama_memory_recurrent::state_clear(llama_seq_id seq_id) {
    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    if (seq_id == -1) {
        clear(true);
        return;
    }

    // the state of the box goes back with the seq, the next one to take it zeroes it
    seq_rm(seq_id, -1, -1);
}

//
// llama_memory_recurrent_context
//

llama_memory_recurrent_context::llama_memory_recurrent_context(llama_memory_status status) : status(status) {}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), is_full(true) {
}

llama_memory_recurrent_context::llama_memory_recurrent_context(
        llama_memory_recurrent * mem,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), mem(mem), ubatches(std::move(ubatches)) {}

llama_memory_recurrent_context::~llama_memory_recurrent_context() = default;

bool llama_memory_recurrent_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_next >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_memory_recurrent_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is an update, which the states in the box never need
    if (ubatches.empty()) {
        return true;
    }

    mem->find_slot(ubatches[i_next]);

    return true;
}

llama_memory_status llama_memory_recurrent_context::get_status() const {
    return status;
}

const llama_ubatch & llama_memory_recurrent_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_next];
}

ggml_tensor * llama_memory_recurrent_context::get_r_l(int32_t il) const {
    return mem->r_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_s_l(int32_t il) const {
    return mem->s_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_p_l(int32_t il) const {
    return mem->p_l[il];
}

ggml_tensor * llama_memory_recurrent_context::get_x_l(int32_t il) const {
    return mem->rs_replay ? mem->x_l[il] : nullptr;
}

uint32_t llama_memory_recurrent_context::get_n_x() const {
    return mem->n_x();
}

bool llama_memory_recurrent_context::is_p_l(const ggml_tensor * t) const {
    for (const auto * p : mem->p_l) {
        if (p == t) {
            return true;
        }
    }

    return false;
}

bool llama_memory_recurrent_context::is_s_l(const ggml_tensor * t) const {
    for (const auto * s : mem->s_l) {
        if (s == t) {
            return true;
        }
    }

    return false;
}

uint32_t llama_memory_recurrent_context::replay_n(int i) const {
    const llama_seq_id seq_id = mem->box_lead[i];

    return !mem->rs_replay || seq_id < 0 ? 0 : mem->n_rep[seq_id];
}

uint32_t llama_memory_recurrent_context::get_n_rs_seq() const {
    return mem->n_rs_seq;
}

uint32_t llama_memory_recurrent_context::box_n_layers() const {
    return mem->box->get_n_recr();
}

int32_t llama_memory_recurrent_context::box_layer(int32_t il) const {
    return mem->box->recr_index(il);
}

int32_t llama_memory_recurrent_context::box_row(int i, uint32_t l, box_row_kind kind, uint32_t k) const {
    const auto * box = mem->box;

    const int32_t src = mem->box_src[i];
    const int32_t dst = mem->box_dst[i];

    switch (kind) {
        case BOX_R_READ:  return src < 0 ? -1 : (int32_t) box->r_row(src, l, k);
        case BOX_R_WRITE: return (int32_t) box->r_row(dst, l, k);
        case BOX_S_READ:  return src < 0 ? -1 : (int32_t) box->s_row(src, l, k);
        case BOX_S_WRITE: return (int32_t) box->s_row(dst, l, k);
        case BOX_X_READ:  return mem->box_x_src[i] < 0 ? -1 : (int32_t) box->x_row(mem->box_x_src[i], l, mem->box_x_half[i]);
        case BOX_X_WRITE: return (int32_t) box->x_row(dst, l, mem->x_cur[mem->box_lead[i]]);
        case BOX_C_READ:  return src < 0 || !box->has_c(l) ? -1 : (int32_t) box->c_row(src, l, k);
        case BOX_C_WRITE: return !box->has_c(l) ? -1 : (int32_t) box->c_row(dst, l, k);
    }

    GGML_ABORT("unknown box row kind");
}

uint32_t llama_memory_recurrent_context::box_take_rollback(int i) const {
    if (mem->n_rs_seq == 0) {
        return 0;
    }

    const llama_seq_id lead = mem->box_lead[i];

    const uint32_t idx = mem->rs_idx[lead];
    mem->rs_idx[lead] = 0; // reset rollback idx

    return idx;
}
