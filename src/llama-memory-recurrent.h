#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <map>
#include <set>
#include <vector>

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                    ggml_type   type_r,
                    ggml_type   type_s,
                         bool   offload,
                     uint32_t   mem_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter);

    ~llama_memory_recurrent() = default;

    //
    // llama_memory_i
    //

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // find a contiguous slot of memory cells and emplace the ubatch there
    bool find_slot(const llama_ubatch & ubatch);

    bool get_can_shift() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    uint32_t head = 0; // the location where the batch will be placed in the cache (see find_slot())
    uint32_t size = 0; // total number of cells, shared across all sequences
    uint32_t used = 0; // used cells (i.e. at least one seq_id)

    // number of recurrent-state snapshots per seq for rollback; tensors are widened to (1 + n_rs_seq) groups
    uint32_t n_rs_seq = 0;

    // rollback by replaying the cached delta-net inputs of the last tokens, instead of keeping one state
    // snapshot per draft position: the state of a rejected draft is recomputed by running the scan again
    // over the accepted tokens, so the memory no longer grows with the speculation depth
    // enabled with LLAMA_RS_REPLAY=1, see also llm_build_delta_net_base::build_recurrent_attn()
    bool rs_replay = false;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    //
    // rs_replay bookkeeping
    //

    // per cell: which scan-state row holds the current state; a rollback flips it back to the previous one
    std::vector<uint8_t> s_cur;

    // per seq: which half of the input cache the last step wrote; a rollback does not change it, the inputs it
    // replays are the ones of that step. the cache is indexed by seq, since cells move and sequences do not
    std::vector<uint8_t> x_cur;

    // per cell: whether the current step flipped s_cur, i.e. whether the state from before the step is now in the
    // other row; cells in the range of the step that it does not compute keep their row
    std::vector<uint8_t> s_flip;

    // per seq: how many tokens of the last step are in the input cache, i.e. how far a replay can go back
    std::vector<uint32_t> n_x_cached;

    // per seq: tokens to replay from the previous state before the next step, see rs_replay
    std::vector<uint32_t> rs_replay_n;

    // LLAMA_RS_REPLAY_DEBUG=1 traces the rollback bookkeeping
    static bool rs_debug() {
        static const bool res = [] {
            const char * env = getenv("LLAMA_RS_REPLAY_DEBUG");
            return env != nullptr && atoi(env) != 0;
        }();
        return res;
    }

    // the scan-state row of cell i: the current state, or the one from before the last step
    uint32_t s_row(uint32_t i, bool prev) const {
        return ((uint32_t) (s_cur[i] ^ (prev ? 1 : 0)))*size + i;
    }

    // the input-cache row of cell i: the half the last step wrote, or the one this step writes
    uint32_t x_row(llama_seq_id seq_id, bool prev) const {
        return ((uint32_t) (x_cur[seq_id] ^ (prev ? 1 : 0)))*n_seq_max + (uint32_t) seq_id;
    }

    // tokens the seq of cell i replays before this step, see rs_replay
    uint32_t replay_n(uint32_t i) const {
        if (!rs_replay || cells[i].seq_id.empty()) {
            return 0;
        }

        const llama_seq_id seq_id = *cells[i].seq_id.begin();

        return seq_id >= 0 && (size_t) seq_id < rs_replay_n.size() ? rs_replay_n[seq_id] : 0;
    }

    // computed before each graph build
    uint32_t n = 0;

    // first zero-ed state
    int32_t rs_z = -1;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from
        int32_t   src0 = -1; // like src, but only used when setting the inputs (allowing to copy once)

        std::set<llama_seq_id> seq_id;

        bool has_seq_id(const llama_seq_id & id) const {
            return seq_id.find(id) != seq_id.end();
        }

        bool is_empty() const {
            return seq_id.empty();
        }

        bool is_same_seq(const mem_cell & other) const {
            return seq_id == other.seq_id;
        }
    };

    std::vector<mem_cell> cells;

    // per-seq index of the cell holding that seq's state, or -1
    // kept apart from `cells` so the cell pool can be smaller than n_seq_max
    std::vector<int32_t> seq_tails;

    // number of free cells, i.e. cells not owned by any seq
    uint32_t n_free_cells() const;

    // per layer
    std::vector<ggml_tensor *> r_l;
    std::vector<ggml_tensor *> s_l;
    // cached delta-net inputs of the last (1 + n_rs_seq) tokens of each cell, see rs_replay
    // [n_embd_gdn_inp*(1 + n_rs_seq), size], empty when rs_replay is off
    std::vector<ggml_tensor *> x_l;
    // a second conv history that must stay replicated across devices, so it cannot share the r row
    std::vector<ggml_tensor *> p_l;

private:
    //const llama_model & model;
    const llama_hparams & hparams;

    const uint32_t n_seq_max = 1;

    // ggml contexts for the KV cache along with the allocated backend buffers:
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> ctxs_bufs;

    size_t total_size() const;

    size_t size_r_bytes() const;
    size_t size_s_bytes() const;
    size_t size_p_bytes() const;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges_s) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache or update context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create a batch processing context from a batch
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_memory_recurrent_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_memory_recurrent_context specific API
    //

    uint32_t get_n_rs() const;
    uint32_t get_head() const;
    int32_t  get_rs_z() const;
    uint32_t get_size() const;

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;

    // cached delta-net inputs of the last tokens, nullptr unless rollback replays them, see llama_memory_recurrent
    ggml_tensor * get_x_l(int32_t il) const;

    // number of token slots in the input cache, 0 when it is not used
    uint32_t get_n_x() const;

    // the row of the scan state of the i-th cell, which has no snapshots when a rollback replays instead
    int32_t s_copy_plain(int i) const;

    // whether t is one of the scan-state tensors, which are read with s_copy_plain()
    bool is_s_l(const ggml_tensor * t) const;

    // the row of the scan state this step writes
    int32_t s_write(int i) const;

    // the other row of the cell, which this step fills with the state after the replayed tokens and before
    // its own ones, so that a rollback into this step replays from there
    int32_t s_prev(int i) const;

    // rows of the input cache read by a replay, and written by this step
    int32_t x_read (int i) const;
    int32_t x_write(int i) const;

    // tokens replayed by the seq of the i-th cell of the ubatch before this step
    uint32_t replay_n(int i) const;

    // the largest number of tokens any seq of this ubatch replays, 0 when no rollback is pending
    uint32_t get_n_replay() const;

    int32_t s_copy(int i) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
