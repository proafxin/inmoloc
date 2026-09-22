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
                         bool   rs_replay_req, // replay if the model supports it, see rs_replay
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

    // rollback by replaying cached delta-net inputs, instead of keeping one state snapshot per draft position:
    // the scan state kept in memory lags behind the tokens of the last speculative step, whose inputs are cached,
    // and every step first replays the ones of them that were accepted. a rollback only lowers how many are
    // replayed, so a cell needs a single scan-state row whatever the speculation depth
    // chosen with llama_context_params::rs_rollback, see also llm_build_delta_net_base::build_recurrent_attn_replay()
    bool rs_replay = false;

    // per-seq rollback index
    std::vector<uint32_t> rs_idx;

    void set_rs_idx(llama_seq_id seq_id, uint32_t idx);

    //
    // rs_replay bookkeeping
    //

    // the input cache is indexed by seq, since cells move and sequences do not; each seq has two halves, so that a
    // step can cache its own inputs while its replay reads the ones of the step before

    // per seq: the half holding the inputs of the last cached step
    std::vector<uint8_t> x_cur;

    // per seq: the tokens of the last cached step that are in the sequence but not yet in the scan state
    std::vector<uint32_t> n_pend;

    // per seq, set by find_slot() for the step being computed: the row its replay reads and how many tokens
    std::vector<uint32_t> x_rd;
    std::vector<uint32_t> n_rep;

    // input-cache rows to copy before the next step, when a seq takes over the pending tokens of another one
    // (seq_cp, or a step decoding several seqs at once); applied by the next memory update, see init_update()
    struct x_copy {
        llama_seq_id dst;
        uint32_t     row_src;
        uint32_t     row_dst;
    };

    std::vector<x_copy> x_copies;

    // number of token slots in the input cache of a seq: a speculative step is at most this long
    uint32_t n_x() const {
        return rs_replay ? n_rs_seq + 1 : 0;
    }

    uint32_t x_row(llama_seq_id seq_id, uint32_t half) const {
        return half*n_seq_max + (uint32_t) seq_id;
    }

    // the row that holds the pending inputs of a seq, which is another seq's while a copy into it is queued
    uint32_t x_row_pend(llama_seq_id seq_id) const;

    // the seq whose rows a cell's replay reads and writes
    llama_seq_id x_seq(uint32_t i) const {
        return cells[i].seq_id.empty() ? -1 : *cells[i].seq_id.begin();
    }

    // copies the input-cache row of seq src into the half of seq dst that holds its pending inputs
    void x_copy_queue(llama_seq_id src, llama_seq_id dst);

    // applies the queued copies, after the computation that writes their source rows
    void x_copy_apply(llama_context * lctx);

    // drops the queued copies into seq_id (all when negative)
    void x_copy_drop(llama_seq_id seq_id);

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
    // cached delta-net inputs of the last (1 + n_rs_seq) tokens of each seq, two halves, see rs_replay
    // [n_embd_gdn_inp*(1 + n_rs_seq), 2*n_seq_max], empty when rs_replay is off
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

    // the pending tokens of each seq and their cached inputs, see rs_replay
    void state_write_pend(llama_io_write_i & io, llama_seq_id seq_id) const;
    bool state_read_pend (llama_io_read_i  & io, llama_seq_id dest_seq_id);
};

class llama_memory_recurrent_context : public llama_memory_context_i {
public:
    // used for errors
    llama_memory_recurrent_context(llama_memory_status status);

    // used to create a full-cache context
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem);

    // used to create an update context, which applies the queued input-cache copies
    llama_memory_recurrent_context(
            llama_memory_recurrent * mem,
                     llama_context * lctx);

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

    // rows of the input cache read by the replay of this step, and written by it
    int32_t x_read (int i) const;
    int32_t x_write(int i) const;

    // tokens replayed by the i-th cell of the ubatch before the tokens of this step
    uint32_t replay_n(int i) const;

    int32_t s_copy(int i) const;

private:
    const llama_memory_status status;

    llama_memory_recurrent * mem;

    // set for an update context
    llama_context * lctx = nullptr;

    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    //
    // data needed for building the compute graph for the current ubatch:
    // TODO: extract all the state like `head` and `n` here
    //

    const bool is_full = false;
};
