#pragma once

#include "llama-batch.h"
#include "llama-graph.h"
#include "llama-memory.h"

#include <functional>
#include <map>
#include <set>
#include <vector>

class llama_box;

//
// llama_memory_recurrent
//

// TODO: extract the cache state used for graph computation into llama_memory_recurrent_context_i
//       see the implementation of llama_kv_cache_context_i for an example how to do it
class llama_memory_recurrent : public llama_memory_i {
public:
    llama_memory_recurrent(
            const llama_model & model,
                     uint32_t   n_rs_seq,
                         bool   rs_replay_req, // replay if the model supports it, see rs_replay
        const layer_filter_cb & filter,
        // the elastic memory that holds the states, see llama_box: a cell is a state of it
                    llama_box * box,
        // the elastic memory this memory owns, a model whose only memory is the recurrent states
        std::shared_ptr<llama_box> box_own = nullptr);

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

    std::shared_ptr<llama_box> get_box_shared() const override { return box_own; }

    bool box_reserve(llama_seq_id seq_id, uint32_t n_input, uint32_t n_output) override;
    void box_reserve_drop(llama_seq_id seq_id) override;

    bool prepare(const std::vector<llama_ubatch> & ubatches);

    // sets up the states the step reads and writes, see box_src and box_dst
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

    // per seq, set by find_slot() for the step being computed: how many tokens its replay reads
    std::vector<uint32_t> n_rep;

    // number of token slots in the input cache of a seq: a speculative step is at most this long
    uint32_t n_x() const {
        return rs_replay ? n_rs_seq + 1 : 0;
    }

    // computed before each graph build
    uint32_t n = 0;

    // TODO: optimize for recurrent state needs
    struct mem_cell {
        llama_pos pos  = -1;
        int32_t   src  = -1; // used to know where states should be copied from

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

    //
    // elastic memory, see llama_box
    //
    // a cell is a state of the box (its state id), made of blocks of the box, and stays where it is: a step reads the
    // states of its seqs and writes them back by their rows (llama_memory_recurrent_context::box_row()), instead of
    // gathering them into a range of cells from head; nothing is cleared: a new state reads zeros in its first step
    //

    llama_box * box = nullptr;

    // the box of a model whose only memory is this one
    std::shared_ptr<llama_box> box_own;

    // per seq: a state that the seq shared (seq_cp) when it took a cell of its own, read by the next step of the seq,
    // which writes the cell of the seq, see box_prepare(); -1 when the seq reads its own cell
    std::vector<int32_t> box_cow;

    // per seq: the cell whose state holds the cached replay inputs of the seq, see rs_replay
    std::vector<int32_t> box_x_cell;

    // per cell: a state that holds nothing yet, which its first step reads as zeros (and then writes)
    std::vector<uint8_t> box_fresh;

    // shared states the seqs of a batch all left, which the batch still reads: they go back to the box with the next one
    std::vector<int32_t> box_free_next;

    // whether seq_id keeps its state when all its tokens are removed (an admitted request, see llama_box::reserve())
    std::function<bool(llama_seq_id)> box_keep;

    // per seq of the ubatch being computed, set by box_find_slot(): its lead seq, the cell its state is read from (-1:
    // zeros), the cell it is written to, and the cell and half of its pending replay inputs (-1: zeros)
    std::vector<llama_seq_id> box_lead;
    std::vector<int32_t>      box_src;
    std::vector<int32_t>      box_dst;
    std::vector<int32_t>      box_x_src;
    std::vector<uint8_t>      box_x_half;

    // gives a state of its own to each seq of the ubatches that has none or shares one
    bool box_prepare(const std::vector<llama_ubatch> & ubatches);

    // a state for seq_id, which must hold none; false when the box has not enough free blocks for it
    bool box_state_take(llama_seq_id seq_id);

    // the state of an emptied cell goes back to the box
    void box_cell_freed(uint32_t i);

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

    uint32_t n_seq_max = 1;

    void state_write_meta(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, llama_seq_id seq_id = -1) const;
    void state_write_data(llama_io_write_i & io, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges, const std::vector<std::pair<uint32_t, uint32_t>> & cell_ranges_s) const;

    bool state_read_meta(llama_io_read_i & io, uint32_t cell_count, llama_seq_id dest_seq_id = -1);
    bool state_read_data(llama_io_read_i & io, uint32_t cell_count);

    void state_clear(llama_seq_id seq_id);

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

    ggml_tensor * get_r_l(int32_t il) const;
    ggml_tensor * get_s_l(int32_t il) const;
    ggml_tensor * get_p_l(int32_t il) const;

    // cached delta-net inputs of the last tokens, nullptr unless rollback replays them, see llama_memory_recurrent
    ggml_tensor * get_x_l(int32_t il) const;

    // number of token slots in the input cache, 0 when it is not used
    uint32_t get_n_x() const;

    // whether t is one of the scan-state or conv history tensors, which have rows of their own in the box
    bool is_s_l(const ggml_tensor * t) const;
    bool is_p_l(const ggml_tensor * t) const;

    // tokens replayed by the i-th seq of the ubatch before the tokens of this step
    uint32_t replay_n(int i) const;

    //
    // the elastic memory, see llama_memory_recurrent::box
    //

    enum box_row_kind {
        BOX_R_READ,  // conv row k (the rollback index) of the state read
        BOX_R_WRITE, // conv row k of the state written
        BOX_S_READ,
        BOX_S_WRITE,
        BOX_X_READ,  // the pending replay inputs
        BOX_X_WRITE, // the replay inputs this step caches
        BOX_C_READ,  // conv history row k (the rollback index) of the state read
        BOX_C_WRITE, // conv history row k of the state written
    };

    uint32_t get_n_rs_seq() const;

    // the recurrent layers of the box, and the index of model layer il among them
    uint32_t box_n_layers() const;
    int32_t  box_layer(int32_t il) const;

    // the row of the i-th seq of the ubatch in the box view of its kind, for recurrent layer l; -1 reads zeros
    int32_t box_row(int i, uint32_t l, box_row_kind kind, uint32_t k = 0) const;

    // the rollback index of the i-th seq of the ubatch, which a step reads once (it is reset)
    uint32_t box_take_rollback(int i) const;

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
