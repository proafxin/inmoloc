#pragma once

#include "llama.h"
#include "llama-cparams.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <set>
#include <utility>
#include <vector>

struct ggml_context;
struct ggml_tensor;
struct llama_model;
struct llama_hparams;

// the elastic memory: one buffer, the box, holds the attention KV and the recurrent states of all the requests, handed
// out by offset as the requests come and go; a model without recurrent layers has no states
//
// the states sit on the left of the box, one after the other, each exactly as large as a state is; the rest is a row of
// blocks of the same size, taken from the right: a block holds n_block_tokens tokens of every attention layer, laid out
// by layer inside the block (the K rows of the first layer, its V rows, then the next layer); with an MTP draft, the rows
// of its layers follow: a draft context reads and writes them for the same sequences, so its tokens take no memory of
// their own, see llama_kv_cache::box_ref
//
// what a request is known to take, its state and its input, is taken when it is admitted; the most it may generate is
// charged as a count, and its blocks are taken as the tokens appear (see reserve()); blocks and states go back to their
// free sets, nothing is cleared or copied; only a block next to the states is ever looked up by its index, which keeps
// the blocks on the right busy and the room for more states free
//
// each layer tensor is a view of the whole box with one row stride, read and written through row indices:
//   - attention: KV cell c (block c/n_block_tokens) is row (c/n_block_tokens)*rows_per_block + c%n_block_tokens of the
//     K and V views of every layer, whose view offsets hold the offset of the layer inside a block; the blocks under
//     the states hold no cells
//   - recurrent: one view per kind (conv, scan, replay inputs) for all the layers, with rows of unit bytes: a row of a
//     layer's state piece is at the unit index of its byte offset in the box, see r_row(), s_row(), x_row()
//
// a model with compressed history (DeepSeek V4) keeps its raw rows for a window and its compressed rows for good, so the
// blocks of a sequence are of several kinds, all of the same size: raw blocks go back as they leave the window, the
// others stay. A compressed block has the layout of a raw block, so compressed row r of a kind is row kv_row() of cell
// b*n_block_tokens + r%n_block_tokens, for every layer of the kind, in the same view as the raw rows of the layer: one
// attention reads both in place. The indexer rows have views of their own, on the blocks of the CSA rows
class llama_box {
public:
    struct attn_layer {
        int32_t   il;
        ggml_type type_k;
        ggml_type type_v;
        uint32_t  n_embd_k;
        uint32_t  n_embd_v;
        bool      has_v;
        bool      comp_free = false; // no compressed rows of the layer in a block of BLOCK_CSA, the indexer rows can use them
    };

    struct recr_layer {
        int32_t   il;
        ggml_type type_r;
        ggml_type type_s;
        uint32_t  n_embd_r;
        uint32_t  n_embd_s;
        uint32_t  n_embd_x; // the cached replay inputs of the last tokens, 0 without replay
        uint32_t  n_rows_r; // conv rows: one per draft position
        uint32_t  n_rows_s; // scan rows: one per draft position, or one when a rollback replays
        uint32_t  n_embd_c = 0; // a conv history next to the conv state (PLE), as many rows, 0 without one
    };

    // a piece of a state that is not a recurrent layer: n_rows rows of n_embd elements, read through a view of its own
    struct state_piece {
        ggml_type type;
        uint32_t  n_embd;
        uint32_t  n_rows;
    };

    enum block_kind : uint32_t {
        BLOCK_RAW = 0,
        BLOCK_CSA = 1,
        BLOCK_HCA = 2,
        BLOCK_KINDS,
    };

    static constexpr uint32_t n_block_tokens = 128;

    // the row unit of the recurrent views, which every state row is padded to
    static constexpr size_t unit = 256;

    // a box of `bytes` bytes, or with the memory of a cache of n_tokens cells and n_states states when bytes is 0; draft
    // are the layers of an MTP draft that keeps its tokens in the blocks, empty for none; no memory is taken until alloc()
    // lid are the indexer layers on the blocks of BLOCK_CSA, pieces the rest of a state
    llama_box(std::vector<attn_layer> attn, std::vector<recr_layer> recr, const std::vector<attn_layer> & draft, size_t bytes, uint32_t n_tokens, uint32_t n_states,
            std::vector<attn_layer> lid = {}, std::vector<state_piece> pieces = {});

    // takes the buffer from buft and creates the views; a dry run (no_alloc) takes none
    void alloc(ggml_backend_buffer_type_t buft, bool no_alloc);

    // the box was created with no buffer: it takes its real size, at most what it was created with, and its buffer; the
    // tensors of the views stay the same ones
    void finalize(size_t bytes);

    //
    // layout
    //

    size_t   get_block_bytes()  const { return block_bytes; }
    uint32_t get_n_blocks()     const { return n_blocks; }

    // sequence ids are kept per block, like the free list: a sequence that runs holds a block
    uint32_t get_n_seq_max()    const { return (uint32_t) seqs.size(); }
    uint32_t get_state_blocks() const { return state_blocks; } // the blocks one state covers, rounded up
    size_t   get_state_bytes()  const { return state_bytes; }
    size_t   get_total_bytes()  const { return (size_t) n_blocks*block_bytes; }

    // the KV cells of the box: n_block_tokens per block, the cells of the blocks that hold KV are used
    uint32_t get_n_cells() const { return n_blocks*n_block_tokens; }

    // the most states the box can hold, the size of the state id space
    uint32_t get_max_states() const { return state_bytes > 0 ? (uint32_t) (get_total_bytes()/state_bytes) : 0; }

    // the recurrent layers, in the order of their pieces of a state
    uint32_t get_n_recr() const { return (uint32_t) recr.size(); }

    // the index of model layer il among the recurrent layers, -1 for the others
    int32_t recr_index(int32_t il) const { return il >= 0 && il < (int32_t) recr_idx.size() ? recr_idx[il] : -1; }

    // row of KV cell c in the K and V views
    int64_t kv_row(uint32_t c) const {
        return (int64_t) (c/n_block_tokens)*rows_per_block + c%n_block_tokens;
    }

    // row of KV cell c in the K and V views of the draft layers
    int64_t kv_row_draft(uint32_t c) const {
        return (int64_t) (c/n_block_tokens)*rows_per_block_draft + c%n_block_tokens;
    }

    bool has_draft() const { return row_bytes_draft > 0; }

    // a draft model of its own whose layers the blocks hold (nullptr for MTP layers), in the slots il + draft_il_offset
    void set_draft(const llama_hparams * hparams, int32_t il_offset) { draft_hparams = hparams; draft_il_offset = il_offset; }
    const llama_hparams * get_draft_hparams()   const { return draft_hparams; }
    int32_t               get_draft_il_offset() const { return draft_il_offset; }

    // whether model layer il is a draft layer of the box
    bool is_draft(int32_t il) const;

    // whether the device reads the rows of a prompt through one list, else every token reads through ranges of its cells
    void set_row_lists(bool value) { row_lists = value; }
    bool has_row_lists() const { return row_lists; }


    // unit index of a row of the state s, recurrent layer l (recr_index()): conv row k, the scan row, replay input half h
    int64_t r_row(int32_t s, uint32_t l, uint32_t k) const;
    int64_t s_row(int32_t s, uint32_t l, uint32_t k = 0) const;
    int64_t x_row(int32_t s, uint32_t l, uint32_t h) const;
    int64_t c_row(int32_t s, uint32_t l, uint32_t k) const;

    // whether recurrent layer l has a conv history, see recr_layer
    bool has_c(uint32_t l) const { return recr[l].n_embd_c > 0; }

    // row k of a state piece in its view, the same for every piece of the size of the row
    int64_t p_row(int32_t s, uint32_t piece, uint32_t k) const {
        return (int64_t) (((size_t) s*state_bytes + k*GGML_PAD(ggml_row_size(pieces[piece].type, pieces[piece].n_embd), unit))/unit);
    }

    // row of compressed row r of seq_id, in the K views of the layers of the kind and in the indexer views
    int64_t comp_row(llama_seq_id seq_id, block_kind kind, uint32_t r) const {
        const auto & blocks = seqs.at(seq_id).comp[kind];
        return kv_row(blocks.at(r/n_block_tokens)*n_block_tokens + r%n_block_tokens);
    }

    // views of the whole box
    ggml_tensor * get_k(int32_t il) const;
    ggml_tensor * get_v(int32_t il) const;
    ggml_tensor * get_lid(int32_t il) const { return t_lid.at(il); }
    ggml_tensor * get_piece(uint32_t piece) const { return t_p.at(piece); } // [n_embd, units]
    ggml_tensor * get_r() const { return t_r; } // [n_embd_r, units]
    ggml_tensor * get_s() const { return t_s; } // [n_embd_s, units]
    ggml_tensor * get_x() const { return t_x; } // [n_embd_x, units], nullptr without replay
    ggml_tensor * get_c() const { return t_c; } // [n_embd_c, units], nullptr without a conv history

    ggml_backend_buffer_t get_buffer() const { return buf.get(); }

    //
    // blocks and states
    //

    // free blocks
    uint32_t n_free()        const { return (uint32_t) free_blocks.size(); }

    // a state: a free one, or the next one on the left when the blocks it covers are free; -1 otherwise
    int32_t state_alloc();
    void    state_free(int32_t s);

    // the KV blocks of a sequence, in the order its tokens fill them
    const std::vector<uint32_t> & seq_blocks(llama_seq_id seq_id) const { return seqs.at(seq_id).blocks; }

    // the rightmost free block for the tokens of seq_id; false when there is none
    bool block_take(llama_seq_id seq_id);

    // the same for the compressed rows of seq_id: until it holds n_blocks blocks of the kind
    bool comp_take(llama_seq_id seq_id, block_kind kind, uint32_t n_blocks);

    uint32_t comp_blocks(llama_seq_id seq_id, block_kind kind) const { return (uint32_t) seqs.at(seq_id).comp[kind].size(); }

    // the raw blocks of seq_id that hold no cell any more go back: they left the window; what an admitted request gives
    // back is charged again, it may need the blocks for its next tokens
    void seq_evict(llama_seq_id seq_id);

    // the caches that keep cells in the blocks: fn tells whether a block holds a used cell of the cache
    void add_user(const void * user, std::function<bool(uint32_t)> fn);
    void remove_user(const void * user);

    // copies of the compressed blocks and the states of sequences, between them or to host memory (block_bytes, state_bytes)
    void comp_copy(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, block_kind kind);
    void comp_get(llama_seq_id seq_id, block_kind kind, uint32_t i, void * data) const;
    void comp_set(llama_seq_id seq_id, block_kind kind, uint32_t i, const void * data);
    void state_copy(int32_t src, int32_t dst);
    void state_get(int32_t s, void * data) const;
    void state_set(int32_t s, const void * data);

    // the blocks of seq_id become free, except those whose cells a cache still holds for other sequences, which go back
    // when they are empty, see reclaim(); an admitted request keeps its blocks: they are what its admission took
    void seq_release(llama_seq_id seq_id);

    // blocks kept by seq_release() that are empty by now
    void reclaim();

    // everything free
    void clear();

    //
    // admission: a request takes its state and its input when it is admitted, and the most it may generate is charged
    // as a count, so that a running request never waits for memory
    //

    // the blocks of the input and of the most output of a request; the output continues in the last block of the input
    uint32_t blocks_input (uint32_t n_input) const;
    uint32_t blocks_output(uint32_t n_input, uint32_t n_output) const;

    // the blocks a sequence of n tokens takes, when they are not one per n_block_tokens tokens
    void set_blocks_fn(std::function<uint32_t(uint32_t)> fn) { blocks_fn = std::move(fn); }

    // admits a request of seq_id: takes the state (state_alloc_fn, when the seq holds none) and the input blocks the seq
    // does not hold yet, and charges its output blocks; false, with nothing taken, when they do not fit next to the
    // charge of the admitted requests
    bool reserve(llama_seq_id seq_id, uint32_t n_input, uint32_t n_output, const std::function<int32_t()> & state_alloc_fn);

    // the output blocks the admitted requests may still take
    uint32_t n_reserved() const { return reserved; }

    // the request of seq_id ended: its charge is dropped, its blocks stay with the seq
    void reserve_drop(llama_seq_id seq_id);

    // whether seq_id is an admitted request
    bool seq_admitted(llama_seq_id seq_id) const { return seqs.at(seq_id).admitted; }

private:
    std::vector<attn_layer> attn;
    std::vector<recr_layer> recr;

    std::vector<int32_t> recr_idx; // model layer -> recurrent layer index, -1 for the others

    uint32_t n_blocks = 0;

    size_t row_bytes      = 0; // the stride of the K and V rows, the longest of them
    size_t block_bytes    = 0;
    size_t rows_per_block = 0; // block_bytes/row_bytes

    uint32_t n_blocks_max = 0; // what the box was created with, the most finalize() takes it to

    bool row_lists = false;

    size_t row_bytes_draft      = 0; // one K or V row of the draft layers, the longest of them, 0 without them
    size_t rows_per_block_draft = 0;

    size_t n_attn_main = 0; // the attention layers before the draft layers

    const llama_hparams * draft_hparams   = nullptr;
    int32_t               draft_il_offset = 0;

    size_t r_row_bytes = 0; // padded to unit
    size_t s_row_bytes = 0;
    size_t x_row_bytes = 0;
    size_t c_row_bytes = 0;

    size_t   state_bytes  = 0;
    uint32_t state_blocks = 0;

    // offsets of the attention layers inside a block
    std::vector<size_t> off_k;
    std::vector<size_t> off_v;

    std::vector<attn_layer> lid;
    std::vector<size_t>     off_lid;

    std::vector<state_piece> pieces;
    std::vector<size_t>      off_p;

    std::function<uint32_t(uint32_t)> blocks_fn;

    // the offset of the pieces of each recurrent layer in a state
    std::vector<size_t> pc_r;
    std::vector<size_t> pc_s;
    std::vector<size_t> pc_x;
    std::vector<size_t> pc_c;

    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor *           pool = nullptr;

    ggml_backend_buffer_type_t buft = nullptr;

    std::vector<ggml_tensor *> t_k; // by model layer, nullptr for the others
    std::vector<ggml_tensor *> t_v;
    std::vector<ggml_tensor *> t_lid;
    std::vector<ggml_tensor *> t_p;

    ggml_tensor * t_r = nullptr;
    ggml_tensor * t_s = nullptr;
    ggml_tensor * t_x = nullptr;
    ggml_tensor * t_c = nullptr;

    // the free blocks, the rightmost taken first
    std::set<uint32_t> free_blocks;

    // the blocks a seq released while a cache still holds some of their cells, see seq_release()
    std::vector<uint32_t> held_blocks;

    std::vector<std::pair<const void *, std::function<bool(uint32_t)>>> users;

    bool block_used(uint32_t b) const;

    // the states on the left, and the free ones among them, the leftmost taken first
    uint32_t          n_states = 0;
    std::set<int32_t> free_states;

    struct seq_info {
        std::vector<uint32_t> blocks;

        // the blocks of the compressed rows, by kind, in the order of the rows
        std::vector<uint32_t> comp[BLOCK_KINDS];

        // an admitted request: the blocks its input takes (the blocks after them are output) and the output blocks it
        // may still take
        bool     admitted       = false;
        uint32_t n_input_blocks = 0;
        uint32_t reserve        = 0;
    };

    std::vector<seq_info> seqs;

    // the output blocks the admitted requests may still take, the sum of their reserves
    uint32_t reserved = 0;

    uint32_t block_pop();

    // a block of a seq was taken: after the input of an admitted request it is output, taken from its charge
    void block_charge(seq_info & seq);

    // the blocks the first n states cover
    uint32_t states_blocks(uint32_t n) const;

    ggml_tensor * view(ggml_type type, int64_t ne0, size_t nb1, size_t offs, const char * name, int32_t il);
};

// the elastic memory of a model; filter_attn selects the attention layers, and filter_recr the recurrent ones (nullptr
// for a model without them, which ignores the state parameters); kv_size cells and rs_size states size it when box_bytes
// is 0; the blocks also hold the layers of box_draft, see llama_box_draft; extra_attn holds pre-built layer descriptors
// for layers whose dimensions differ from the main hparams (e.g. an indexer cache with a different head size), appended
// after the main attention layers
std::unique_ptr<llama_box> llama_box_make(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                 uint32_t   kv_size,
                ggml_type   type_r,
                ggml_type   type_s,
                 uint32_t   rs_size,
                 uint32_t   n_rs_seq,
                     bool   rs_replay_req,
                     bool   offload,
                   size_t   box_bytes,
                     bool   box_deferred,
                     const llama_box_draft & box_draft,
    const std::function<bool(int32_t)> & filter_attn,
    const std::function<bool(int32_t)> & filter_recr,
    std::vector<llama_box::attn_layer>   extra_attn = {},
    std::vector<llama_box::attn_layer>   lid = {},
    std::vector<llama_box::state_piece>  pieces = {},
    const llama_hparams * hparams_attn = nullptr);
