#pragma once

#include "llama.h"

#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <set>
#include <utility>
#include <vector>

struct ggml_context;
struct ggml_tensor;

// the elastic memory of a hybrid model: one buffer, the box, holds the attention KV and the recurrent states of all the
// requests, handed out by offset as the requests come and go
//
// the states sit on the left of the box, one after the other, each exactly as large as a state is; the rest is a row of
// blocks of the same size, taken from the right: a block holds n_block_tokens tokens of every attention layer, laid out
// by layer inside the block (the K rows of the first layer, its V rows, then the next layer); with an MTP draft on the
// same model, the rows of its layer follow: a draft context reads and writes them for the same sequences, so its tokens
// take no memory of their own, see llama_kv_cache::box_ref
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
class llama_box {
public:
    struct attn_layer {
        int32_t   il;
        ggml_type type_k;
        ggml_type type_v;
        uint32_t  n_embd_k;
        uint32_t  n_embd_v;
        bool      has_v;
    };

    struct recr_layer {
        int32_t   il;
        ggml_type type_r;
        ggml_type type_s;
        uint32_t  n_embd_r;
        uint32_t  n_embd_s;
        uint32_t  n_embd_x; // the cached replay inputs of the last tokens, 0 without replay
        uint32_t  n_rows_r; // conv rows: one per draft position
    };

    static constexpr uint32_t n_block_tokens = 128;

    // the row unit of the recurrent views, which every state row is padded to
    static constexpr size_t unit = 256;

    // whether a model with these layers can use the box: one row size for all K and V rows and one size per kind of
    // state row
    static bool supported(const std::vector<attn_layer> & attn, const std::vector<recr_layer> & recr, const attn_layer * draft);

    // a box of `bytes` bytes, or with the memory of a cache of n_tokens cells and n_states states when bytes is 0; draft
    // is the layer of an MTP draft that keeps its tokens in the blocks, nullptr for none; no memory is taken until alloc()
    llama_box(std::vector<attn_layer> attn, std::vector<recr_layer> recr, const attn_layer * draft, size_t bytes, uint32_t n_tokens, uint32_t n_states, uint32_t n_seq_max);

    // takes the buffer from buft and creates the views; a dry run (no_alloc) takes none
    void alloc(ggml_backend_buffer_type_t buft, bool no_alloc);

    //
    // layout
    //

    size_t   get_block_bytes()  const { return block_bytes; }
    uint32_t get_n_blocks()     const { return n_blocks; }
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

    // row of KV cell c in the K and V views of the draft layer
    int64_t kv_row_draft(uint32_t c) const {
        return (int64_t) (c/n_block_tokens)*rows_per_block_draft + c%n_block_tokens;
    }

    bool has_draft() const { return row_bytes_draft > 0; }

    // unit index of a row of the state s, recurrent layer l (recr_index()): conv row k, the scan row, replay input half h
    int64_t r_row(int32_t s, uint32_t l, uint32_t k) const;
    int64_t s_row(int32_t s, uint32_t l) const;
    int64_t x_row(int32_t s, uint32_t l, uint32_t h) const;

    // views of the whole box
    ggml_tensor * get_k(int32_t il) const;
    ggml_tensor * get_v(int32_t il) const;
    ggml_tensor * get_r() const { return t_r; } // [n_embd_r, units]
    ggml_tensor * get_s() const { return t_s; } // [n_embd_s, units]
    ggml_tensor * get_x() const { return t_x; } // [n_embd_x, units], nullptr without replay

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

    // the caches that keep cells in the blocks: fn tells whether a block holds a used cell of the cache
    void add_user(const void * user, std::function<bool(uint32_t)> fn);
    void remove_user(const void * user);

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

    size_t row_bytes      = 0; // one K or V row, the same for all layers
    size_t block_bytes    = 0;
    size_t rows_per_block = 0; // block_bytes/row_bytes

    size_t row_bytes_draft      = 0; // one K or V row of the draft layer, 0 without one
    size_t rows_per_block_draft = 0;

    size_t r_row_bytes = 0; // padded to unit
    size_t s_row_bytes = 0;
    size_t x_row_bytes = 0;

    size_t   state_bytes  = 0;
    uint32_t state_blocks = 0;

    // offsets of the attention layers inside a block
    std::vector<size_t> off_k;
    std::vector<size_t> off_v;

    // the offset of the pieces of each recurrent layer in a state
    std::vector<size_t> pc_r;
    std::vector<size_t> pc_s;
    std::vector<size_t> pc_x;

    ggml_context_ptr        ctx;
    ggml_backend_buffer_ptr buf;
    ggml_tensor *           pool = nullptr;

    std::vector<ggml_tensor *> t_k; // by model layer, nullptr for the others
    std::vector<ggml_tensor *> t_v;

    ggml_tensor * t_r = nullptr;
    ggml_tensor * t_s = nullptr;
    ggml_tensor * t_x = nullptr;

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

    // the blocks the first n states cover
    uint32_t states_blocks(uint32_t n) const;

    ggml_tensor * view(ggml_type type, int64_t ne0, size_t nb1, size_t offs, const char * name, int32_t il);
};
