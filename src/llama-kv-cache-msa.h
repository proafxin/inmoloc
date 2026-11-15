#pragma once

#include "llama-kv-cache.h"

#include <vector>

// llama_kv_cache_msa

// uses two instances of llama_kv_cache, one for K/V tensors, and one for the MSA indexer tensors
// both receive identical sequence operations and identical ubatches, so their cell layouts stay in synced.
// the context lays out the rows of the sequences of a ubatch in the elastic memory, see set_input_box()

class llama_kv_cache_msa : public llama_memory_i {
public:
    llama_kv_cache_msa(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   offload,
                     uint32_t   kv_size,
                     uint32_t   n_seq_max,
                     uint32_t   n_pad,
                     uint32_t   n_swa,
               llama_swa_type   swa_type,
        const layer_filter_cb & filter,
        const layer_filter_cb & filter_idx,
        const  layer_reuse_cb & reuse,
                       size_t   box_bytes = 0,
                         bool   box_deferred = false);

    ~llama_kv_cache_msa() = default;

    // llama_memory_i

    llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) override;

    llama_memory_context_ptr init_full() override;

    llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) override;

    bool get_can_shift() const override;

    void clear(bool data) override;

    bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) override;
    void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) override;
    void seq_keep(llama_seq_id seq_id)                                                          override;
    void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) override;
    void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) override;

    llama_pos seq_pos_min(llama_seq_id seq_id) const override;
    llama_pos seq_pos_max(llama_seq_id seq_id) const override;

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const override;

    // state write/load

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    // llama_kv_cache_msa specific API

    llama_kv_cache * get_base() const;
    llama_kv_cache * get_idx () const;

    // the elastic memory, owned by the main cache: the indexer keeps its keys in the same blocks
    std::shared_ptr<llama_box> get_box_shared() const override { return kv_base->get_box_shared(); }
    bool box_reserve(llama_seq_id seq_id, uint32_t n_input, uint32_t n_output) override { return kv_base->box_reserve(seq_id, n_input, n_output); }
    void box_reserve_drop(llama_seq_id seq_id) override { kv_base->box_reserve_drop(seq_id); }
    void box_finalized() override { kv_base->box_finalized(); kv_idx->box_finalized(); }

    uint32_t       get_n_pad()    const { return n_pad; }
    uint32_t       get_n_seq_max() const { return n_seq_max; }
    uint32_t       get_n_swa()    const { return n_swa; }
    llama_swa_type get_swa_type() const { return swa_type; }
    uint32_t       get_n_ctx_train() const { return hparams_idx.n_ctx_train; }

private:
    // keep the indexer KV cache hparams instance here as llama_kv_cache stores only a reference
    llama_hparams hparams_idx;

    const uint32_t n_seq_max = 1;
    const uint32_t n_pad     = 1;

    const uint32_t       n_swa    = 0;
    const llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE;

    std::unique_ptr<llama_kv_cache> kv_base;
    std::unique_ptr<llama_kv_cache> kv_idx;
};

class llama_kv_cache_msa_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    // used for errors
    llama_kv_cache_msa_context(llama_memory_status status);

    // used to create a full-cache context
    llama_kv_cache_msa_context(
            llama_kv_cache_msa * kv);

    // used to create an update context
    llama_kv_cache_msa_context(
            llama_kv_cache_msa * kv,
            llama_context * lctx,
            bool optimize);

    // used to create a batch processing context from a batch
    llama_kv_cache_msa_context(
            llama_kv_cache_msa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_idx,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_msa_context();

    // llama_memory_context_i

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    // llama_kv_cache_msa_context specific API

    const llama_kv_cache_context * get_base() const;
    const llama_kv_cache_context * get_idx () const;

    // the elastic memory: the list of the slots of the sequences of the ubatch, one after the other, and the window of a
    // token in it, which starts at the slots of its sequence; the cell of rank r of a sequence (causal order) is the slot
    // base + p0 % blk + r, p0 its first position, so that the blocks are those of the positions while they follow each
    // other; a reserve takes the most there can be
    //   n_list: the slots of the list, a whole number of blocks
    //   n_win:  the slots of a window, a whole number of blocks
    void box_layout(const llama_ubatch & ubatch, int blk, int64_t & n_list, int64_t & n_win) const;

    //   rows:   I32 [n_list, 1], the row of each slot
    //   win:    I32 [3, n_tokens], the window of a token in rows and the slots it sees, see ggml_lightning_indexer_rows()
    //   off:    I32 [1, 1, n_tokens], the block of rows the window starts at
    //   bias:   F32 [n_win/blk, n_tokens], the blocks of the window a token always reads
    //   q_base: I32 [2, n_tokens], the list of a token and where its range starts
    void set_input_box(ggml_tensor * rows, ggml_tensor * win, ggml_tensor * off, ggml_tensor * bias, ggml_tensor * q_base,
            const llama_ubatch * ubatch, int blk, int local) const;

private:
    llama_kv_cache_msa * kv;

    // the index of the next ubatch to process
    size_t i_next = 0;

    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_base;
    const llama_memory_context_ptr ctx_idx;

    const llama_memory_status status;
};
