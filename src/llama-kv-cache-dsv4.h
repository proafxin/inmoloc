#pragma once

#include "llama-box.h"
#include "llama-kv-cache.h"
#include "llama-kv-cache-iswa.h"

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

// the compressor state of a sequence is a state of the elastic memory, at pieces piece0, 2 per layer (kv, score)
class llama_dsv4_comp_state {
public:
    using layer_id = uint32_t;

    llama_dsv4_comp_state(
            llama_box * box,
            uint32_t    ratio,
            uint32_t    state_size,
            uint32_t    n_rs_seq,
            uint32_t    piece0,
        const std::vector<uint32_t> & ils);

    // row r of plane (0 is the state, the others are the rollback snapshots) of box state s
    int32_t box_row(int32_t s, uint32_t plane, uint32_t r) const;

    uint32_t get_ratio()      const { return ratio; }
    uint32_t get_state_size() const { return state_size; }
    uint32_t get_n_rs_seq()   const { return n_rs_seq; }

    const std::vector<layer_id> & get_ils() const { return ils; }

    // the pieces of the box that hold the states of every sequence
    ggml_tensor * get_kv_all   (int32_t il) const;
    ggml_tensor * get_score_all(int32_t il) const;

    ggml_tensor * cpy_kv   (ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const;
    ggml_tensor * cpy_score(ggml_context * ctx, ggml_tensor * cur, ggml_tensor * idxs, int32_t il) const;

private:
    struct layer {
        uint32_t il;

        ggml_tensor * kv;
        ggml_tensor * score;
    };

    const uint32_t ratio;
    const uint32_t state_size;
    const uint32_t n_rs_seq;

    llama_box * box    = nullptr;
    uint32_t    piece0 = 0;

    std::vector<layer_id> ils;

    std::vector<layer> layers;

    std::unordered_map<int32_t, int32_t> map_layer_ids;
};

//
// llama_kv_cache_dsv4
//

// DSV4 uses a normal raw/SWA token cache plus compressed K-only block caches.
// The compressed rows and the compressor states are in the elastic memory; the compressed caches of one cell keep the
// rotation matrices of the graph. DSV4-specific visibility and block planning are handled by
// llama_kv_cache_dsv4_context / llm_graph_input_dsv4.
// FIXME: we currently conflate token_pos and buffer contents. See https://github.com/ggml-org/llama.cpp/pull/25521#discussion_r3558173819

class llama_kv_cache_dsv4 : public llama_memory_i {
public:
    llama_kv_cache_dsv4(
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
                       size_t   box_bytes = 0,
                         bool   box_deferred = false,
                     uint32_t   n_batch = 0,
                         const llama_box_draft & box_draft = {});

    ~llama_kv_cache_dsv4() = default;

    //
    // llama_memory_i
    //

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

    void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const override;
    void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) override;

    std::shared_ptr<llama_box> get_box_shared() const override { return box; }

    bool box_reserve(llama_seq_id seq_id, uint32_t n_input, uint32_t n_output) override;
    void box_reserve_drop(llama_seq_id seq_id) override;
    void box_finalized() override;

    //
    // llama_kv_cache_dsv4 specific API
    //

    llama_box * get_box() const { return box.get(); }

    // the box state of a sequence, -1 without one
    int32_t get_seq_state(llama_seq_id seq_id) const { return seq_states.at(seq_id); }

    uint32_t get_n_swa() const { return hparams_raw.n_swa; }
    uint32_t get_n_top() const { return hparams_raw.indexer_top_k; }

    // the rows a graph is sized for: what a sequence of the model can hold, or the box when that is smaller
    uint32_t get_n_cells_max() const { return std::min(box->get_n_cells(), hparams_raw.n_ctx_train); }

    llama_kv_cache_iswa * get_raw() const;
    llama_kv_cache      * get_csa() const;
    llama_kv_cache      * get_hca() const;
    llama_kv_cache      * get_lid() const;
    llama_dsv4_comp_state * get_csa_state() const;
    llama_dsv4_comp_state * get_hca_state() const;
    llama_dsv4_comp_state * get_lid_state() const;

    uint32_t get_n_rs_seq() const;
    const std::vector<uint32_t> & get_rs_idx() const;
    void reset_rs_idx_for_ubatches(const std::vector<llama_ubatch> & ubatches);

    // the copies of shared compressed blocks the last box_prepare() made own blocks of: the first graph of the batch makes
    // them, see llama_box::comp_own()
    std::vector<std::pair<uint32_t, uint32_t>> take_block_copies() { return std::move(block_copies); }

private:
    std::vector<std::pair<uint32_t, uint32_t>> block_copies;

    llama_hparams hparams_raw;
    llama_hparams hparams_csa;
    llama_hparams hparams_hca;
    llama_hparams hparams_lid;

    uint32_t n_seq_max;
    const uint32_t n_rs_seq;

    std::vector<uint32_t> rs_idx;

    std::unique_ptr<llama_kv_cache_iswa> kv_raw;
    std::unique_ptr<llama_kv_cache>      kv_csa;
    std::unique_ptr<llama_kv_cache>      kv_hca;
    std::unique_ptr<llama_kv_cache>      kv_lid;
    std::unique_ptr<llama_dsv4_comp_state> csa_state;
    std::unique_ptr<llama_dsv4_comp_state> hca_state;
    std::unique_ptr<llama_dsv4_comp_state> lid_state;

    // frees the compressor state of seq_id (all for -1) unless an admitted request holds it
    void clear_compressed(llama_seq_id seq_id);

    // the elastic memory, see llama_box; it holds raw blocks for the window of a sequence and compressed blocks for all
    // of its tokens; the caches of the compressed rows only keep their rotation matrices then
    std::shared_ptr<llama_box> box;

    // the box state of each sequence, -1 without one
    std::vector<int32_t> seq_states;

    // gives the sequences of the ubatches their states and their compressed blocks, and frees what left the window
    bool box_prepare(const std::vector<llama_ubatch> & ubatches);

    bool box_state_take(llama_seq_id seq_id);

    void box_state_write(llama_io_write_i & io, llama_seq_id seq_id, bool partial_only) const;
    void box_state_read (llama_io_read_i  & io, llama_seq_id seq_id, bool partial_only);

    // copies between sequences for seq_cp()
    void box_seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst);
};

// the hparams of a K-only cache, which a cache keeps a reference to: a base of the draft cache, to outlive its construction
struct llama_dsv4_draft_hparams {
    llama_hparams hparams_draft;

    explicit llama_dsv4_draft_hparams(const llama_hparams & hparams);
};

// the MTP draft of the elastic memory: its K rows are in the raw blocks of its target, for the same sequences, so it takes
// no memory of its own and gives up its cells as they leave the window, which lets the target give the blocks back
class llama_kv_cache_dsv4_draft : private llama_dsv4_draft_hparams, public llama_kv_cache_iswa {
public:
    llama_kv_cache_dsv4_draft(
            const llama_model & model,
                    ggml_type   type_k,
                    ggml_type   type_v,
                         bool   offload,
                     uint32_t   n_seq_max,
                     uint32_t   n_ubatch,
                     uint32_t   n_rs_seq,
        const layer_filter_cb & filter,
        const std::shared_ptr<llama_box> & box);
};

// DSV4 raw attention only uses the SWA half of kv_raw. The base half is kept
// for generic ISWA bookkeeping, but it has no DSV4 layers to expose here.
class llama_kv_cache_dsv4_raw_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    llama_kv_cache_dsv4_raw_context(llama_kv_cache_iswa * kv);

    llama_kv_cache_dsv4_raw_context(
            llama_kv_cache_iswa * kv,
            llama_context * lctx,
            bool optimize);

    llama_kv_cache_dsv4_raw_context(
            llama_kv_cache_iswa * kv,
            slot_info_vec_t sinfos_base,
            slot_info_vec_t sinfos_swa,
            std::vector<llama_ubatch> ubatches);

    bool next() override;
    bool apply() override;

    llama_memory_status get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    uint32_t get_n_kv() const;

    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;

    // the raw cells a token of the ubatch can read, sorted by position: the elastic memory
    void seq_rows(llama_seq_id seq_id, bool is_2d, std::vector<int32_t> & rows, std::vector<llama_pos> & pos) const;

    ggml_tensor * build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const;
    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;

    void set_input_k_idxs(ggml_tensor * dst) const;
    void set_input_k_rot(ggml_tensor * dst) const;

private:
    size_t i_next = 0;

    llama_kv_cache * kv_swa = nullptr;

    slot_info_vec_t sinfos;
    std::vector<llama_ubatch> ubatches;

    const llama_memory_context_ptr ctx_base_mem;
    const llama_memory_context_ptr ctx_swa_mem;

    uint32_t n_kv = 0;

    const llama_memory_status status;
};

// DSV4 compressed KV rows are graph outputs, not normal token KV writes.
// Keep a small context that exposes K tensors without generic apply() semantics.
class llama_kv_cache_dsv4_comp_context {
public:
    // the rows are in the box (the indexer rows when lid), kv only holds the rotation matrix; n_kv is the most rows a
    // sequence holds
    llama_kv_cache_dsv4_comp_context(llama_kv_cache * kv, llama_box * box, bool lid, uint32_t n_kv);

    uint32_t get_n_kv() const;

    ggml_tensor * get_k(ggml_context * ctx, int32_t il) const;
    ggml_tensor * cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const;

    ggml_tensor * build_input_k_rot(ggml_context * ctx) const;
    void set_input_k_rot(ggml_tensor * dst) const;

private:
    llama_box * box = nullptr;
    bool        lid = false;

    llama_kv_cache * kv;

    uint32_t n_kv;
};

class llama_kv_cache_dsv4_context : public llama_memory_context_i {
public:
    using slot_info_vec_t = llama_kv_cache::slot_info_vec_t;

    struct comp_plan {
        // Per-ubatch recipe for updating compressor state, committing completed
        // compressed rows, and masking the compressed attention source.

        // APE row ids, i.e. pos % ratio, for the compressor-state updates.
        std::vector<int32_t> state_pos;

        // Current-ubatch source row ids and unique persistent-state
        // destination row ids for deterministic ring-state updates.
        std::vector<int32_t> state_persist_src_idxs;
        std::vector<int32_t> state_persist_dst_idxs;

        // Device-side rollback restore copies snapshot planes back to the
        // current compressor-state plane before the graph reads it.
        std::vector<int32_t> state_restore_src_idxs;

        // Device-side rollback snapshots copy rows from the graph-local
        // [persistent_state | current_ubatch_scratch] tensor into rollback
        // planes after the graph has computed current-token compressor state.
        std::vector<int32_t> state_snapshot_src_idxs;
        std::vector<int32_t> state_snapshot_dst_idxs;

        // Flattened source row ids used for state-backed commits. Source rows
        // index the graph-local [persistent_state | current_ubatch_scratch]
        // tensor. For overlapped compression the first half is previous rows
        // and the second half is current rows; a final synthetic zero/-inf row
        // may be addressed for the first block's previous half.
        std::vector<int32_t> state_read_idxs;

        // Final compressed-cache row ids written by state-backed commits.
        // A non-boundary CSA/LID decode step can target a masked scratch row.
        std::vector<int64_t> state_write_idxs;

        // RoPE positions for state-backed commits.
        std::vector<int32_t> state_write_pos;

        // Number of completed compressed rows visible for each query token.
        std::vector<int32_t> n_visible;

        // The first row each query token can see, nothing for 0; the elastic
        // memory, where the indexer list holds the sequences of the ubatch one after the other
        std::vector<int32_t> n_skip;

        // Graph-width for compressed rows. This can be larger than n_visible
        // so masked padding rows do not force a new graph at every CSA block.
        int64_t n_kv = 0;
    };

    llama_kv_cache_dsv4_context(llama_memory_status status);

    llama_kv_cache_dsv4_context(
            llama_kv_cache_dsv4 * kv);

    llama_kv_cache_dsv4_context(
            llama_kv_cache_dsv4 * kv,
            llama_context * lctx,
            bool optimize);

    llama_kv_cache_dsv4_context(
            llama_kv_cache_dsv4 * kv,
            slot_info_vec_t sinfos_raw_base,
            slot_info_vec_t sinfos_raw_swa,
            std::vector<llama_ubatch> ubatches);

    virtual ~llama_kv_cache_dsv4_context();

    //
    // llama_memory_context_i
    //

    bool next()  override;
    bool apply() override;

    llama_memory_status  get_status() const override;
    const llama_ubatch & get_ubatch() const override;

    //
    // llama_kv_cache_dsv4_context specific API
    //

    const llama_kv_cache_dsv4_raw_context * get_raw() const;
    const llama_kv_cache_dsv4_comp_context * get_csa() const;
    const llama_kv_cache_dsv4_comp_context * get_hca() const;
    const llama_kv_cache_dsv4_comp_context * get_lid() const;
    const llama_dsv4_comp_state       * get_csa_state() const;
    const llama_dsv4_comp_state       * get_hca_state() const;
    const llama_dsv4_comp_state       * get_lid_state() const;

    const comp_plan & get_csa_plan() const;
    const comp_plan & get_hca_plan() const;
    const comp_plan & get_lid_plan() const;

    const comp_plan & get_csa_plan(const llama_ubatch & ubatch) const;
    const comp_plan & get_hca_plan(const llama_ubatch & ubatch) const;
    const comp_plan & get_lid_plan(const llama_ubatch & ubatch) const;

    // the elastic memory
    llama_box * get_box() const { return kv_box; }

    // the size of the lists of an attention of the elastic memory: raw rows, then compressed rows
    uint32_t get_n_raw() const;
    uint32_t get_box_top() const { return kv_dsv4->get_n_top(); }
    int64_t  get_n_hca(const llama_ubatch & ubatch) const;

    // the lists of the attentions of the elastic memory, see llm_graph_input_dsv4::box_input:
    //   raw:     I32 [n_raw, n_tokens], the raw rows a token reads, at the end of its slots
    //   hca:     I32 [n_hca, n_tokens], the HCA rows a token reads, the live ones first
    //   lid:     I32 [n_lid, 1], the CSA and indexer rows of the sequences, one after the other
    //   rng_*:   I32 [3, n_tokens], the ranges of a token in its lists [raw|hca] and [raw|top-k]
    //   win_lid: I32 [3, n_tokens], the window of a token in lid, see ggml_lightning_indexer_rows(), nullptr without one
    void set_input_box(ggml_tensor * raw, ggml_tensor * hca, ggml_tensor * lid, ggml_tensor * rng_raw, ggml_tensor * rng_hca, ggml_tensor * rng_csa,
            ggml_tensor * win_lid, uint32_t n_top, const llama_ubatch * ubatch) const;

    // the entries of the window of a token in lid: the most CSA rows a token of the ubatch sees, padded so that graphs can
    // be reused; a reserve takes the most a sequence holds
    int64_t get_n_win_lid(const llama_ubatch & ubatch) const;

    // the copies of shared compressed blocks (block, own block) the graph of the ubatch makes before it writes, the first
    // one of the batch only, see llama_kv_cache_dsv4::take_block_copies()
    const std::vector<std::pair<uint32_t, uint32_t>> & get_block_copies() const;

private:
    llama_kv_cache_dsv4 * kv_dsv4 = nullptr;
    llama_box * kv_box = nullptr;

    size_t i_next = 0;

    std::vector<std::pair<uint32_t, uint32_t>> block_copies;

    std::vector<llama_ubatch> ubatches;

    std::vector<comp_plan> plans_csa;
    std::vector<comp_plan> plans_hca;
    std::vector<comp_plan> plans_lid;

    const std::unique_ptr<llama_kv_cache_dsv4_raw_context> ctx_raw;
    const llama_memory_context_ptr ctx_csa_mem;
    const llama_memory_context_ptr ctx_hca_mem;
    const llama_memory_context_ptr ctx_lid_mem;

    const std::unique_ptr<llama_kv_cache_dsv4_comp_context> ctx_csa;
    const std::unique_ptr<llama_kv_cache_dsv4_comp_context> ctx_hca;
    const std::unique_ptr<llama_kv_cache_dsv4_comp_context> ctx_lid;

    llama_dsv4_comp_state * csa_state = nullptr;
    llama_dsv4_comp_state * hca_state = nullptr;
    llama_dsv4_comp_state * lid_state = nullptr;

    static std::unique_ptr<llama_kv_cache_dsv4_comp_context> comp_context(
            llama_kv_cache_dsv4 * kv, llama_kv_cache * kv_comp, bool lid, uint32_t ratio);

    bool reserve_plans = false;
    mutable comp_plan reserve_plan_csa;
    mutable comp_plan reserve_plan_hca;
    mutable comp_plan reserve_plan_lid;

    const llama_memory_status status;
};
