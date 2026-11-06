#include "llama-kv-cache.h"

#include "llama-box.h"
#include "llama-impl.h"
#include "llama-io.h"
#include "llama-model.h"
#include "llama-context.h"

#include <algorithm>
#include <bitset>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

static bool ggml_is_power_of_2(int n) {
    return (n & (n - 1)) == 0;
}

// orthonormal Walsh-Hadamard rotation matrix
// note: res^2 == I
static void ggml_gen_hadamard(ggml_tensor * tensor) {
    assert(tensor->type == GGML_TYPE_F32);

    const int n = tensor->ne[0];

    assert(ggml_is_power_of_2(n));
    assert(tensor->ne[1] == n);
    assert(tensor->ne[2] == 1);
    assert(tensor->ne[3] == 1);

    std::vector<float> data_f32;

    float * data = (float *) tensor->data;

    if (tensor->type != GGML_TYPE_F32) {
        data_f32.resize(n*n);
        data = data_f32.data();
    }

    data[0*n + 0] = 1.0 / sqrtf(n);

    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < s; i++) {
            for (int j = 0; j < s; j++) {
                const float val = data[i*n + j];

                data[(i + s)*n + (j    )] =  val;
                data[(i    )*n + (j + s)] =  val;
                data[(i + s)*n + (j + s)] = -val;
            }
        }
    }

    if (tensor->type != GGML_TYPE_F32) {
        ggml_quantize_chunk(tensor->type, data, tensor->data, 0, 1, n*n, nullptr);
    }
}

//
// llama_kv_cache
//

llama_kv_cache::llama_kv_cache(
        const llama_model & model,
        const llama_hparams & hparams,
                ggml_type   type_k,
                ggml_type   type_v,
                     bool   offload,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
                 uint32_t   n_pad,
                 uint32_t   n_swa,
           llama_swa_type   swa_type,
           llama_memory_t   mem_other,
    const layer_filter_cb & filter,
    const  layer_reuse_cb & reuse,
    const  layer_share_cb & share,
             const char *   name_tag,
                llama_box *   box_in,
                  int32_t     box_il_offset_in,
    std::shared_ptr<llama_box>  box_ref_in,
    std::shared_ptr<llama_box>  box_own_in) :
    model(model), hparams(hparams), box(box_ref_in ? box_ref_in.get() : (box_own_in ? box_own_in.get() : box_in)),
    box_il_offset(box_il_offset_in),
    box_ref(std::move(box_ref_in)), box_own(std::move(box_own_in)),
    n_seq_max(n_seq_max), n_pad(n_pad), n_swa(n_swa), swa_type(swa_type),
    other(static_cast<llama_kv_cache *>(mem_other)),
    cells_impl(other ? other->cells_impl : std::make_shared<llama_kv_cells>()),
    cells(*cells_impl) {
    // with the elastic memory, the sequences are what fits in it
    if (box) {
        this->n_seq_max = box->get_n_seq_max();
    }

    // shared cells view the source cache's K/V tensors, so the cell count
    // follows the source allocation: a fitted target can be smaller than the
    // draft default and oversized views would overflow the source tensors
    if (other) {
        const uint32_t size_other = other->get_size();
        if (kv_size != size_other) {
            LLAMA_LOG_WARN("%s: kv_size = %u overridden to %u to match the shared source cache\n", __func__, kv_size, size_other);
            kv_size = size_other;
        }

        // the cells of the source are in its elastic memory, read in its rows: this cache only reads them
        GGML_ASSERT(box == nullptr);
        box = other->box;
        if (box) {
            this->n_seq_max = box->get_n_seq_max();
        }
    }

    GGML_ASSERT(kv_size % n_pad == 0);

    // the cells of the box: every cell a row of the box
    GGML_ASSERT(!box || kv_size == box->get_n_cells());

    // the cells of a shared cache are those of its source, which holds the blocks
    if (box && !other) {
        box->add_user(this, [this](uint32_t b) { return box_block_used(b); });
    }

    const uint32_t n_layer = hparams.n_layer_all;

    // define a comparator for the buft -> ctx map to ensure that the order is well-defined:
    struct ggml_backend_buft_comparator {
        bool operator()(const ggml_backend_buffer_type_t & lhs, const ggml_backend_buffer_type_t & rhs) const {
            return strcmp(ggml_backend_buft_name(lhs), ggml_backend_buft_name(rhs)) < 0;
        }
    };
    std::map<ggml_backend_buffer_type_t, ggml_context_ptr, ggml_backend_buft_comparator> ctx_map;

    // create a context for each buffer type
    auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
        auto it = ctx_map.find(buft);
        if (it == ctx_map.end()) {
            ggml_init_params params = {
                /*.mem_size   =*/ size_t(2u*n_layer*ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            ggml_context * ctx = ggml_init(params);
            if (!ctx) {
                return nullptr;
            }

            ctx_map.emplace(buft, ctx);

            return ctx;
        }

        return it->second.get();
    };

    cells.resize(kv_size);

    const bool is_mla = hparams.is_mla();

    for (uint32_t il = 0; il < n_layer; il++) {
        if (!hparams.has_kv(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: does not have KV cache\n", __func__, il);
            continue;
        }

        if (filter && !filter(il)) {
            LLAMA_LOG_DEBUG("%s: layer %3d: filtered\n", __func__, il);
            continue;
        }

        if (share && other) {
            const int32_t il_share = share(il);

            if (il_share >= 0) {
                const auto & layer_share = other->layers[other->map_layer_ids[il_share]];

                LLAMA_LOG_WARN("%s: layer %3d: sharing with layer %d. k = %p, v = %p\n", __func__, il, il_share,
                        layer_share.k->data, layer_share.v->data);

                map_layer_ids[il] = layers.size();

                layers.push_back(layer_share);
                layers.back().il = il;

                continue;
            }
        }

        if (n_embd_head_k_all == 0) {
            n_embd_head_k_all = (int32_t) hparams.n_embd_head_k(il);
        } else if (n_embd_head_k_all > 0 && n_embd_head_k_all != (int32_t) hparams.n_embd_head_k(il)) {
            n_embd_head_k_all = -1;
        }

        if (!is_mla) {
            if (n_embd_head_v_all == 0) {
                n_embd_head_v_all = (int32_t) hparams.n_embd_head_v(il);
            } else if (n_embd_head_v_all > 0 && n_embd_head_v_all != (int32_t) hparams.n_embd_head_v(il)) {
                n_embd_head_v_all = -1;
            }
        }

        // [TAG_V_CACHE_VARIABLE] the V rows are not transposed, each layer holds its own size
        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);
        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

        const char * dev_name = "CPU";

        ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();

        if (offload) {
            auto * dev = model.dev_layer(il);
            buft = ggml_backend_dev_buffer_type(dev);

            dev_name = ggml_backend_dev_name(dev);
        }

        LLAMA_LOG_DEBUG("%s: layer %3d: dev = %s\n", __func__, il, dev_name);

        ggml_context * ctx = ctx_for_buft(buft);
        if (!ctx) {
            throw std::runtime_error("failed to create ggml context for kv cache");
        }

        const bool has_k = true;
        const bool has_v = !is_mla;

        if (box) {
            // views of the whole box, read and written through the rows of the cells, see cell_row()
            ggml_tensor * k = box->get_k(il + box_il_offset);
            ggml_tensor * v = box->get_v(il + box_il_offset);
            GGML_ASSERT(k && (v || !has_v) && k->type == type_k && k->ne[0] == n_embd_k_gqa);
            GGML_ASSERT(!has_v || (v->type == type_v && v->ne[0] == n_embd_v_gqa));

            map_layer_ids[il] = layers.size();

            layers.push_back({ il, k, v });
            continue;
        }

        ggml_tensor * k = has_k ? ggml_new_tensor_2d(ctx, type_k, n_embd_k_gqa, kv_size) : nullptr;
        ggml_tensor * v = has_v ? ggml_new_tensor_2d(ctx, type_v, n_embd_v_gqa, kv_size) : nullptr;

        has_k && ggml_format_name(k, "cache_%sk_l%d", name_tag, il);
        has_v && ggml_format_name(v, "cache_%sv_l%d", name_tag, il);

        map_layer_ids[il] = layers.size();

        layers.push_back({ il, k, v });
    }

    if (reuse) {
        LLAMA_LOG_DEBUG("%s: reusing layers:\n", __func__);

        for (uint32_t il = 0; il < n_layer; il++) {
            const int32_t il_reuse = reuse(il);

            if (il_reuse < 0) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: no reuse\n", __func__, il);
                continue;
            }

            if (filter && !filter(il)) {
                LLAMA_LOG_DEBUG("%s: - layer %3d: filtered\n", __func__, il);
                continue;
            }

            GGML_ASSERT(map_layer_ids.find(il_reuse) != map_layer_ids.end());

            map_layer_ids[il] = map_layer_ids[il_reuse];

            LLAMA_LOG_DEBUG("%s: - layer %3d: reuse layer %d, is_swa = %d\n", __func__, il, il_reuse, hparams.is_swa(il));
        }
    }

    // allocate tensors and initialize the buffers to avoid NaNs in the padding; the box holds the cells of a box cache
    if (box) {
        ctx_map.clear();
    }
    for (auto & [buft, ctx] : ctx_map) {
        ggml_backend_buffer_t buf;
        if (hparams.no_alloc) {
            buf = ggml_backend_buft_alloc_buffer(buft, /*size =*/ 0); // dummy buffer
            for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
                t->buffer = buf; // set dummy buffer for KV cache so that the backend scheduler won't try to allocate it
            }
        } else {
            buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft); // real buffer
        }
        if (!buf) {
            throw std::runtime_error("failed to allocate buffer for kv cache");
        }

        LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", __func__, ggml_backend_buffer_name(buf), ggml_backend_buffer_get_size(buf)/1024.0/1024.0);

        ggml_backend_buffer_clear(buf, 0);
        ctxs_bufs.emplace_back(std::move(ctx), buf);
    }

    {
        const size_t memory_size_k = size_k_bytes();
        const size_t memory_size_v = size_v_bytes();

        LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__,
                (float)(memory_size_k + memory_size_v) / (1024.0f * 1024.0f), kv_size, (int) layers.size(), n_seq_max,
                ggml_type_name(type_k), (float)memory_size_k / (1024.0f * 1024.0f),
                ggml_type_name(type_v), (float)memory_size_v / (1024.0f * 1024.0f));
    }

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        n_embd_head_k_all = other->n_embd_head_k_all;
        n_embd_head_v_all = other->n_embd_head_v_all;

        attn_rot_k = other->attn_rot_k;
        attn_rot_v = other->attn_rot_v;
    } else {
        const char * LOCAL_INFERENCE_ATTN_ROT_DISABLE = getenv("LOCAL_INFERENCE_ATTN_ROT_DISABLE");
        const bool attn_rot_disable = LOCAL_INFERENCE_ATTN_ROT_DISABLE ? atoi(LOCAL_INFERENCE_ATTN_ROT_DISABLE) : false;
        if (attn_rot_disable) {
            LLAMA_LOG_WARN("%s: attention rotation force disabled (LOCAL_INFERENCE_ATTN_ROT_DISABLE)\n", __func__);
        }

        attn_rot_k =
            !attn_rot_disable &&
            n_embd_head_k_all > 0 &&
            ggml_is_quantized(type_k) &&
            hparams.n_embd_head_k() % 64 == 0;

        // always create Hadamard rotation tensors for DeepSeek lightning indexers
        if ((model.arch == LLM_ARCH_DEEPSEEK32 || model.arch == LLM_ARCH_DEEPSEEK4 ||
                model.arch == LLM_ARCH_GLM_DSA || model.arch == LLM_ARCH_DOTS3NOTE) &&
                hparams.n_embd_head_k_full == hparams.indexer_head_size) {
            attn_rot_k = true;
        }

        attn_rot_v =
            !attn_rot_disable &&
            n_embd_head_v_all > 0 &&
            ggml_is_quantized(type_v) &&
            hparams.n_embd_head_v() % 64 == 0;
    }

    LLAMA_LOG_INFO("%s: attn_rot_k = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_k, n_embd_head_k_all);
    LLAMA_LOG_INFO("%s: attn_rot_v = %d, n_embd_head_k_all = %d\n", __func__, attn_rot_v, n_embd_head_v_all);

    // pre-compute the haramard matrices and keep them in host memory
    // TODO: in the future, we can make copies in the backend buffers to avoid host -> device transfers
    if (attn_rot_k || attn_rot_v) {
        for (int64_t n = 64; n <= std::max(n_embd_head_k_all, n_embd_head_v_all); n *= 2) {
            attn_rot_hadamard[n] = std::vector<float>(n*n);

            ggml_init_params params = {
                /* .mem_size   = */ 1*ggml_tensor_overhead(),
                /* .mem_buffer = */ nullptr,
                /* .no_alloc   = */ true,
            };

            ggml_context_ptr ctx { ggml_init(params) };

            ggml_tensor * tmp = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, n, n);
            tmp->data = attn_rot_hadamard[n].data();

            ggml_gen_hadamard(tmp);
        }
    }

    const char * LOCAL_INFERENCE_KV_CACHE_DEBUG = getenv("LOCAL_INFERENCE_KV_CACHE_DEBUG");
    debug = LOCAL_INFERENCE_KV_CACHE_DEBUG ? atoi(LOCAL_INFERENCE_KV_CACHE_DEBUG) : 0;
}

llama_kv_cache::~llama_kv_cache() {
    if (box && !other) {
        box->remove_user(this);
    }
}

void llama_kv_cache::clear(bool data) {
    cells.reset();
    head = 0;

    if (data) {
        for (auto & [_, buf] : ctxs_bufs) {
            ggml_backend_buffer_clear(buf.get(), 0);
        }
    }

    if (box_own) {
        box_own->clear();
    }
}

bool llama_kv_cache::seq_rm(llama_seq_id seq_id, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (uint32_t) seq_id < n_seq_max));

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (seq_id >= 0) {
            if (!cells.seq_has(i, seq_id) || !cells.seq_rm(i, seq_id)) {
                continue;
            }
        } else {
            // match any sequence
            cells.rm(i);
        }

        if (new_head == cells.size()) {
            new_head = i;
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }

    if (box_own) {
        // all positions of a seq: its blocks go back to the box, except those holding cells of other seqs
        if (p0 == 0 && p1 == std::numeric_limits<llama_pos>::max()) {
            for (llama_seq_id s = seq_id >= 0 ? seq_id : 0; s < (seq_id >= 0 ? seq_id + 1 : (llama_seq_id) n_seq_max); ++s) {
                box_own->seq_release(s);
            }
        }

        // blocks held for other seqs may be empty now
        box_own->reclaim();
    }

    return true;
}

uint32_t llama_kv_cache::attn_n_cells_max() const {
    return get_size();
}

void llama_kv_cache::box_finalized() {
    if (box) {
        cells.resize(box->get_n_cells());
        head = 0;
    }
}

bool llama_kv_cache::box_reserve(llama_seq_id seq_id, uint32_t n_input, uint32_t n_output) {
    GGML_ASSERT(box_own);

    // no recurrent layers, no state
    return box_own->reserve(seq_id, n_input, n_output, nullptr);
}

void llama_kv_cache::box_reserve_drop(llama_seq_id seq_id) {
    GGML_ASSERT(box_own);

    box_own->reserve_drop(seq_id);
}

void llama_kv_cache::seq_cp(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id_src >= 0 && (uint32_t) seq_id_src < n_seq_max);
    GGML_ASSERT(seq_id_dst >= 0 && (uint32_t) seq_id_dst < n_seq_max);

    // the cells are shared by the sequences: no data copy, only the cells meta data
    if (seq_id_src == seq_id_dst) {
        return;
    }

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id_src)) {
            cells.seq_add(i, seq_id_dst);
        }
    }
}

void llama_kv_cache::seq_keep(llama_seq_id seq_id) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max);

    uint32_t new_head = cells.size();

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_keep(i, seq_id)) {
            if (new_head == cells.size()) {
                new_head = i;
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    if (new_head != cells.size() && new_head < head) {
        head = new_head;
    }

    if (box_own) {
        for (llama_seq_id s = 0; s < (llama_seq_id) n_seq_max; ++s) {
            if (s != seq_id) {
                box_own->seq_release(s);
            }
        }
    }
}

void llama_kv_cache::seq_add(llama_seq_id seq_id, llama_pos p0, llama_pos p1, llama_pos shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max);
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_add() is only supported for n_pos_per_embd() == 1");

    if (shift == 0) {
        return;
    }

    uint32_t new_head = cells.size();

    if (p0 < 0) {
        p0 = 0;
    }

    if (p1 < 0) {
        p1 = std::numeric_limits<llama_pos>::max();
    }

    // If there is no range then return early to avoid looping over all cells.
    if (p0 == p1) {
        return;
    }

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            if (cells.pos_add(i, shift)) {
                if (new_head == cells.size()) {
                    new_head = i;
                }
            }
        }
    }

    // If we freed up a slot, set head to it so searching can start there.
    // Otherwise we just start the next search from the beginning.
    head = new_head != cells.size() ? new_head : 0;
}

void llama_kv_cache::seq_div(llama_seq_id seq_id, llama_pos p0, llama_pos p1, int d) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max);
    GGML_ASSERT(hparams.n_pos_per_embd() == 1 && "seq_div() is only supported for n_pos_per_embd() == 1");

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

    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.pos_in(i, p0, p1)) {
            continue;
        }

        if (cells.seq_has(i, seq_id)) {
            cells.pos_div(i, d);
        }
    }
}

llama_pos llama_kv_cache::seq_pos_min(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_min(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max);

    return cells.seq_pos_min(seq_id);
}

llama_pos llama_kv_cache::seq_pos_max(llama_seq_id seq_id) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return other->seq_pos_max(seq_id);
    }

    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max);

    return cells.seq_pos_max(seq_id);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_kv_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> ret;
    for (const auto & [ctx, buf] : ctxs_bufs) {
        ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf.get());

        if (hparams.no_alloc) {
            GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) == nullptr);
            ret[buft] += ggml_backend_alloc_ctx_tensors_from_buft_size(ctx.get(), buft);
        } else {
            // GGML_ASSERT(ggml_backend_buffer_get_base(buf.get()) != nullptr); // multi_buffer does not have a defined base
            ret[buft] += ggml_backend_buffer_get_size(buf.get());
        }
    }

    if (box_own) {
        ret[ggml_backend_buffer_get_type(box_own->get_buffer())] += box_own->get_total_bytes();
    }

    return ret;
}

llama_memory_context_ptr llama_kv_cache::init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) {
    GGML_UNUSED(embd_all);

    do {
        balloc.split_reset();

        std::vector<llama_ubatch> ubatches;
        while (true) {
            auto ubatch = balloc.split_simple(n_ubatch);

            if (ubatch.n_tokens == 0) {
                break;
            }

            ubatches.push_back(std::move(ubatch)); // NOLINT
        }

        if (balloc.get_n_used() < balloc.get_n_tokens()) {
            // failed to find a suitable split
            break;
        }

        auto sinfos = prepare(ubatches);
        if (sinfos.empty()) {
            break;
        }

        return std::make_unique<llama_kv_cache_context>(
                this, std::move(sinfos), std::move(ubatches));
    } while (false);

    return std::make_unique<llama_kv_cache_context>(LLAMA_MEMORY_STATUS_FAILED_PREPARE);
}

llama_memory_context_ptr llama_kv_cache::init_full() {
    return std::make_unique<llama_kv_cache_context>(this);
}

llama_memory_context_ptr llama_kv_cache::init_update(llama_context * lctx, bool optimize) {
    GGML_UNUSED(optimize);

    bool do_shift = get_has_shift();

    return std::make_unique<llama_kv_cache_context>(this, lctx, do_shift);
}

llama_kv_cache::slot_info_vec_t llama_kv_cache::prepare(const std::vector<llama_ubatch> & ubatches) {
    llama_kv_cache::slot_info_vec_t res;

    // a draft context, or the cache that owns the box, fills the blocks of its sequences: a sequence short of empty cells
    // takes a block
    if (box_ref || box_own) {
        std::map<llama_seq_id, uint32_t> n_new;
        std::map<llama_seq_id, llama_pos> pos_first;
        for (const auto & ubatch : ubatches) {
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const llama_seq_id seq_id = ubatch.seq_id[i][0];

                n_new[seq_id]++;

                const auto it = pos_first.emplace(seq_id, ubatch.pos[i]).first;
                it->second = std::min(it->second, ubatch.pos[i]);
            }
        }

        for (const auto & [seq_id, p] : pos_first) {
            if (box_keep > 0 && p > (llama_pos) box_keep) {
                seq_rm(seq_id, 0, p - (llama_pos) box_keep);
            }
        }

        for (const auto & [seq_id, n] : n_new) {
            while (box_seq_free(seq_id) < n) {
                if (!box->block_take(seq_id)) {
                    LLAMA_LOG_WARN("%s: no free block in the box for the tokens of seq %d\n", __func__, seq_id);
                    return res;
                }
            }
        }
    }

    struct state_t {
        slot_info sinfo; // slot info for the ubatch

        uint32_t head_old; // old position of the head, before placing the ubatch

        llama_kv_cells cells; // copy of the old cells, before placing the ubatch
    };

    // remember the old state of the cells so we can restore it in the end
    std::vector<state_t> states;

    bool success = true;

    for (const auto & ubatch : ubatches) {
        // only find a suitable slot for the ubatch. don't modify the cells yet
        const auto sinfo_new = find_slot(ubatch, false);
        if (sinfo_new.empty()) {
            success = false;
            break;
        }

        // remember the position that we found
        res.push_back(sinfo_new);

        // store the old state of the cells in the recovery stack
        states.push_back({ sinfo_new, head, cells.cp(sinfo_new.idxs) });

        // now emplace the ubatch
        apply_ubatch(sinfo_new, ubatch);
    }

    GGML_ASSERT(!states.empty() || !success);

    // iterate backwards and restore the cells to their original state
    for (auto it = states.rbegin(); it != states.rend(); ++it) {
        cells.set(it->sinfo.idxs, it->cells);
        head = it->head_old;
    }

    if (!success) {
        return {};
    }

    return res;
}

bool llama_kv_cache::update(llama_context * lctx, bool do_shift) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return true;
    }

    bool updated = false;

    auto * sched = lctx->get_sched();

    if (do_shift) {
        if (!get_can_shift()) {
            GGML_ABORT("The current KV cache / model configuration does not support K-shift");
        }

        LLAMA_LOG_DEBUG("%s: applying K-shift\n", __func__);

        // apply K-shift if needed
        if (hparams.rope_type != LLAMA_ROPE_TYPE_NONE) {
            ggml_backend_sched_reset(sched);

            auto * res = lctx->get_gf_res_reserve();

            res->reset();

            auto * gf = build_graph_shift(res, lctx);
            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                LLAMA_LOG_ERROR("%s: failed to allocate compute graph for K-shift\n", __func__);
                return updated;
            }

            res->set_inputs(nullptr);

            if (lctx->graph_compute(gf, false) != GGML_STATUS_SUCCESS) {
                LLAMA_LOG_ERROR("%s: failed to compute K-shift\n", __func__);
                return updated;
            }

            updated = true;
        }

        cells.reset_shift();
    }

    return updated;
}

// places the tokens of the ubatch so that the cells of each sequence form runs: a token takes the cell after the last cell
// of its sequence, else the first cell of the next empty chunk; a sequence holds the rest of its chunk this way, until no
// chunk is empty: then a token takes any empty cell from head_cur, also in the rest of the chunk of another sequence
// returns false when the cells cannot hold the ubatch
static bool find_slot_chunked(const llama_ubatch & ubatch, const llama_kv_cells & cells, uint32_t head_cur, std::vector<uint32_t> & idxs) {
    constexpr uint32_t chunk_size = llama_kv_cells::chunk_size;

    const uint32_t n_cells = cells.size();

    // the cells and the chunks taken by the tokens before, and the last cell of each sequence
    std::unordered_set<uint32_t> taken;
    std::unordered_set<uint32_t> chunks_taken;
    std::unordered_map<llama_seq_id, uint32_t> last;

    const auto is_free = [&](uint32_t i) {
        return cells.is_empty(i) && taken.find(i) == taken.end();
    };
    const auto chunk_empty = [&](uint32_t c) {
        return cells.chunk_n_used(c) == 0 && chunks_taken.find(c) == chunks_taken.end();
    };

    // the empty chunks are taken in order
    uint32_t chunk_next = 0;
    uint32_t n_tested   = 0;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch.seq_id[i][0];

        int64_t idx = -1;

        auto it = last.find(seq_id);
        if (it == last.end() && cells.seq_n_cells(seq_id) > 0) {
            it = last.emplace(seq_id, cells.seq_pos_cells(seq_id).rbegin()->second).first;
        }
        if (it != last.end()) {
            const uint32_t next = it->second + 1;
            if (next < n_cells && is_free(next) && (next/chunk_size == it->second/chunk_size || chunk_empty(next/chunk_size))) {
                idx = next;
            }
        }

        if (idx < 0) {
            while (chunk_next < cells.n_chunks() && !chunk_empty(chunk_next)) {
                chunk_next++;
            }
            if (chunk_next < cells.n_chunks()) {
                idx = chunk_next*chunk_size;
            }
        }

        while (idx < 0 && n_tested < n_cells) {
            if (head_cur >= n_cells) {
                head_cur = 0;
            }
            if (is_free(head_cur)) {
                idx = head_cur;
            }
            head_cur++;
            n_tested++;
        }

        if (idx < 0) {
            return false;
        }

        taken.insert(idx);
        chunks_taken.insert(idx/chunk_size);
        last[seq_id] = idx;

        idxs.push_back(idx);
    }

    return true;
}

static_assert(llama_box::n_block_tokens % llama_kv_cells::chunk_size == 0, "a block of the box is whole chunks of the cells");

bool llama_kv_cache::find_slot_box(const llama_ubatch & ubatch, std::vector<uint32_t> & idxs) const {
    constexpr uint32_t n_block = llama_box::n_block_tokens;

    // per sequence, the block of its list and the slot in it to look at next: the tokens of a sequence fill its blocks
    // in order, an empty cell left by a removed token included
    std::unordered_map<llama_seq_id, std::pair<size_t, uint32_t>> cursor;

    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch.seq_id[i][0];

        const auto & blocks = box->seq_blocks(seq_id);

        auto & [ip, slot] = cursor[seq_id];

        int64_t idx = -1;
        while (idx < 0 && ip < blocks.size()) {
            const uint32_t c = blocks[ip]*n_block + slot;
            if (++slot == n_block) {
                slot = 0;
                ++ip;
            }
            if (cells.is_empty(c)) {
                idx = c;
            }
        }

        if (idx < 0) {
            // the memory takes the blocks before the cells are searched, see llama_memory_hybrid::box_prepare()
            LLAMA_LOG_ERROR("%s: the blocks of seq %d hold no empty cell\n", __func__, seq_id);
            return false;
        }

        idxs.push_back((uint32_t) idx);
    }

    return true;
}

int64_t llama_kv_cache::cell_row(uint32_t c) const {
    return box ? (box_ref ? box->kv_row_draft(c) : box->kv_row(c)) : (int64_t) c;
}

// the used cells of block b of the box, from the counts of its chunks of cells
static uint32_t box_block_n_used(const llama_kv_cells & cells, uint32_t b) {
    constexpr uint32_t n_chunks = llama_box::n_block_tokens/llama_kv_cells::chunk_size;

    uint32_t n = 0;
    for (uint32_t c = 0; c < n_chunks; ++c) {
        n += cells.chunk_n_used(b*n_chunks + c);
    }
    return n;
}

bool llama_kv_cache::box_block_used(uint32_t b) const {
    return box_block_n_used(cells, b) > 0;
}

uint32_t llama_kv_cache::box_seq_free(llama_seq_id seq_id) const {
    uint32_t n = 0;
    for (const uint32_t b : box->seq_blocks(seq_id)) {
        n += llama_box::n_block_tokens - box_block_n_used(cells, b);
    }
    return n;
}

void llama_kv_cache::cell_row_runs(uint32_t c0, uint32_t c1, const std::function<void(int64_t, uint32_t)> & fn) const {
    if (!box) {
        if (c1 > c0) {
            fn(c0, c1 - c0);
        }
        return;
    }

    constexpr uint32_t n_block = llama_box::n_block_tokens;

    while (c0 < c1) {
        const uint32_t n = std::min(c1, (c0/n_block + 1)*n_block) - c0;
        fn(cell_row(c0), n);
        c0 += n;
    }
}

llama_kv_cache::slot_info llama_kv_cache::find_slot(const llama_ubatch & ubatch, bool cont) const {

    if (debug > 0) {
        LLAMA_LOG_DEBUG("%s: n = %5d, used = %5d, head = %5d, size = %5d, n_swa = %5d\n",
                __func__, cells.used_max_p1(), cells.get_used(), head, get_size(), n_swa);

        if ((debug == 2 && n_swa > 0) || debug > 2) {
            std::string ss;
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (cells.is_empty(i)) {
                    ss += '.';
                } else {
                    assert(cells.seq_count(i) >= 1);

                    if (cells.seq_count(i) == 1) {
                        ss += std::to_string(cells.seq_get(i));
                    } else {
                        ss += 'M';
                    }
                }
                if (i%256 == 255) {
                    ss += " *";
                    ss += '\n';
                }
            }
            LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
        }

        if ((debug == 2 && n_swa > 0) || debug > 2) {
            std::string ss;
            for (uint32_t i = 0; i < cells.size(); ++i) {
                std::string cur;
                if (cells.is_empty(i)) {
                    cur = '.';
                } else {
                    cur = std::to_string(cells.pos_get(i));
                }
                const int n = cur.size();
                for (int j = 0; j < 5 - n; ++j) {
                    cur += ' ';
                }
                ss += cur;
                if (i%256 == 255) {
                    ss += " *";
                }
                if (i%64 == 63) {
                    ss += '\n';
                }
            }
            LLAMA_LOG_DEBUG("\n%s\n", ss.c_str());
        }

        for (int s = 0; s < (int) n_seq_max; ++s) {
            if (cells.seq_pos_min(s) < 0) {
                continue;
            }

            LLAMA_LOG_DEBUG("%s: min[%d] = %5d, max[%d] = %5d\n", __func__, s, cells.seq_pos_min(s), s, cells.seq_pos_max(s));
        }
    }

    const uint32_t n_tokens = ubatch.n_tokens;

    slot_info res;
    res.idxs.reserve(n_tokens);

    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    // a shared cache writes no cells (see apply_ubatch()): its tokens take no slot
    if (other) {
        res.idxs.assign(n_tokens, 0);
        return res;
    }

    uint32_t head_cur = head;

    // if we have enough unused cells before the current head ->
    //   better to start searching from the beginning of the cache, hoping to fill it
    if (head_cur > cells.get_used() + 2*n_tokens) {
        head_cur = 0;
    }

    if (n_tokens > cells.size()) {
        LLAMA_LOG_ERROR("%s: n_tokens = %d > size = %u\n", __func__, n_tokens, cells.size());
        return { };
    }

    // the cells of the box: each token takes a cell in the blocks of its sequence
    if (box) {
        if (!find_slot_box(ubatch, res.idxs)) {
            return { };
        }
        return res;
    }

    // keep the cells of each sequence together; a SWA cache reuses the masked cells of a sequence in place
    if (!cont && swa_type == LLAMA_SWA_TYPE_NONE) {
        if (!find_slot_chunked(ubatch, cells, head_cur, res.idxs)) {
            return { };
        }
        return res;
    }

    {
        uint32_t n_tested = 0;

        // for continuous slots, we test that all tokens in the ubatch fit, starting from the current head
        // for non-continuous slots, we test the tokens one by one
        const uint32_t n_test = cont ? n_tokens : 1;

        while (true) {
            if (head_cur + n_test > cells.size()) {
                n_tested += cells.size() - head_cur;
                head_cur = 0;
                continue;
            }

            for (uint32_t i = 0; i < n_test; i++) {
                const auto idx = head_cur;

                head_cur++;
                n_tested++;

                //const llama_pos    pos    = ubatch.pos[i];
                //const llama_seq_id seq_id = ubatch.seq_id[i][0];

                // can we use this cell? either:
                //  - the cell is empty
                //  - the cell is occupied only by one sequence:
                //    - (disabled) mask causally, if the sequence is the same as the one we are inserting
                //    - mask SWA, using current max pos for that sequence in the cache
                //                always insert in the cell with minimum pos
                bool can_use = cells.is_empty(idx);

                if (!can_use && cells.seq_count(idx) == 1) {
                    const llama_pos pos_cell = cells.pos_get(idx);

                    // (disabled) causal mask
                    // note: it's better to purge any "future" tokens beforehand
                    //if (cells.seq_has(idx, seq_id)) {
                    //    can_use = pos_cell >= pos;
                    //}

                    if (!can_use) {
                        const llama_seq_id seq_id_cell = cells.seq_get(idx);

                        // SWA mask
                        if (llama_hparams::is_masked_swa(n_swa, swa_type, pos_cell, cells.seq_pos_max(seq_id_cell) + 1)) {
                            can_use = true;
                        }
                    }
                }

                if (can_use) {
                    res.idxs.push_back(idx);
                } else {
                    if (cont) {
                        break;
                    }
                }
            }

            if (res.idxs.size() == n_tokens) {
                break;
            }

            if (cont) {
                res.idxs.clear();
            }

            if (n_tested >= cells.size()) {
                //LLAMA_LOG_ERROR("%s: failed to find a slot for %d tokens\n", __func__, n_tokens);
                return { };
            }
        }

        // we didn't find a suitable slot - return empty result
        if (res.idxs.size() < n_tokens) {
            return { };
        }
    }

    return res;
}

void llama_kv_cache::apply_ubatch(const slot_info & sinfo, const llama_ubatch & ubatch) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    // keep track of the max sequence position that we would overwrite with this ubatch
    // for non-SWA cache, this would be always empty
    std::vector<llama_seq_id> seq_pos_max_rm(n_seq_max);
    for (uint32_t s = 0; s < n_seq_max; ++s) {
        seq_pos_max_rm[s] = -1;
    }

    assert(ubatch.n_tokens == sinfo.size());

    for (uint32_t i = 0; i < sinfo.size(); ++i) {
        const auto idx = sinfo.idxs[i];

        if (!cells.is_empty(idx)) {
            assert(cells.seq_count(idx) == 1);

            const llama_seq_id seq_id = cells.seq_get(idx);
            const llama_pos    pos    = cells.pos_get(idx);

            seq_pos_max_rm[seq_id] = std::max(seq_pos_max_rm[seq_id], pos);

            cells.rm(idx);
        }

        cells.pos_set(idx, ubatch.pos[i]);

        if (ubatch.is_pos_2d() || ubatch.token || hparams.ple_n_heads > 0) {
            llama_kv_cell_ext ext;

            if (ubatch.is_pos_2d()) {
                ext.x = ubatch.pos[i + ubatch.n_tokens*2];
                ext.y = ubatch.pos[i + ubatch.n_tokens];
            }

            if (ubatch.token) {
                ext.tok = ubatch.token[i];
            } else if (hparams.ple_n_heads > 0) {
                // embd batch (multimodal input) has no token ids, need to pad it with the correct ID for PLE layers
                // TODO @ngxson : check if we can do the same as gemma 3n / gemma 4
                ext.tok = hparams.ple_image_token_id != 0
                    ? (llama_token) hparams.ple_image_token_id
                    : (llama_token) hparams.ple_eos_token_id;
            }

            cells.ext_set(idx, ext);
        }

        for (int32_t s = 0; s < ubatch.n_seq_id[i]; s++) {
            cells.seq_add(idx, ubatch.seq_id[i][s]);
        }
    }

    // note: we want to preserve the invariant that all positions between [pos_min, pos_max] for each sequence
    //       will be present in the cache. so we have to purge any position which is less than those we would overwrite
    //       ref: https://github.com/ggml-org/llama.cpp/pull/13746#issuecomment-2916057092
    for (uint32_t s = 0; s < n_seq_max; ++s) {
        if (seq_pos_max_rm[s] == -1) {
            continue;
        }

        if (cells.seq_pos_min(s) <= seq_pos_max_rm[s]) {
            LLAMA_LOG_DEBUG("%s: purging positions [%d, %d] of sequence %d from KV cache\n",
                    __func__, cells.seq_pos_min(s), seq_pos_max_rm[s], s);

            seq_rm(s, cells.seq_pos_min(s), seq_pos_max_rm[s] + 1);
        }
    }

    // move the head at the end of the slot
    head = sinfo.idxs.back() + 1;
}

bool llama_kv_cache::get_can_shift() const {
    // the rows of a sequence in the box are not one view to shift
    if (box) {
        return false;
    }
    // Step35 uses per-layer RoPE dims; K-shift assumes a single global n_rot.
    if (model.arch == LLM_ARCH_STEP35) {
        return false;
    }
    if (hparams.n_pos_per_embd() > 1) {
        return false;
    }
    return true;
}

uint32_t llama_kv_cache::get_size() const {
    return cells.size();
}

llama_kv_cache::attn_stats llama_kv_cache::get_attn_stats() const {
    return stats_attn;
}

void llama_kv_cache::attn_read_counted_by_graph() const {
    stats_read_by_graph = true;
}

void llama_kv_cache::add_attn_read(uint64_t n_cells) const {
    stats_attn.read += n_cells;
}

void llama_kv_cache::get_usage(uint32_t & n_used, uint32_t & n_span) const {
    n_used = cells.get_used();
    n_span = cells.used_max_p1();
}

bool llama_kv_cache::get_has_shift() const {
    return cells.get_has_shift();
}

ggml_type llama_kv_cache::type_k() const {
    return layers[0].k->type;
}

ggml_type llama_kv_cache::type_v() const {
    return layers[0].v->type;
}

std::vector<uint32_t> llama_kv_cache::get_layer_ids() const {
    std::vector<uint32_t> res;
    res.reserve(layers.size());

    for (const auto & layer : layers) {
        res.push_back(layer.il);
    }

    return res;
}

ggml_tensor * llama_kv_cache::get_k_storage(int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    return layers[ikv].k;
}

const llama_kv_cells & llama_kv_cache::get_cells() const {
    return cells;
}

uint32_t llama_kv_cache::get_n_kv() const {
    // the cells of the box are only read through lists, the view they would take is the whole box
    if (box) {
        return get_size();
    }

    // pad the n_kv value so that the graph remains constant across batches and can be reused
    // note: this also helps some backends with performance (f.ex https://github.com/ggml-org/llama.cpp/pull/16812#issuecomment-3455112220)
    const uint32_t n_pad_cur = std::max(n_pad, 256u);

    return std::min(cells.size(), std::max(n_pad_cur, GGML_PAD(cells.used_max_p1(), n_pad_cur)));
}

ggml_tensor * llama_kv_cache::get_k(ggml_context * ctx, int32_t il, uint32_t n_kv) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * k = layers[ikv].k;

    // the rows of the box, of every block; the attention reads them through the lists of the cells
    if (box) {
        GGML_UNUSED(n_kv);
        return ggml_view_4d(ctx, k,
                hparams.n_embd_head_k(il), hparams.n_head_kv(il), k->ne[1], 1,
                ggml_row_size(k->type, hparams.n_embd_head_k(il)),
                k->nb[1], k->nb[1]*k->ne[1], 0);
    }

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_k_gqa = k->ne[0];

    assert(n_embd_k_gqa == hparams.n_embd_k_gqa(il));

    return ggml_view_4d(ctx, k,
            hparams.n_embd_head_k(il), hparams.n_head_kv(il), n_kv, 1,
            ggml_row_size(k->type, hparams.n_embd_head_k(il)),
            ggml_row_size(k->type, n_embd_k_gqa),
            ggml_row_size(k->type, n_embd_k_gqa*kv_size),
            0);
}

ggml_tensor * llama_kv_cache::get_v(ggml_context * ctx, int32_t il, uint32_t n_kv) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    if (box) {
        return ggml_view_4d(ctx, v,
                hparams.n_embd_head_v(il), hparams.n_head_kv(il), v->ne[1], 1,
                ggml_row_size(v->type, hparams.n_embd_head_v(il)),
                v->nb[1], v->nb[1]*v->ne[1], 0);
    }

    const uint64_t kv_size      = get_size();
    const uint64_t n_embd_v_gqa = v->ne[0];

    // [TAG_V_CACHE_VARIABLE]
    assert(n_embd_v_gqa >= hparams.n_embd_v_gqa(il));

    return ggml_view_4d(ctx, v,
            hparams.n_embd_head_v(il), hparams.n_head_kv(il), n_kv, 1,
            ggml_row_size(v->type, hparams.n_embd_head_v(il)),
            ggml_row_size(v->type, n_embd_v_gqa),
            ggml_row_size(v->type, n_embd_v_gqa*kv_size),
            0);
}

// runs shorter than this are decode steps, read through their ranges
static constexpr uint32_t attn_gather_min_tokens = 32;

// each gather run adds an attention op per layer, this bounds the graph size and the compute buffer: the outputs of the
// runs are joined in token order, more runs take more memory
static constexpr uint32_t attn_gather_max_runs = 4;

bool llama_kv_cache::attn_gather_ok(bool in_place) const {
    GGML_UNUSED(in_place);
    return swa_type == LLAMA_SWA_TYPE_NONE;
}

std::vector<llama_kv_attn_run> llama_kv_cache::get_attn_runs(const llama_ubatch & ubatch, uint32_t n_kv, bool in_place, bool rest_indexed) const {
    GGML_UNUSED(n_kv);

    const uint32_t min_tokens = attn_gather_min_tokens;
    const uint32_t max_runs   = attn_gather_max_runs;

    if (!attn_gather_ok(in_place)) {
        return {};
    }

    // the box is read through lists only: a prompt of one sequence is gathered in place (up to max_runs of them), every
    // other token reads the cells of its sequence through indexed ranges
    {
        GGML_ASSERT(box && in_place && rest_indexed);

        std::vector<llama_kv_attn_run> res;
        uint32_t n_gather = 0;

        for (uint32_t t0 = 0; t0 < ubatch.n_tokens;) {
            const bool single = ubatch.n_seq_id[t0] == 1;
            const llama_seq_id seq_id = ubatch.seq_id[t0][0];

            uint32_t t1 = t0 + 1;
            while (t1 < ubatch.n_tokens && ubatch.n_seq_id[t1] == ubatch.n_seq_id[t0] && ubatch.seq_id[t1][0] == seq_id) {
                t1++;
            }

            llama_kv_attn_run run = { t0, t1, false, seq_id, 0 };

            if (single && t1 - t0 >= min_tokens && n_gather < max_runs) {
                run.gather = true;
                run.n_idx  = std::max<uint32_t>(256, GGML_PAD(cells.seq_n_cells(seq_id), 256));
                n_gather++;
            } else {
                run.indexed = true;
                run.seq_id  = -1;
            }

            // neighboring indexed runs share the lists
            if (run.indexed && !res.empty() && res.back().indexed) {
                res.back().t1 = run.t1;
            } else {
                res.push_back(run);
            }

            t0 = t1;
        }

        return res;
    }
}

// the largest gather runs a ubatch of this size can have, for the compute buffers reserved at startup: one run of one
// sequence over the whole ubatch, its attention takes the most scratch (e.g. the partial results of the CUDA kernels
// grow with the tokens of a run); several sequences: the most runs over the fewest tokens each, the rest indexed
std::vector<llama_kv_attn_run> llama_kv_cache::get_attn_runs_reserve(const llama_ubatch & ubatch, uint32_t n_kv, bool in_place, bool rest_indexed) const {
    if (!attn_gather_ok(in_place)) {
        return {};
    }

    GGML_ASSERT(box && in_place && rest_indexed);

    // a run is gathered up to all the cells of the view
    const uint32_t n_idx  = GGML_PAD(n_kv, 256);
    const uint32_t n_runs = std::min(attn_gather_max_runs, ubatch.n_tokens/attn_gather_min_tokens);

    if (n_runs == 0) {
        return { { 0, ubatch.n_tokens, false, -1, 0, true } };
    }
    if (ubatch.n_seqs_unq == 1) {
        return { { 0, ubatch.n_tokens, true, 0, n_idx } };
    }

    std::vector<llama_kv_attn_run> res;
    for (uint32_t r = 0; r < n_runs; ++r) {
        const uint32_t t0 = r*attn_gather_min_tokens;
        res.push_back({ t0, t0 + attn_gather_min_tokens, true, 0, n_idx });
    }
    if (n_runs*attn_gather_min_tokens < ubatch.n_tokens) {
        res.push_back({ n_runs*attn_gather_min_tokens, ubatch.n_tokens, false, -1, 0, true });
    }
    return res;
}

void llama_kv_cache::set_input_attn_run(ggml_tensor * idxs, ggml_tensor * hi, const llama_ubatch * ubatch, const llama_kv_attn_run & run) const {
    GGML_ASSERT(run.gather);
    GGML_ASSERT(ggml_backend_buffer_is_host(idxs->buffer));
    GGML_ASSERT(run.t1 <= ubatch->n_tokens);

    const int64_t n_idx = run.n_idx;

    GGML_ASSERT(idxs->ne[0] == n_idx);

    auto * dst_idxs = (int32_t *) idxs->data;

    // the rows of the sequence in the order of the causal mask, each token sees a prefix of them (see
    // ggml_flash_attn_ext_set_kv_rows_hi()); a padding entry of -1 is left out
    {
        GGML_ASSERT(run.seq_id >= 0);
        GGML_ASSERT(ggml_backend_buffer_is_host(hi->buffer));
        GGML_ASSERT(hi->ne[0] >= (int64_t) (run.t1 - run.t0));

        std::vector<int32_t>   cid;
        std::vector<llama_pos> pos;
        seq_cells_causal(run.seq_id, ubatch->is_pos_2d(), cid, pos);

        const int32_t n = (int32_t) cid.size();
        GGML_ASSERT(n <= n_idx);

        for (int64_t c = 0; c < n_idx; ++c) {
            dst_idxs[c] = c < n ? (int32_t) cell_row(cid[c]) : -1;
        }

        llama_pos seq_pos_min = std::numeric_limits<llama_pos>::max();
        for (uint32_t t = run.t0; t < run.t1; ++t) {
            seq_pos_min = std::min(seq_pos_min, ubatch->pos[t]);
        }

        auto * dst_hi = (int32_t *) hi->data;
        std::fill(dst_hi, dst_hi + hi->ne[0], 0);

        for (uint32_t t = run.t0; t < run.t1; ++t) {
            int32_t lo = 0;
            int32_t hi_t = 0;
            causal_range(ubatch, t, true, cid.data(), pos.data(), n, seq_pos_min, lo, hi_t);
            GGML_ASSERT(lo == 0); // no sliding window, see attn_gather_ok()

            dst_hi[t - run.t0] = hi_t;

            if (!stats_read_by_graph) {
                stats_attn.read += hi_t;
            }
            stats_attn.range += n;
            stats_attn.owned += n;
        }
    }
}

ggml_tensor * llama_kv_cache::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    ggml_tensor * k = layers[ikv].k;

    const int64_t n_embd_head = k_cur->ne[0];
    const int64_t n_head      = k_cur->ne[1];
    const int64_t n_tokens    = k_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    // TODO: add ggml helper function for this?
    GGML_ASSERT(ggml_row_size(k_cur->type, n_embd_head) == k_cur->nb[1]);

    k_cur = ggml_view_2d(ctx, k_cur, n_embd_gqa, n_tokens, k_cur->nb[2], 0);

    // store the current K values into the cache
    return ggml_set_rows(ctx, k, k_cur, k_idxs);
}

ggml_tensor * llama_kv_cache::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    const int32_t ikv = map_layer_ids.at(il);

    auto * v = layers[ikv].v;

    const int64_t n_embd_head = v_cur->ne[0];
    const int64_t n_head      = v_cur->ne[1];
    const int64_t n_tokens    = v_cur->ne[2];

    const int64_t n_embd_gqa = n_embd_head*n_head;

    // we can merge dims 0 and 1
    GGML_ASSERT(ggml_row_size(v_cur->type, n_embd_head) == v_cur->nb[1]);

    v_cur = ggml_view_2d(ctx, v_cur, n_embd_gqa, n_tokens, v_cur->nb[2], 0);

    return ggml_set_rows(ctx, v, v_cur, v_idxs);
}

ggml_tensor * llama_kv_cache::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * k_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(k_idxs);

    return k_idxs;
}

ggml_tensor * llama_kv_cache::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    ggml_tensor * v_idxs = ggml_new_tensor_1d(ctx, GGML_TYPE_I64, n_tokens);

    ggml_set_input(v_idxs);

    return v_idxs;
}

ggml_tensor * llama_kv_cache::build_input_k_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_k) {
        int nrot = 64;

        // TODO: investigate if using the smallest rotation matrix is beneficial also for K (similar as for V)
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4141323088
        do {
            nrot *= 2;
        } while (n_embd_head_k_all % nrot == 0);
        nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_k_rot");
    }

    return res;
}

ggml_tensor * llama_kv_cache::build_input_v_rot(ggml_context * ctx) const {
    ggml_tensor * res = nullptr;

    if (attn_rot_v) {
        int nrot = 64;
        // using smaller rotation matrices for V seems beneficial
        // ref: https://github.com/ggml-org/llama.cpp/pull/21038#issuecomment-4146397570
        //do {
        //    nrot *= 2;
        //} while (hparams.n_embd_head_v() % nrot == 0);
        //nrot /= 2;

        res = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, nrot, nrot);
        ggml_set_input(res);
        ggml_set_name(res, "attn_inp_v_rot");
    }

    return res;
}

void llama_kv_cache::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    GGML_ASSERT(ubatch->n_tokens == sinfo.size());

    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));
    int64_t * data = (int64_t *) dst->data;

    for (uint32_t i = 0; i < sinfo.size(); ++i) {
        data[i] = cell_row(sinfo.idxs[i]);
    }
}

void llama_kv_cache::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch, const slot_info & sinfo) const {
    set_input_k_idxs(dst, ubatch, sinfo);
}

void llama_kv_cache::set_input_k_shift(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    int32_t * data = (int32_t *) dst->data;

    for (uint32_t i = 0; i < cells.size(); ++i) {
        data[i] = cells.is_empty(i) ? 0 : cells.get_shift(i);
    }
}

uint32_t llama_kv_cache::kv_idx_planes() const {
    // ALiBi and the relative bias by distance (T5) read the positions of the rows
    return hparams.use_alibi || hparams.n_rel_attn_bkts > 0 ? 2 : 1;
}

void llama_kv_cache::get_kv_idx_shape(const llama_ubatch & ubatch, uint32_t & n_idx, uint32_t & n_group) const {
    std::vector<bool> seen(n_seq_max, false);

    // one list of the cells of all the sequences of the ubatch, one after the other, each token reading a range of it
    // (see set_input_kv_idx()); sized by the cells read, whatever the number of sequences
    {
        uint32_t n_all = 0;
        for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
            const llama_seq_id seq_id = ubatch.seq_id[i][0];
            if (!seen[seq_id]) {
                seen[seq_id] = true;
                n_all += cells.seq_n_cells(seq_id);
            }
        }

        // a cell shared by several sequences of the ubatch is in each of their lists
        n_idx   = std::max<uint32_t>(256, GGML_PAD(n_all, 256));
        n_group = 1;
    }
}

void llama_kv_cache::seq_cells_causal(llama_seq_id seq_id, bool is_2d, std::vector<int32_t> & cells_out, std::vector<llama_pos> & pos_out) const {
    const auto & sp = cells.seq_pos_cells(seq_id);

    cells_out.clear();
    pos_out.clear();
    cells_out.reserve(sp.size());
    pos_out.reserve(sp.size());

    for (const auto & [p, c] : sp) {
        cells_out.push_back((int32_t) c);
        pos_out.push_back(p);
    }

    // M-RoPE: cells with the same position are ordered by (y, x), the order of the 2D causal mask
    if (is_2d) {
        const int32_t n = (int32_t) cells_out.size();
        for (int32_t a = 0; a < n;) {
            int32_t b = a + 1;
            while (b < n && pos_out[b] == pos_out[a]) {
                b++;
            }
            if (b - a > 1) {
                std::sort(cells_out.begin() + a, cells_out.begin() + b, [&](int32_t c0, int32_t c1) {
                    const auto & e0 = cells.ext_get(c0);
                    const auto & e1 = cells.ext_get(c1);
                    return e0.y < e1.y || (e0.y == e1.y && e0.x < e1.x);
                });
            }
            a = b;
        }
    }
}

void llama_kv_cache::causal_range(const llama_ubatch * ubatch, uint32_t i, bool causal_attn, const int32_t * cid, const llama_pos * pos, int32_t n,
        llama_pos seq_pos_min, int32_t & lo, int32_t & hi) const {
    const uint32_t n_tokens = ubatch->n_tokens;

    const bool swa      = swa_type != LLAMA_SWA_TYPE_NONE;
    const bool is_2d    = ubatch->is_pos_2d();
    const bool swa_full = !causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_FULL;

    const llama_pos p1 = ubatch->pos[i];

    // first entry with a position after p1
    const int32_t m = (int32_t) (std::upper_bound(pos, pos + n, p1) - pos);

    lo = 0;
    hi = causal_attn ? m : n;

    // M-RoPE causal mask: of the cells at p1, keep the ones not after the token in 2D order
    if (causal_attn && is_2d) {
        const llama_pos p1_x = ubatch->pos[i + n_tokens*2];
        const llama_pos p1_y = ubatch->pos[i + n_tokens];

        const int32_t a = (int32_t) (std::lower_bound(pos, pos + m, p1) - pos);

        hi = (int32_t) (std::partition_point(cid + a, cid + m, [&](int32_t c) {
            return !cells.ext_get(c).is_2d_gt(p1_x, p1_y);
        }) - cid);
    }

    if (swa) {
        const auto masked = [&](llama_pos p0) {
            return llama_hparams::is_masked_swa(n_swa, swa_type, p0, p1);
        };

        // the masked cells before p1 are a prefix, the ones after p1 a suffix
        lo = (int32_t) (std::partition_point(pos, pos + m, masked) - pos);

        if (!causal_attn) {
            hi = (int32_t) (std::partition_point(pos + m, pos + n, [&](llama_pos p0) { return !masked(p0); }) - pos);
        }

        // see llama_non_causal_type: the cells of the sequence from its first token in the ubatch stay visible
        if (swa_full) {
            lo = std::min(lo, (int32_t) (std::lower_bound(pos, pos + n, seq_pos_min) - pos));
            hi = n;
        }
    }

    lo = std::min(lo, hi);
}

void llama_kv_cache::set_input_kv_idx(ggml_tensor * kv_idx, ggml_tensor * q_rng, const llama_ubatch * ubatch, bool causal_attn) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(q_rng->buffer));
    GGML_ASSERT(q_rng->ne[0] == 3 && q_rng->ne[1] == ubatch->n_tokens);

    fill_kv_idx(kv_idx, (int32_t *) q_rng->data, ubatch, causal_attn);
}

// a selection over the list of rows: the windows say what a token sees of the list, and each query reads its n_top best
// entries through the list the graph builds from the sorted top-k, of which only the visible ones are live
void llama_kv_cache::set_input_kv_sel(ggml_tensor * kv_idx, ggml_tensor * q_rng, ggml_tensor * win_rng, uint32_t n_top,
        const llama_ubatch * ubatch, bool causal_attn) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(q_rng->buffer));
    GGML_ASSERT(ggml_backend_buffer_is_host(win_rng->buffer));

    const uint32_t n_tokens = ubatch->n_tokens;
    const int64_t  n_idx    = kv_idx->ne[0];

    GGML_ASSERT(win_rng->ne[0] == 3 && win_rng->ne[1] == n_tokens);
    GGML_ASSERT(n_idx <= INT32_MAX);

    std::vector<int32_t> rng(3*n_tokens);

    fill_kv_idx(kv_idx, rng.data(), ubatch, causal_attn);

    int32_t * sel = (int32_t *) q_rng->data;
    int32_t * win = (int32_t *) win_rng->data;

    // the window of a token starts at the first entry it sees, see ggml_lightning_indexer_rows()
    for (uint32_t i = 0; i < n_tokens; ++i) {
        win[3*i + 0] = rng[3*i + 1];
        win[3*i + 1] = rng[3*i + 1];
        win[3*i + 2] = rng[3*i + 2];
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        sel[3*i + 0] = (int32_t) i;
        sel[3*i + 1] = 0;
        sel[3*i + 2] = std::min<int32_t>(rng[3*i + 2] - rng[3*i + 1], (int32_t) std::min<int64_t>(n_top, n_idx));
    }
}

uint32_t llama_kv_cache::get_n_win(const llama_ubatch & ubatch) const {
    uint32_t n_max = 0;
    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
        n_max = std::max<uint32_t>(n_max, cells.seq_n_cells(ubatch.seq_id_unq[s]));
    }

    // the window of a token covers what it sees of its sequence
    GGML_ASSERT(n_max <= n_win_max());

    return std::min(n_win_max(), std::max<uint32_t>(256, GGML_PAD(n_max, 256)));
}

uint32_t llama_kv_cache::n_win_max() const {
    // a request holds at most the context window of the model
    return std::max<uint32_t>(256, GGML_PAD(std::min(get_size(), hparams.n_ctx_train), 256));
}

void llama_kv_cache::fill_kv_idx(ggml_tensor * kv_idx, int32_t * rng, const llama_ubatch * ubatch, bool causal_attn) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(kv_idx->buffer));

    const uint32_t n_tokens = ubatch->n_tokens;

    // see llama_non_causal_type
    if (!causal_attn && hparams.non_causal_type == LLAMA_NON_CAUSAL_TYPE_SWA_ONLY) {
        causal_attn = swa_type == LLAMA_SWA_TYPE_NONE;
    }

    const bool is_2d = ubatch->is_pos_2d();

    const int64_t n_idx   = kv_idx->ne[0];
    const int64_t n_group = kv_idx->ne[1];

    int32_t * idx = (int32_t *) kv_idx->data;

    // ALiBi: the positions of the rows in a second plane, see ggml_flash_attn_ext_set_kv_idx()
    int32_t * idx_pos = kv_idx->ne[2] == 2 ? idx + n_idx*n_group : nullptr;

    std::fill(idx, idx + n_idx*n_group*kv_idx->ne[2], 0);

    // group of each sequence of the ubatch, its cells in causal order with their positions, and the min position of
    // the sequence in the ubatch
    std::vector<int32_t>   seq_group(n_seq_max, -1);
    std::vector<llama_pos> seq_pos_min(n_seq_max, std::numeric_limits<llama_pos>::max());

    std::vector<std::vector<int32_t>>   group_cells;
    std::vector<std::vector<llama_pos>> group_pos;
    std::vector<int32_t>                group_start; // where the list of a group starts in group 0
    int64_t                             group_off = 0;

    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        seq_pos_min[seq_id] = std::min(seq_pos_min[seq_id], ubatch->pos[i]);

        if (seq_group[seq_id] >= 0) {
            continue;
        }

        GGML_ASSERT(n_group == 1);

        const int32_t g = (int32_t) group_cells.size();
        seq_group[seq_id] = g;

        group_cells.emplace_back();
        group_pos.emplace_back();
        seq_cells_causal(seq_id, is_2d, group_cells[g], group_pos[g]);

        // the lists one after the other in group 0, see get_kv_idx_shape()
        int32_t * dst = idx + group_off;

        GGML_ASSERT((int64_t) group_cells[g].size() <= n_idx - group_off);

        // the rows of the cells: the cells themselves, or their rows of the box
        for (size_t c = 0; c < group_cells[g].size(); ++c) {
            dst[c] = (int32_t) cell_row(group_cells[g][c]);
        }
        if (idx_pos) {
            std::copy(group_pos[g].begin(), group_pos[g].end(), idx_pos + group_off);
        }

        group_start.push_back((int32_t) group_off);
        group_off += (int64_t) group_cells[g].size();
    }

    // a copy of the rows reads the padding too, masked out: it has to be a row that holds a value, a NaN of a row nothing
    // wrote stays a NaN under the mask
    if (group_off > 0) {
        std::fill(idx + group_off, idx + n_idx*n_group, idx[0]);
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        const llama_seq_id seq_id = ubatch->seq_id[i][0];

        const int32_t g = seq_group[seq_id];
        const int32_t n = (int32_t) group_cells[g].size();

        int32_t lo = 0;
        int32_t hi = 0;
        causal_range(ubatch, i, causal_attn, group_cells[g].data(), group_pos[g].data(), n, seq_pos_min[seq_id], lo, hi);

        // the ranges are offsets into the one list
        rng[3*i + 0] = 0;
        rng[3*i + 1] = group_start[g] + lo;
        rng[3*i + 2] = group_start[g] + hi;

        if (!stats_read_by_graph) {
            stats_attn.read += hi - lo;
        }
        stats_attn.range += n;
        stats_attn.owned += n;
    }
}

void llama_kv_cache::set_input_k_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

void llama_kv_cache::set_input_v_rot(ggml_tensor * dst) const {
    GGML_ASSERT(ggml_backend_buffer_is_host(dst->buffer));

    const auto n_rot = dst->ne[0];
    GGML_ASSERT(attn_rot_hadamard.count(dst->ne[0]));

    memcpy(dst->data, attn_rot_hadamard.at(n_rot).data(), ggml_nbytes(dst));
}

bool llama_kv_cache::has_cell_ext() const {
    // M-RoPE needs the 2D position, the PLE n-gram hash needs the token id
    return hparams.n_pos_per_embd() > 1 || hparams.ple_n_heads > 0;
}

void llama_kv_cache::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    const uint32_t n_tokens = ubatch.n_tokens;

    res.clear();
    res.resize(n_tokens*n, LLAMA_TOKEN_NULL);

    if (n == 0) {
        return;
    }

    // note: apply_ubatch() has already stored the current ubatch, so the cells cover the tokens
    //       of this very ubatch as well, which is what we want
    // the nearest cell at or before a position also resolves M-RoPE gaps, where multiple tokens
    // share the same temporal pos

    // an embd (multimodal) ubatch can repeat one position for a whole image, so positions
    // do not encode the token order; resolve its predecessors by ubatch order instead
    std::vector<uint32_t> ord; // index among the ubatch tokens of the same seq
    std::unordered_map<llama_seq_id, std::vector<uint32_t>> seq_idx;

    if (!ubatch.token) {
        ord.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; ++i) {
            auto & v = seq_idx[ubatch.seq_id[i][0]];
            ord[i] = v.size();
            v.push_back(i);
        }
    }

    for (uint32_t i = 0; i < n_tokens; ++i) {
        // TODO: a token that belongs to more than one sequence has an ambiguous history.
        //       the n-gram architectures have to reject such batches
        const llama_seq_id seq_id = ubatch.seq_id[i][0];

        for (uint32_t j = 0; j < n; ++j) {
            const llama_pos d = (llama_pos) (n - j);

            llama_pos p;
            if (!ubatch.token) {
                const auto & v = seq_idx[seq_id];
                const int64_t k = (int64_t) ord[i] - d;
                // k >= 0: an earlier token of this very ubatch; k < 0: before the chunk
                p = k >= 0 ? ubatch.pos[v[k]] : ubatch.pos[v[0]] + (llama_pos) k;
            } else {
                p = ubatch.pos[i] - d;
            }

            if (p < 0) {
                continue;
            }

            res[i*n + j] = cells.seq_pos_tok_le(seq_id, p);
        }
    }
}

size_t llama_kv_cache::total_size() const {
    size_t size = 0;

    for (const auto & [_, buf] : ctxs_bufs) {
        size += ggml_backend_buffer_get_size(buf.get());
    }

    return size;
}

size_t llama_kv_cache::size_k_bytes() const {
    size_t size_k_bytes = 0;

    for (const auto & layer : layers) {
        // the views of the box span it all, the cells take a row each
        size_k_bytes += box ? ggml_row_size(layer.k->type, layer.k->ne[0])*get_size() : ggml_nbytes(layer.k);
    }

    return size_k_bytes;
}

size_t llama_kv_cache::size_v_bytes() const {
    size_t size_v_bytes = 0;

    for (const auto & layer : layers) {
        if (!layer.v) {
            continue;
        }
        size_v_bytes += box ? ggml_row_size(layer.v->type, layer.v->ne[0])*get_size() : ggml_nbytes(layer.v);
    }

    return size_v_bytes;
}

ggml_tensor * llama_kv_cache::build_rope_shift(
        const llama_cparams & cparams,
               ggml_context * ctx,
                ggml_tensor * cur,
                ggml_tensor * shift,
                ggml_tensor * rot,
                ggml_tensor * factors,
                      float   freq_base,
                      float   freq_scale,
                   uint32_t   il) const {
    const auto & n_ctx_orig = cparams.n_ctx_orig_yarn;

    const auto & yarn_ext_factor  = cparams.yarn_ext_factor;
    const auto & yarn_beta_fast   = cparams.yarn_beta_fast;
    const auto & yarn_beta_slow   = cparams.yarn_beta_slow;
    const auto & yarn_attn_factor = cparams.yarn_attn_factor;

    const auto & n_rot     = hparams.n_rot(il);
    const auto & rope_type = hparams.rope_type == LLAMA_ROPE_TYPE_MROPE || hparams.rope_type == LLAMA_ROPE_TYPE_IMROPE
                                // @ngxson : this is a workaround
                                // for M-RoPE, we want to rotate the whole vector when doing KV shift
                                // a normal RoPE should work, we just need to use the correct ordering
                                // ref: https://github.com/ggml-org/llama.cpp/pull/13870
                                ? LLAMA_ROPE_TYPE_NEOX
                                : hparams.rope_type;
    ggml_tensor * tmp;

    if (ggml_is_quantized(cur->type)) {
        // dequantize to f32 -> RoPE -> quantize back
        tmp = ggml_cast(ctx, cur, GGML_TYPE_F32);

        // rotate back
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_rope_ext(ctx, tmp,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);

        // rotate fwd
        tmp = llama_mul_mat_hadamard(ctx, tmp, rot);

        tmp = ggml_cpy(ctx, tmp, cur);
    } else {
        // we rotate only the first n_rot dimensions
        tmp = ggml_rope_ext_inplace(ctx, cur,
                shift, factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                yarn_ext_factor, yarn_attn_factor, yarn_beta_fast, yarn_beta_slow);
    }

    return tmp;
}

class llm_graph_input_k_shift : public llm_graph_input_i {
public:
    llm_graph_input_k_shift(const llama_kv_cache * kv_self) : kv_self(kv_self) {}
    virtual ~llm_graph_input_k_shift() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * k_shift; // I32 [kv_size]

    // note: assumes k_rot^2 == I
    ggml_tensor * k_rot = nullptr;

    const llama_kv_cache * kv_self;
};

void llm_graph_input_k_shift::set_input(const llama_ubatch * ubatch) {
    GGML_UNUSED(ubatch);

    if (k_shift) {
        kv_self->set_input_k_shift(k_shift);
    }

    if (k_rot && k_rot->buffer) {
        kv_self->set_input_k_rot(k_rot);
    }
}

ggml_cgraph * llama_kv_cache::build_graph_shift(llm_graph_result * res, llama_context * lctx) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    GGML_ASSERT(!other);

    auto * ctx = res->get_ctx();
    auto * gf  = res->get_gf();

    auto inp = std::make_unique<llm_graph_input_k_shift>(this);

    inp->k_shift = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, (int64_t) get_size());
    ggml_set_input(inp->k_shift);

    inp->k_rot = build_input_k_rot(ctx);

    const auto & cparams = lctx->get_cparams();

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        if (!hparams.has_rope(il)) {
            continue;
        }

        const int64_t n_head_kv    = hparams.n_head_kv(il);
        const int64_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        const auto n_rot         = hparams.n_rot(il);
        const auto n_embd_head_k = hparams.n_embd_head_k(il);
        const auto n_embd_nope   = hparams.n_lora_kv > 0 ? n_embd_head_k - n_rot : 0;

        const float freq_base_l  = model.get_rope_freq_base (cparams, il);
        const float freq_scale_l = model.get_rope_freq_scale(cparams, il);

        ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);

        ggml_tensor * k =
            ggml_view_3d(ctx, layer.k,
                n_rot, n_head_kv, get_size(),
                ggml_row_size(layer.k->type, n_embd_head_k),
                ggml_row_size(layer.k->type, n_embd_k_gqa),
                ggml_row_size(layer.k->type, n_embd_nope));

        ggml_tensor * cur = build_rope_shift(cparams, ctx, k, inp->k_shift, inp->k_rot, rope_factors, freq_base_l, freq_scale_l, il);

        ggml_build_forward_expand(gf, cur);
    }

    res->add_input(std::move(inp));

    return gf;
}

void llama_kv_cache::state_write(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) const {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    // the format keeps the number of streams, which is one
    const uint32_t n_stream = 1;
    io.write(&n_stream, sizeof(n_stream));

    {
        cell_ranges_t cr;

        uint32_t cell_count = 0;

        // Count the number of cells with the specified seq_id
        // Find all the ranges of cells with this seq id (or all, when -1)
        uint32_t cell_range_begin = cells.size();

        for (uint32_t i = 0; i < cells.size(); ++i) {
            bool add_cell = true;

            add_cell = add_cell && !cells.is_empty(i);
            add_cell = add_cell && (seq_id == -1 || cells.seq_has(i, seq_id));

            // check the cell is not SWA-masked
            if (add_cell && seq_id != -1) {
                const bool is_masked = llama_hparams::is_masked_swa(n_swa, swa_type, cells.pos_get(i), cells.seq_pos_max(seq_id));

                add_cell = !is_masked;
            }

            if (add_cell) {
                ++cell_count;
                if (cell_range_begin == cells.size()) {
                    cell_range_begin = i;
                }
            } else {
                if (cell_range_begin != cells.size()) {
                    cr.data.emplace_back(cell_range_begin, i);
                    cell_range_begin = cells.size();
                }
            }
        }

        if (cell_range_begin != cells.size()) {
            cr.data.emplace_back(cell_range_begin, cells.size());
        }

        // DEBUG CHECK: Sum of cell counts in ranges should equal the total cell count
        uint32_t cell_count_check = 0;
        for (const auto & range : cr.data) {
            cell_count_check += range.second - range.first;
        }
        GGML_ASSERT(cell_count == cell_count_check);

        io.write(&cell_count, sizeof(cell_count));

        if (cell_count == 0) {
            return;
        }

        state_write_meta(io, cr, seq_id);
        state_write_data(io, cr);
    }
}

void llama_kv_cache::state_read(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    state_read_sinfo(io, seq_id, flags, nullptr, nullptr);
}

void llama_kv_cache::state_read_sinfo(
        llama_io_read_i & io,
           llama_seq_id   seq_id,
  llama_state_seq_flags   flags,
            slot_info *   sinfo_out,
      const slot_info *   sinfo_in) {
    // TODO: refactor [TAG_KV_CACHE_SHARE_CELLS]
    if (other) {
        return;
    }

    GGML_UNUSED(flags);

    // TODO: fix incosistent handling of `seq_id < 0` and `seq_id == -1` in the codebase [TAG_LLAMA_SEQ_ID_NEG]
    GGML_ASSERT(seq_id == -1 || (seq_id >= 0 && (uint32_t) seq_id < n_seq_max));

    if (sinfo_out) {
        *sinfo_out = slot_info{};
    }

    uint32_t n_stream;
    io.read(&n_stream, sizeof(n_stream));
    if (n_stream != 1) {
        throw std::runtime_error("failed to restore kv cache: the state has more than one stream");
    }

    // a whole-context restore replaces the cache
    if (seq_id == -1) {
        clear(true);
    }

    uint32_t cell_count;
    io.read(&cell_count, sizeof(cell_count));

    if (cell_count == 0) {
        // a mirrored cache must be empty here as well, or the two no longer agree cell for cell
        if (sinfo_in && !sinfo_in->empty()) {
            throw std::runtime_error("failed to restore kv cache: mirrored cache holds cells this one does not");
        }
        return;
    }

    slot_info sinfo;

    bool res = true;
    res = res && state_read_meta(io, cell_count, sinfo, seq_id, sinfo_in);

    try {
        res = res && state_read_data(io, cell_count, sinfo);
    } catch (...) {
        res = false;
    }

    if (!res) {
        state_clear(seq_id, sinfo);
        throw std::runtime_error("failed to restore kv cache");
    }

    if (sinfo_out) {
        *sinfo_out = sinfo;
    }
}

void llama_kv_cache::state_write_meta(llama_io_write_i & io, const cell_ranges_t & cr, llama_seq_id seq_id) const {
    for (const auto & range : cr.data) {
        for (uint32_t i = range.first; i < range.second; ++i) {
            std::vector<llama_seq_id> seq_ids;

            for (llama_seq_id cur = 0; cur < (int) n_seq_max; ++cur) {
                if (cur == seq_id || seq_id == -1) {
                    if (cells.seq_has(i, cur)) {
                        seq_ids.push_back(cur);
                    }
                }
            }

            const llama_pos pos     = cells.pos_get(i);
            const uint32_t n_seq_id = seq_ids.size();

            io.write(&pos,      sizeof(pos));
            io.write(&n_seq_id, sizeof(n_seq_id));

            if (has_cell_ext()) {
                const llama_kv_cell_ext ext = cells.ext_get(i);
                io.write(&ext, sizeof(ext));
            }

            for (const auto & seq_id : seq_ids) {
                io.write(&seq_id, sizeof(seq_id));
            }
        }
    }
}

// the bytes of rows [row, row + n) of a K or V tensor, rows of row_size bytes: one run when the rows are dense, else one
// run per row, at the stride of the tensor (the views of the box space their rows by the row of the box, see llama_box)
static void kv_rows_bytes(const ggml_tensor * t, size_t row_size, int64_t row, uint32_t n, const std::function<void(size_t, size_t)> & fn) {
    if ((size_t) t->nb[1] == row_size) {
        fn((size_t) row*row_size, (size_t) n*row_size);
        return;
    }

    for (uint32_t j = 0; j < n; ++j) {
        fn((size_t) (row + j)*t->nb[1], row_size);
    }
}

void llama_kv_cache::state_write_data(llama_io_write_i & io, const cell_ranges_t & cr) const {
    // the format keeps the flag of a transposed V, which the cache never is
    const uint32_t v_trans = 0;
    const uint32_t n_layer = layers.size();

    io.write(&v_trans, sizeof(v_trans));
    io.write(&n_layer, sizeof(n_layer));

    // Iterate and write all the keys first, each row is a cell
    // Get whole range at a time
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k;

        // Write key type
        const int32_t k_type_i = (int32_t) k->type;
        io.write(&k_type_i, sizeof(k_type_i));

        // Write row size of key
        const uint64_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        io.write(&k_size_row, sizeof(k_size_row));

        // Read each range of cells of k_size length and write out
        // the rows of a range of cells are contiguous, in the box within a block
        for (const auto & range : cr.data) {
            cell_row_runs(range.first, range.second, [&](int64_t row, uint32_t n) {
                kv_rows_bytes(k, k_size_row, row, n, [&](size_t offs, size_t size) { io.write_tensor(k, offs, size); });
            });
        }
    }

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

        auto * v = layer.v;
        if (!v) {
            continue;
        }

        // Write value type
        const int32_t v_type_i = (int32_t) v->type;
        io.write(&v_type_i, sizeof(v_type_i));

        // Write row size of value
        const uint64_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
        io.write(&v_size_row, sizeof(v_size_row));

        // Read each range of cells of v_size length and write out
        for (const auto & range : cr.data) {
            cell_row_runs(range.first, range.second, [&](int64_t row, uint32_t n) {
                kv_rows_bytes(v, v_size_row, row, n, [&](size_t offs, size_t size) { io.write_tensor(v, offs, size); });
            });
        }
    }
}

bool llama_kv_cache::state_read_meta(llama_io_read_i & io, uint32_t cell_count, slot_info & sinfo, llama_seq_id dest_seq_id, const slot_info * sinfo_in) {
    if (dest_seq_id != -1) {
        // single sequence
        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        if (box) {
            // only the cells of the sequence go, its blocks stay: the restored cells take them again, and the blocks it
            // holds for others (the compressed rows of a DSV4 cache, which a partial state leaves in place) are kept
            for (uint32_t i = 0; i < cells.size(); ++i) {
                if (cells.seq_has(i, dest_seq_id)) {
                    cells.seq_rm(i, dest_seq_id);
                }
            }
        } else {
            seq_rm(dest_seq_id, -1, -1);
        }

        // the restored tokens go into the blocks of the sequence, which takes the blocks it lacks, see llama_box
        if (box && !sinfo_in) {
            while (box_seq_free(dest_seq_id) < cell_count) {
                if (!box->block_take(dest_seq_id)) {
                    LLAMA_LOG_ERROR("%s: no free block in the box for %u cells\n", __func__, cell_count);
                    return false;
                }
            }
        }

        llama_batch_allocr balloc(hparams.n_pos_per_embd());

        llama_ubatch ubatch = balloc.ubatch_reserve(cell_count, 1);

        ubatch.seq_id_unq[0] = dest_seq_id;

        // the ext as it was saved, to put back after apply_ubatch()
        std::vector<llama_kv_cell_ext> exts;
        if (has_cell_ext()) {
            exts.resize(cell_count);
        }

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            if (n_seq_id != 1) {
                LLAMA_LOG_ERROR("%s: invalid seq_id-agnostic kv cell\n", __func__);
                return false;
            }

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));

                if (hparams.n_pos_per_embd() > 1) {
                    ubatch.pos[i + ubatch.n_tokens]   = ext.y;
                    ubatch.pos[i + ubatch.n_tokens*2] = ext.x;
                }

                // apply_ubatch() below restores ext.tok from the ubatch tokens
                ubatch.token[i] = ext.tok;

                exts[i] = ext;
            }

            // read the sequence id, but directly discard it - we will use dest_seq_id instead
            {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));
            }

            ubatch.pos[i]      = pos;
            ubatch.n_seq_id[i] = n_seq_id;
            ubatch.seq_id[i]   = &dest_seq_id;
        }

        if (sinfo_in) {
            // this cache mirrors another one, so it takes that cache's layout instead of searching for its own cells
            if (sinfo_in->size() != cell_count) {
                LLAMA_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                        (int) sinfo_in->size(), cell_count);
                return false;
            }

            sinfo = *sinfo_in;

            // seq_rm above freed exactly the cells this sequence held
            // anything else in the way is a cache that had already drifted, which this restore must not hide
            for (uint32_t i = 0; i < cell_count; ++i) {
                const uint32_t idx = sinfo.idxs[i];

                if (idx >= cells.size() || !cells.is_empty(idx)) {
                    LLAMA_LOG_ERROR("%s: cell %u of the mirrored slot layout is not free\n", __func__, idx);
                    return false;
                }
            }
        } else {
            sinfo = find_slot(ubatch, false);
            if (sinfo.empty()) {
                LLAMA_LOG_ERROR("%s: failed to find %d available cells in kv cache\n", __func__,  cell_count);
                return false;
            }
        }

        // note: apply_ubatch() rebuilds llama_kv_cell_ext from the ubatch
        //       only ext.tok and the M-RoPE 2D position round-trip through it
        //       see: https://github.com/ggml-org/llama.cpp/pull/16825#issuecomment-3460868350
        apply_ubatch(sinfo, ubatch);

        // apply_ubatch() takes the 2D position from the ubatch, and that ubatch is built with this
        // cache's own n_pos_per_embd. a cache that does not use M-RoPE itself but mirrors one that
        // does (the qwen4exp QSA indexer) would drop x and y. put the saved ext back instead, which
        // is what the whole-context path below already does.
        for (uint32_t i = 0; i < (uint32_t) exts.size(); ++i) {
            cells.ext_set(sinfo.idxs[i], exts[i]);
        }

        LLAMA_LOG_DEBUG("%s: cell_count = %d, dest_seq_id = %d\n", __func__, cell_count, dest_seq_id);

        // DEBUG CHECK: verify that all cells were allocated and have correct seq_id and pos values
        GGML_ASSERT(sinfo.size() == cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            const uint32_t idx = sinfo.idxs[i];
            GGML_ASSERT(cells.pos_get(idx) == ubatch.pos[i]);
            GGML_ASSERT(cells.seq_has(idx, dest_seq_id));
        }
    } else {
        // whole KV cache restore

        // the cells of the box belong to the blocks of their sequences
        if (box) {
            LLAMA_LOG_ERROR("%s: the box restores single sequences only\n", __func__);
            return false;
        }

        if (cell_count > cells.size()) {
            LLAMA_LOG_ERROR("%s: not enough cells in kv cache\n", __func__);
            return false;
        }

        // the cells go in from 0, so a mirrored cache lands on the same ones as long as it restores the same count. the layout itself carries no more information here
        if (sinfo_in && sinfo_in->size() != cell_count) {
            LLAMA_LOG_ERROR("%s: mirrored slot layout holds %d cells, this cache restores %d\n", __func__,
                    (int) sinfo_in->size(), cell_count);
            return false;
        }

        for (uint32_t i = 0; i < cell_count; ++i) {
            llama_pos pos;
            uint32_t  n_seq_id;

            io.read(&pos,      sizeof(pos));
            io.read(&n_seq_id, sizeof(n_seq_id));

            cells.pos_set(i, pos);

            if (has_cell_ext()) {
                llama_kv_cell_ext ext;
                io.read(&ext, sizeof(ext));
                cells.ext_set(i, ext);
            }

            for (uint32_t j = 0; j < n_seq_id; ++j) {
                llama_seq_id seq_id;
                io.read(&seq_id, sizeof(seq_id));

                if (seq_id < 0 || (uint32_t) seq_id >= n_seq_max) {
                    LLAMA_LOG_ERROR("%s: invalid seq_id, %d is out of range [0, %u)\n", __func__, seq_id, n_seq_max);
                    return false;
                }

                cells.seq_add(i, seq_id);
            }
        }

        // Create contiguous slot_info for whole cache restore
        sinfo.idxs.resize(cell_count);
        for (uint32_t i = 0; i < cell_count; ++i) {
            sinfo.idxs[i] = i;
        }

        head = 0;
    }

    return true;
}

bool llama_kv_cache::state_read_data(llama_io_read_i & io, uint32_t cell_count, const slot_info & sinfo) {
    // batch the scatter reads per contiguous run of destination indices
    // from inclusive, to exclusive - same convention as cell_ranges_t
    // contiguous cells yield a single run covering the whole block
    struct cell_run { uint32_t from; uint32_t to; };
    std::vector<cell_run> runs;
    if (cell_count > 0) {
        const auto & idxs = sinfo.idxs;
        uint32_t i0 = 0;
        while (i0 < cell_count) {
            uint32_t i1 = i0 + 1;
            while (i1 < cell_count && idxs[i1] == idxs[i1 - 1] + 1) {
                ++i1;
            }
            runs.push_back({idxs[i0], idxs[i1 - 1] + 1});
            i0 = i1;
        }
    }

    uint32_t v_trans;
    uint32_t n_layer;

    io.read(&v_trans, sizeof(v_trans));
    io.read(&n_layer, sizeof(n_layer));

    if (n_layer != layers.size()) {
        LLAMA_LOG_ERROR("%s: mismatched layer count (%u instead of %u)\n", __func__, n_layer, (uint32_t) layers.size());
        return false;
    }

    if (cell_count > cells.size()) {
        LLAMA_LOG_ERROR("%s: not enough cells in kv cache to restore state (%u > %u)\n", __func__, cell_count, cells.size());
        return false;
    }

    if (v_trans != 0) {
        LLAMA_LOG_ERROR("%s: the state has a transposed V, which the cache never has\n", __func__);
        return false;
    }

    // For each layer, read the keys for each cell, one row is one cell, read as one contiguous block
    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_k_gqa = hparams.n_embd_k_gqa(il);

        auto * k = layer.k;

        // Read type of key
        int32_t k_type_i_ref;
        io.read(&k_type_i_ref, sizeof(k_type_i_ref));
        const int32_t k_type_i = (int32_t) k->type;
        if (k_type_i != k_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key type (%d != %d, layer %d)\n", __func__, k_type_i, k_type_i_ref, il);
            return false;
        }

        // Read row size of key
        uint64_t k_size_row_ref;
        io.read(&k_size_row_ref, sizeof(k_size_row_ref));
        const size_t k_size_row = ggml_row_size(k->type, n_embd_k_gqa);
        if (k_size_row != k_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched key row size (%zu != %zu, layer %d)\n", __func__, k_size_row, (size_t) k_size_row_ref, il);
            return false;
        }

        for (const auto & r : runs) {
            cell_row_runs(r.from, r.to, [&](int64_t row, uint32_t n) {
                kv_rows_bytes(k, k_size_row, row, n, [&](size_t offs, size_t size) { io.read_tensor(k, offs, size); });
            });
        }
    }

    for (const auto & layer : layers) {
        const uint32_t il = layer.il;

        const uint32_t n_embd_v_gqa = hparams.n_embd_v_gqa(il);

        auto * v = layer.v;
        if (!v) {
            continue;
        }

        // Read type of value
        int32_t v_type_i_ref;
        io.read(&v_type_i_ref, sizeof(v_type_i_ref));
        const int32_t v_type_i = (int32_t) v->type;
        if (v_type_i != v_type_i_ref) {
            LLAMA_LOG_ERROR("%s: mismatched value type (%d != %d, layer %d)\n", __func__, v_type_i, v_type_i_ref, il);
            return false;
        }

        // Read row size of value
        uint64_t v_size_row_ref;
        io.read(&v_size_row_ref, sizeof(v_size_row_ref));
        const size_t v_size_row = ggml_row_size(v->type, n_embd_v_gqa);
        if (v_size_row != v_size_row_ref) {
            LLAMA_LOG_ERROR("%s: mismatched value row size (%zu != %zu, layer %d)\n", __func__, v_size_row, (size_t) v_size_row_ref, il);
            return false;
        }

        for (const auto & r : runs) {
            cell_row_runs(r.from, r.to, [&](int64_t row, uint32_t n) {
                kv_rows_bytes(v, v_size_row, row, n, [&](size_t offs, size_t size) { io.read_tensor(v, offs, size); });
            });
        }
    }

    return true;
}

void llama_kv_cache::state_clear(llama_seq_id seq_id) {
    if (seq_id == -1) {
        clear(true);
        return;
    }

    GGML_ASSERT(seq_id >= 0 && (uint32_t) seq_id < n_seq_max);

    slot_info sinfo;

    // a cell that another sequence still uses keeps its data
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (cells.seq_has(i, seq_id) && cells.seq_count(i) == 1) {
            sinfo.idxs.push_back(i);
        }
    }

    state_clear(seq_id, sinfo);
}

// the cleared ranges mirror the write pattern of state_read_data() - keep both in sync
void llama_kv_cache::state_clear(llama_seq_id seq_id, const slot_info & sinfo) {
    if (seq_id == -1) {
        clear(true);
        return;
    }

    seq_rm(seq_id, -1, -1);

    // zero the K/V data of the failed restore attempt - the attention can still read the data of free cells
    if (sinfo.empty()) {
        return;
    }

    // the cells in runs of consecutive indices, as state_read_data() writes them
    struct run { uint32_t from, to; };
    std::vector<run> runs;
    {
        const auto & idxs = sinfo.idxs;
        for (size_t i0 = 0; i0 < idxs.size();) {
            size_t i1 = i0 + 1;
            while (i1 < idxs.size() && idxs[i1] == idxs[i1 - 1] + 1) {
                ++i1;
            }
            runs.push_back({idxs[i0], idxs[i1 - 1] + 1});
            i0 = i1;
        }
    }

    for (const auto & layer : layers) {
        auto * k = layer.k;

        const size_t k_size_row = ggml_row_size(k->type, hparams.n_embd_k_gqa(layer.il));

        for (const auto & r : runs) {
            cell_row_runs(r.from, r.to, [&](int64_t row, uint32_t n) {
                kv_rows_bytes(k, k_size_row, row, n, [&](size_t offs, size_t size) { llama_clear_tensor_data(k, offs, size); });
            });
        }
    }

    for (const auto & layer : layers) {
        auto * v = layer.v;
        if (!v) {
            continue;
        }

        const size_t v_size_row = ggml_row_size(v->type, hparams.n_embd_v_gqa(layer.il));

        for (const auto & r : runs) {
            cell_row_runs(r.from, r.to, [&](int64_t row, uint32_t n) {
                kv_rows_bytes(v, v_size_row, row, n, [&](size_t offs, size_t size) { llama_clear_tensor_data(v, offs, size); });
            });
        }
    }
}

//
// llama_kv_cache_context
//

llama_kv_cache_context::llama_kv_cache_context(llama_memory_status status) : status(status) {}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), is_full(true) {
    n_kv = kv->get_size();

    // create a dummy slot info - the actual data is irrelevant. we just need to build the graph
    sinfos.resize(1);
    sinfos[0].idxs.resize(1, 0);
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_context * lctx,
        bool do_shift) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), lctx(lctx), do_shift(do_shift) {
    if (!do_shift) {
        status = LLAMA_MEMORY_STATUS_NO_UPDATE;
    }
}

llama_kv_cache_context::llama_kv_cache_context(
        llama_kv_cache * kv,
        llama_kv_cache::slot_info_vec_t sinfos,
        std::vector<llama_ubatch> ubatches) : status(LLAMA_MEMORY_STATUS_SUCCESS), kv(kv), sinfos(std::move(sinfos)), ubatches(std::move(ubatches)) {
}

llama_kv_cache_context::~llama_kv_cache_context() = default;

bool llama_kv_cache_context::next() {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    if (++i_cur >= ubatches.size()) {
        return false;
    }

    return true;
}

bool llama_kv_cache_context::apply() {
    assert(!llama_memory_status_is_fail(status));

    // no ubatches -> this is a KV cache update
    if (ubatches.empty()) {
        kv->update(lctx, do_shift);

        return true;
    }

    kv->apply_ubatch(sinfos[i_cur], ubatches[i_cur]);
    n_kv = kv->get_n_kv();

    return true;
}

llama_memory_status llama_kv_cache_context::get_status() const {
    return status;
}

const llama_ubatch & llama_kv_cache_context::get_ubatch() const {
    assert(status == LLAMA_MEMORY_STATUS_SUCCESS);

    return ubatches[i_cur];
}

uint32_t llama_kv_cache_context::get_n_kv() const {
    return n_kv;
}

ggml_type llama_kv_cache_context::type_k() const {
    return kv->type_k();
}

ggml_type llama_kv_cache_context::type_v() const {
    return kv->type_v();
}

ggml_tensor * llama_kv_cache_context::get_k(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, n_kv);
}

ggml_tensor * llama_kv_cache_context::get_v(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, n_kv);
}

std::vector<llama_kv_attn_run> llama_kv_cache_context::get_attn_runs(const llama_ubatch & ubatch, bool in_place, bool rest_indexed) const {
    // the context of the whole cache only builds the graphs reserved at startup, which must cover the largest runs
    return is_full ? kv->get_attn_runs_reserve(ubatch, n_kv, in_place, rest_indexed) : kv->get_attn_runs(ubatch, n_kv, in_place, rest_indexed);
}

void llama_kv_cache_context::set_input_attn_run(ggml_tensor * idxs, ggml_tensor * hi, const llama_ubatch * ubatch, const llama_kv_attn_run & run) const {
    kv->set_input_attn_run(idxs, hi, ubatch, run);
}

uint32_t llama_kv_cache_context::kv_idx_planes() const {
    return kv->kv_idx_planes();
}

void llama_kv_cache_context::get_kv_idx_shape(const llama_ubatch & ubatch, uint32_t & n_idx, uint32_t & n_group) const {
    kv->get_kv_idx_shape(ubatch, n_idx, n_group);

    // a reserve has no cells yet: its lists take the most a ubatch reads, so that the input buffers do not grow once the
    // sequences do
    if (is_full) {
        n_idx = kv->attn_n_cells_max();
    }
}

void llama_kv_cache_context::set_input_kv_idx(ggml_tensor * kv_idx, ggml_tensor * q_rng, const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kv_idx(kv_idx, q_rng, ubatch, causal_attn);
}

void llama_kv_cache_context::set_input_kv_sel(ggml_tensor * kv_idx, ggml_tensor * q_rng, ggml_tensor * win_rng, uint32_t n_top,
        const llama_ubatch * ubatch, bool causal_attn) const {
    kv->set_input_kv_sel(kv_idx, q_rng, win_rng, n_top, ubatch, causal_attn);
}

uint32_t llama_kv_cache_context::get_n_win(const llama_ubatch & ubatch) const {
    // a reserve has no cells yet: its windows take the most a sequence holds
    return is_full ? kv->n_win_max() : kv->get_n_win(ubatch);
}

void llama_kv_cache_context::attn_read_counted_by_graph() const {
    kv->attn_read_counted_by_graph();
}

void llama_kv_cache_context::add_attn_read(uint64_t n_cells) const {
    kv->add_attn_read(n_cells);
}

ggml_tensor * llama_kv_cache_context::get_k_all(ggml_context * ctx, int32_t il) const {
    return kv->get_k(ctx, il, kv->get_size());
}

ggml_tensor * llama_kv_cache_context::get_v_all(ggml_context * ctx, int32_t il) const {
    return kv->get_v(ctx, il, kv->get_size());
}

ggml_tensor * llama_kv_cache_context::cpy_k(ggml_context * ctx, ggml_tensor * k_cur, ggml_tensor * k_idxs, int32_t il) const {
    return kv->cpy_k(ctx, k_cur, k_idxs, il);
}

ggml_tensor * llama_kv_cache_context::cpy_v(ggml_context * ctx, ggml_tensor * v_cur, ggml_tensor * v_idxs, int32_t il) const {
    return kv->cpy_v(ctx, v_cur, v_idxs, il);
}

ggml_tensor * llama_kv_cache_context::build_input_k_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_k_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_v_idxs(ggml_context * ctx, const llama_ubatch & ubatch) const {
    return kv->build_input_v_idxs(ctx, ubatch);
}

ggml_tensor * llama_kv_cache_context::build_input_k_rot(ggml_context * ctx) const {
    return kv->build_input_k_rot(ctx);
}

ggml_tensor * llama_kv_cache_context::build_input_v_rot(ggml_context * ctx) const {
    return kv->build_input_v_rot(ctx);
}

void llama_kv_cache_context::set_input_k_shift(ggml_tensor * dst) const {
    kv->set_input_k_shift(dst);
}

void llama_kv_cache_context::set_input_k_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_k_idxs(dst, ubatch, sinfos[i_cur]);
}

void llama_kv_cache_context::set_input_v_idxs(ggml_tensor * dst, const llama_ubatch * ubatch) const {
    kv->set_input_v_idxs(dst, ubatch, sinfos[i_cur]);
}



void llama_kv_cache_context::set_input_k_rot(ggml_tensor * dst) const {
    kv->set_input_k_rot(dst);
}

void llama_kv_cache_context::set_input_v_rot(ggml_tensor * dst) const {
    kv->set_input_v_rot(dst);
}

void llama_kv_cache_context::get_prev_tokens(const llama_ubatch & ubatch, uint32_t n, std::vector<llama_token> & res) const {
    kv->get_prev_tokens(ubatch, n, res);
}
