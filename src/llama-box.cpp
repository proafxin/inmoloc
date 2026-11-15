#include "llama-box.h"

#include "llama-graph.h"
#include "llama-impl.h"
#include "llama-model.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>
#include <stdexcept>

// the row stride of layers: their longest K or V row, which every row is padded to, so that a cell has one row index in
// all of them
static size_t llama_box_row_stride(const llama_box::attn_layer * l, size_t n) {
    size_t row = 0;
    for (size_t i = 0; i < n; ++i) {
        row = std::max(row, ggml_row_size(l[i].type_k, l[i].n_embd_k));
        if (l[i].has_v) {
            row = std::max(row, ggml_row_size(l[i].type_v, l[i].n_embd_v));
        }
    }
    return row;
}

// the bytes of the rows of the draft layers in a block
static size_t llama_box_draft_bytes(const std::vector<llama_box::attn_layer> & draft, size_t rowd) {
    size_t size = 0;
    for (const auto & l : draft) {
        size += (l.has_v ? 2 : 1)*llama_box::n_block_tokens*rowd;
    }
    return size;
}

// the bytes of a block: n_block_tokens rows of every attention layer and of the draft layers, padded to whole rows of
// each (row indices) and whole units (state row indices)
static size_t llama_box_block_bytes(const std::vector<llama_box::attn_layer> & attn, const std::vector<llama_box::attn_layer> & draft) {
    // no attention layers: the box only holds states, see the constructor
    if (attn.empty()) {
        return 0;
    }

    const size_t row  = llama_box_row_stride(attn.data(), attn.size());
    const size_t rowd = draft.empty() ? row : llama_box_row_stride(draft.data(), draft.size());

    size_t size = 0;
    for (const auto & l : attn) {
        size += (l.has_v ? 2 : 1)*llama_box::n_block_tokens*row;
    }
    size += llama_box_draft_bytes(draft, rowd);

    const size_t mult = std::lcm(std::lcm(row, rowd), llama_box::unit); // not a power of 2, GGML_PAD does not apply

    return (size + mult - 1)/mult*mult;
}

llama_box::llama_box(std::vector<attn_layer> attn_in, std::vector<recr_layer> recr_in, const std::vector<attn_layer> & draft, size_t bytes, uint32_t n_tokens,
        uint32_t n_seq_max, uint32_t n_states_in, std::vector<attn_layer> lid_in, std::vector<state_piece> pieces_in) :
    attn(std::move(attn_in)), recr(std::move(recr_in)), lid(std::move(lid_in)), pieces(std::move(pieces_in)) {
    row_bytes      = llama_box_row_stride(attn.data(), attn.size());
    block_bytes    = llama_box_block_bytes(attn, draft);

    // the indexer rows are at the row stride of the K views, on the layers without compressed rows and after the layers
    // when they run out; a block holds them whatever its kind, so that all blocks are of one size
    if (!lid.empty()) {
        row_bytes = std::max(row_bytes, llama_box_row_stride(lid.data(), lid.size()));

        size_t n_free = 0;
        for (const auto & l : attn) {
            n_free += l.comp_free ? 1 : 0;
        }

        const size_t n_extra = lid.size() > n_free ? lid.size() - n_free : 0;

        const size_t rowd = draft.empty() ? row_bytes : llama_box_row_stride(draft.data(), draft.size());
        const size_t mult = std::lcm(std::lcm(row_bytes, rowd), unit);

        block_bytes = (attn.size() + n_extra)*n_block_tokens*row_bytes;
        block_bytes += llama_box_draft_bytes(draft, rowd);
        block_bytes = (block_bytes + mult - 1)/mult*mult;
    }

    rows_per_block = row_bytes > 0 ? block_bytes/row_bytes : 0;

    size_t off = 0;
    for (const auto & l : attn) {
        off_k.push_back(off);
        off += n_block_tokens*row_bytes;
        off_v.push_back(off);
        off += l.has_v ? n_block_tokens*row_bytes : 0;
    }

    // the free layers in order, then the room after the layers
    for (size_t i = 0, j = 0; j < lid.size(); ++j) {
        while (i < attn.size() && !attn[i].comp_free) {
            ++i;
        }
        if (i < attn.size()) {
            off_lid.push_back(off_k[i++]);
        } else {
            off_lid.push_back(off);
            off += n_block_tokens*row_bytes;
        }
    }

    // the draft layers after the others, at a row stride of their own
    n_attn_main = attn.size();
    if (!draft.empty()) {
        row_bytes_draft      = llama_box_row_stride(draft.data(), draft.size());
        rows_per_block_draft = block_bytes/row_bytes_draft;

        for (const auto & l : draft) {
            attn.push_back(l);
            off_k.push_back(off);
            off += n_block_tokens*row_bytes_draft;
            off_v.push_back(off);
            off += l.has_v ? n_block_tokens*row_bytes_draft : 0;
        }
    }

    if (!recr.empty()) {
        const auto & l0 = recr[0];

        r_row_bytes = GGML_PAD(ggml_row_size(l0.type_r, l0.n_embd_r), unit);
        s_row_bytes = GGML_PAD(ggml_row_size(l0.type_s, l0.n_embd_s), unit);
        x_row_bytes = l0.n_embd_x > 0 ? GGML_PAD(ggml_row_size(l0.type_r, l0.n_embd_x), unit) : 0;

        // the layers with a conv history all have the same one
        for (const auto & l : recr) {
            c_row_bytes = std::max(c_row_bytes, l.n_embd_c > 0 ? GGML_PAD(ggml_row_size(l.type_r, l.n_embd_c), unit) : 0);
        }

        int32_t il_max = 0;
        for (const auto & l : recr) {
            il_max = std::max(il_max, l.il);
        }
        recr_idx.assign(il_max + 1, -1);

        // the pieces of the layers one after the other
        for (size_t i = 0; i < recr.size(); ++i) {
            recr_idx[recr[i].il] = (int32_t) i;

            pc_r.push_back(state_bytes);
            state_bytes += recr[i].n_rows_r*r_row_bytes;
            pc_s.push_back(state_bytes);
            state_bytes += recr[i].n_rows_s*s_row_bytes;
            pc_x.push_back(state_bytes);
            state_bytes += 2*x_row_bytes;
            pc_c.push_back(state_bytes);
            state_bytes += recr[i].n_embd_c > 0 ? recr[i].n_rows_r*c_row_bytes : 0;
        }

        // a model without attention has no tokens to hold: a block is a state
        if (block_bytes == 0) {
            block_bytes = state_bytes;
        }

        state_blocks = states_blocks(1);
    }

    if (!pieces.empty()) {
        GGML_ASSERT(recr.empty());

        for (const auto & p : pieces) {
            off_p.push_back(state_bytes);
            state_bytes += p.n_rows*GGML_PAD(ggml_row_size(p.type, p.n_embd), unit);
        }

        state_blocks = states_blocks(1);
    }

    // without bytes: the n_tokens tokens in any split over the sequences, each of which leaves at most one block partly
    // empty, and the states. room for one state and a block of tokens, at least
    const uint32_t n_blocks_tokens = n_tokens > 0 ? (n_tokens + n_block_tokens - 1)/n_block_tokens + std::max(n_seq_max, 1u) - 1 : 0;

    n_blocks = bytes > 0 ? (uint32_t) (bytes/block_bytes) : n_blocks_tokens + states_blocks(n_states_in);
    n_blocks = std::max(n_blocks, states_blocks(1) + 1);

    n_blocks_max = n_blocks;

    seqs.resize(n_blocks);

    clear();

    LLAMA_LOG_INFO("%s: %u blocks of %.2f MiB (%u tokens), %.2f MiB; a state takes %.2f MiB\n", __func__,
            n_blocks, block_bytes/1024.0/1024.0, n_block_tokens, get_total_bytes()/1024.0/1024.0, state_bytes/1024.0/1024.0);
}

ggml_tensor * llama_box::view(ggml_type type, int64_t ne0, size_t nb1, size_t offs, const char * name, int32_t il) {
    const size_t total    = get_total_bytes();
    const size_t row_size = ggml_row_size(type, ne0);

    GGML_ASSERT(offs + row_size <= total);

    // every row that fits in the box from the view offset, the last one included
    const int64_t ne1 = (int64_t) ((total - offs - row_size)/nb1) + 1;

    // the rows overlap when nb1 is less than a row (the state views), which the shape checks of a regular view refuse,
    // so the view is set up by hand; it is initialized as a view of the pool when the pool is allocated
    ggml_tensor * t = ggml_new_tensor_2d(ctx.get(), type, ne0, ne1);
    t->nb[1] = nb1;
    t->nb[2] = nb1*ne1;
    t->nb[3] = nb1*ne1;
    t->view_src  = pool;
    t->view_offs = offs;

    if (il >= 0) {
        ggml_format_name(t, "box_%s_l%d", name, il);
    } else {
        ggml_format_name(t, "box_%s", name);
    }

    return t;
}

void llama_box::finalize(size_t bytes) {
    GGML_ASSERT(buft != nullptr);

    n_blocks = std::max(std::min(n_blocks_max, (uint32_t) (bytes/block_bytes)), states_blocks(1) + 1);

    const size_t total = get_total_bytes();

    // the size of the views follows the box, the tensors stay: the caches hold them
    for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
        t->buffer = nullptr;
        t->data   = nullptr;

        if (t == pool) {
            t->ne[0] = (int64_t) total;
            t->nb[1] = t->nb[0]*total;
            t->nb[2] = t->nb[1];
            t->nb[3] = t->nb[2];
            continue;
        }

        const size_t nb1 = t->nb[1];

        t->ne[1] = (int64_t) ((total - t->view_offs - ggml_row_size(t->type, t->ne[0]))/nb1) + 1;
        t->nb[2] = nb1*t->ne[1];
        t->nb[3] = nb1*t->ne[1];
    }

    buf.reset();
    buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
    if (!buf) {
        throw std::runtime_error("failed to allocate the box");
    }

    clear();

    LLAMA_LOG_INFO("%s: %u blocks of %.2f MiB (%u tokens), %.2f MiB; a state takes %.2f MiB\n", __func__,
            n_blocks, block_bytes/1024.0/1024.0, n_block_tokens, total/1024.0/1024.0, state_bytes/1024.0/1024.0);
}

void llama_box::alloc(ggml_backend_buffer_type_t buft_in, bool no_alloc) {
    buft = buft_in;

    ggml_init_params params = {
        /*.mem_size   =*/ (5 + 2*attn.size() + lid.size() + pieces.size())*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ctx.reset(ggml_init(params));
    if (!ctx) {
        throw std::runtime_error("failed to create the ggml context of the box");
    }

    // the pool comes first in the context, so that it is allocated before its views are initialized
    pool = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I8, (int64_t) get_total_bytes());
    ggml_set_name(pool, "box");

    int32_t il_max = 0;
    for (const auto & l : attn) {
        il_max = std::max(il_max, l.il);
    }

    t_k.assign(il_max + 1, nullptr);
    t_v.assign(il_max + 1, nullptr);

    for (size_t i = 0; i < attn.size(); ++i) {
        const auto & l = attn[i];
        const size_t row = i >= n_attn_main ? row_bytes_draft : row_bytes;
        t_k[l.il] = view(l.type_k, l.n_embd_k, row, off_k[i], "k", l.il);
        if (l.has_v) {
            t_v[l.il] = view(l.type_v, l.n_embd_v, row, off_v[i], "v", l.il);
        }
    }

    t_lid.assign(il_max + 1, nullptr);
    for (size_t i = 0; i < lid.size(); ++i) {
        t_lid[lid[i].il] = view(lid[i].type_k, lid[i].n_embd_k, row_bytes, off_lid[i], "lid", lid[i].il);
    }

    t_p.resize(pieces.size());
    for (size_t i = 0; i < pieces.size(); ++i) {
        t_p[i] = view(pieces[i].type, pieces[i].n_embd, unit, off_p[i], "state", (int32_t) i);
    }

    GGML_ASSERT(block_bytes % sizeof(float) == 0);
    t_blocks = view(GGML_TYPE_F32, (int64_t) (block_bytes/sizeof(float)), block_bytes, 0, "blocks", -1);

    if (!recr.empty()) {
        const auto & l0 = recr[0];
        t_r = view(l0.type_r, l0.n_embd_r, unit, 0, "r", -1);
        t_s = view(l0.type_s, l0.n_embd_s, unit, 0, "s", -1);
        if (l0.n_embd_x > 0) {
            t_x = view(l0.type_r, l0.n_embd_x, unit, 0, "x", -1);
        }
        for (const auto & l : recr) {
            if (l.n_embd_c > 0) {
                t_c = view(l.type_r, l.n_embd_c, unit, 0, "c", -1);
                break;
            }
        }
    }

    if (no_alloc) {
        // a dry run: a dummy buffer, so that the scheduler does not try to allocate the views
        buf.reset(ggml_backend_buft_alloc_buffer(buft, 0));
        for (ggml_tensor * t = ggml_get_first_tensor(ctx.get()); t != nullptr; t = ggml_get_next_tensor(ctx.get(), t)) {
            t->buffer = buf.get();
        }
    } else {
        buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx.get(), buft));
        if (!buf) {
            throw std::runtime_error("failed to allocate the box");
        }
    }

    LLAMA_LOG_INFO("%s: %10s box buffer size = %8.2f MiB\n", __func__, ggml_backend_buft_name(buft), get_total_bytes()/1024.0/1024.0);
}

ggml_tensor * llama_box::get_k(int32_t il) const { return t_k.at(il); }

bool llama_box::is_draft(int32_t il) const {
    for (size_t i = n_attn_main; i < attn.size(); ++i) {
        if (attn[i].il == il) {
            return true;
        }
    }
    return false;
}
ggml_tensor * llama_box::get_v(int32_t il) const { return t_v.at(il); }

int64_t llama_box::r_row(int32_t s, uint32_t l, uint32_t k) const {
    return (int64_t) (((size_t) s*state_bytes + pc_r[l] + k*r_row_bytes)/unit);
}

int64_t llama_box::s_row(int32_t s, uint32_t l, uint32_t k) const {
    return (int64_t) (((size_t) s*state_bytes + pc_s[l] + k*s_row_bytes)/unit);
}

int64_t llama_box::x_row(int32_t s, uint32_t l, uint32_t h) const {
    return (int64_t) (((size_t) s*state_bytes + pc_x[l] + h*x_row_bytes)/unit);
}

int64_t llama_box::c_row(int32_t s, uint32_t l, uint32_t k) const {
    return (int64_t) (((size_t) s*state_bytes + pc_c[l] + k*c_row_bytes)/unit);
}

uint32_t llama_box::states_blocks(uint32_t n) const {
    return (uint32_t) (((size_t) n*state_bytes + block_bytes - 1)/block_bytes);
}

uint32_t llama_box::block_pop() {
    GGML_ASSERT(!free_blocks.empty());

    const auto it = std::prev(free_blocks.end());
    const uint32_t b = *it;
    free_blocks.erase(it);

    return b;
}

int32_t llama_box::state_alloc() {
    if (state_bytes == 0) {
        return -1;
    }

    if (!free_states.empty()) {
        const int32_t s = *free_states.begin();
        free_states.erase(free_states.begin());
        return s;
    }

    const uint32_t b0 = states_blocks(n_states);
    const uint32_t b1 = states_blocks(n_states + 1);

    if (b1 > n_blocks) {
        return -1;
    }
    for (uint32_t b = b0; b < b1; ++b) {
        if (free_blocks.count(b) == 0) {
            return -1;
        }
    }
    for (uint32_t b = b0; b < b1; ++b) {
        free_blocks.erase(b);
    }

    return (int32_t) n_states++;
}

void llama_box::state_free(int32_t s) {
    GGML_ASSERT(s >= 0 && (uint32_t) s < n_states && free_states.count(s) == 0);

    free_states.insert(s);

    // the free states at the right end go back to being blocks
    while (n_states > 0 && free_states.count((int32_t) n_states - 1) > 0) {
        free_states.erase((int32_t) n_states - 1);

        const uint32_t b1 = states_blocks(n_states);
        n_states--;
        for (uint32_t b = states_blocks(n_states); b < b1; ++b) {
            free_blocks.insert(b);
        }
    }
}

bool llama_box::block_take(llama_seq_id seq_id) {
    auto & seq = seqs.at(seq_id);

    if (free_blocks.empty()) {
        reclaim();
    }
    if (free_blocks.empty()) {
        return false;
    }

    seq.blocks.push_back(block_pop());

    block_charge(seq_id);

    return true;
}

uint32_t llama_box::seq_n_held(llama_seq_id seq_id) const {
    const auto & seq = seqs.at(seq_id);

    uint32_t n_held = (uint32_t) seq.blocks.size();
    for (const auto & blocks : seq.comp) {
        for (const uint32_t b : blocks) {
            n_held += comp_owner[b] == seq_id;
        }
    }
    return n_held;
}

void llama_box::block_charge(llama_seq_id seq_id) {
    auto & seq = seqs.at(seq_id);

    const uint32_t n_held = seq_n_held(seq_id);

    if (seq.admitted && n_held > seq.n_input_blocks && seq.reserve > 0) {
        seq.reserve--;
        reserved--;
    }
}

bool llama_box::comp_take(llama_seq_id seq_id, block_kind kind, uint32_t n_blocks_kind) {
    auto & seq = seqs.at(seq_id);

    GGML_ASSERT(kind != BLOCK_RAW);

    while (seq.comp[kind].size() < n_blocks_kind) {
        if (free_blocks.empty()) {
            reclaim();
        }
        if (free_blocks.empty()) {
            return false;
        }

        // a state of the seq covers block 0, under which the row views of the layers would not reach
        seq.comp[kind].push_back(block_pop());
        GGML_ASSERT(seq.comp[kind].back() > 0);

        comp_refs [seq.comp[kind].back()] = 1;
        comp_owner[seq.comp[kind].back()] = seq_id;

        block_charge(seq_id);
    }

    return true;
}

void llama_box::comp_release(llama_seq_id seq_id, uint32_t b) {
    GGML_ASSERT(comp_refs[b] > 0);

    if (comp_owner[b] == seq_id) {
        comp_owner[b] = -1;
    }
    if (--comp_refs[b] == 0) {
        free_blocks.insert(b);
    }
}

void llama_box::comp_clear(llama_seq_id seq_id, block_kind kind) {
    auto & seq = seqs.at(seq_id);

    for (const uint32_t b : seq.comp[kind]) {
        comp_release(seq_id, b);
    }
    seq.comp[kind].clear();
}

bool llama_box::comp_own(llama_seq_id seq_id, block_kind kind, uint32_t i0, std::vector<std::pair<uint32_t, uint32_t>> & copies) {
    auto & blocks = seqs.at(seq_id).comp[kind];

    for (size_t i = i0; i < blocks.size(); ++i) {
        const uint32_t b = blocks[i];
        if (comp_refs[b] < 2) {
            continue;
        }

        if (free_blocks.empty()) {
            reclaim();
        }
        if (free_blocks.empty()) {
            return false;
        }

        const uint32_t c = block_pop();
        GGML_ASSERT(c > 0);

        comp_refs [c] = 1;
        comp_owner[c] = seq_id;

        comp_release(seq_id, b);
        blocks[i] = c;

        copies.emplace_back(b, c);

        block_charge(seq_id);
    }

    return true;
}

void llama_box::comp_copy(const std::vector<std::pair<uint32_t, uint32_t>> & copies) {
    std::vector<uint8_t> data(copies.empty() ? 0 : block_bytes);

    for (const auto & [b, c] : copies) {
        ggml_backend_tensor_get(pool, data.data(), (size_t) b*block_bytes, block_bytes);
        ggml_backend_tensor_set(pool, data.data(), (size_t) c*block_bytes, block_bytes);
    }
}

bool llama_box::comp_share(llama_seq_id seq_id_src, llama_seq_id seq_id_dst, block_kind kind, uint32_t i_own) {
    GGML_ASSERT(kind != BLOCK_RAW && seq_id_src != seq_id_dst);

    auto & dst = seqs.at(seq_id_dst);

    comp_clear(seq_id_dst, kind);

    const std::vector<uint32_t> blocks = seqs.at(seq_id_src).comp[kind];

    std::vector<uint8_t> data;

    for (size_t i = 0; i < blocks.size(); ++i) {
        const uint32_t b = blocks[i];

        if (i < i_own) {
            comp_refs[b]++;
            dst.comp[kind].push_back(b);
            continue;
        }

        if (free_blocks.empty()) {
            reclaim();
        }
        if (free_blocks.empty()) {
            return false;
        }

        const uint32_t c = block_pop();
        GGML_ASSERT(c > 0);

        comp_refs [c] = 1;
        comp_owner[c] = seq_id_dst;

        data.resize(block_bytes);
        ggml_backend_tensor_get(pool, data.data(), (size_t) b*block_bytes, block_bytes);
        ggml_backend_tensor_set(pool, data.data(), (size_t) c*block_bytes, block_bytes);

        dst.comp[kind].push_back(c);

        block_charge(seq_id_dst);
    }

    return true;
}

void llama_box::comp_get(llama_seq_id seq_id, block_kind kind, uint32_t i, void * data) const {
    ggml_backend_tensor_get(pool, data, (size_t) seqs.at(seq_id).comp[kind].at(i)*block_bytes, block_bytes);
}

void llama_box::comp_set(llama_seq_id seq_id, block_kind kind, uint32_t i, const void * data) {
    ggml_backend_tensor_set(pool, data, (size_t) seqs.at(seq_id).comp[kind].at(i)*block_bytes, block_bytes);
}

void llama_box::state_get(int32_t s, void * data) const {
    ggml_backend_tensor_get(pool, data, (size_t) s*state_bytes, state_bytes);
}

void llama_box::state_set(int32_t s, const void * data) {
    ggml_backend_tensor_set(pool, data, (size_t) s*state_bytes, state_bytes);
}

void llama_box::state_copy(int32_t src, int32_t dst) {
    std::vector<uint8_t> data(state_bytes);

    state_get(src, data.data());
    state_set(dst, data.data());
}

void llama_box::seq_evict(llama_seq_id seq_id) {
    auto & seq = seqs.at(seq_id);

    for (size_t i = 0; i < seq.blocks.size();) {
        const uint32_t b = seq.blocks[i];

        if (block_used(b)) {
            ++i;
            continue;
        }

        free_blocks.insert(b);
        seq.blocks.erase(seq.blocks.begin() + i);

        if (seq.admitted) {
            seq.n_input_blocks -= seq.n_input_blocks > 0 ? 1 : 0;
            seq.reserve++;
            reserved++;
        }
    }
}

void llama_box::add_user(const void * user, std::function<bool(uint32_t)> fn) {
    users.emplace_back(user, std::move(fn));
}

void llama_box::remove_user(const void * user) {
    users.erase(std::remove_if(users.begin(), users.end(), [&](const auto & u) { return u.first == user; }), users.end());
}

bool llama_box::block_used(uint32_t b) const {
    return std::any_of(users.begin(), users.end(), [&](const auto & u) { return u.second(b); });
}

void llama_box::seq_release(llama_seq_id seq_id) {
    auto & seq = seqs.at(seq_id);

    if (seq.admitted) {
        return;
    }

    for (const uint32_t b : seq.blocks) {
        if (block_used(b)) {
            held_blocks.push_back(b);
        } else {
            free_blocks.insert(b);
        }
    }

    seq.blocks.clear();

    // a compressed block other seqs hold stays theirs
    for (auto & blocks : seq.comp) {
        for (const uint32_t b : blocks) {
            comp_release(seq_id, b);
        }
        blocks.clear();
    }
}

void llama_box::reclaim() {
    for (size_t i = 0; i < held_blocks.size();) {
        if (block_used(held_blocks[i])) {
            ++i;
            continue;
        }
        free_blocks.insert(held_blocks[i]);
        held_blocks[i] = held_blocks.back();
        held_blocks.pop_back();
    }
}

void llama_box::clear() {
    free_blocks.clear();
    for (uint32_t b = 0; b < n_blocks; ++b) {
        free_blocks.insert(free_blocks.end(), b);
    }

    held_blocks.clear();
    free_states.clear();
    n_states = 0;

    for (auto & seq : seqs) {
        seq = {};
    }

    comp_refs .assign(n_blocks, 0);
    comp_owner.assign(n_blocks, -1);

    reserved = 0;
}

uint32_t llama_box::blocks_input(uint32_t n_input) const {
    if (blocks_fn) {
        return blocks_fn(n_input);
    }

    return attn.empty() ? 0 : (n_input + n_block_tokens - 1)/n_block_tokens;
}

uint32_t llama_box::blocks_output(uint32_t n_input, uint32_t n_output) const {
    if (attn.empty()) {
        return 0;
    }

    const uint64_t n_all = (uint64_t) n_input + n_output;

    if (blocks_fn) {
        return blocks_fn((uint32_t) std::min<uint64_t>(n_all, UINT32_MAX)) - blocks_fn(n_input);
    }

    return (uint32_t) ((n_all + n_block_tokens - 1)/n_block_tokens) - blocks_input(n_input);
}

bool llama_box::reserve(llama_seq_id seq_id, uint32_t n_input, uint32_t n_output, const std::function<int32_t()> & state_alloc_fn) {
    auto & seq = seqs.at(seq_id);

    // the input blocks the seq does not hold yet (it may hold a reused prompt), and the output after them; a block shared
    // with it is charged to another seq, its own copies of the blocks it will write are not, see comp_share()
    const uint32_t n_held = seq_n_held(seq_id);

    const uint32_t n_take_input = blocks_input(n_input) > n_held ? blocks_input(n_input) - n_held : 0;

    // blocks of several kinds are taken as the tokens show which ones: all of them are a charge
    const uint32_t n_take = blocks_fn ? 0 : n_take_input;
    const uint32_t n_out  = blocks_output(n_input, n_output) + (blocks_fn ? n_take_input : 0);

    // a state between the states in use takes no block
    const bool     need_state = state_alloc_fn != nullptr;
    const uint32_t n_state    = need_state && free_states.empty() ? states_blocks(n_states + 1) - states_blocks(n_states) : 0;
    const uint32_t n_need     = n_state + n_take + n_out;

    const uint32_t reserved_other = reserved - (seq.admitted ? seq.reserve : 0);

    // a draft context gives its cells back after the target context
    if (free_blocks.size() < (size_t) reserved_other + n_need) {
        reclaim();
    }
    if (free_blocks.size() < (size_t) reserved_other + n_need) {
        return false;
    }

    if (need_state && state_alloc_fn() < 0) {
        return false;
    }

    if (seq.admitted) {
        reserved -= seq.reserve;
    }
    seq.admitted = false; // the input blocks are taken outside of the charge

    for (uint32_t i = 0; i < n_take; ++i) {
        const bool ok = block_take(seq_id);
        GGML_ASSERT(ok); // counted above
    }

    seq.admitted       = true;
    seq.n_input_blocks = n_held + n_take;
    seq.reserve        = n_out;

    reserved += n_out;

    return true;
}

void llama_box::reserve_drop(llama_seq_id seq_id) {
    auto & seq = seqs.at(seq_id);

    if (seq.admitted) {
        reserved -= seq.reserve;
    }

    seq.admitted       = false;
    seq.n_input_blocks = 0;
    seq.reserve        = 0;
}

// the V head attention reads of a layer: a layer without V rows (MLA) reads the compressed KV at the start of its K rows
static uint32_t llama_box_head_v(const llama_hparams & hparams, const llama_box::attn_layer & attn) {
    const uint32_t n_lora_kv = hparams.is_swa(attn.il) ? hparams.n_lora_kv_swa : hparams.n_lora_kv;

    return !attn.has_v && n_lora_kv > 0 ? n_lora_kv : hparams.n_embd_head_v(attn.il);
}

// whether the device reads the rows of a layer in place, through ranges of the cells of each token, as the graph builds
// its attention (head sizes, softcap, ALiBi); row_lists tells whether it also reads the rows of a prompt through one
// list, each query seeing a prefix of it, which the tensor core kernels do; the context only describes the ops, nothing
// is allocated
static bool llama_box_reads_rows(const llama_hparams & hparams, ggml_backend_buffer_type_t buft, const llama_box::attn_layer & attn, bool * row_lists) {
    ggml_init_params params = {
        /*.mem_size   =*/ 32*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx { ggml_init(params) };
    if (!ctx) {
        return false;
    }

    // the CPU buffer type has no device of its own
    ggml_backend_dev_t dev = ggml_backend_buft_get_device(buft);
    if (dev == nullptr && ggml_backend_buft_is_host(buft)) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    if (dev == nullptr) {
        return false;
    }

    // an MLA layer reads its V rows as the first of its K rows, the compressed KV
    const int64_t hs_k  = hparams.n_embd_head_k(attn.il);
    const int64_t hs_v  = llama_box_head_v(hparams, attn);
    const int64_t nh    = hparams.n_head_kv(attn.il);
    const int64_t n_q   = hparams.n_head(attn.il);
    const int64_t nb    = 32;
    const int64_t n_kv  = 1024;
    const int64_t n_idx = 256;

    const float max_bias = hparams.use_alibi ? hparams.f_max_alibi_bias : 0.0f;
    const float softcap  = hparams.attn_soft_cap ? hparams.f_attn_logit_softcapping : 0.0f;

    const auto attention = [&]() {
        ggml_tensor * q = ggml_new_tensor_4d(ctx.get(), GGML_TYPE_F32, hs_k, nb, n_q, 1);
        ggml_tensor * k = ggml_new_tensor_4d(ctx.get(), attn.type_k, hs_k, n_kv, nh, 1);
        ggml_tensor * v = attn.has_v ? ggml_new_tensor_4d(ctx.get(), attn.type_v, hs_v, n_kv, nh, 1) :
            ggml_view_4d(ctx.get(), k, hs_v, n_kv, nh, 1, k->nb[1], k->nb[2], k->nb[3], 0);

        ggml_tensor * res = ggml_flash_attn_ext(ctx.get(), q, k, v, nullptr, 1.0f, max_bias, softcap);
        ggml_prec_set_acc(res, GGML_PREC_F32);
        return res;
    };

    // the relative bias by distance of a T5 decoder, see build_pos_bias_dist()
    const bool rel_bias = hparams.n_rel_attn_bkts > 0;

    // a prompt reads one list of rows without a softcap, ALiBi and a relative bias only, with them it reads through
    // ranges like every token, whose lists hold the positions of the rows for ALiBi and the relative bias
    ggml_tensor * fa_rows = nullptr;
    if (softcap == 0.0f && max_bias == 0.0f && !rel_bias) {
        fa_rows = attention();
        ggml_flash_attn_ext_set_kv_rows_hi(fa_rows,
                ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_idx), ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, nb));
    }

    ggml_tensor * fa_idx = attention();
    ggml_flash_attn_ext_set_kv_idx(fa_idx,
            ggml_new_tensor_3d(ctx.get(), GGML_TYPE_I32, n_idx, 1, max_bias != 0.0f || rel_bias ? 2 : 1),
            ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, 3, nb));
    if (rel_bias) {
        ggml_flash_attn_ext_set_rel_bias(fa_idx, ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, LLAMA_REL_POS_N_DIST, n_q));
    }

    if (row_lists) {
        *row_lists = fa_rows != nullptr && ggml_backend_dev_supports_op(dev, fa_rows);

        if (const char * env = getenv("LOCAL_INFERENCE_BOX_ROW_LISTS"); env && atoi(env) == 0) {
            *row_lists = false;
        }
    }

    return ggml_backend_dev_supports_op(dev, fa_idx);
}

std::unique_ptr<llama_box> llama_box_make(
        const llama_model & model,
                ggml_type   type_k,
                ggml_type   type_v,
                 uint32_t   kv_size,
                 uint32_t   n_seq_max,
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
    std::vector<llama_box::attn_layer>   extra_attn,
    std::vector<llama_box::attn_layer>   lid,
    std::vector<llama_box::state_piece>  pieces,
    const llama_hparams * hparams_attn) {
    const auto & hparams = hparams_attn ? *hparams_attn : model.hparams;

    const bool has_recr = filter_recr != nullptr;

    // the state of a seq has one scan row: no rollback, or a rollback that replays (see llama_memory_recurrent::rs_replay)
    const bool rs_replay = rs_replay_req && n_rs_seq > 0 && llm_arch_supports_rs_replay(model.arch) && hparams.n_embd_gdn_inp() > 0;

    std::vector<llama_box::attn_layer> attn;
    std::vector<llama_box::recr_layer> recr;

    std::vector<llama_box::attn_layer> draft;

    bool row_lists = false;

    ggml_backend_buffer_type_t buft = nullptr;

    const auto layer_buft = [&](uint32_t il) {
        return offload ? ggml_backend_dev_buffer_type(model.dev_layer(il)) : ggml_backend_cpu_buffer_type();
    };

    for (uint32_t il = 0; il < hparams.n_layer_all; ++il) {
        if (!hparams.has_kv(il) || (filter_attn && !filter_attn(il))) {
            continue;
        }
        buft = layer_buft(il);
        attn.push_back({ (int32_t) il, type_k, type_v, hparams.n_embd_k_gqa(il), hparams.n_embd_v_gqa(il), !hparams.is_mla() });
    }

    for (uint32_t il = 0; has_recr && il < hparams.n_layer(); ++il) {
        if (!filter_recr(il)) {
            continue;
        }
        buft = layer_buft(il);
        const uint32_t n_embd_x = rs_replay ? hparams.n_embd_gdn_inp()*(1 + n_rs_seq) : 0;
        recr.push_back({ (int32_t) il, type_r, type_s, hparams.n_embd_r(), hparams.n_embd_s(), n_embd_x, 1 + n_rs_seq, rs_replay ? 1 : 1 + n_rs_seq,
                hparams.is_ple(il) ? hparams.ple_conv_state() : 0 });
    }

    // the layers of a draft keep their rows in the blocks, next to the layers of the same tokens: the MTP layers of the
    // model or of an MTP head in a GGUF of its own, at their own slots (the layers of the model tell where its head
    // starts, hparams_attn may describe the layers of one kind only), or all the layers of a draft model of its own, in
    // the slots after those of the model and of its indexers (il + n_layer_all)
    const llama_hparams & hd = box_draft.hparams ? *box_draft.hparams : hparams;

    const int32_t  draft_offset = box_draft.standalone ? 2*(int32_t) model.hparams.n_layer_all : 0;
    const uint32_t draft_begin  = box_draft.standalone ? 0 : box_draft.hparams ? hd.n_layer() : model.hparams.n_layer();

    GGML_ASSERT(!box_draft.standalone || box_draft.hparams);

    for (uint32_t il = draft_begin; box_draft.enabled && il < hd.n_layer_all; ++il) {
        if (!hd.has_kv(il)) {
            continue;
        }
        const int32_t slot = (int32_t) il + draft_offset;
        GGML_ASSERT(std::none_of(attn.begin(), attn.end(), [&](const auto & l) { return l.il == slot; }));

        draft.push_back({ slot, box_draft.type_k, box_draft.type_v, hd.n_embd_k_gqa(il), hd.n_embd_v_gqa(il), !hd.is_mla() });
    }
    if (box_draft.enabled && draft.empty()) {
        throw std::runtime_error(box_draft.standalone ? "the draft model has no layer with a KV cache" :
                "an MTP draft was asked for, but the model has no MTP layer with a KV cache");
    }

    // the indexer rows take the blocks of the layers without CSA rows
    for (auto & l : attn) {
        l.comp_free = !lid.empty() && std::none_of(lid.begin(), lid.end(), [&](const auto & li) { return li.il == l.il; });
    }


    // every attention layer is read in place: there is nothing else; the extra layers are read by their own ops
    const auto check = [&](const llama_hparams & hp, const llama_box::attn_layer & l, size_t i) {
        bool lists = false;
        if (!llama_box_reads_rows(hp, buft, l, &lists)) {
            throw std::runtime_error(format("the device %s cannot read the KV rows of layer %d in place (K/V heads %u/%u, types %s/%s%s%s): "
                    "the elastic memory needs flash attention that reads lists of rows for it",
                    ggml_backend_buft_name(buft), l.il, hp.n_embd_head_k(l.il), llama_box_head_v(hp, l),
                    ggml_type_name(l.type_k), ggml_type_name(l.has_v ? l.type_v : l.type_k),
                    hp.attn_soft_cap ? ", logit softcap" : "", hp.use_alibi ? ", ALiBi" : ""));
        }
        row_lists = (i == 0 || row_lists) && lists;
    };

    for (size_t i = 0; i < attn.size(); ++i) {
        check(hparams, attn[i], i);
    }
    // the hparams of a draft layer are those of its layer in the draft
    for (size_t i = 0; i < draft.size(); ++i) {
        llama_box::attn_layer l = draft[i];
        l.il -= draft_offset;
        check(hd, l, attn.size() + i);
    }

    // the memory of box_bytes bytes, or of kv_size cells and rs_size states, elastic
    // append extra attention layers (e.g. indexer cache with different head dimensions)
    for (auto & l : extra_attn) {
        attn.push_back(std::move(l));
    }

    auto box = std::make_unique<llama_box>(std::move(attn), std::move(recr), draft, box_bytes, kv_size, n_seq_max, rs_size, std::move(lid), std::move(pieces));
    box->set_row_lists(row_lists);
    box->set_draft(box_draft.standalone ? box_draft.hparams : nullptr, draft_offset);
    box->alloc(buft, hparams.no_alloc || box_deferred);

    return box;
}
