#include "llama-box.h"

#include "llama-impl.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

// the bytes of a block: n_block_tokens rows of every attention layer and of the draft layer, padded to whole rows of each
// (row indices) and whole units (state row indices)
static size_t llama_box_block_bytes(const std::vector<llama_box::attn_layer> & attn, const llama_box::attn_layer * draft) {
    const size_t row  = ggml_row_size(attn[0].type_k, attn[0].n_embd_k);
    const size_t rowd = draft ? ggml_row_size(draft->type_k, draft->n_embd_k) : row;
    const size_t size = 2*llama_box::n_block_tokens*(attn.size()*row + (draft ? rowd : 0));
    const size_t mult = std::lcm(std::lcm(row, rowd), llama_box::unit); // not a power of 2, GGML_PAD does not apply

    return (size + mult - 1)/mult*mult;
}

bool llama_box::supported(const std::vector<attn_layer> & attn, const std::vector<recr_layer> & recr, const attn_layer * draft) {
    if (attn.empty()) {
        return false;
    }

    // the K and V rows of every layer share the row indices of a cell, so they have one size
    const size_t row = ggml_row_size(attn[0].type_k, attn[0].n_embd_k);
    for (const auto & l : attn) {
        if (!l.has_v || ggml_row_size(l.type_k, l.n_embd_k) != row || ggml_row_size(l.type_v, l.n_embd_v) != row) {
            return false;
        }
    }

    if (draft && (!draft->has_v || ggml_row_size(draft->type_k, draft->n_embd_k) != ggml_row_size(draft->type_v, draft->n_embd_v))) {
        return false;
    }

    if (recr.empty()) {
        return true;
    }

    // the state rows of every layer are read through one view per kind, so each kind has one size
    const auto & l0 = recr[0];
    for (const auto & l : recr) {
        if (l.type_r != l0.type_r || l.type_s != l0.type_s || l.n_embd_r != l0.n_embd_r || l.n_embd_s != l0.n_embd_s ||
            l.n_embd_x != l0.n_embd_x || l.n_rows_r != l0.n_rows_r || l.n_rows_r == 0) {
            return false;
        }
    }

    return true;
}

llama_box::llama_box(std::vector<attn_layer> attn_in, std::vector<recr_layer> recr_in, const attn_layer * draft, size_t bytes, uint32_t n_tokens, uint32_t n_states_in, uint32_t n_seq_max) :
    attn(std::move(attn_in)), recr(std::move(recr_in)) {
    GGML_ASSERT(supported(attn, recr, draft));

    row_bytes      = ggml_row_size(attn[0].type_k, attn[0].n_embd_k);
    block_bytes    = llama_box_block_bytes(attn, draft);
    rows_per_block = block_bytes/row_bytes;

    for (size_t i = 0; i < attn.size(); ++i) {
        off_k.push_back(i*2*n_block_tokens*row_bytes);
        off_v.push_back(i*2*n_block_tokens*row_bytes + n_block_tokens*row_bytes);
    }

    // the draft layer after the others
    if (draft) {
        row_bytes_draft      = ggml_row_size(draft->type_k, draft->n_embd_k);
        rows_per_block_draft = block_bytes/row_bytes_draft;

        const size_t off = attn.size()*2*n_block_tokens*row_bytes;

        attn.push_back(*draft);
        off_k.push_back(off);
        off_v.push_back(off + n_block_tokens*row_bytes_draft);
    }

    if (!recr.empty()) {
        const auto & l0 = recr[0];

        r_row_bytes = GGML_PAD(ggml_row_size(l0.type_r, l0.n_embd_r), unit);
        s_row_bytes = GGML_PAD(ggml_row_size(l0.type_s, l0.n_embd_s), unit);
        x_row_bytes = l0.n_embd_x > 0 ? GGML_PAD(ggml_row_size(l0.type_r, l0.n_embd_x), unit) : 0;

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
            state_bytes += s_row_bytes;
            pc_x.push_back(state_bytes);
            state_bytes += 2*x_row_bytes;
        }

        state_blocks = states_blocks(1);
    }

    // room for one state and a block of tokens, at least
    n_blocks = bytes > 0 ? (uint32_t) (bytes/block_bytes) : (n_tokens + n_block_tokens - 1)/n_block_tokens + states_blocks(n_states_in);
    n_blocks = std::max(n_blocks, states_blocks(1) + 1);

    seqs.resize(n_seq_max);

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

void llama_box::alloc(ggml_backend_buffer_type_t buft, bool no_alloc) {
    ggml_init_params params = {
        /*.mem_size   =*/ (4 + 2*attn.size())*ggml_tensor_overhead(),
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
        const size_t row = ggml_row_size(l.type_k, l.n_embd_k);
        t_k[l.il] = view(l.type_k, l.n_embd_k, row, off_k[i], "k", l.il);
        t_v[l.il] = view(l.type_v, l.n_embd_v, row, off_v[i], "v", l.il);
    }

    if (!recr.empty()) {
        const auto & l0 = recr[0];
        t_r = view(l0.type_r, l0.n_embd_r, unit, 0, "r", -1);
        t_s = view(l0.type_s, l0.n_embd_s, unit, 0, "s", -1);
        if (l0.n_embd_x > 0) {
            t_x = view(l0.type_r, l0.n_embd_x, unit, 0, "x", -1);
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
ggml_tensor * llama_box::get_v(int32_t il) const { return t_v.at(il); }

int64_t llama_box::r_row(int32_t s, uint32_t l, uint32_t k) const {
    return (int64_t) (((size_t) s*state_bytes + pc_r[l] + k*r_row_bytes)/unit);
}

int64_t llama_box::s_row(int32_t s, uint32_t l) const {
    return (int64_t) (((size_t) s*state_bytes + pc_s[l])/unit);
}

int64_t llama_box::x_row(int32_t s, uint32_t l, uint32_t h) const {
    return (int64_t) (((size_t) s*state_bytes + pc_x[l] + h*x_row_bytes)/unit);
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

    // a block after the input of an admitted request is output, taken from its charge
    if (seq.admitted && seq.blocks.size() > seq.n_input_blocks && seq.reserve > 0) {
        seq.reserve--;
        reserved--;
    }

    return true;
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

    reserved = 0;
}

uint32_t llama_box::blocks_input(uint32_t n_input) const {
    return (n_input + n_block_tokens - 1)/n_block_tokens;
}

uint32_t llama_box::blocks_output(uint32_t n_input, uint32_t n_output) const {
    const uint64_t n_all = (uint64_t) n_input + n_output;

    return (uint32_t) ((n_all + n_block_tokens - 1)/n_block_tokens) - blocks_input(n_input);
}

bool llama_box::reserve(llama_seq_id seq_id, uint32_t n_input, uint32_t n_output, const std::function<int32_t()> & state_alloc_fn) {
    auto & seq = seqs.at(seq_id);

    // the input blocks the seq does not hold yet (it may hold a reused prompt), and the output after them
    const uint32_t n_held = (uint32_t) seq.blocks.size();
    const uint32_t n_take = blocks_input(n_input) > n_held ? blocks_input(n_input) - n_held : 0;
    const uint32_t n_out  = blocks_output(n_input, n_output);

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
    seq.n_input_blocks = (uint32_t) seq.blocks.size();
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
