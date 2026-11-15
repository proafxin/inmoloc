// the cost of graph inputs in the backend scheduler: a GPU graph whose only work is n small row gathers, each reading
// its own index input (separate), a view of one input of all the indices (views), or a view of indices that are on the
// device already (device: the same compute without any input copy, the baseline)
//
// usage: test-sched-inputs [n_inputs ...]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char ** argv) {
    ggml_backend_load_all();

    ggml_backend_t gpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_GPU, nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!gpu || !cpu) {
        fprintf(stderr, "a GPU and a CPU backend are needed\n");
        return 1;
    }

    std::vector<int> ns;
    for (int i = 1; i < argc; ++i) {
        ns.push_back(atoi(argv[i]));
    }
    if (ns.empty()) {
        ns = { 1, 8, 48, 96, 300 };
    }

    // the rows the gathers read, on the device
    const int64_t n_embd = 256;
    const int64_t n_rows = 4096;
    const int64_t n_idx  = 32;

    ggml_init_params wp = { ggml_tensor_overhead(), nullptr, true };
    ggml_context_ptr ctx_w(ggml_init(wp));
    ggml_tensor * w = ggml_new_tensor_2d(ctx_w.get(), GGML_TYPE_F32, n_embd, n_rows);
    ggml_backend_buffer_ptr buf_w(ggml_backend_alloc_ctx_tensors(ctx_w.get(), gpu));
    {
        std::vector<float> data(n_embd*n_rows);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = (float) (i % 97)/97.0f;
        }
        ggml_backend_tensor_set(w, data.data(), 0, ggml_nbytes(w));
    }

    const int n_warmup = 20;
    const int n_iter   = 200;

    printf("%-9s %7s %12s %14s %7s\n", "mode", "inputs", "us/compute", "us/input", "splits");

    enum mode_t { SEPARATE, VIEWS, DEVICE };

    for (const mode_t mode : { SEPARATE, VIEWS, DEVICE }) {
        const bool views = mode != SEPARATE;
        for (const int n : ns) {
            const size_t graph_size = 4*n + 16;

            ggml_init_params gp = { (4*n + 16)*ggml_tensor_overhead() + ggml_graph_overhead_custom(graph_size, false), nullptr, true };
            ggml_context_ptr ctx(ggml_init(gp));

            std::vector<ggml_tensor *> idx(n);
            ggml_tensor * all = nullptr;
            ggml_context_ptr ctx_dev;
            ggml_backend_buffer_ptr buf_dev;
            if (mode == DEVICE) {
                // the indices in a device buffer, set once: no input to copy
                ggml_init_params dp = { ggml_tensor_overhead(), nullptr, true };
                ctx_dev.reset(ggml_init(dp));
                all = ggml_new_tensor_1d(ctx_dev.get(), GGML_TYPE_I32, n_idx*n);
                buf_dev.reset(ggml_backend_alloc_ctx_tensors(ctx_dev.get(), gpu));
                for (int i = 0; i < n; ++i) {
                    idx[i] = ggml_view_1d(ctx.get(), all, n_idx, i*n_idx*ggml_element_size(all));
                }
            } else if (views) {
                all = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_idx*n);
                ggml_set_input(all);
                for (int i = 0; i < n; ++i) {
                    idx[i] = ggml_view_1d(ctx.get(), all, n_idx, i*n_idx*ggml_element_size(all));
                }
            } else {
                for (int i = 0; i < n; ++i) {
                    idx[i] = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, n_idx);
                    ggml_set_input(idx[i]);
                }
            }

            ggml_tensor * out = nullptr;
            for (int i = 0; i < n; ++i) {
                ggml_tensor * rows = ggml_get_rows(ctx.get(), w, idx[i]);
                out = out ? ggml_add(ctx.get(), out, rows) : rows;
            }
            ggml_set_output(out);

            ggml_cgraph * gf = ggml_new_graph_custom(ctx.get(), graph_size, false);
            ggml_build_forward_expand(gf, out);

            ggml_backend_t backends[2] = { gpu, cpu };
            ggml_backend_sched_t sched = ggml_backend_sched_new(backends, nullptr, 2, graph_size, false, true);

            if (!ggml_backend_sched_alloc_graph(sched, gf)) {
                fprintf(stderr, "failed to allocate the graph\n");
                return 1;
            }

            std::vector<int32_t> data(n_idx*n);
            const auto set_inputs = [&](int it) {
                for (size_t i = 0; i < data.size(); ++i) {
                    data[i] = (int32_t) ((i*7 + it) % n_rows);
                }
                if (mode == DEVICE) {
                    if (it == 0) {
                        ggml_backend_tensor_set(all, data.data(), 0, ggml_nbytes(all));
                    }
                } else if (views) {
                    ggml_backend_tensor_set(all, data.data(), 0, ggml_nbytes(all));
                } else {
                    for (int i = 0; i < n; ++i) {
                        ggml_backend_tensor_set(idx[i], data.data() + i*n_idx, 0, ggml_nbytes(idx[i]));
                    }
                }
            };

            for (int it = 0; it < n_warmup; ++it) {
                set_inputs(it);
                ggml_backend_sched_graph_compute(sched, gf);
            }

            const auto t0 = std::chrono::steady_clock::now();
            for (int it = 0; it < n_iter; ++it) {
                set_inputs(it);
                ggml_backend_sched_graph_compute(sched, gf);
            }
            const auto t1 = std::chrono::steady_clock::now();

            const double us = std::chrono::duration<double, std::micro>(t1 - t0).count()/n_iter;

            printf("%-9s %7d %12.1f %14.2f %7d\n", mode == SEPARATE ? "separate" : mode == VIEWS ? "views" : "device", n, us, us/n,
                    ggml_backend_sched_get_n_splits(sched));

            ggml_backend_sched_free(sched);
        }
    }

    ggml_backend_free(cpu);
    ggml_backend_free(gpu);

    return 0;
}
