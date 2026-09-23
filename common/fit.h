#pragma once

#include "ggml.h"
#include "llama.h"

#include <vector>

enum common_params_fit_status {
    COMMON_PARAMS_FIT_STATUS_SUCCESS = 0, // found allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_FAILURE = 1, // could not find allocations that are projected to fit
    COMMON_PARAMS_FIT_STATUS_ERROR   = 2, // a hard error occurred, e.g. because no model could be found at the specified path
};

// a second model that shares the devices of the main model, e.g. a draft model
//   - its context follows the context of the main model, so its memory is measured again whenever that context changes
//   - shares_model tells the fit that the weights are already counted in the main model, as for an MTP context
struct common_fit_extra_model {
    const char * path_model;
    llama_model_params * mparams;
    llama_context_params * cparams;
    bool shares_model;
};

// fits mparams and cparams to free device memory (assumes system memory is unlimited)
//   - returns true if the parameters could be successfully modified to fit device memory
//   - this function is NOT thread safe because it modifies the global llama logger state
//   - only parameters that have the same value as in llama_default_model_params are modified
//     with the exception of the context size which is modified if and only if equal to 0
common_params_fit_status common_fit_params(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams,
                              float * tensor_split,          // writable buffer for tensor split, needs at least llama_max_devices elements
   llama_model_tensor_buft_override * tensor_buft_overrides, // writable buffer for overrides, needs at least llama_max_tensor_buft_overrides elements
                             size_t * margins,               // margins of memory to leave per device in bytes
                           uint32_t   n_ctx_min,             // minimum context size to set when trying to reduce memory use
      const common_fit_extra_model * extra,                  // model to fit alongside the main one, nullptr if there is none
                     ggml_log_level   log_level);            // minimum log level to print during fitting, lower levels go to debug log

// sizes the number of concurrent sequences to a device memory budget (summed over the devices the model runs on):
//   - finds the largest cparams->n_seq_max, at most its value on entry, for which the model weights, the memory for
//     cparams->n_ctx tokens in a unified cache, the per-sequence memory (e.g. recurrent state, sliding-window cache)
//     and the worst-case compute buffers of the main and the extra context fit in budget - budget_used
//   - the memory is measured by creating the contexts without allocating them, so it holds for any architecture
//   - budget 0 means the free memory of the devices; a model on the host alone is left as is
//   - budget_used is memory the caller takes from the same budget, e.g. a vision encoder
//   - returns false when not even one sequence fits, leaving cparams unchanged
//   - predicted and free_before, when given, report the memory the configuration is projected to use and the free
//     device memory it was measured against, so that the caller can compare them with what is used in the end
bool common_budget_params(
                   const char * path_model,
     const llama_model_params * mparams,
         llama_context_params * cparams,
 const common_fit_extra_model * extra,
                         size_t   budget,
                         size_t   budget_used,
                 ggml_log_level   log_level,
                       size_t *   predicted = nullptr,
                       size_t *   free_before = nullptr);

// the free device memory of the devices a model runs on, summed
size_t common_device_memory_free(const llama_model * model);

// logs the device memory taken since the memory budget measured free_at_start, after a stage of the startup
void common_device_memory_log(const llama_model * model, size_t free_at_start, const char * stage);

// print estimated memory to stdout
void common_fit_print(
                         const char * path_model,
                 llama_model_params * mparams,
               llama_context_params * cparams);

void common_memory_breakdown_print(const llama_context * ctx);

struct common_device_memory_data {
    int64_t total;
    int64_t free;
    size_t  model;
    size_t  context;
    size_t  compute;
};

using common_device_memory_data_vec = std::vector<common_device_memory_data>;

// Load a model + context with no_alloc and return the per-device memory breakdown.
common_device_memory_data_vec common_get_device_memory_data(
                         const char * path_model,
           const llama_model_params * mparams,
         const llama_context_params * cparams,
    std::vector<ggml_backend_dev_t> & devs,
                           uint32_t & hp_ngl,
                           uint32_t & hp_n_ctx_train,
                           uint32_t & hp_n_expert,
                     ggml_log_level   log_level);
