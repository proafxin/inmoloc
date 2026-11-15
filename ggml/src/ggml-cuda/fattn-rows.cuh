#pragma once

#include "common.cuh"

// the per query ranges of ggml_flash_attn_ext_set_kv_idx() at the head sizes the vector and tensor core kernels have no
// instance for: any head size up to FATTN_ROWS_D_MAX, f16, bf16 and f32 rows, read in place through the lists
#define FATTN_ROWS_D_MAX 512

bool ggml_cuda_flash_attn_ext_rows_supported(const ggml_tensor * dst);

void ggml_cuda_flash_attn_ext_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
