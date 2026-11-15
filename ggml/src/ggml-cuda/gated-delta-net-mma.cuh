#pragma once

#include "common.cuh"

#include <cstdint>

struct ggml_cuda_gdn_mma_args {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    const float * state;
    float *       dst;
    float *       state_out;

    int64_t H;
    int64_t H_k;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t rq3;
    int64_t sq1;
    int64_t sq2;
    int64_t sq3;
    int64_t sv1;
    int64_t sv2;
    int64_t sv3;
    int64_t sb1;
    int64_t sb2;
    int64_t sb3;
    float   scale;
    bool    eligible;

    // the rows mode (ggml_gated_delta_net_rows): the state of seq s is read from row rows_in[s] of state (-1: zeros) and
    // written to row rows_out[s] of state_out (-1 or no rows_out: not written), rows row_stride floats apart
    const int32_t * rows_in    = nullptr;
    const int32_t * rows_out   = nullptr;
    int64_t         row_stride = 0;
};

bool ggml_cuda_gdn_mma_launch(int device, const ggml_cuda_gdn_mma_args & args, cudaStream_t stream);
