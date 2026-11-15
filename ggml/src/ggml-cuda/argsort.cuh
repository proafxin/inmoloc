#include "common.cuh"

#define CUDA_ARGSORT_BLOCK_SIZE 1024

void ggml_cuda_op_argsort(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

int ggml_cuda_chunk_nrows(const size_t row_bytes, const int64_t nrows);

// the offsets of ggml_argsort_top_k_off(): the indices of a row plus the offset of the row, none without data
struct argsort_off {
    const char * data = nullptr;
    int64_t ne01 = 1; // the rows of the sorted tensor
    int64_t ne02 = 1;
    int64_t ne1  = 1; // the offsets, broadcast over the rows
    int64_t ne2  = 1;
    int64_t ne3  = 1;
    size_t  nb1  = 0;
    size_t  nb2  = 0;
    size_t  nb3  = 0;
    int64_t row0 = 0; // the first row of the rows sorted at once
};

static __device__ __forceinline__ int argsort_off_get(const argsort_off & off, int64_t row) {
    if (off.data == nullptr) {
        return 0;
    }
    row += off.row0;
    const int64_t i1 = row % off.ne01;
    const int64_t i2 = (row / off.ne01) % off.ne02;
    const int64_t i3 = row / (off.ne01*off.ne02);
    return *(const int *) (off.data + (i1 % off.ne1)*off.nb1 + (i2 % off.ne2)*off.nb2 + (i3 % off.ne3)*off.nb3);
}

#ifdef GGML_CUDA_USE_CUB
void argsort_f32_i32_cuda_cub(ggml_cuda_pool & pool,
                              const float *    x,
                              int *            dst,
                              const int        ncols,
                              const int        nrows,
                              ggml_sort_order  order,
                              cudaStream_t     stream,
                              argsort_off      off = {});
#endif  // GGML_CUDA_USE_CUB
void argsort_f32_i32_cuda_bitonic(const float *   x,
                                  int *           dst,
                                  const int       ncols,
                                  const int       nrows,
                                  ggml_sort_order order,
                                  cudaStream_t    stream,
                                  argsort_off     off = {});
