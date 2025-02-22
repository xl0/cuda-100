#include <stdint.h>
#include <stdio.h>

// We will use square blocks to keep things sane.
#define BLOCK_WIDTH 16

__global__ void matmul_fp32_tiled(float *m1, float *m2, float *res, uint32_t out_shape_0,
                                  uint32_t out_shape_1, uint32_t inner_dim, uint32_t) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ float m1_tile[BLOCK_WIDTH][BLOCK_WIDTH];
    __shared__ float m2_tile[BLOCK_WIDTH][BLOCK_WIDTH];

    int m1_x = inner_dim;
    int m2_x = out_shape_1;

    // Assume the matrices are multiples my block size on both dims.

    float R = 0;
    for (int tile = 0; tile < inner_dim / BLOCK_WIDTH; tile++) {
        m1_tile[threadIdx.y][threadIdx.x] = m1[y * m1_x + tile * BLOCK_WIDTH + threadIdx.x];
        m2_tile[threadIdx.y][threadIdx.x] = m2[(tile * BLOCK_WIDTH + threadIdx.y) * m2_x + x];

        __syncthreads();

        for (int i = 0; i < BLOCK_WIDTH; i++) {
            R += m1_tile[threadIdx.y][i] * m2_tile[i][threadIdx.x];
        }

        __syncthreads();
    }

    res[y * out_shape_1 + x] = R;
}

// Non-tiled version
__global__ void matmul_f32(float *m1, float *m2, float *res, uint32_t out_shape_0,
                           uint32_t out_shape_1, uint32_t inner_dim, uint32_t) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int m1_width = inner_dim;
    int m2_width = out_shape_1;

    float out;
    if (x < out_shape_1 && y < out_shape_0) {
        out = 0;
        for (int i = 0; i < inner_dim; i++) {
            out += m1[y * m1_width + i] * m2[i * m2_width + x];
        }
        res[y * out_shape_1 + x] = out;
    }
}
