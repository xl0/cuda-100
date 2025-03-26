#include <stdint.h>
#include <stdio.h>

#ifndef TILE_WIDTH
#ifdef __INTELLISENSE__
#define TILE_WIDTH 16
#else
#error "TILE_WIDTH must be defined"
#endif
#endif

#ifndef THREAD_COARSENING
#ifdef __INTELLISENSE__
#define THREAD_COARSENING 2
#else
#error "THREAD_COARSENING must be defined"
#endif
#endif

__global__ void matmul_fp32_tiled_coarse(float *m1, float *m2, float *res, uint32_t out_shape_0,
                                         uint32_t out_shape_1, uint32_t inner_dim, uint32_t) {
    int x = blockIdx.x * blockDim.x * THREAD_COARSENING + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // if (threadIdx.x == 0 && threadIdx.y == 0) {
    //     printf("blockIdx = (%d, %d), mx = %d, y = %d\n", blockIdx.x, blockIdx.y, x, y);
    // }

    __shared__ float m1_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float m2_tile[TILE_WIDTH][TILE_WIDTH];

    float R[THREAD_COARSENING];
    for (int i = 0; i < THREAD_COARSENING; i++) {
        R[i] = 0;
    }

    int m1_x = inner_dim;
    int m2_x = out_shape_1;

    // We are going to coarse the thread over x, so let's load the tile from the
    // second matrix.

    for (int tile = 0; tile < inner_dim / TILE_WIDTH; tile++) {
        m1_tile[threadIdx.y][threadIdx.x] = m1[y * m1_x + tile * TILE_WIDTH + threadIdx.x];

        // Now, we are going to calculate a bunch consecutive tiles one by one,
        // so we need to load the
        for (int c = 0; c < THREAD_COARSENING; c++) {
            m2_tile[threadIdx.y][threadIdx.x] =
                m2[(tile * TILE_WIDTH + threadIdx.y) * m2_x + c * TILE_WIDTH + x];

            __syncthreads();

            for (int i = 0; i < TILE_WIDTH; i++) {
                R[c] += m1_tile[threadIdx.y][i] * m2_tile[i][threadIdx.x];
            }

            __syncthreads();
        }
    }

    for (int c = 0; c < THREAD_COARSENING; c++) {
        res[y * out_shape_1 + c * TILE_WIDTH + x] = R[c];
    }
}
