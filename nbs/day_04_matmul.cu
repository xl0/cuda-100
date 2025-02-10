#include <stdint.h>
#include <stdio.h>

__global__ void matmul_f32(float *m1, float *m2, float *res,
    uint32_t out_shape_0,
    uint32_t out_shape_1,
    uint32_t inner_dim,
    uint32_t ) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int m1_width = inner_dim;
    int m2_width = out_shape_1;

    double out;
    if (x < out_shape_1 && y < out_shape_0) {
        out = 0;
        for (int i = 0; i < inner_dim; i++) {
            out += m1[y*m1_width + i] * m2[i*m2_width + x];
        }
        res[y*out_shape_1 + x] = out;
    }
}

