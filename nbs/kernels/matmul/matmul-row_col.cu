#include <stdint.h>
#include <stdio.h>

__global__ void matmul_f32(float* m1, float* m2, float* res,
                           uint32_t out_shape_0,
                           uint32_t out_shape_1,
                           uint32_t inner_dim)
{

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int m1_width = inner_dim;
    int m2_width = out_shape_1;

    double out;
    if (x < out_shape_1 && y < out_shape_0) {
        out = 0;
        for (int i = 0; i < inner_dim; i++) {
            out += m1[y * m1_width + i] * m2[i * m2_width + x];
        }
        res[y * out_shape_1 + x] = out;
    }
}

__global__ void matmul_f32_row(float* m1, float* m2, float* res,
                               uint32_t out_shape_0,
                               uint32_t out_shape_1,
                               uint32_t inner_dim,
                               uint32_t)

{
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int m1_width = inner_dim;
    int m2_width = out_shape_1;

    if (y < out_shape_0) {
        for (int x = 0; x < out_shape_1; x++) {
            double out = 0;
            for (int i = 0; i < inner_dim; i++) {
                out += m1[y * m1_width + i] * m2[i * m2_width + x];
            }
            res[y * out_shape_1 + x] = out;
        }
    }

}

__global__ void matmul_f32_col(float* m1, float* m2, float* res,
                               uint32_t out_shape_0,
                               uint32_t out_shape_1,
                               uint32_t inner_dim,
                               uint32_t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int m1_width = inner_dim;
    int m2_width = out_shape_1;

    if (x < out_shape_1) {
        for (int y = 0; y < out_shape_1; y++) {
            double out = 0;
            for (int i = 0; i < inner_dim; i++) {
                out += m1[y * m1_width + i] * m2[i * m2_width + x];
            }
            res[y * out_shape_1 + x] = out;
        }
    }
}
