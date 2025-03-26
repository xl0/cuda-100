#include <stdint.h>
#include <stdio.h>

__global__ void mat_vec_mul(float* m, float* v, float* res,
                            uint32_t m_height,
                            uint32_t m_width) {

    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float out;
    if (y < m_height) {
        out = 0;
        for (int i = 0; i < m_width; i++) {
            out += m[y * m_width + i] * v[i];
        }
        res[y] = out;
    }
}
