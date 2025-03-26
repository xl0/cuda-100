#include <stdint.h>
#include <stdio.h>

__global__ void rgb_blur(uint8_t *in, uint8_t *out, uint32_t w, uint32_t h, uint32_t blur) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < w && y < h) {
        int idx = (y * w + x);

        for (int ch = 0; ch < 3; ch++) {
            uint32_t v = 0;
            for (int j = -blur; j <= (int)blur; j++) {
                for (int i = -blur; i <= (int)blur; i++) {
                    if (y + j >= 0   &&
                        y + j < h    &&
                        x + i >= 0   &&
                        x + i < w) {
                            v += in[ ((y + j) * w + x + i)*3 + ch];
                        }
                }
            }

            out[idx*3+ch] = (uint8_t)(v / ((2*blur + 1) * (2*blur + 1)));
        }
    }
}

