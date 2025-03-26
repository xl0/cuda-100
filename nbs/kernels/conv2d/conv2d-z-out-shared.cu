#include <stdint.h>
#include <stdio.h>

#include "conv2d-helpers.h"

// This version copies each input channel into shared memory before performing the
// convolution. Grid Z is used for output channels, so each thread only handles one
// output channel
__global__ void conv2d_pad_z_out_shared(float *in,
                                        float *out,
                                        float *filter,
                                        int h,
                                        int w,
                                        int in_channels,
                                        int out_channels,
                                        int filter_size /* Must be an odd number */,
                                        float pad) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int out_ch = blockIdx.z;

    int filter_r = (filter_size - 1) / 2;

    extern __shared__ float cell[];

    // In and Out data dimensions:
    // 0 - channel
    // 1 - height
    // 2 - width

    // Filter dimensions:
    // 0 - out channels
    // 1 - in channels
    // 2 - height
    // 3 - width

    if (x >= w || y >= h) return;

#ifdef DEBUG
    if (x == 0 && y == 0) PRINT_INPUTS();
#endif

    // Loop over the output channels

    // // Pointer to the 2d slice of the output

    float *sub_output = out + out_ch * w * h;
    ACCUM_DTYPE R = 0;
    // Loop over the input channels
    for (int in_c = 0; in_c < in_channels; in_c++) {
        // Pointer to the 2d slice of the filter that corresponds to the active input and output
        // channels
        float *sub_filter = filter + (filter_size * filter_size * in_channels * out_ch) +
                            (filter_size * filter_size * in_c);
        // Pinter to the current channel in the input
        float *sub_input = in + (w * h * in_c);

        cell[threadIdx.y * blockDim.x + threadIdx.x] = sub_input[y * w + x];
        __syncthreads();  // Wait for all threads to load the input

        // Apply the filter to the input or the pad value for outside indices.
        for (int filter_y = 0; filter_y < filter_size; filter_y++) {
            for (int filter_x = 0; filter_x < filter_size; filter_x++) {
                int tile_x = threadIdx.x - filter_r + filter_x;
                int tile_y = threadIdx.y - filter_r + filter_y;

                int input_x = x - filter_r + filter_x;
                int input_y = y - filter_r + filter_y;

                if (tile_x >= 0 && tile_x < blockDim.x && tile_y >= 0 && tile_y < blockDim.y) {
                    R += cell[tile_y * blockDim.x + tile_x] *
                         sub_filter[filter_y * filter_size + filter_x];
                } else if (input_x >= 0 && input_x < w && input_y >= 0 && input_y < h) {
                    R += sub_input[input_y * w + input_x] *
                         sub_filter[filter_y * filter_size + filter_x];
                } else {
                    R += pad * sub_filter[filter_y * filter_size + filter_x];
                }
            }
        }

        __syncthreads();  // Wait for all threads to complete before we load the next input
    }

    sub_output[y * w + x] = R;
}
