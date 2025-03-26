#include <stdint.h>
#include <stdio.h>

#include "conv2d-helpers.h"

/* In this version, we spawn extra threads to copy the tile into cache. */
__global__ void conv2d_pad_z_out_shared_halo(
    float *in,
    float *out,
    float *filter,
    int h,
    int w,
    int in_channels,
    int out_channels,
    int filter_size /* Must be an odd number */,
    float pad
#ifdef SINGLE_BLOCK
    ,
    int *debug_counter /* This allows you to run the kernel one block at a time */
#endif

) {
    int filter_r = (filter_size - 1) / 2;

    int output_suze = TILE_SIZE - filter_size + 1;

    int out_x = blockIdx.x * output_suze + threadIdx.x - filter_r;
    int out_y = blockIdx.y * output_suze + threadIdx.y - filter_r;

    int out_ch = blockIdx.z;

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

#ifdef SINGLE_BLOCK
    int blockId = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        while (atomicAdd(debug_counter, 0) != blockId) {
        }
    }
    __syncthreads();  // This is needed because only the first thread of the block is watinig for
                      // the blocks turn to run

#endif

#ifdef DEBUG
    if (!threadIdx.x && !threadIdx.y && !blockIdx.x && !blockIdx.y && !blockIdx.z) {
        PRINT_INPUTS();
    }
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

        if (out_x >= 0 && out_y >= 0 && out_x < w && out_y < h) {
            cell[threadIdx.y * TILE_SIZE + threadIdx.x] = sub_input[(out_y)*w + out_x];
        } else {
            cell[threadIdx.y * TILE_SIZE + threadIdx.x] = pad;
        }
        __syncthreads();  // Wait for all threads to load the input

#ifdef DEBUG
        if (!threadIdx.x && !threadIdx.y) {
            printf("Cell contents at (%d, %d, %d):\n", blockIdx.z, blockIdx.y, blockIdx.x);
            print_data_float(cell, h, w);
        }
#endif

        // Apply the filter to the cell, which should be padded if it lands outside of the input.

        if (threadIdx.x >= filter_r && threadIdx.x < TILE_SIZE - filter_r &&
            threadIdx.y >= filter_r && threadIdx.y < TILE_SIZE - filter_r) {
            for (int filter_y = 0; filter_y < filter_size; filter_y++) {
                for (int filter_x = 0; filter_x < filter_size; filter_x++) {
                    R += cell[(threadIdx.y + filter_y - filter_r) * TILE_SIZE +
                              (threadIdx.x + filter_x - filter_r)] *
                         sub_filter[filter_y * filter_size + filter_x];
                }
            }
        }

        __syncthreads();  // Wait for all threads to complete before we load the next input
    }
    if ((threadIdx.x >= filter_r && threadIdx.x < TILE_SIZE - filter_r) &&
        (out_x >= 0 && out_x < w) &&
        (threadIdx.y >= filter_r && threadIdx.y < TILE_SIZE - filter_r) &&
        (out_y >= 0 && out_y < h)) {
        sub_output[(out_y)*w + out_x] = R;
    }

#ifdef SINGLE_BLOCK
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        atomicAdd(debug_counter, 1);
    }
#endif
}
