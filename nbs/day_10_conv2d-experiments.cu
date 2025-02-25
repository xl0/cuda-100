#include <stdint.h>
#include <stdio.h>

/* 2D convolution, with padding to valid shape. Channel-first */
__global__ void conv2d_pad(float *in,
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

    int filter_r = (filter_size - 1) / 2;

    // In and Out data dimensions:
    // 0 - channel
    // 1 - height
    // 2 - width

    // Filter dimensions:
    // 0 - out channels
    // 1 - in channels
    // 2 - height
    // 3 - width

    // if (x == 0 && y == 0) {
    //     printf("h: %d\n", h);
    //     printf("w: %d\n", w);
    //     printf("in_channels: %d\n", in_channels);
    //     printf("out_channels: %d\n", out_channels);
    //     printf("filter_size: %d\n", filter_size);
    //     printf("filter r: %d\n", filter_r);
    //     printf("pad: %f\n", pad);

    //     printf("Filter:\n");
    //     for (int oc = 0; oc < out_channels; oc++) {
    //         printf("Output channel %d:\n", oc);
    //         for (int ic = 0; ic < in_channels; ic++) {
    //             printf("  Input channel %d:\n", ic);
    //             float *sub_filter = filter + (filter_size * filter_size * in_channels * oc) +
    //                                 (filter_size * filter_size * ic);
    //             for (int i = 0; i < filter_size; i++) {
    //                 printf("    ");
    //                 for (int j = 0; j < filter_size; j++) {
    //                     printf("%f ", sub_filter[i * filter_size + j]);
    //                 }
    //                 printf("\n");
    //             }
    //         }
    //     }
    // }

    if (x >= w || y >= h) return;

    // Loop over the output channels
    for (int out_c = 0; out_c < out_channels; out_c++) {
        float R = 0;

        // Pointer to the 2d slice of the output
        float *sub_output = out + out_c * w * h;

        // Loop over the input channels
        for (int in_c = 0; in_c < in_channels; in_c++) {
            // Pointer to the 2d slice of the filter that corresponds to the active input and
            // output channels
            float *sub_filter = filter + (filter_size * filter_size * in_channels * out_c) +
                                (filter_size * filter_size * in_c);
            // Pinter to the current channel in the input
            float *sub_input = in + (w * h * in_c);

            // Apply the filter to the input or the pad value for outside indices.
            for (int filter_y = 0; filter_y < filter_size; filter_y++) {
                for (int filter_x = 0; filter_x < filter_size; filter_x++) {
                    float v = pad;
                    int input_x = x - filter_r + filter_x;
                    int input_y = y - filter_r + filter_y;

                    if (input_x >= 0 && input_x < w && input_y >= 0 && input_y < h) {
                        v = sub_input[input_y * w + input_x];
                    }
                    R += v * sub_filter[filter_y * filter_size + filter_x];
                }
            }
        }
        sub_output[y * w + x] = R;
    }
}

/* This version uses the z grid dimensions for out channels. */

__global__ void conv2d_pad_z_out(float *in,
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

    // Loop over the output channels

    float R = 0;

    // // Pointer to the 2d slice of the output
    float *sub_output = out + out_ch * w * h;
    // Loop over the input channels
    for (int in_c = 0; in_c < in_channels; in_c++) {
        // Pointer to the 2d slice of the filter that corresponds to the active input and output
        // channels
        float *sub_filter = filter + (filter_size * filter_size * in_channels * out_ch) +
                            (filter_size * filter_size * in_c);
        // Pinter to the current channel in the input
        float *sub_input = in + (w * h * in_c);

        // Apply the filter to the input or the pad value for outside indices.
        for (int filter_y = 0; filter_y < filter_size; filter_y++) {
            for (int filter_x = 0; filter_x < filter_size; filter_x++) {
                float v = pad;
                int input_x = x - filter_r + filter_x;
                int input_y = y - filter_r + filter_y;

                if (input_x >= 0 && input_x < w && input_y >= 0 && input_y < h) {
                    v = sub_input[input_y * w + input_x];
                }
                R += v * sub_filter[filter_y * filter_size + filter_x];
            }
        }
    }
    sub_output[y * w + x] = R;
}
