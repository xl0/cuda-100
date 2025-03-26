#ifdef __INTELLISENSE__
#define DEBUG
#define DEBUG_FILTER
#define ACCUM_DTYPE float
#endif

#ifndef ACCUM_DTYPE
#error "ACUM_DTYPE should be defined (usually 'float' or 'double')"
#endif

#ifdef DEBUG


// Some kernels don't define a tile size, define it here for a unified debug print.
#ifndef TILE_SIZE
#define TILE_SIZE 0
#endif

#include <stdio.h>

#ifdef DEBUG_FILTER

// This version prints out the filter. It's very verbose, so there is also a version that only
// ptints out the inputs
#define PRINT_INPUTS()                                                                        \
    do {                                                                                      \
        printf("h: %d\n", h);                                                                 \
        printf("w: %d\n", w);                                                                 \
        printf("in_channels: %d\n", in_channels);                                             \
        printf("out_channels: %d\n", out_channels);                                           \
        printf("filter_size: %d\n", filter_size);                                             \
        printf("filter r: %d\n", filter_r);                                                   \
        printf("tile size: %d\n", TILE_SIZE);                                                 \
        printf("pad: %f\n", pad);                                                             \
        printf("Filter:\n");                                                                  \
        for (int oc = 0; oc < out_channels; oc++) {                                           \
            printf("Output channel %d:\n", oc);                                               \
            for (int ic = 0; ic < in_channels; ic++) {                                        \
                printf("  Input channel %d:\n", ic);                                          \
                float *sub_filter = filter + (filter_size * filter_size * in_channels * oc) + \
                                    (filter_size * filter_size * ic);                         \
                for (int i = 0; i < filter_size; i++) {                                       \
                    printf("    ");                                                           \
                    for (int j = 0; j < filter_size; j++) {                                   \
                        printf("%f ", sub_filter[i * filter_size + j]);                       \
                    }                                                                         \
                    printf("\n");                                                             \
                }                                                                             \
            }                                                                                 \
        }                                                                                     \
    } while (0)

#else  // DEBUG_FILTER

#define PRINT_INPUTS()                              \
    do {                                            \
        printf("h: %d\n", h);                       \
        printf("w: %d\n", w);                       \
        printf("in_channels: %d\n", in_channels);   \
        printf("out_channels: %d\n", out_channels); \
        printf("filter_size: %d\n", filter_size);   \
        printf("filter r: %d\n", filter_r);         \
        printf("tile size: %d\n", TILE_SIZE);       \
        printf("pad: %f\n", pad);                   \
    } while (0)

#endif  // DEBUG_FILTER

__device__ void print_data_float(float *data, int h, int w) {
    for (int i = 0; i < h; i++) {
        printf("Row %d: ", i);
        for (int j = 0; j < w; j++) {
            printf("%.3f ", data[i * w + j]);
        }
        printf("\n");
    }
}

#endif  // DEBUG




