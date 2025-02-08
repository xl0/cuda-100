#include <stdio.h>
#include <cuda_runtime.h>


#ifndef N_THREADS
    #define N_THREADS 512
#endif


__global__ void vecAddKernel(float *a, float *b, float *c, uint n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i]+ b[i];
    }

}


void vecAdd_f32(float *A, float *B, float *C, uint n) {
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    cudaMalloc((void **) &A_d, size);
    cudaMalloc((void **) &B_d, size);
    cudaMalloc((void **) &C_d, size);


    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    vecAddKernel <<<(n + N_THREADS - 1) / N_THREADS, N_THREADS>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}



void add_vectors_cpu(float *a, float *b, float *c, uint n) {
    for (uint i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int verify_equal(float *a, float *b, uint n)
{
    for (uint i = 0; i < n; i++) {
        if (a[i] != b[i]) return 0;
    }
    return 1;

}


int main() {
    uint n = 1024*1024;

    float *A = (float*)malloc(n * sizeof(float));
    float *B = (float*)malloc(n * sizeof(float));
    float *C = (float*)malloc(n * sizeof(float));


    for(uint i = 0; i < n; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    vecAdd_f32(A, B, C, n);

    float *C_cpu = (float *)malloc(n * sizeof(float));

    add_vectors_cpu(A, B, C_cpu, n);

    printf("Do they match? %s!\n", verify_equal(C, C_cpu, n) ? "Yes" : "No" );

    return 0;
}