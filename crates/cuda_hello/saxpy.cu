#include <cstdio>
#include <cuda_runtime.h>
#include <random>

__global__ void saxpy(int n, float a, float *x, float *y){
    // threadIdx.x: thread index within the block
    // blockIdx.x: block index within the grid
    // blockDim.x: number of threads per block
    // gridDim.x: number of blocks in the grid

    // global_id: unique index for each thread in the entire grid
    int global_id = threadIdx.x + blockDim.x * blockIdx.x;

    // Example: gridDim.x = 2, blockDim.x = 4

    // Block 0: threadIdx.x = [0,1,2,3] → global_id = [0,1,2,3]
    // Block 1: threadIdx.x = [0,1,2,3] → global_id = [4,5,6,7]

    // stride: total number of threads in the grid
    int stride = blockDim.x * gridDim.x;

    // Each thread processes multiple elements, striding by the total number of threads
    // Striding ensures all elements are processed even if n > total threads
    for (int i=global_id; i < n; i += stride)
    {
        y[i] = a * x[i] + y[i];
    }
}

int main() {
    // Set up data
    const int N = 100;
    float alpha = 3.14f;
    float *h_x, *h_y;
    float *d_x, *d_y;
    size_t size = N * sizeof(float);

    // Allocate device memory
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);

    // Initialize host data
    h_x = (float*)malloc(size);
    h_y = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_x[i] = rand() / (float)RAND_MAX;
        h_y[i] = rand() / (float)RAND_MAX;
    }

    // Copy data to device
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    // Define block size (number of threads per block)
    int blockSize = 4;

    // Calculate number of blocks needed
    int numBlocks = (N + blockSize - 1) / blockSize;

    // Launch kernel
    saxpy<<<numBlocks, blockSize>>>(N, alpha, d_x, d_y);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        printf("h_y[%d] = %f\n", i, h_y[i]);
    }

    // Clean up
    free(h_x);
    free(h_y);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}