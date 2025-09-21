#include <cstdio>

// Kernel function that runs on the GPU
__global__ void hello_kernel() {
    printf("Hello from GPU thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}

int main() {
    // Launch 2 blocks with 4 threads each
    hello_kernel<<<2, 4>>>();
    cudaDeviceSynchronize();  // wait for GPU to finish

    printf("Hello from CPU!\n");
    return 0;
}
