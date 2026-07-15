#include <iostream>
#include <cuda_runtime.h>
#include "change_detector.h"

// ==========================================
// CUDA KERNEL
// Computes Absolute Difference and Thresholding on the GPU
// ==========================================
__global__ void change_detection_kernel(const unsigned char* img1, 
                                        const unsigned char* img2, 
                                        unsigned char* output, 
                                        int num_pixels, 
                                        int threshold) {
    // Calculate the global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Ensure we don't go out of bounds
    if (idx < num_pixels) {
        // Pixel Difference & Absolute Difference
        int diff = abs((int)img1[idx] - (int)img2[idx]);
        
        // Thresholding
        if (diff > threshold) {
            output[idx] = 255; // Changed (White)
        } else {
            output[idx] = 0;   // Unchanged (Black)
        }
    }
}

// ==========================================
// HOST FUNCTION (C++ Wrapper)
// Manages memory and launches the kernel
// ==========================================
void run_change_detection_cuda(const unsigned char* img1, 
                               const unsigned char* img2, 
                               unsigned char* output, 
                               int width, int height, int channels, 
                               int threshold) {
                                   
    int num_pixels = width * height * channels;
    size_t size = num_pixels * sizeof(unsigned char);

    // 1. Allocate memory on the GPU (Device)
    unsigned char *d_img1, *d_img2, *d_output;
    cudaMalloc((void**)&d_img1, size);
    cudaMalloc((void**)&d_img2, size);
    cudaMalloc((void**)&d_output, size);

    // 2. Copy data from CPU (Host) to GPU (Device)
    cudaMemcpy(d_img1, img1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, img2, size, cudaMemcpyHostToDevice);

    // 3. Define Grid and Block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;

    // 4. Launch the CUDA Kernel
    change_detection_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_img1, d_img2, d_output, num_pixels, threshold);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // 5. Copy the result back from GPU to CPU
    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    // 6. Free GPU memory
    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_output);
}

// Stub main function for standalone compilation/testing on HPC
int main(int argc, char** argv) {
    std::cout << "CUDA Change Detection Module Compiled Successfully." << std::endl;
    std::cout << "Ready for Python Integration on HPC Cluster." << std::endl;
    return 0;
}