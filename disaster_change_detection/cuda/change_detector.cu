#include <iostream>
#include <cuda_runtime.h>
#include "change_detector.h"

using namespace std;

__global__ void change_detection_kernel(const unsigned char* img1, 
                                        const unsigned char* img2, 
                                        unsigned char* output, 
                                        int num_pixels, 
                                        int threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_pixels) {
        int diff = abs((int)img1[idx] - (int)img2[idx]);
        
        if (diff > threshold) {
            output[idx] = 255;
        } else {
            output[idx] = 0;
        }
    }
}

void run_change_detection_cuda(const unsigned char* img1, 
                               const unsigned char* img2, 
                               unsigned char* output, 
                               int width, int height, int channels, 
                               int threshold) {
                                   
    int num_pixels = width * height * channels;
    size_t size = num_pixels * sizeof(unsigned char);

    unsigned char *d_img1, *d_img2, *d_output;
    cudaMalloc((void**)&d_img1, size);
    cudaMalloc((void**)&d_img2, size);
    cudaMalloc((void**)&d_output, size);

    cudaMemcpy(d_img1, img1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_img2, img2, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_pixels + threadsPerBlock - 1) / threadsPerBlock;

    change_detection_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_img1, d_img2, d_output, num_pixels, threshold);

    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    cudaFree(d_img1);
    cudaFree(d_img2);
    cudaFree(d_output);
}

int main(int argc, char** argv) {
    cout<< "CUDA Change Detection Module Compiled Successfully." << endl;
    cout<< "Ready for Python Integration on HPC Cluster." << endl;
    return 0;
}