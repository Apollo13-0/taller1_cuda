#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define SOBEL_FILTER_WIDTH 3


// Define the Sobel kernel for x and y directions
__constant__ int SobelX[SOBEL_FILTER_WIDTH][SOBEL_FILTER_WIDTH] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

__constant__ int SobelY[SOBEL_FILTER_WIDTH][SOBEL_FILTER_WIDTH] = {
    {-1, 2, 1},
    { 0, 0, 0},
    { 1, 2, 1}
};

// This kernel computes the Sobel filter on the input image
__global__ void sobelFilterKernel(unsigned char *inputImage, unsigned char *outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row

    // Ensure threads are within the image bounds
    if (x >= width || y >= height) {
        return;
    }

    float dx = 0.0f, dy = 0.0f;

    // Apply the Sobel filter
    if( x > 0 && y > 0 && x < width-1 && y < height-1) {
        dx = (-1* inputImage[(y-1)*width + (x-1)]) + (-2*inputImage[y*width+(x-1)]) + (-1*inputImage[(y+1)*width+(x-1)]) +
             (    inputImage[(y-1)*width + (x+1)]) + ( 2*inputImage[y*width+(x+1)]) + (   inputImage[(y+1)*width+(x+1)]);
        dy = (    inputImage[(y-1)*width + (x-1)]) + ( 2*inputImage[(y-1)*width+x]) + (   inputImage[(y-1)*width+(x+1)]) +
             (-1* inputImage[(y+1)*width + (x-1)]) + (-2*inputImage[(y+1)*width+x]) + (-1*inputImage[(y+1)*width+(x+1)]);

        // Compute the gradient magnitude
        float edgeVal = sqrtf(dx * dx + dy * dy);

        // Normalize and set the output pixel
        outputImage[y * width + x] = (unsigned char)min(max(edgeVal, 0.0f), 255.0f);
    }
}

int main(int argc, char* argv[]) {
    int width, height, channels;

    if (argc < 2) {
        printf("Error: Please provide the image filename as a parameter!");
        return -1;
    }

    const char* imageName = argv[1];
    
    // Load the image using stb_image
    unsigned char* h_inputImage = stbi_load(imageName, &width, &height, &channels, 1); // Load as grayscale
    if (!h_inputImage) {
        printf("Error: Could not load the image!");
        return -1;
    }

    printf("Image loaded: %ix%i, channels: %x\n", width, height, channels);

    clock_t start = clock();

    // Allocate host memory for the output image
    unsigned char* h_outputImage = (unsigned char*)malloc(width * height * sizeof(unsigned char));

    // Allocate device memory
    unsigned char* d_inputImage;
    unsigned char* d_outputImage;
    cudaMalloc((void**)&d_inputImage, width * height * sizeof(unsigned char));
    cudaMalloc((void**)&d_outputImage, width * height * sizeof(unsigned char));

    // Copy input image from host to device
    cudaMemcpy(d_inputImage, h_inputImage, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // Launch the Sobel kernel
    sobelFilterKernel<<<gridDim, blockDim>>>(d_inputImage, d_outputImage, width, height);

    clock_t end = clock();

    double time = (double)(end-start)/CLOCKS_PER_SEC;

    printf("GPU time = %fs\n", time);

    // Copy the result back to host
    cudaMemcpy(h_outputImage, d_outputImage, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Save the Sobel result to a file (optional)
    stbi_write_png("sobel_output.png", width, height, 1, h_outputImage, width);

    // Free device and host memory
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    free(h_inputImage);
    free(h_outputImage);

    return 0;
}
