#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <memory>
#include "stb_image.h"
#include "stb_image_write.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <type_traits>

using u8 = unsigned char;

__constant__ float d_filter[3][3];

template<typename T, typename U>
requires std::is_arithmetic_v<T> &&
         std::is_arithmetic_v<U>
__global__
void conv(T* out, const U* in, int w, int h);
__global__
void colorToGreyscaleConversion(u8 * Pout, u8 * Pin, int width, int height, int channels);
__global__
void norm(u8* out, const float* d_A, const float* d_B, int w, int h);

void edge(u8* out, const u8* in, int width, int height);
void greyscale(u8 * Pout, const u8 * Pin, int width, int height, int channels);
void gaussian_blur(u8* out, const u8* in, int width, int height);

int main()
{
    int width, height, channels;

    u8* data = stbi_load(
        "resources/turin.jpg",   
        &width,        
        &height,       
        &channels,     
        0              
    );

    if (!data) {
        std::cout << "Failed to load image\n";
        return 1;
    }

    
    auto out = std::make_unique<unsigned char[]>(width*height);
    auto blur_out = std::make_unique<unsigned char[]>(width*height);
    auto edge_out = std::make_unique<unsigned char[]>(width*height);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cudaEventRecord(start);
    greyscale(out.get(), data, width, height, channels);
    // stbi_write_jpg(
    //     "resources/grey.jpg",
    //     width,
    //     height,
    //     1,
    //     out.get(),
    //     width
    // );

    gaussian_blur(blur_out.get(), out.get(), width, height);

    // stbi_write_jpg(
    //     "resources/blurred.jpg",
    //     width,
    //     height,
    //     1,
    //     blur_out.get(),
    //     width
    // );

    edge(edge_out.get(), blur_out.get(), width, height);
    cudaEventRecord(end);

    cudaEventSynchronize(end);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, end);
    std::cout << "GPU kernel: " << ms << " ms\n";

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    stbi_write_jpg(
        "resources/edges_gpu.jpg",
        width,
        height,
        1,
        edge_out.get(),
        width
    );

    stbi_image_free(data);
    return 0;
}

__global__
void norm(u8* out, const float* d_A, const float* d_B, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    out[y*w + x] = (sqrtf((d_A[y*w+x]*d_A[y*w+x])+ (d_B[y*w+x]*d_B[y*w+x])) > 100) ? 255 : 0;
}


template<typename T, typename U>
requires std::is_arithmetic_v<T> &&
         std::is_arithmetic_v<U>
__global__
void conv(T* out, const U* in, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    float sum = 0.0f;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (y+dy >= 0 && y+dy < h && x+dx >= 0 && x+dx < w)
                sum += d_filter[dy+1][dx+1] * (float)in[(y+dy)*w+ x+dx];
        }
    }

    out[y*w + x] = (T)sum;
}

void edge(u8* out, const u8* in, int width, int height) {
    // greyscale has one channel
    size_t input_size = width * height * sizeof(u8); 
    size_t out_size = width * height * sizeof(u8); 

    u8* d_in = nullptr;
    float* d_gx = nullptr;
    float* d_gy = nullptr;
    u8* d_out = nullptr;

    cudaMalloc(&d_in, input_size);
    cudaMalloc(&d_gx, input_size*sizeof(float));
    cudaMalloc(&d_gy, input_size*sizeof(float));
    cudaMalloc(&d_out, out_size);

    cudaMemcpy(d_in, in, input_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);

    dim3 gridSize(
        (width  + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    const float Gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const float Gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    // convolution for gx, gy, then sum
    // can be improved by computing Gx and Gy in parallel
    cudaMemcpyToSymbol(d_filter, Gx, sizeof(float)*9);
    conv<<<gridSize, blockSize>>>(d_gx, d_in, width, height);
    

    cudaDeviceSynchronize();

    cudaMemcpyToSymbol(d_filter, Gy, sizeof(float)*9);
    conv<<<gridSize, blockSize>>>(d_gy, d_in, width, height);

    cudaDeviceSynchronize();
    norm<<<gridSize, blockSize>>>(d_out, d_gx, d_gy, width, height);

    cudaMemcpy(out, d_out, out_size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_gx);
    cudaFree(d_gy);
}

void gaussian_blur(u8* out, const u8* in, int width, int height) {
    
    size_t input_size = width * height * sizeof(u8); 
    // greyscale has one channel
    size_t out_size = width * height * sizeof(u8); 

    u8* d_in = nullptr;
    u8* d_out = nullptr;

    cudaMalloc(&d_in, input_size);
    cudaMalloc(&d_out, out_size);

    cudaMemcpy(d_in, in, input_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);

    dim3 gridSize(
        (width  + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    float blur_filter[3][3] = {
        {0.0625, 0.125, 0.0625},
        {0.1250, 0.250, 0.1250},
        {0.0625, 0.125, 0.0625},
    };    
    cudaMemcpyToSymbol(d_filter, blur_filter, sizeof(float)*9);
    conv<<<gridSize, blockSize>>>(d_out, d_in,  width, height);
    
    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, out_size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

void greyscale(u8 * Pout, const u8 * Pin, int width, int height, int channels) {
    size_t input_size = width * height * sizeof(u8) * channels; 
    // greyscale has one channel
    size_t out_size = width * height * sizeof(u8); 

    u8* d_in = nullptr;
    u8* d_out = nullptr;

    cudaMalloc(&d_in, input_size);
    cudaMalloc(&d_out, out_size);

    cudaMemcpy(d_in, Pin, input_size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);

    dim3 gridSize(
        (width  + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y
    );

    colorToGreyscaleConversion<<<gridSize, blockSize>>>(d_out, d_in, width, height, channels);
    cudaDeviceSynchronize();

    cudaMemcpy(Pout, d_out, out_size, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

// implementation from PMPP
__global__
void colorToGreyscaleConversion(u8 * Pout, u8 * Pin, int width, int height, int channels)
{
    int Col = threadIdx.x + blockIdx.x * blockDim.x;
    int Row = threadIdx.y + blockIdx.y * blockDim.y;

    if (Col < width && Row < height) {
        // get 1D coordinate for the grayscale image
        int greyOffset = Row*width + Col;
        // one can think of the RGB image having
        // CHANNEL times columns than the grayscale image
        int rgbOffset = greyOffset*channels;
        unsigned char r = Pin[rgbOffset    ]; // red value for pixel
        unsigned char g = Pin[rgbOffset + 1]; // green value for pixel
        unsigned char b = Pin[rgbOffset + 2]; // blue value for pixel
        // perform the rescaling and store it
        // We multiply by floating point constants
        Pout[greyOffset] = (u8)( 0.21f*r + 0.72f*g + 0.07f*b);
    }
}
