#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <cmath>
#include <chrono>
#include <memory>
#include "stb_image.h"
#include "stb_image_write.h"


using u8 = unsigned char;

template<typename T>
float conv(const T* mat, const float filter[3][3], int w, int h, int i, int j);

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

    auto image = std::make_unique<float[]>(height*width); 
    
    // Sobel operators
    const float Gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const float Gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    const float blur_filter[3][3] = {
        {0.0625, 0.125, 0.0625},
        {0.1250, 0.250, 0.1250},
        {0.0625, 0.125, 0.0625},
    };    
    
    auto res_x = std::make_unique<float[]>(height*width); 
    auto res_y = std::make_unique<float[]>(height*width); 

    auto G = std::make_unique<float[]>(height*width); 
    auto blurred = std::make_unique<float[]>(height*width); 
    auto out = std::make_unique<u8[]>(height*width); 

    auto start = std::chrono::high_resolution_clock::now();
    // Greyscale conversion formula per PMPP
    // L = r * 0.21 + g * 0.72 + b * 0.07
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int i = (row * width + col) * channels;

            image.get()[row*width + col] = data[i] * 0.21f + data[i+1] * 0.72f + data[i+2] * 0.07f;
        }
    } 

    // Applying blur
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            blurred.get()[row*width+col] = conv(image.get(), blur_filter, width, height, row, col);
        }
    }

    // Applying Sobel
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            res_x.get()[row*width+col] = conv(blurred.get(), Gx, width, height, row, col);
            res_y.get()[row*width+col] = conv(blurred.get(), Gy, width, height, row, col);
        }
    }


    // Computing the norm
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            G.get()[row*width+col] = sqrtf((res_x.get()[row*width+col]*res_x[row*width+col]) +(res_y.get()[row*width+col]*res_y[row*width+col]));

            out.get()[row*width+col] =  (G.get()[row*width+col] > 100) ? 255 : 0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    float ms = std::chrono::duration<float, std::milli>(end - start).count();

    std::cout << "CPU: " << ms << " ms\n";
    stbi_image_free(data);
    stbi_write_jpg(
        "resources/edges_cpu.jpg",
        width,
        height,
        1, // since it's greyscale 1 channel is fine
        out.get(),
        width
    );
    return 0;
}


template<typename T>
float conv(const T* mat, const float filter[3][3], int w, int h, int i, int j) {
    float res = 0.0f;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (i+dy >= 0 && i+dy < h && j+dx >= 0 && j+dx < w)
                res += filter[dy+1][dx+1] * mat[(i+dy)*w+ j+dx];
        }
    }
    return res;
}
