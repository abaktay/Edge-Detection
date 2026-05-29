#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <cmath>
#include <vector>
#include "stb_image.h"
#include "stb_image_write.h"


using u8 = unsigned char;

float conv(const std::vector<float>& mat, const float filter[3][3], int w, int h, int i, int j);

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

    std::vector<float> image(height*width, 0.0f); 
    
    // Greyscale conversion formula per PMPP
    // L = r * 0.21 + g * 0.72 + b * 0.07
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int i = (row * width + col) * channels;

            image[row*width + col] = data[i] * 0.21f + data[i+1] * 0.72f + data[i+2] * 0.07f;
        }
    } 
    stbi_image_free(data);

    // Sobel operators
    const float Gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const float Gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    std::vector<float> res_x(height*width, 0.0f); 
    std::vector<float> res_y(height*width, 0.0f); 

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            res_x[row*width+col] = conv(image, Gx, width, height, row, col);
            res_y[row*width+col] = conv(image, Gy, width, height, row, col);
        }
    }

    std::vector<float> G(width*height,0.0f);
    std::vector<u8> out(width*height, 0.0f);

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            G[row*width+col] = sqrtf((res_x[row*width+col]*res_x[row*width+col]) +(res_y[row*width+col]*res_y[row*width+col]));

            out[row*width+col] =  (G[row*width+col] > 100) ? 255 : 0;
        }
    }

    stbi_write_jpg(
        "resources/cpu.jpg",
        width,
        height,
        1, // since it's greyscale 1 channel is fine
        out.data(),
        width
    );
    return 0;
}

float conv(const std::vector<float>& mat, const float filter[3][3], int w, int h, int i, int j) {
    float res = 0.0f;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (i+dy >= 0 && i+dy < h && j+dx >= 0 && j+dx < w)
                res += filter[dy+1][dx+1] * mat[(i+dy)*w+ j+dx];
        }
    }
    return res;
}
