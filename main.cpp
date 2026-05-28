#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <iostream>
#include <cmath>
#include "inc/stb_image.h"
#include "inc/stb_image_write.h"

#define N 16
float conv(const float mat[N][N], const float filter[3][3], int w, int h, int i, int j);

int main()
{
    int width, height, channels;

    stbi_uc* data = stbi_load(
        "resources/amogus.png",   
        &width,        
        &height,       
        &channels,     
        4              
    );

    if (!data) {
        std::cout << "Failed to load image\n";
        return 1;
    }


    // For a 16x16 image no need for heap allocation
    float image[N][N];

    
    // Greyscale conversion formula per PMPP
    // L = r * 0.21 + g * 0.72 + b * 0.07
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            int i = (row * width + col) * channels;

            image[row][col] = data[i] * 0.21f + data[i+1] * 0.72f + data[i+2] * 0.07f;
        }
    } 
    stbi_image_free(data);

    // Sobel operators
    const float Gx[3][3] = { {-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const float Gy[3][3] = { {-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
    
    float res_x[N][N];
    float res_y[N][N];

    // Now, for all indices in image apply the 3x3 convolution
    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            res_x[row][col] = conv(image, Gx, width, height, row, col);
            res_y[row][col] = conv(image, Gy, width, height, row, col);
        }
    }

    float G[N][N];
    stbi_uc out[N*N];

    for (int row = 0; row < height; row++) {
        for (int col = 0; col < width; col++) {
            G[row][col] = sqrtf((res_x[row][col]*res_x[row][col]) +(res_y[row][col]*res_y[row][col]));
            // std::printf("%.2f\t",  G[row][col]);

            out[row*width+col] =  (G[row][col] > 1000) ? 255 : 0;
        }
        // std::cout << "\n";
    }

    stbi_write_png(
        "edges.png",
        width,
        height,
        1,
        out,
        width
    );
    return 0;
}

float conv(const float mat[N][N], const float filter[3][3], int w, int h, int i, int j) {
    float res = 0.0f;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (i+dy >= 0 && i+dy < h && j+dx >= 0 && j+dx < w)
                res += filter[dy+1][dx+1] * mat[i+dy][j+dx];
        }
    }
    return res;
}
