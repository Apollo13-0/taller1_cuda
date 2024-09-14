#include <stdio.h>
#include <arm_neon.h>
#include <math.h>
#include <stdlib.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Sobel kernels
int8_t SobelX[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

int8_t SobelY[3][3] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1}
};

void sobelFilterNeon(const uint8_t* inputImage, uint8_t* outputImage, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int16x8_t dx = vdupq_n_s16(0);
            int16x8_t dy = vdupq_n_s16(0);

            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    // Cargar los valores de los píxeles sin signo
                    uint8x8_t pixelVals = vld1_u8(&inputImage[(y + j) * width + (x + i)]);

                    // Reinterpretar los valores de píxeles como int8x8_t (con signo)
                    int8x8_t signedPixelVals = vreinterpret_s8_u8(pixelVals);

                    // Aplicar el filtro Sobel
                    dx = vmlal_s8(dx, signedPixelVals, vdup_n_s8(SobelX[j + 1][i + 1]));
                    dy = vmlal_s8(dy, signedPixelVals, vdup_n_s8(SobelY[j + 1][i + 1]));
                }
            }

            // Calcular la magnitud del gradiente
            int16x8_t mag = vaddq_s16(vabsq_s16(dx), vabsq_s16(dy));

            // Convertir y almacenar el resultado normalizado en el rango [0, 255]
            uint8x8_t result = vqmovun_s16(mag);
            outputImage[y * width + x] = vget_lane_u8(result, 0);
        }
    }
}

int main(int argc, char* argv[]) {
    int width, height, channels;

    if (argc < 2) {
        printf("Error: Please provide the image filename as a parameter!");
        return -1;
    }

    const char* imageName = argv[1];
    
    // Cargar la imagen usando stb_image
    uint8_t* inputImage = stbi_load(imageName, &width, &height, &channels, 1); // Cargar en escala de grises
    if (!inputImage) {
        printf("Error: Could not load the image!");
        return -1;
    }

    printf("Image loaded: %ix%i, channels: %x\n", width, height, channels);

    // Crear el buffer de salida
    uint8_t* outputImage = (uint8_t*)malloc(width * height * sizeof(uint8_t));

    // Aplicar el filtro Sobel con NEON
    sobelFilterNeon(inputImage, outputImage, width, height);

    // Guardar el resultado como imagen PNG
    stbi_write_png("sobel_output_neon.png", width, height, 1, outputImage, width);

    // Liberar memoria
    stbi_image_free(inputImage);
    free(outputImage);

    return 0;
}
