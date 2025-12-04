// src/core/image.cpp
#include "image.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

#include <stdexcept>
#include <string>
#include <vector>

Image load_image(const std::string& path)
{
    int w = 0, h = 0, ch = 0;
    // Force 3 channels to normalize downstream CUDA kernels.
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, 3);
    if (!data)
    {
        throw std::runtime_error("Failed to load image: " + path);
    }

    Image img;
    img.width = w;
    img.height = h;
    img.channels = 3;
    img.pixels.assign(data, data + (w * h * img.channels));
    stbi_image_free(data);
    return img;
}

void save_image(const std::string& path, const Image& img)
{
    if (img.channels != 3)
    {
        throw std::runtime_error("save_image expects RGB image (3 channels).");
    }

    int stride = img.width * img.channels;
    int result = stbi_write_png(path.c_str(), img.width, img.height, img.channels, img.pixels.data(), stride);
    if (result == 0)
    {
        throw std::runtime_error("Failed to save image: " + path);
    }
}
