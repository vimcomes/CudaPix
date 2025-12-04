// src/core/image.h
#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Simple 8-bit interleaved RGB image stored row-major.
struct Image
{
    int width = 0;
    int height = 0;
    int channels = 3; // we normalize to RGB
    std::vector<uint8_t> pixels;
};

// Load an image from disk. Alpha (if present) is dropped and data is converted to RGB.
Image load_image(const std::string& path);

// Save an image to disk as PNG.
void save_image(const std::string& path, const Image& img);
