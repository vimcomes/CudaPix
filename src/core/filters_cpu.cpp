// src/core/filters_cpu.cpp
#include "filters_cpu.h"

#include <algorithm>
#include <cmath>

namespace
{
inline uint8_t clamp_byte(float v)
{
    v = std::clamp(v, 0.0f, 255.0f);
    return static_cast<uint8_t>(v + 0.5f);
}

inline float to_gray(uint8_t r, uint8_t g, uint8_t b)
{
    return 0.299f * r + 0.587f * g + 0.114f * b;
}
} // namespace

void cpu_grayscale(Image& img)
{
    const int stride = img.channels;
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            const int idx = (y * img.width + x) * stride;
            float gray = to_gray(img.pixels[idx], img.pixels[idx + 1], img.pixels[idx + 2]);
            uint8_t g = clamp_byte(gray);
            img.pixels[idx] = img.pixels[idx + 1] = img.pixels[idx + 2] = g;
        }
    }
}

void cpu_brightness(Image& img, float delta)
{
    delta = std::clamp(delta, -1.0f, 1.0f);
    const int stride = img.channels;
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            const int idx = (y * img.width + x) * stride;
            for (int c = 0; c < img.channels; ++c)
            {
                float v = static_cast<float>(img.pixels[idx + c]) / 255.0f;
                v = std::clamp(v + delta, 0.0f, 1.0f);
                img.pixels[idx + c] = clamp_byte(v * 255.0f);
            }
        }
    }
}

void cpu_contrast(Image& img, float factor)
{
    factor = std::max(factor, 0.0f);
    const int stride = img.channels;
    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            const int idx = (y * img.width + x) * stride;
            for (int c = 0; c < img.channels; ++c)
            {
                float v = static_cast<float>(img.pixels[idx + c]) / 255.0f;
                v = (v - 0.5f) * factor + 0.5f;
                v = std::clamp(v, 0.0f, 1.0f);
                img.pixels[idx + c] = clamp_byte(v * 255.0f);
            }
        }
    }
}

void cpu_box_blur(Image& img)
{
    const int stride = img.channels;
    std::vector<uint8_t> output(img.pixels.size(), 0);

    auto sample = [&](int x, int y, int c) -> uint8_t
    {
        x = std::clamp(x, 0, img.width - 1);
        y = std::clamp(y, 0, img.height - 1);
        return img.pixels[(y * img.width + x) * stride + c];
    };

    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            for (int c = 0; c < img.channels; ++c)
            {
                int sum = 0;
                int count = 0;
                for (int ky = -1; ky <= 1; ++ky)
                {
                    for (int kx = -1; kx <= 1; ++kx)
                    {
                        sum += sample(x + kx, y + ky, c);
                        ++count;
                    }
                }
                output[(y * img.width + x) * stride + c] = static_cast<uint8_t>(sum / count);
            }
        }
    }

    img.pixels.swap(output);
}

void cpu_sobel(Image& img)
{
    const int stride = img.channels;
    std::vector<uint8_t> output(img.pixels.size(), 0);

    const int gx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
    const int gy[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

    auto sample_gray = [&](int x, int y) -> float
    {
        x = std::clamp(x, 0, img.width - 1);
        y = std::clamp(y, 0, img.height - 1);
        int idx = (y * img.width + x) * stride;
        return to_gray(img.pixels[idx], img.pixels[idx + 1], img.pixels[idx + 2]);
    };

    for (int y = 0; y < img.height; ++y)
    {
        for (int x = 0; x < img.width; ++x)
        {
            float sum_x = 0.0f;
            float sum_y = 0.0f;

            for (int ky = -1; ky <= 1; ++ky)
            {
                for (int kx = -1; kx <= 1; ++kx)
                {
                    float g = sample_gray(x + kx, y + ky);
                    sum_x += g * gx[ky + 1][kx + 1];
                    sum_y += g * gy[ky + 1][kx + 1];
                }
            }

            float mag = std::sqrt(sum_x * sum_x + sum_y * sum_y);
            uint8_t out = clamp_byte(mag);

            int idx = (y * img.width + x) * stride;
            output[idx] = output[idx + 1] = output[idx + 2] = out;
        }
    }

    img.pixels.swap(output);
}
