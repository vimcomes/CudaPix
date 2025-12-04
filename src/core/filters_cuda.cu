// src/core/filters_cuda.cu
#include "filters_cuda.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(expr)                                                                             \
    do                                                                                               \
    {                                                                                                \
        cudaError_t err__ = (expr);                                                                  \
        if (err__ != cudaSuccess)                                                                    \
        {                                                                                            \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err__));       \
        }                                                                                            \
    } while (0)

namespace
{
__device__ __forceinline__ uint8_t clamp_to_byte(float v)
{
    v = v < 0.0f ? 0.0f : (v > 255.0f ? 255.0f : v);
    return static_cast<uint8_t>(v + 0.5f);
}

__device__ __forceinline__ float to_grayscale(uint8_t r, uint8_t g, uint8_t b)
{
    // Luma coefficients for a quick grayscale conversion.
    return 0.299f * r + 0.587f * g + 0.114f * b;
}

__global__ void grayscale_kernel(uint8_t* data, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;
    float gray = to_grayscale(data[idx], data[idx + 1], data[idx + 2]);
    uint8_t g = clamp_to_byte(gray);
    data[idx] = data[idx + 1] = data[idx + 2] = g;
}

__global__ void brightness_kernel(uint8_t* data, int width, int height, int channels, float delta)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;
    for (int c = 0; c < channels; ++c)
    {
        float v = static_cast<float>(data[idx + c]) / 255.0f;
        v = fminf(fmaxf(v + delta, 0.0f), 1.0f);
        data[idx + c] = clamp_to_byte(v * 255.0f);
    }
}

__global__ void contrast_kernel(uint8_t* data, int width, int height, int channels, float factor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = (y * width + x) * channels;
    for (int c = 0; c < channels; ++c)
    {
        float v = static_cast<float>(data[idx + c]) / 255.0f;
        v = (v - 0.5f) * factor + 0.5f;
        v = fminf(fmaxf(v, 0.0f), 1.0f);
        data[idx + c] = clamp_to_byte(v * 255.0f);
    }
}

__global__ void box_blur_kernel(const uint8_t* input, uint8_t* output, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    for (int c = 0; c < channels; ++c)
    {
        int sum = 0;
        int count = 0;
        for (int ky = -1; ky <= 1; ++ky)
        {
            int yy = y + ky;
            if (yy < 0 || yy >= height) continue;
            for (int kx = -1; kx <= 1; ++kx)
            {
                int xx = x + kx;
                if (xx < 0 || xx >= width) continue;
                int src_idx = (yy * width + xx) * channels + c;
                sum += input[src_idx];
                ++count;
            }
        }
        output[(y * width + x) * channels + c] = static_cast<uint8_t>(sum / count);
    }
}

__global__ void sobel_kernel(const uint8_t* input, uint8_t* output, int width, int height, int channels)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    // Sobel kernels.
    const int gx[3][3] = { { -1, 0, 1 }, { -2, 0, 2 }, { -1, 0, 1 } };
    const int gy[3][3] = { { 1, 2, 1 }, { 0, 0, 0 }, { -1, -2, -1 } };

    float sum_x = 0.0f;
    float sum_y = 0.0f;

    for (int ky = -1; ky <= 1; ++ky)
    {
        int yy = min(max(y + ky, 0), height - 1);
        for (int kx = -1; kx <= 1; ++kx)
        {
            int xx = min(max(x + kx, 0), width - 1);
            int idx = (yy * width + xx) * channels;
            float gray = to_grayscale(input[idx], input[idx + 1], input[idx + 2]);
            sum_x += gray * gx[ky + 1][kx + 1];
            sum_y += gray * gy[ky + 1][kx + 1];
        }
    }

    float magnitude = sqrtf(sum_x * sum_x + sum_y * sum_y);
    magnitude = fminf(magnitude, 255.0f);
    uint8_t out = clamp_to_byte(magnitude);

    int out_idx = (y * width + x) * channels;
    output[out_idx] = output[out_idx + 1] = output[out_idx + 2] = out;
}

dim3 make_grid(int width, int height, dim3 block = dim3(16, 16))
{
    return dim3((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
}

size_t image_size_bytes(const Image& img)
{
    return static_cast<size_t>(img.width) * static_cast<size_t>(img.height) * static_cast<size_t>(img.channels);
}

} // namespace

void apply_grayscale(Image& img)
{
    size_t bytes = image_size_bytes(img);
    uint8_t* d_img = nullptr;
    CUDA_CHECK(cudaMalloc(&d_img, bytes));
    CUDA_CHECK(cudaMemcpy(d_img, img.pixels.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid = make_grid(img.width, img.height, block);
    grayscale_kernel<<<grid, block>>>(d_img, img.width, img.height, img.channels);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(img.pixels.data(), d_img, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_img));
}

void apply_brightness(Image& img, float delta)
{
    delta = std::clamp(delta, -1.0f, 1.0f);

    size_t bytes = image_size_bytes(img);
    uint8_t* d_img = nullptr;
    CUDA_CHECK(cudaMalloc(&d_img, bytes));
    CUDA_CHECK(cudaMemcpy(d_img, img.pixels.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid = make_grid(img.width, img.height, block);
    brightness_kernel<<<grid, block>>>(d_img, img.width, img.height, img.channels, delta);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(img.pixels.data(), d_img, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_img));
}

void apply_contrast(Image& img, float factor)
{
    if (factor < 0.0f) factor = 0.0f;

    size_t bytes = image_size_bytes(img);
    uint8_t* d_img = nullptr;
    CUDA_CHECK(cudaMalloc(&d_img, bytes));
    CUDA_CHECK(cudaMemcpy(d_img, img.pixels.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid = make_grid(img.width, img.height, block);
    contrast_kernel<<<grid, block>>>(d_img, img.width, img.height, img.channels, factor);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(img.pixels.data(), d_img, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_img));
}

void apply_box_blur(Image& img)
{
    size_t bytes = image_size_bytes(img);
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, img.pixels.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid = make_grid(img.width, img.height, block);
    box_blur_kernel<<<grid, block>>>(d_input, d_output, img.width, img.height, img.channels);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(img.pixels.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}

void apply_sobel(Image& img)
{
    size_t bytes = image_size_bytes(img);
    uint8_t* d_input = nullptr;
    uint8_t* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, img.pixels.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16, 16);
    dim3 grid = make_grid(img.width, img.height, block);
    sobel_kernel<<<grid, block>>>(d_input, d_output, img.width, img.height, img.channels);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(img.pixels.data(), d_output, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
}
