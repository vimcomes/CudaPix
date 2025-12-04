// src/core/filters_cpu.h
#pragma once

#include "image.h"

// CPU implementations of the same filters used on the GPU. Operate in-place on RGB8 data.
void cpu_grayscale(Image& img);
void cpu_brightness(Image& img, float delta); // [-1, 1]
void cpu_contrast(Image& img, float factor);  // > 0
void cpu_box_blur(Image& img);                // 3x3 blur
void cpu_sobel(Image& img);                   // edge detection, grayscale
