// src/core/filters_cuda.h
#pragma once

#include "image.h"

// Apply filters in-place on the GPU. Image data is assumed to be interleaved RGB8.
void apply_grayscale(Image& img);
void apply_brightness(Image& img, float delta); // delta in [-1, 1]
void apply_contrast(Image& img, float factor);  // e.g. 0.5, 1.0, 1.5, 2.0
void apply_box_blur(Image& img);                 // naive 3x3 blur
void apply_sobel(Image& img);                    // edge detection, output grayscale
