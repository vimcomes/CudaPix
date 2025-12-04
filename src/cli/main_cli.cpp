// src/cli/main_cli.cpp
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "core/filters_cuda.h"
#include "core/image.h"

namespace
{
void print_usage()
{
    std::cout << "Usage: cuda_image_filters_cli <input> <output> <filter> [params]\n";
    std::cout << "Filters:\n";
    std::cout << "  grayscale\n";
    std::cout << "  brightness <delta>    (delta in [-1.0, 1.0])\n";
    std::cout << "  contrast <factor>     (factor > 0, e.g., 0.5, 1.0, 1.5, 2.0)\n";
    std::cout << "  blur\n";
    std::cout << "  sobel\n";
}
} // namespace

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        print_usage();
        return 1;
    }

    const std::string input_path = argv[1];
    const std::string output_path = argv[2];
    const std::string filter = argv[3];

    try
    {
        Image img = load_image(input_path);
        std::cout << "Loaded " << input_path << " (" << img.width << "x" << img.height << ")\n";

        auto start = std::chrono::high_resolution_clock::now();

        if (filter == "grayscale")
        {
            apply_grayscale(img);
        }
        else if (filter == "brightness")
        {
            if (argc < 5)
            {
                std::cerr << "brightness requires <delta>\n";
                print_usage();
                return 1;
            }
            float delta = std::stof(argv[4]);
            apply_brightness(img, delta);
        }
        else if (filter == "contrast")
        {
            if (argc < 5)
            {
                std::cerr << "contrast requires <factor>\n";
                print_usage();
                return 1;
            }
            float factor = std::stof(argv[4]);
            apply_contrast(img, factor);
        }
        else if (filter == "blur")
        {
            apply_box_blur(img);
        }
        else if (filter == "sobel")
        {
            apply_sobel(img);
        }
        else
        {
            std::cerr << "Unknown filter: " << filter << "\n";
            print_usage();
            return 1;
        }

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "Filter '" << filter << "' done in " << ms << " ms\n";

        save_image(output_path, img);
        std::cout << "Saved result to " << output_path << "\n";
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
