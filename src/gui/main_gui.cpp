// src/gui/main_gui.cpp
#include <array>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstring>

#include <GL/glew.h> // must precede SDL_opengl.h to avoid gl.h before glew.h
#include <SDL.h>
#include <SDL_opengl.h>

#include "backends/imgui_impl_opengl3.h"
#include "backends/imgui_impl_sdl2.h"
#include "imgui.h"

#include "core/filters_cpu.h"
#include "core/filters_cuda.h"
#include "core/image.h"

struct GLTexture
{
    GLuint id = 0;
    int width = 0;
    int height = 0;

    void reset()
    {
        if (id != 0)
        {
            glDeleteTextures(1, &id);
            id = 0;
        }
        width = height = 0;
    }
};

bool upload_image_to_texture(const Image& img, GLTexture& texture)
{
    if (img.pixels.empty() || img.width <= 0 || img.height <= 0) return false;

    if (texture.id == 0)
    {
        glGenTextures(1, &texture.id);
    }
    texture.width = img.width;
    texture.height = img.height;

    glBindTexture(GL_TEXTURE_2D, texture.id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img.pixels.data());
    glBindTexture(GL_TEXTURE_2D, 0);
    return true;
}

ImVec2 fit_size(int img_w, int img_h, float max_w, float max_h)
{
    float scale = std::min(max_w / static_cast<float>(img_w), max_h / static_cast<float>(img_h));
    scale = std::min(scale, 1.0f);
    return ImVec2(img_w * scale, img_h * scale);
}

int main(int, char**)
{
    if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) != 0)
    {
        std::cerr << "SDL_Init failed: " << SDL_GetError() << "\n";
        return 1;
    }

    // GL 3.2 core
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
    SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
    SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);

    SDL_Window* window = SDL_CreateWindow("CUDA Image Filters", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED,
                                          1280, 720, SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
    if (!window)
    {
        std::cerr << "Failed to create window: " << SDL_GetError() << "\n";
        SDL_Quit();
        return 1;
    }

    SDL_GLContext gl_context = SDL_GL_CreateContext(window);
    if (!gl_context)
    {
        std::cerr << "Failed to create GL context: " << SDL_GetError() << "\n";
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_GL_MakeCurrent(window, gl_context);
    SDL_GL_SetSwapInterval(1);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "Failed to init GLEW\n";
        SDL_GL_DeleteContext(gl_context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;

    ImGui::StyleColorsDark();

    ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
    ImGui_ImplOpenGL3_Init("#version 150");

    std::array<char, 512> load_path_buf{};
    std::array<char, 512> save_path_buf{};
    std::strncpy(load_path_buf.data(), "input.png", load_path_buf.size() - 1);
    std::strncpy(save_path_buf.data(), "output.png", save_path_buf.size() - 1);

    enum class FilterType
    {
        None = 0,
        Grayscale,
        Brightness,
        Contrast,
        Blur,
        Sobel
    };

    FilterType current_filter = FilterType::None;
    float brightness_delta = 0.0f;
    float contrast_factor = 1.0f;
    double last_cpu_ms = 0.0;
    double last_gpu_ms = 0.0;
    double last_speedup = 0.0;
    bool have_timings = false;

    Image original_image;
    Image processed_image;
    bool has_image = false;

    GLTexture original_tex;
    GLTexture processed_tex;

    bool running = true;
    while (running)
    {
        SDL_Event event;
        while (SDL_PollEvent(&event))
        {
            ImGui_ImplSDL2_ProcessEvent(&event);
            if (event.type == SDL_QUIT)
            {
                running = false;
            }
            if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE &&
                event.window.windowID == SDL_GetWindowID(window))
            {
                running = false;
            }
        }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplSDL2_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Controls");
        ImGui::InputText("Load path", load_path_buf.data(), load_path_buf.size());
        if (ImGui::Button("Load image"))
        {
            try
            {
                original_image = load_image(load_path_buf.data());
                processed_image = original_image;
                has_image = true;
                upload_image_to_texture(original_image, original_tex);
                upload_image_to_texture(processed_image, processed_tex);
            }
            catch (const std::exception& ex)
            {
                std::cerr << "Load failed: " << ex.what() << "\n";
            }
        }

        const char* filter_labels[] = { "None", "Grayscale", "Brightness", "Contrast", "Blur", "Sobel" };
        int filter_idx = static_cast<int>(current_filter);
        if (ImGui::Combo("Filter", &filter_idx, filter_labels, IM_ARRAYSIZE(filter_labels)))
        {
            current_filter = static_cast<FilterType>(filter_idx);
        }

        ImGui::SliderFloat("Brightness delta", &brightness_delta, -1.0f, 1.0f);
        ImGui::SliderFloat("Contrast factor", &contrast_factor, 0.5f, 2.0f);

        if (ImGui::Button("Apply filter") && has_image)
        {
            Image cpu_image = original_image;
            Image gpu_image = original_image;
            try
            {
                if (current_filter == FilterType::None)
                {
                    processed_image = original_image;
                    have_timings = false;
                    upload_image_to_texture(processed_image, processed_tex);
                }
                else
                {
                    auto start_cpu = std::chrono::high_resolution_clock::now();
                    switch (current_filter)
                    {
                    case FilterType::Grayscale: cpu_grayscale(cpu_image); break;
                    case FilterType::Brightness: cpu_brightness(cpu_image, brightness_delta); break;
                    case FilterType::Contrast: cpu_contrast(cpu_image, contrast_factor); break;
                    case FilterType::Blur: cpu_box_blur(cpu_image); break;
                    case FilterType::Sobel: cpu_sobel(cpu_image); break;
                    case FilterType::None: break;
                    }
                    auto end_cpu = std::chrono::high_resolution_clock::now();
                    last_cpu_ms = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();

                    auto start_gpu = std::chrono::high_resolution_clock::now();
                    switch (current_filter)
                    {
                    case FilterType::Grayscale: apply_grayscale(gpu_image); break;
                    case FilterType::Brightness: apply_brightness(gpu_image, brightness_delta); break;
                    case FilterType::Contrast: apply_contrast(gpu_image, contrast_factor); break;
                    case FilterType::Blur: apply_box_blur(gpu_image); break;
                    case FilterType::Sobel: apply_sobel(gpu_image); break;
                    case FilterType::None: break;
                    }
                    auto end_gpu = std::chrono::high_resolution_clock::now();
                    last_gpu_ms = std::chrono::duration<double, std::milli>(end_gpu - start_gpu).count();

                    last_speedup = (last_gpu_ms > 0.0) ? (last_cpu_ms / last_gpu_ms) : 0.0;
                    have_timings = true;

                    processed_image = std::move(gpu_image);
                    upload_image_to_texture(processed_image, processed_tex);
                }
            }
            catch (const std::exception& ex)
            {
                std::cerr << "Filter failed: " << ex.what() << "\n";
            }
        }

        if (have_timings)
        {
            ImGui::Separator();
            ImGui::Text("Timings:");
            ImGui::Text("CPU: %.3f ms", last_cpu_ms);
            ImGui::Text("GPU: %.3f ms", last_gpu_ms);
            if (last_gpu_ms > 0.0)
                ImGui::Text("Speedup: %.2fx", last_speedup);
        }

        ImGui::InputText("Save path", save_path_buf.data(), save_path_buf.size());
        if (ImGui::Button("Save result") && has_image)
        {
            try
            {
                save_image(save_path_buf.data(), processed_image);
                std::cout << "Saved to " << save_path_buf.data() << "\n";
            }
            catch (const std::exception& ex)
            {
                std::cerr << "Save failed: " << ex.what() << "\n";
            }
        }
        ImGui::End();

        ImGui::Begin("Images");
        if (has_image)
        {
            ImVec2 avail = ImGui::GetContentRegionAvail();
            float half_w = avail.x * 0.5f - 10.0f;
            ImVec2 size_orig = fit_size(original_image.width, original_image.height, half_w, avail.y);
            ImVec2 size_proc = fit_size(processed_image.width, processed_image.height, half_w, avail.y);

            ImGui::BeginGroup();
            ImGui::Text("Original");
            ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<intptr_t>(original_tex.id)), size_orig);
            ImGui::EndGroup();

            ImGui::SameLine();

            ImGui::BeginGroup();
            ImGui::Text("Processed");
            ImGui::Image(reinterpret_cast<ImTextureID>(static_cast<intptr_t>(processed_tex.id)), size_proc);
            ImGui::EndGroup();
        }
        else
        {
            ImGui::Text("Load an image to begin.");
        }
        ImGui::End();

        ImGui::Render();
        int display_w, display_h;
        SDL_GL_GetDrawableSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        SDL_GL_SwapWindow(window);
    }

    original_tex.reset();
    processed_tex.reset();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplSDL2_Shutdown();
    ImGui::DestroyContext();

    SDL_GL_DeleteContext(gl_context);
    SDL_DestroyWindow(window);
    SDL_Quit();
    return 0;
}
