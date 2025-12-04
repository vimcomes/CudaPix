# Root CMakeLists.txt for CUDA image filters demo
cmake_minimum_required(VERSION 3.20)
project(cuda_image_filters LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(USE_FETCHCONTENT_IMGUI "Fetch Dear ImGui with FetchContent instead of third_party/imgui" OFF)
option(USE_FETCHCONTENT_STB "Fetch stb headers with FetchContent instead of third_party copies" OFF)

include(FetchContent)

if(USE_FETCHCONTENT_STB)
    FetchContent_Declare(
        stb
        URL https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
        URL_HASH SHA256=3e1e3540a0fbba60c184ab2217630a8b5eabd6d8caf2a4309a57d7d7bc1a7d1a
    )
    FetchContent_Declare(
        stb_write
        URL https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h
        URL_HASH SHA256=4d05b43d5bbeb13005bd6a826b104a7c12956d33f8e27b3faf4c2c28fb8f7e20
    )
    FetchContent_MakeAvailable(stb stb_write)
    set(STB_INCLUDE_DIR ${stb_SOURCE_DIR} ${stb_write_SOURCE_DIR})
else()
    set(STB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
endif()

if(USE_FETCHCONTENT_IMGUI)
    FetchContent_Declare(
        imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG v1.90.8
    )
    FetchContent_MakeAvailable(imgui)
    set(IMGUI_DIR ${imgui_SOURCE_DIR})
else()
    set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui)
endif()

find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(GLEW REQUIRED)

add_library(cuda_image_filters_core STATIC
    src/core/image.cpp
    src/core/filters_cuda.cu
    src/core/filters_cpu.cpp
)
set_target_properties(cuda_image_filters_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(cuda_image_filters_core
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${STB_INCLUDE_DIR}
)
target_link_libraries(cuda_image_filters_core PUBLIC CUDA::cudart)

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

add_library(imgui STATIC ${IMGUI_SOURCES})
target_include_directories(imgui
    PUBLIC
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
)
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLEW)
target_link_libraries(imgui PUBLIC SDL2::SDL2 OpenGL::GL GLEW::GLEW dl)

add_executable(cuda_image_filters_cli src/cli/main_cli.cpp)
target_link_libraries(cuda_image_filters_cli PRIVATE cuda_image_filters_core)

add_executable(cuda_image_filters_gui src/gui/main_gui.cpp)
target_include_directories(cuda_image_filters_gui PRIVATE ${IMGUI_DIR} ${IMGUI_DIR}/backends)
target_link_libraries(cuda_image_filters_gui
    PRIVATE
        cuda_image_filters_core
        imgui
        SDL2::SDL2
        OpenGL::GL
)
