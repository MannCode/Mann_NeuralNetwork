cmake_minimum_required(VERSION 3.10)
project(Mann_NeuralNetwork VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES "arm64")

# Include directories
include_directories(src dependencies/includes/GLFW dependencies/includes/imgui .)

# Source files
# all .cpp files in src/ except for those starting with test_
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "src/*.mm"
)
list(FILTER SOURCES EXCLUDE REGEX "src/test_.*\\.cpp$")

set_source_files_properties(src/SystemProfiler.mm PROPERTIES
    COMPILE_FLAGS "-ObjC++"
)

set(IMGUI_SOURCES
    dependencies/includes/imgui/imgui.cpp
    dependencies/includes/imgui/imgui_demo.cpp
    dependencies/includes/imgui/imgui_draw.cpp
    dependencies/includes/imgui/imgui_tables.cpp
    dependencies/includes/imgui/imgui_widgets.cpp
    dependencies/includes/imgui/backends/imgui_impl_glfw.cpp
    dependencies/includes/imgui/backends/imgui_impl_opengl3.cpp
)

set(IMPLOT_SOURCES  # Renamed to IMPLOT_SOURCES for clarity (was IMPLOT, which was a variable name)
    dependencies/includes/implot/implot.cpp
    dependencies/includes/implot/implot_items.cpp
)

# Find required packages
find_package(OpenGL REQUIRED)

if(APPLE)
    # On macOS, assume GLFW is installed (e.g., via Homebrew)
    find_package(glfw3 3.3 REQUIRED)
elseif(WIN32)
    # On Windows, try to find glfw3 normally
    find_package(glfw3 3.3 QUIET)

    if(NOT glfw3_FOUND)
        message(STATUS "glfw3 not found, fetching GLFW via FetchContent")

        include(FetchContent)
        FetchContent_Declare(
            glfw
            GIT_REPOSITORY https://github.com/glfw/glfw.git
            GIT_TAG        3.3.8
        )
        FetchContent_MakeAvailable(glfw)
    endif()
else()
    # On Linux or other platforms, try to find glfw3 normally
    find_package(glfw3 3.3 REQUIRED)
endif()

add_executable(NN
    ${SOURCES}
    ${IMGUI_SOURCES}
    ${IMPLOT_SOURCES}
)

# Link libraries
target_link_libraries(NN PRIVATE OpenGL::GL m)

if(WIN32 AND NOT glfw3_FOUND)
    # Link to the FetchContent glfw target
    target_link_libraries(NN PRIVATE glfw)
else()
    # Link to system glfw3
    target_link_libraries(NN PRIVATE glfw)
endif()

# macOS Metal support
if(APPLE)
    target_link_libraries(NN PRIVATE
        "-framework Metal"
        "-framework IOKit"
        "-framework CoreFoundation"
    )
endif()

# Windows NVML
if(WIN32)
    set(NVML_PATH "C:/Program Files/NVIDIA Corporation/NVSMI")
    target_include_directories(NN PRIVATE "${NVML_PATH}")
    target_link_libraries(NN PRIVATE "${NVML_PATH}/nvml.lib")
endif()

# Linux NVML
if(UNIX AND NOT APPLE)
    find_library(NVML_LIB nvidia-ml)
    if(NVML_LIB)
        target_link_libraries(NN PRIVATE ${NVML_LIB})
    else()
        message(WARNING "NVML not found. GPU monitoring disabled.")
    endif()
endif()

