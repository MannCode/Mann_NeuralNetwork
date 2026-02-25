cmake_minimum_required(VERSION 3.10)
project(Mann_NeuralNetwork VERSION 1.0 LANGUAGES CXX)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fno-sanitize=address")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-sanitize=address")

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fsanitize=address")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OSX_ARCHITECTURES "arm64")

# Include directories
include_directories(src dependencies/includes/GLFW dependencies/includes/imgui dependencies/includes/metal-cpp .)

# Source files
# all .cpp files in src/ except for those starting with test_
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "src/*.mm"
)
list(FILTER SOURCES EXCLUDE REGEX "src/test_.*\\.cpp$")
list(FILTER SOURCES EXCLUDE REGEX "src/platforms/MAC/.*")
list(FILTER SOURCES EXCLUDE REGEX "src/platforms/WINDOWS/.*")

if(APPLE)
    file(GLOB PLATFORM_SOURCES "src/platforms/MAC/*.cpp")
endif()

if(WIN32)
    file(GLOB PLATFORM_SOURCES "src/platforms/WINDOWS/*.cpp")
endif()

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
    #link metal cpp
    find_package(glfw3 3.3 REQUIRED)
elseif(WIN32)
    # On Windows, try to find glfw3 normally
    find_package(glfw3 3.3 QUIET)

    if(NOT glfw3_FOUND)
        message(STATUS "glfw3 not found, fetching GLFW via FetchContent")

        include(FetchContent)

        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
        set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
        
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
    ${PLATFORM_SOURCES}
    ${IMGUI_SOURCES}
    ${IMPLOT_SOURCES}
)

if(WIN32 AND NOT glfw3_FOUND)
    # Link to the FetchContent glfw target
    target_link_libraries(NN PRIVATE glfw opengl32 dxgi pdh)
endif()

# macOS Metal support
if(APPLE)
    target_link_libraries(NN PRIVATE OpenGL::GL "glfw"
        "-framework Metal"
        "-framework Foundation"
        "-framework MetalKit"
        "-framework QuartzCore"
    )
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