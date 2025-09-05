cmake_minimum_required(VERSION 3.10)
project(Mann_NeuralNetwork VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(src dependencies/includes/GLFW dependencies/includes/imgui .)

# Source files
set(SOURCES
    src/main.cpp
    src/mannui.cpp
    src/mann.cpp
    src/mnist.cpp
    src/MNNetwork.cpp
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

# set(boost_min_ver 1.50.0)
# set(boost_libs system filesystem)

# Find required packages
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
# find_package(Boost ${boost_min_ver})


# if(Boost_FOUND)
#     find_package(Boost ${boost_min_ver} COMPONENTS ${boost_libs})
# endif()

# Add executable
add_executable(NN ${SOURCES} ${IMGUI_SOURCES})
# Link libraries
target_link_libraries(NN PRIVATE OpenGL::GL glfw)