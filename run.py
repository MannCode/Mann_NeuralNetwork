# import subprocess
# from sys import platform
# import os

# executable = "NN"

# def get_cpp_files(directory='src'):
#     return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.cpp')]

# def compile_cpp_files(cpp_files):
#     if platform == "darwin":
#         print("MacOS detected")
#         command = f"g++ --std=c++26 {' '.join(cpp_files)} -o {executable} -lglfw -framework OpenGL"
#     elif platform == "win32" or platform == "win64":
#         print("Windows detected")
#         command = f"g++ {' '.join(cpp_files)} -o {executable}.exe -lglfw -framework OpenGL"
#     else:
#         raise NotImplementedError("This script is not implemented for this platform.")
#     return subprocess.run(command, shell=True, capture_output=True, text=True)

# def main():
#     print("====================================")
#     print("Compiling C++ files")

#     output = compile_cpp_files(get_cpp_files())
#     if output.returncode != 0:
#         print("Compilation failed with the following error:")
#         print(output.stderr)
#         print("====================================")
#     else:
#         print("Compilation successful.")
#         print("Running the program...")
#         print("====================================")
#         print("Mann Neural Network Started")
#         print("====================================\n\n")
#         run_cmd = [f"./{executable}"]
#         subprocess.run(run_cmd, shell=True) 

# if __name__ == "__main__":
#     main()

# import subprocess
# import sys
# import os

# executable = "NN"

# def get_cpp_files(directory='src'):
#     return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.cpp')]

# def compile_cpp_files(cpp_files):
#     if not cpp_files:
#         raise FileNotFoundError("No C++ files found in the src directory.")

#     if sys.platform == "darwin":
#         print("MacOS detected")
#         # Adjust paths based on Homebrew installation (ARM64 or Intel)
#         glfw_include = "/opt/homebrew/include" if os.path.exists("/opt/homebrew/include/GLFW") else "/usr/local/include"
#         glfw_lib = "/opt/homebrew/lib" if os.path.exists("/opt/homebrew/lib/libglfw.dylib") else "/usr/local/lib"
#         command = (
#             f"g++ --std=c++26 {' '.join(cpp_files)} "
#             f"-I{glfw_include} -L{glfw_lib} -lglfw -framework OpenGL -o {executable}"
#         )
#     elif sys.platform.startswith("win"):
#         print("Windows detected")
#         # Adjust paths for your GLFW installation
#         glfw_include = "C:/glfw/include"  # Update to your GLFW include path
#         glfw_lib = "C:/glfw/lib"         # Update to your GLFW library path
#         command = (
#             f"g++ {' '.join(cpp_files)} "
#             f"-I{glfw_include} -L{glfw_lib} -lglfw3 -lopengl32 -lgdi32 -o {executable}.exe"
#         )
#     else:
#         raise NotImplementedError("This script is not implemented for this platform.")
    
#     print(f"Running command: {command}")
#     return subprocess.run(command, shell=True, capture_output=True, text=True)

# def main():
#     print("====================================")
#     print("Compiling C++ files")
    
#     try:
#         output = compile_cpp_files(get_cpp_files())
#         if output.returncode != 0:
#             print("Compilation failed with the following error:")
#             print(output.stderr)
#             print("====================================")
#             sys.exit(1)
#         else:
#             print("Compilation successful.")
#             print("Running the program...")
#             print("====================================")
#             print("Mann Neural Network Started")
#             print("====================================\n\n")
#             run_cmd = f"./{executable}" if sys.platform == "darwin" else f"{executable}.exe"
#             result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
#             print(result.stdout)
#             if result.stderr:
#                 print("Runtime errors:")
#                 print(result.stderr)
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         sys.exit(1)
#     except NotImplementedError as e:
#         print(f"Error: {e}")
#         sys.exit(1)

# if __name__ == "__main__":
#     main()

import subprocess
import sys
import os

executable = "NN"

def get_cpp_files(directory='src'):
    # Get C++ files from src directory
    cpp_files = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.cpp')]
    # Add ImGui source files from dependencies
    imgui_dir = '../dependencies/includes/imgui'
    if os.path.exists(imgui_dir):
        cpp_files += [os.path.join(imgui_dir, file) for file in os.listdir(imgui_dir) if file.endswith('.cpp')]
    return cpp_files

def compile_cpp_files(cpp_files):
    if not cpp_files:
        raise FileNotFoundError("No C++ files found in the src or imgui directories.")

    include_path = "../dependencies/includes"
    lib_path = "../dependencies/lib"

    if sys.platform == "darwin":
        print("MacOS detected")
        command = (
            f"clang++ --std=c++20 -arch arm64 {' '.join(cpp_files)} "
            f"-I{include_path} -I{include_path}/imgui -L{lib_path} -lglfw -framework OpenGL -o {executable}"
        )
    elif sys.platform.startswith("win"):
        print("Windows detected")
        command = (
            f"g++ {' '.join(cpp_files)} "
            f"-I{include_path} -I{include_path}/imgui -L{lib_path} -lglfw3 -lopengl32 -lgdi32 -o {executable}.exe"
        )
    else:
        raise NotImplementedError("This script is not implemented for this platform.")
    
    print(f"Found C++ files: {cpp_files}")
    print(f"Running command: {command}")
    return subprocess.run(command, shell=True, capture_output=True, text=True)

def main():
    print("====================================")
    print("Compiling C++ files")
    
    try:
        cpp_files = get_cpp_files()
        output = compile_cpp_files(cpp_files)
        if output.returncode != 0:
            print("Compilation failed with the following error:")
            print(output.stderr)
            print("====================================")
            sys.exit(1)
        else:
            print("Compilation successful.")
            print("Running the program...")
            print("====================================")
            print("Mann Neural Network Started")
            print("====================================\n\n")
            run_cmd = f"./{executable}" if sys.platform == "darwin" else f"{executable}.exe"
            result = subprocess.run(run_cmd, shell=True, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print("Runtime errors:")
                print(result.stderr)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except NotImplementedError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()