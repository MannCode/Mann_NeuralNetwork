import subprocess
from sys import platform
import os

executable = "NN"

def get_cpp_files(directory='src'):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.cpp')]

def compile_cpp_files(cpp_files):
    if platform == "darwin":
        print("MacOS detected")
        command = f"g++ --std=c++26 {' '.join(cpp_files)} -o {executable}"
    elif platform == "win32" or platform == "win64":
        print("Windows detected")
        command = f"g++ {' '.join(cpp_files)} -o {executable}.exe"
    else:
        raise NotImplementedError("This script is not implemented for this platform.")
    return subprocess.run(command, shell=True, capture_output=True, text=True)

def main():
    print("====================================")
    print("Compiling C++ files")

    output = compile_cpp_files(get_cpp_files())
    if output.returncode != 0:
        print("Compilation failed with the following error:")
        print(output.stderr)
        print("====================================")
    else:
        print("Compilation successful.")
        print("Running the program...")
        print("====================================")
        print("Mann Neural Network Started")
        print("====================================\n\n")
        run_cmd = [f"./{executable}"]
        subprocess.run(run_cmd, shell=True) 

if __name__ == "__main__":
    main()