import subprocess
import os

executable = "NN"

def get_cpp_files(directory='src'):
    return [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.cpp')]

def compile_cpp_files(cpp_files):
    command = f"g++ --std=c++26 {' '.join(cpp_files)} -o {executable}"
    return subprocess.run(command, shell=True, capture_output=True, text=True)

compile_cpp_files(get_cpp_files())

run_cmd = [f"./{executable}"]
run_result = subprocess.run(run_cmd, capture_output=True, text=True)

print("Program output:")
print(run_result.stdout)
if run_result.stderr:
    print("Program errors:")
    print(run_result.stderr)