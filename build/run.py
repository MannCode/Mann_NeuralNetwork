import subprocess
import sys
import os

def clean_project():
    if os.name == "nt":
        cmd = (
            "powershell -Command \"Get-ChildItem -Recurse | "
            "Where-Object { $_.Name -ne 'run.py' } | Remove-Item -Recurse -Force\""
        )
    else:
        "find . ! -name run.py -delete",
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print("Failed to clean project")
        sys.exit(1)

def build_project():
    result = subprocess.run("cmake ..", shell=True)
    if result.returncode != 0:
        print("CMake configuration failed")
        sys.exit(1)

    if os.name == 'nt':
        # On Windows, specify config Release (or Debug)
        build_cmd = "cmake --build . --config Release"
    else:
        build_cmd = "cmake --build ."

    result = subprocess.run(build_cmd, shell=True)
    if result.returncode != 0:
        print("Build failed")
        sys.exit(1)

def run_executable():
    if os.name == 'nt':
        exe = "NN.exe"
    else:
        exe = "./NN"
    print(f"Running: {exe}")
    result = subprocess.run(exe, shell=True)
    if result.returncode != 0:
        print("Executable failed")
        sys.exit(1)
if __name__ == "__main__":
    clean_project()
    build_project()
    run_executable()

# commands = [
#     "find . ! -name run.py -delete",
#     "cmake ..",
#     "cmake --build .",
#     "./NN"
# ]

# for cmd in commands:
#     print(f"Running: {cmd}")
#     result = subprocess.run(cmd, shell=True)
#     if result.returncode != 0:
#         print(f"Command failed: {cmd}")
#         break

