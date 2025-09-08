import subprocess

commands = [
    "find . ! -name run.py -delete",
    "cmake ..",
    "cmake --build .",
    "./NN"
]

for cmd in commands:
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Command failed: {cmd}")
        break