import subprocess
import sys
import os

def platform_dependent_execution():
    if sys.platform == 'darwin':
        commands = [
            "find . ! -name run.py -delete",
            "cmake ..",
            "cmake --build .",
            "./NN"
        ]
        shell = "/bin/bash"
    elif sys.platform == 'win32':
        commands = [
            'for /d %i in (*) do if not "%i"=="_deps" rmdir /s /q "%i" & for %i in (*) do if not "%i"=="run.py" del /q "%i"',
            "cmake ..",
            "cmake --build . --config Release",
            "cd Release && NN.exe"  # Combine cd and execution
        ]
        shell = None  # Use default cmd.exe
    else:
        print(f"Running on an unsupported os: {sys.platform}")
        return

    for cmd in commands:
        print(f"Running: {cmd}")
        try:
            subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed: {cmd}", file=sys.stderr)
            print(f"Error: {e.stderr}", file=sys.stderr)
            sys.exit(e.returncode)

def main():
    platform_dependent_execution()

if __name__ == "__main__":
    main()