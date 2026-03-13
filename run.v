// Read Me: This V script performs platform-dependent execution of build and run commands
// Copy this file and put it in the build/ directory as run.v to use it.
// goto build/ in your terminal and run `v run run.v`


import os

fn platform_dependent_execution() {
    // Create build directory if it doesn't exist
    if !os.exists('build') {
        os.mkdir('build') or {
            eprintln('Failed to create build directory: $err')
            return
        }
    }

    os.chdir('build') or {
        eprintln('Failed to change directory to build: $err')
        return
    }


    mut commands := []string{}

    if os.user_os() == 'macos' {
        commands = [
            'find . -delete',
            "xcrun -sdk macosx metal -frecord-sources=flat ../src/platforms/MAC/gpufunc.metal",
            'cmake ..',
            // make NN executable in root folder
            'make -j8',
            'go to root',
            './NN',
        ]
    } else if os.user_os() == 'windows' {
        commands = [
            'for /d %i in (*) do if not "%i"=="_deps" rmdir /s /q "%i" & for %i in (*) do if not "%i"=="run.v" del /q "%i"',
            'cmake .. -DCMAKE_BUILD_TYPE=Debug',
            'cmake --build . --config Debug --verbose',
            'cmake ..',
            // 'cmake --build . --config Release',
            // 'cmake ..',
            'cd Debug',
            'NN.exe',
        ]
    } else {
        eprintln('Running on an unsupported OS: ${os.user_os()}')
        return
    }

    for cmd in commands {
        println('Running: $cmd')

        if cmd.starts_with('go to root') {
            os.chdir('..') or {
                eprintln('Failed to change directory to root: $err')
                return
            }
            continue
        }


        result := os.execute(cmd)
        if result.exit_code != 0 {
            eprintln('Command failed: $cmd')
            eprintln('Error output: \n$result.output')
            exit(result.exit_code)
        }
        else {
            println(result.output)
        }
    }
}

fn create_backup() {
    // Create a backup of Whole project before running build commands
    backup_dir := 'backup/'
    if !os.exists(backup_dir) {
        os.mkdir(backup_dir) or {
            eprintln('Failed to create backup directory: $err')
            return
        }
    }
    else {
        // clear existing backup directory
        os.rmdir_all(backup_dir) or {
            eprintln('Failed to clear existing backup directory: $err')
            return
        }
        os.mkdir(backup_dir) or {
            eprintln('Failed to recreate backup directory: $err')
            return
        }
    }
    // backup all files and directories except the backup directory itself
    for item in os.ls('.') or { [] } {
        if item == 'backup' || item == 'build' {
            continue
        }
        src_path := item
        dest_path := os.join_path(backup_dir, item)
        if os.is_dir(src_path) {
            // create dest_path directory
            os.mkdir(dest_path) or {
                eprintln('Failed to create backup subdirectory $dest_path: $err')
                return
            }

            os.cp_all(src_path, dest_path, true) or {
                eprintln('Failed to backup directory $src_path: $err')
                return
            }
        } else {
            os.cp(src_path, dest_path) or {
                eprintln('Failed to backup file $src_path: $err')
                return
            }
        }
    }
    println('Backup created at $backup_dir')
}

fn main() {
    println('Starting platform-dependent execution...')
    create_backup()
    platform_dependent_execution()
}
