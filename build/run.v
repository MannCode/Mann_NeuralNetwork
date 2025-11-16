import os

fn platform_dependent_execution() {
    mut commands := []string{}

    if os.user_os() == 'macos' {
        commands = [
            'find . ! -name run.v -delete',
            'cmake ..',
            'cmake --build .',
            './NN',
    } else if os.user_os() == 'windows' {
        commands = [
            'for /d %i in (*) do if not "%i"=="_deps" rmdir /s /q "%i" & for %i in (*) do if not "%i"=="run.v" del /q "%i"',
            'cmake ..',
            'cmake --build . --config Release',
            'cd Release && NN.exe',
        ]
    } else {
        eprintln('Running on an unsupported OS: ${os.user_os()}')
        return
    }

    for cmd in commands {
        println('Running: $cmd')
        result := os.execute(cmd)
        if result.exit_code != 0 {
            eprintln('Command failed: $cmd')
            eprintln('Error output: $result.output')
            exit(result.exit_code)
        }
    }
}

fn main() {
    println('Starting platform-dependent execution...')
    platform_dependent_execution()
}
