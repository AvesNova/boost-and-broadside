**WSL Shell Protocol**:
1. All Linux/bash commands MUST be executed within a persistent WSL session.
2. To begin, use `run_command` with exactly `CommandLine: "wsl"` and set `WaitMsBeforeAsync: 2000` (allowing time for the shell to initialize).
3. Do NOT use `wsl -e <command>` or try to chain commands in a single `run_command` call, as this often fails to load the correct environment or paths.
4. Capture the `CommandId` from the initial `wsl` call.
5. All subsequent interactions must use `send_command_input(CommandId: <id>, Input: "<command>\n", ...)` with the captured ID.
6. Always check the terminal output using `command_status` or `read_terminal` if you aren't sure if the command finished.