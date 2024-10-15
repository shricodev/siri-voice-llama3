#!/usr/bin/env fish
# Using this above way of writing shebang can have some security concerns.
# See this stackoverflow thread: https://stackoverflow.com/a/72332845
# Since, I want this script to be portable for most of the users, instead of hardcoding like '#!/usr/bin/fish', I am using this way.

# Constants for error messages
set -l ERROR_USAGE "ERROR: Usage: fish {path_to_main.py}"
set -l ERROR_FILE_NOT_FOUND "ERROR: The main.py file does not exist or is not a valid file."
set -l ERROR_PYTHON_NOT_FOUND "ERROR: No suitable Python executable found."
set -l ERROR_FISH_NOT_INSTALLED "ERROR: Fish shell is not installed. Please install Fish."
set -l ERROR_ACTIVATE_NOT_FOUND "ERROR: activate.fish not found in '$VENV_DIR/bin'"
set -l ERROR_UNSUPPORTED_SHELL "ERROR: Unsupported shell: '$SHELL'"
set -l ERROR_REQUIREMENTS_NOT_FOUND "ERROR: 'requirements.txt' file not found in '$SCRIPT_DIR'"

# Determine the script directory, virtual environment directory, and log file
set -l SCRIPT_DIR (dirname (realpath (status -f)))
set -l VENV_DIR (realpath "$SCRIPT_DIR/../../.venv")
set -l REQUIREMENTS_FILE_PATH (realpath "$SCRIPT_DIR/../../requirements.txt")
set -g LOG_FILE (realpath "$SCRIPT_DIR/../../logs/shell-error-fish.log")

# Function to log messages
function log_and_exit
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $argv[1]" | tee -a $LOG_FILE
    exit 1
end

# Check if two arguments are provided
if test (count $argv) -ne 1
    log_and_exit $ERROR_USAGE
end

# Function to check if a file exists and has the correct extension
function check_file
    set -l file_path "$argv[1]"
    set -l expected_extension "$argv[2]"

    if not test -f "$file_path"
        log_and_exit $ERROR_FILE_NOT_FOUND
    end

    if not string match -q "*.$expected_extension" "$file_path"
        log_and_exit "ERROR: The file '$file_path' must be a '.$expected_extension' file."
    end
end

check_file "$argv[1]" "py"

# Extract and validate arguments
set -l MAIN_FILE_PATH (realpath "$argv[1]")

# Find the appropriate Python executable (python3 or python) in the system
set -l PYTHON_EXEC (command -v python3; or command -v python)

# Ensure that the Python executable is available before creating the virtual environment
if not test -d "$VENV_DIR"
    if test -z "$PYTHON_EXEC"
        log_and_exit $ERROR_PYTHON_NOT_FOUND
    end

    "$PYTHON_EXEC" -m venv "$VENV_DIR"

        # Activate the virtual environment after creating it
    if test -f "$VENV_DIR/bin/activate.fish"
        source "$VENV_DIR/bin/activate.fish"
    else
        log_and_exit $ERROR_ACTIVATE_NOT_FOUND
    end

    set -l PIP_EXEC_VENV (command -v pip3; or command -v pip)

    # Check if requirements.txt exists and install dependencies
    if test -f "$REQUIREMENTS_FILE_PATH"
        $PIP_EXEC_VENV install -r $REQUIREMENTS_FILE_PATH
    else
        log_and_exit $ERROR_REQUIREMENTS_NOT_FOUND
    end
end

# Ensure that the Fish shell is installed
if not test -x (command -v fish)
    log_and_exit $ERROR_FISH_NOT_INSTALLED
end

# Activate the virtual environment if the shell is Fish
if test "$SHELL" = (command -v fish)
    # Check if the activate.fish file exists before sourcing it
    if test -f "$VENV_DIR/bin/activate.fish"
        source "$VENV_DIR/bin/activate.fish"
    else
        log_and_exit $ERROR_ACTIVATE_NOT_FOUND
    end
else
    log_and_exit $ERROR_UNSUPPORTED_SHELL
end

# Get the path to the python3 executable from the virtual environment
set -l PYTHON_EXEC (command -v python3; or command -v python)

# Run the main.py file
"$PYTHON_EXEC" "$MAIN_FILE_PATH"
