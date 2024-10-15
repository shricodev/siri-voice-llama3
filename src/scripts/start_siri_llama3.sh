#!/usr/bin/env bash
# Using this above way of writing shebang can have some security concerns.
# See this stackoverflow thread: https://stackoverflow.com/a/72332845
# Since, I want this script to be portable for most of the users, instead of hardcoding like '#!/usr/bin/bash', I am using this way.

ERROR_USAGE="ERROR: Usage: bash {path_to_main.py}"
ERROR_FILE_NOT_FOUND="ERROR: The main.py file does not exist or is not a valid file."
ERROR_PYTHON_NOT_FOUND="ERROR: No suitable Python executable found."
ERROR_BASH_NOT_INSTALLED="ERROR: Bash shell is not installed. Please install Bash."
ERROR_ACTIVATE_NOT_FOUND="ERROR: activate file not found in '$VENV_DIR/bin'"
ERROR_UNSUPPORTED_SHELL="ERROR: Unsupported shell: '$SHELL'"
ERROR_REQUIREMENTS_NOT_FOUND="ERROR: requirements.txt file not found in '$SCRIPT_DIR'"

# Determine the script directory, virtual environment directory, and log file
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VENV_DIR="$(realpath "$SCRIPT_DIR/../../.venv")"
LOG_FILE="$(realpath "$SCRIPT_DIR/../../logs/shell-error-bash.log")"
REQUIREMENTS_FILE_PATH="$(realpath "$SCRIPT_DIR/../../requirements.txt")"

log_and_exit() {
  local message="$1"

  echo "[$(date +"%Y-%m-%d %H:%M:%S")] $message" | tee -a $LOG_FILE
  exit 1
}

# Check if the main.py file is provided as an argument
if [ $# -ne 1 ]; then
  log_and_exit "$ERROR_USAGE"
fi

# Function to check if a file exists and has the correct extension
check_file() {
    local file_path="$1"
    local expected_extension="$2"

    if [ ! -f "$file_path" ]; then
        log_and_exit "$ERROR_FILE_NOT_FOUND"
    fi

    if ! [[ "$file_path" == *".$expected_extension" ]]; then
        log_and_exit "The file '$file_path' must be a '.$expected_extension' file."
    fi
}

# Validate the provided main.py file
check_file "$1" "py"

# Extract and validate arguments
MAIN_FILE_PATH="$(realpath "$1")"

# Find the appropriate Python executable
PYTHON_EXEC="$(command -v python3 || command -v python)"

# Ensure that the Python executable is available before creating the virtual environment
if [ ! -d "$VENV_DIR" ]; then
    if [ -z "$PYTHON_EXEC" ]; then
        log_and_exit "$ERROR_PYTHON_NOT_FOUND"
    fi

    "$PYTHON_EXEC" -m venv "$VENV_DIR"

    # Activate the virtual environment after creating it
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    else
        log_and_exit "$ERROR_ACTIVATE_NOT_FOUND"
    fi

    PIP_EXEC_VENV = "$(command -v pip3 || command -v pip)"

    # Check if requirements.txt exists and install dependencies
    if [ -f "$REQUIREMENTS_FILE_PATH" ]; then
        "$PIP_EXEC_VENV" install -r "$REQUIREMENTS_FILE_PATH"
    else
        log_and_exit "$ERROR_REQUIREMENTS_NOT_FOUND"
    fi
fi

# Ensure that the Bash shell is installed.
if ! command -v bash &> /dev/null; then
    log_and_exit "$ERROR_BASH_NOT_INSTALLED"
fi

# Activate the virtual environment based on the shell type
if [[ "$SHELL" == *"/bash" ]]; then
    # Check if the activate file exists before sourcing it
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    else
        log_and_exit "$ERROR_ACTIVATE_NOT_FOUND"
    fi
else
    log_and_exit "$ERROR_UNSUPPORTED_SHELL"
fi

# Set the python executable to the one from the virtual environment
PYTHON_EXEC="$(command -v python3 || command -v python)"

# Run the main.py file
"$PYTHON_EXEC" "$MAIN_FILE_PATH"
