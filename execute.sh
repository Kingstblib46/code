#!/bin/bash

# --- Configuration ---
# Directory where logs will be saved (relative to this script's location)
LOG_DIR="log"
# Python script to execute
PYTHON_SCRIPT="predict.py"
# Path to the Python executable within your conda environment
# Make sure this path is correct for your system! Found using 'which python' after activating the environment.
PYTHON_EXECUTABLE="/root/miniconda3/envs/capcha/bin/python"
# Alternatively, if 'conda activate capcha' reliably sets up your PATH, you *might* just use:
# PYTHON_EXECUTABLE="python"
# --- End Configuration ---

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR" || exit 1 # Change to script directory, exit if failed

echo "Current working directory: $(pwd)"

# Ensure the log directory exists
mkdir -p "$LOG_DIR"
echo "Ensuring log directory exists: $SCRIPT_DIR/$LOG_DIR"

# Check if the Python executable exists
if [ ! -f "$PYTHON_EXECUTABLE" ]; then
    echo "Error: Python executable not found at $PYTHON_EXECUTABLE"
    echo "Please verify the PYTHON_EXECUTABLE path in the script."
    # If using just "python", check if conda environment is activated.
    if [ "$PYTHON_EXECUTABLE" == "python" ]; then
        echo "If using PYTHON_EXECUTABLE=\"python\", ensure the 'capcha' environment is activated before running this script."
    fi
    exit 1
fi

# Check if the predict script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Prediction script not found: $SCRIPT_DIR/$PYTHON_SCRIPT"
    exit 1
fi


# Generate the log filename: YYYY-MM-DD_UnixTimestamp.log
LOG_FILENAME=$(date +%Y-%m-%d)_$(date +%s).log
LOG_FILE="$LOG_DIR/$LOG_FILENAME"

echo "--------------------------------------------------"
echo "Starting prediction at: $(date)"
echo "Running command: $PYTHON_EXECUTABLE $PYTHON_SCRIPT"
echo "Saving output (stdout & stderr) to: $LOG_FILE"
echo "--------------------------------------------------"

# Execute the python script and redirect both stdout and stderr to the log file
# The 'stdbuf -oL -eL' part tries to make output less buffered, so you might see logs sooner if tailing the file. Optional.
stdbuf -oL -eL "$PYTHON_EXECUTABLE" "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1

# Capture the exit status of the python script
EXIT_STATUS=$?

echo "--------------------------------------------------"
echo "Prediction finished at: $(date)"

# Check the exit status and report
if [ $EXIT_STATUS -eq 0 ]; then
  echo "Script finished successfully."
  echo "Log file located at: $SCRIPT_DIR/$LOG_FILE"
else
  echo "Script finished with errors (Exit Status: $EXIT_STATUS)."
  echo "Please check the log file for details: $SCRIPT_DIR/$LOG_FILE"
fi
echo "--------------------------------------------------"

exit $EXIT_STATUS