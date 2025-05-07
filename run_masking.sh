#!/bin/bash

# Usage: ./run_masking.sh /full/path/to/input/file

if [ $# -ne 1 ]; then
  echo "Usage: $0 <input_file_path>"
  exit 1
fi

INPUT_FILE="$1"
SCRIPT_DIR="/data/aadhaarmask"
PYTHON_SCRIPT="/data/aadhaarmask/aadhaar.py"
VENV_PATH="/data/aadhaarmask/masking/bin/activate"

# Activate the virtual environment
source "$VENV_PATH"

# Run the Python script and capture output JSON file path
OUTPUT_FILE=$(python3.11 "$PYTHON_SCRIPT" "$INPUT_FILE")

# Read and return content
if [ -f "$OUTPUT_FILE" ]; then
  cat "$OUTPUT_FILE"
  rm -f "$OUTPUT_FILE"
else
  echo "Error: Output file not found at $OUTPUT_FILE"
  exit 1
fi
