#!/bin/bash

# Define the requirements file
REQUIREMENTS_FILE="requirements.txt"

# Check if the requirements file exists
if [[ -f "$REQUIREMENTS_FILE" ]]; then
    echo "Installing Python libraries from $REQUIREMENTS_FILE"
    pip install -r "$REQUIREMENTS_FILE"
else
    echo "Error: $REQUIREMENTS_FILE not found."
    exit 1
fi