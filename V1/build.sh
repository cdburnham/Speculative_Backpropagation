#!/bin/bash
set -e

# -----------------------------------------
# Interactive Build Script
# -----------------------------------------

echo "Select output model demo file: (xor_demo)"
read -p "> " DEMO_NAME

# Default if empty
if [[ -z "$DEMO_NAME" ]]; then
  DEMO_NAME="xor_demo"
fi

SRC_FILE="${DEMO_NAME}.c"
OUTPUT="${DEMO_NAME}"

# Core library sources (add all relevant .c files)
SRC_FILES=(
    "$SRC_FILE"
    network.c
    layer.c
    activations.c
    speculative_bp.c
    performance.c
    memory.c
)

echo ""
echo "Building demo: $SRC_FILE"
echo "Output binary: $OUTPUT"
echo ""

# Compile with clang, C11 standard, optimization, warnings
clang -O2 -std=c11 -Wall -Wextra "${SRC_FILES[@]}" -o "$OUTPUT"

echo ""
echo "Build complete -> ./$OUTPUT"

# Optional: run automatically
read -p "Run now? (y/n): " RUN_CHOICE
if [[ "$RUN_CHOICE" == "y" || "$RUN_CHOICE" == "Y" ]]; then
    echo ""
    echo "Running..."
    ./"$OUTPUT"
fi