#!/bin/bash
# ==============================================================================
# QRC Pipeline Orchestrator
# ==============================================================================
# Usage: ./run_pipeline.sh [config_file.py]
# Default: 01_config_qrc_ladder.py

CONFIG=${1:-01_config_qrc_ladder.py}

echo "=========================================================="
echo "Starting QRC Pipeline with Config: $CONFIG"
echo "=========================================================="

# 1. Simulation Runner
echo "[1/3] Running Simulation..."
python3 00_runner_parallel_CPU.py "$CONFIG"
if [ $? -ne 0 ]; then
    echo "Error in Simulation Step. Exiting."
    exit 1
fi

# 2. Analysis
echo "[2/3] Analyzing Results..."
python3 02_analyze_qrc_ladder.py "$CONFIG"
if [ $? -ne 0 ]; then
    echo "Error in Analysis Step. Exiting."
    exit 1
fi

# 3. Plotting
echo "[3/3] Generating Figures..."
python3 03_plot_qrc_ladder.py "$CONFIG"
if [ $? -ne 0 ]; then
    echo "Error in Plotting Step. Exiting."
    exit 1
fi

echo "=========================================================="
echo "Pipeline Completed Successfully."
echo "=========================================================="
