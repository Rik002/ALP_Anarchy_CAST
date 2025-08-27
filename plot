#!/bin/bash

# ALP Anarchy CAST Plotting Script
# Runs the Python analysis and plotting pipeline, displaying constants and configuration

echo "ALP Anarchy CAST Analysis and Plotting Pipeline"
echo "Date and Time: $(date)"
echo "-----------------------------------"

# Check for constants.dat and data files
if [ ! -f "config/constants.dat" ]; then
    echo "Diagnostic: config/constants.dat not found!"
    exit 1
fi
echo "Diagnostic: config/constants.dat found"
if [ ! -d "data" ] || [ -z "$(ls -A data)" ]; then
    echo "Diagnostic: No data files found in data/! Run ./simulate first."
    exit 1
fi
echo "Diagnostic: Data files found in data/"

# Parse constants.dat and display in a table
    echo "Configuration Details:"
    echo "+-----------------------------------+-----------------------+----------------------+"
    echo "| Parameter                         | Value                 | Units                |"
    echo "+-----------------------------------+-----------------------+----------------------+"
    while IFS='=' read -r key value; do
        if [[ $key =~ ^[[:space:]]*# || $key =~ ^[[:space:]]*------- || $key =~ ^[[:space:]]*$ || $key == "[Constants]" ]]; then
            continue
        fi
        key=$(echo "$key" | xargs)
        value_full=$(echo "$value" | xargs)
        value=$(echo "$value_full" | cut -d':' -f1 | xargs)
        units=$(echo "$value_full" | cut -d':' -f2- | cut -d',' -f2- | xargs)
        if [ -z "$units" ]; then
            units="No units"
        fi
        printf "| %-33s | %-21s | %-20s |\n" "$key" "$value" "$units"
    done < config/constants.dat
    echo "+-----------------------------------+-----------------------+--------------------+"
    echo "Data Generation Resolution:"
    g_e_points=$(echo "$(grep '^G_E_POINTS_PER_DECADE' config/constants.dat | cut -d'=' -f2 | cut -d':' -f1 | xargs) * ($(grep '^G_E_LOG_MAX' config/constants.dat | cut -d'=' -f2 | cut -d':' -f1 | xargs) - $(grep '^G_E_LOG_MIN' config/constants.dat | cut -d'=' -f2 | cut -d':' -f1 | xargs))" | bc)
    g_gamma_points=$(echo "$(grep '^G_GAMMA_POINTS_PER_DECADE' config/constants.dat | cut -d'=' -f2 | cut -d':' -f1 | xargs) * ($(grep '^G_GAMMA_LOG_MAX' config/constants.dat | cut -d'=' -f2 | cut -d':' -f1 | xargs) - $(grep '^G_GAMMA_LOG_MIN' config/constants.dat | cut -d'=' -f2 | cut -d':' -f1 | xargs))" | bc)
    max_n=$(grep '^MAX_N' config/constants.dat | cut -d'=' -f2 | cut -d':' -f1 | xargs)
    echo "- Figure 2: $g_e_points g_e points"
    echo "           $g_gamma_points g_gamma points"
    echo "- Figure 3: 300 g_gamma points per N (2 to $max_n)"
    echo "-----------------------------------"

# Check Python dependencies
echo "Checking Python dependencies..."
if ! python3 -c "import numpy, scipy, matplotlib, tabulate, tqdm" 2>/dev/null; then
    echo "Installing required Python packages..."
    pip install numpy scipy matplotlib tabulate tqdm pandas
    if [ $? -ne 0 ]; then
        echo "Diagnostic: Failed to install Python packages!"
        exit 1
    fi
fi
echo "Diagnostic: Python dependencies satisfied"

# Run Python plotting
echo "Running Python analysis and plotting..."
python3 simulations/main.py
if [ $? -ne 0 ]; then
    echo "Diagnostic: Plotting failed!"
    exit 1
fi
echo "Diagnostic: Plotting completed"

# Summarize output files
echo -e "\nOutput Files Summary:"
echo "+----------------------------------------------------+--------+"
echo "| File                                               | Status |"
echo "+----------------------------------------------------+--------+"
for file in plots/*.png; do
    if [ -f "$file" ]; then
        printf "| %-50s | %-8s |\n" "$(basename $file)" "Found"
    else
        printf "| %-50s | %-8s |\n" "$(basename $file)" "Missing"
    fi
done
echo "+----------------------------------------------------+--------+"
