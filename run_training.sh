#!/bin/bash
# Script to run RoboPianist training Python scripts in parallel

set -e

# Configuration
CONDA_ENV_NAME="pianist"
SCRIPT_DIR="/dss/dsstbyfs02/pn52ru/pn52ru-dss-0000/di97jur/robopianist_evaluation"
SCRIPT1="evaluation.py"
SCRIPT2="evaluation_golden_hour.py"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Starting staggered parallel Python script execution${NC}"
echo "Environment: $CONDA_ENV_NAME"
echo "Started at: $(date)"

# Source conda and activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV_NAME

# Change to script directory
cd "$SCRIPT_DIR"

# Function to run a Python script
run_python_script() {
    local script=$1
    local delay=$2
    local log_file="${script%.py}_$(date +%Y%m%d_%H%M%S).log"
    
    if [ "$delay" -gt 0 ]; then
        echo -e "${YELLOW}Waiting ${delay}s before starting $script${NC}"
        sleep $delay
    fi
    
    echo -e "${GREEN}Starting $script at $(date)${NC}"
    echo -e "${GREEN}Log file: $log_file${NC}"
    
    # Run Python script with logging
    if python "$script" 2>&1 | tee "$log_file"; then
        echo -e "${GREEN}Completed $script successfully at $(date)${NC}"
    else
        echo -e "${RED}Failed to execute $script at $(date)${NC}"
        return 1
    fi
}

# Export function for parallel execution
export -f run_python_script
export GREEN YELLOW RED NC

# Run Python scripts with staggered start
echo -e "${YELLOW}Launching staggered parallel execution...${NC}"
echo -e "${YELLOW}$SCRIPT1 starts immediately${NC}"
echo -e "${YELLOW}$SCRIPT2 starts in 10 seconds${NC}"

run_python_script "$SCRIPT1" 0 &
PID1=$!

run_python_script "$SCRIPT2" 10 &
PID2=$!

# Wait for both to complete
wait $PID1
EXIT_CODE1=$?

wait $PID2
EXIT_CODE2=$?

echo -e "${GREEN}All Python scripts completed at: $(date)${NC}"

# Check exit codes
if [ $EXIT_CODE1 -eq 0 ] && [ $EXIT_CODE2 -eq 0 ]; then
    echo -e "${GREEN}✓ Both scripts completed successfully!${NC}"
elif [ $EXIT_CODE1 -eq 0 ]; then
    echo -e "${YELLOW}✓ $SCRIPT1 completed successfully${NC}"
    echo -e "${RED}✗ $SCRIPT2 failed${NC}"
    exit 1
elif [ $EXIT_CODE2 -eq 0 ]; then
    echo -e "${RED}✗ $SCRIPT1 failed${NC}"
    echo -e "${YELLOW}✓ $SCRIPT2 completed successfully${NC}"
    exit 1
else
    echo -e "${RED}✗ Both scripts failed${NC}"
    exit 1
fi

