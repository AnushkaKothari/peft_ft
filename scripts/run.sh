#!/bin/bash

# Define paths for saving output and error logs
OUTPUT_LOG="./output_log.txt"
ERROR_LOG="./error_log.txt"
SYSTEM_INFO_LOG="./system_info_log.txt"

# Record start time
echo "Script started at $(date)" > $OUTPUT_LOG
echo "Script started at $(date)" > $ERROR_LOG

# Save system information
echo "System Information:" > $SYSTEM_INFO_LOG
echo "Date: $(date)" >> $SYSTEM_INFO_LOG
echo "" >> $SYSTEM_INFO_LOG
uname -a >> $SYSTEM_INFO_LOG
echo "" >> $SYSTEM_INFO_LOG
echo "CPU Info:" >> $SYSTEM_INFO_LOG
lscpu >> $SYSTEM_INFO_LOG
echo "" >> $SYSTEM_INFO_LOG
echo "Memory Info:" >> $SYSTEM_INFO_LOG
free -h >> $SYSTEM_INFO_LOG
echo "" >> $SYSTEM_INFO_LOG
echo "Disk Usage:" >> $SYSTEM_INFO_LOG
df -h >> $SYSTEM_INFO_LOG
echo "" >> $SYSTEM_INFO_LOG
echo "Python Environment:" >> $SYSTEM_INFO_LOG
conda list >> $SYSTEM_INFO_LOG  # If using Conda
# pip freeze >> $SYSTEM_INFO_LOG  # If using pip directly

# Use nohup and & to execute the Python script in the background and save stdout and stderr
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=1,2,3,4
nohup python vicuna_scinew_ada.py >> $OUTPUT_LOG 2>> $ERROR_LOG &
# vicuna_scinew_ada_0shot.py
# vicuna_scinew_ada_1shot.py
# Record end time not needed here as the script runs in the background

echo "Finished running. Check $OUTPUT_LOG, $ERROR_LOG, and $SYSTEM_INFO_LOG for details."