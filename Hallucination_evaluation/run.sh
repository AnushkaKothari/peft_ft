#!/bin/bash

# Define paths for saving output and error logs
OUTPUT_LOG="./output_log_particular_adalora_r4.txt"
ERROR_LOG="./error_log_particular_adalora_r4.txt"
SYSTEM_INFO_LOG="./system_info_log_particular_adalora_r4.txt"
# First part is the peft technique, and second part is the rank
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
# export NCCL_P2P_DISABLE=1  # Disables peer-to-peer access across different nodes
# export NCCL_IB_DISABLE=1  # Disable Infiniband if not available
# export NCCL_DEBUG=INFO  # This enables detailed NCCL logging to debug potential issues
export CUDA_VISIBLE_DEVICES=0
nohup python eval_particular_adalora_r4.py >> $OUTPUT_LOG 2>> $ERROR_LOG &
# First part is the peft technique, and second part is the rank

echo "Finished running. Check $OUTPUT_LOG, $ERROR_LOG, and $SYSTEM_INFO_LOG for details."