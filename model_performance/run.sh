#!/bin/bash

# Define paths for saving output and error logs
OUTPUT_LOG="/home/anko00006/mp/output_log.txt"
ERROR_LOG="/home/anko00006/mp/error_log.txt"
SYSTEM_INFO_LOG="/home/anko00006/mp/system_info_log.txt"

python -m pip install wandb peft bitsandbytes torch datasets transformers huggingface_hub numpy

python -m pip install jupyter ipykernel torchvision torchaudio scikit-learn transformers sentencepiece thop
 
python /home/anko00006/mp/vicuna_scinews_qlora.py >> $OUTPUT_LOG 2>> $ERROR_LOG
