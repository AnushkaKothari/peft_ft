peft_ft
This repository explores the fine-tuning of large language models using various Parameter-Efficient Fine-Tuning (PEFT) techniques, focusing on the impact of different PEFT methods, rank variations, and dataset choices.

Project Overview
Our project investigates the performance of several PEFT methods across different configurations:

PEFT Methods:

AdaLoRA
QLoRA
LoRA
IA3
Rank Variations:

r=4
r=32
r=512
Datasets:

Elife: https://elifesciences.org/
Scinews: https://www.sci.news/

Training Data Size:

Experiments are conducted with varying percentages of the training data, including 8, 16, 32, so on and full dataset sizes.

Scinews Dataset Updates
For the Scinews dataset, the mapping of content fields has been updated to align with the dataset's structure.

Paper_Body has been changed to summary in the function files.
News_Body has been changed to paper in the function files.
Important: Please ensure these changes are reflected in the relevant run files and scripts that process the Scinews dataset to avoid data loading errors. You'll need to modify the corresponding data loading and preprocessing logic wherever the original column names were used.