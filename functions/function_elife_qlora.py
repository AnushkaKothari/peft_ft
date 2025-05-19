import os
import torch
import numpy as np
import random
from datasets import load_dataset
import wandb
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling)
from peft import prepare_model_for_kbit_training, get_peft_model
import logging
import json
from huggingface_hub import login

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def notebook_login(hugging_face_token):
    try:
        login(token=hugging_face_token)
    except Exception as e:
        logger.error(f"Failed to log in Hugging Face: {e}")

def setup_environment(wandb_project):
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    try:
        wandb.login()
    except Exception as e:
        logger.error(f"Failed to log in to Weights & Biases: {e}")

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def load_and_prepare_dataset(dataset_name, sizes, full):
    try:
        dataset = load_dataset(dataset_name)
        for split, size in sizes.items():
            if split in dataset and size is not None:
                if split == 'train' and full == True:
                    dataset[split] = dataset[split]
                else:
                    dataset[split] = dataset[split].select(range(size))
        return dataset
    except Exception as e:
        logger.error(f"Failed to load or prepare dataset: {e}")
        return None

def training_prompt(prefix, sample, suffix=""):
    instruction = "Based on the provided academic paper, generate a corresponding scientific news report."
    input_text = sample.get("Paper_Body", "")
    response = sample.get("News_Body", "")
    return f"{prefix}Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nResponse:\n{response}{suffix}"

def tokenize_function(tokenizer, prompt_function, max_length):
    def tokenize(examples):
        result = tokenizer(prompt_function(examples), truncation=True, max_length=max_length, padding="max_length", return_attention_mask=True)
        result["labels"] = result["input_ids"].copy()
        return result
    return tokenize

def model_and_tokenizer_setup(base_model,quant_config,  peft_config):
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, 
                                                     quantization_config=quant_config, 
                                                     use_cache=False)
        tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        #model = LoraModel(model, peft_config, "default")
        return model, tokenizer
    except Exception as e:
        logger.error(f"Failed to setup model: {e}")
        return None, None

def train_model(dataset, model, tokenizer, training_args, max_length):
    training_args = TrainingArguments(**training_args)
    tokenized_datasets = dataset.map(
        tokenize_function(tokenizer, lambda x: training_prompt("<s>", x, "</s>"), max_length)
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    return trainer

def inference_prompt(prefix, sample, suffix=""):
    instruction = "Based on the provided academic paper, generate a corresponding scientific news report."
    input_text = sample.get("Paper_Body", "")
    return f"{prefix}Instruction:\n{instruction}\n\nInput:\n{input_text}\n\nResponse:\n"

def generate_answer(model, tokenizer, sample, max_length, beams):
    inputs_dict = tokenizer(inference_prompt("<s>", sample), truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = inputs_dict.input_ids.cuda()
    predicted_ids = model.generate(input_ids, do_sample=False, repetition_penalty=1.2, no_repeat_ngram_size=3, max_new_tokens = 768, num_beams=beams)[0]
    prediction = tokenizer.decode(predicted_ids, skip_special_tokens=False)
    sample["Prediction"] = prediction
    return sample

def postprocess_prediction(sample):
    prediction = sample["Prediction"]
    response_prefix = "Response:\n"
    if response_prefix in prediction:
        prediction = prediction.split(response_prefix)[-1].strip()
    sample["Prediction"] = prediction
    return sample

def save_prediction_to_file(predictions, file_path):
    with open(file_path, "w") as f:
        json.dump(predictions, f, indent=4)
        
def inference_prompt_1shot(prefix, random_example, sample, suffix=""):
    instruction = "Based on the provided academic paper, generate a corresponding scientific news report."
    input_random_text = random_example.get("Paper_Body", "")
    random_response = random_example.get("News_Body", "")
    
    input_text = sample.get("Paper_Body", "")
    return f"{prefix}Instruction:\n{instruction}\n\nInput:\n{input_random_text}\n\nResponse:\n{random_response}\n\nInstruction:\n{instruction}\n\nInput:\n{input_text}\n\nResponse:\n"

def inference_prompt_2shot(prefix, random_example1, random_example2, sample, suffix=""):
    instruction = "Based on the provided academic paper, generate a corresponding scientific news report."
    input_random_text1 = random_example1.get("Paper_Body", "")
    random_response1 = random_example1.get("News_Body", "")
    input_random_text2 = random_example2.get("Paper_Body", "")
    random_response2 = random_example2.get("News_Body", "")

    input_text = sample.get("Paper_Body", "")
    return f"{prefix}Instruction:\n{instruction}\n\nInput:\n{input_random_text1}\n\nResponse:\n{random_response1}\n\nInstruction:\n{instruction}\n\nInput:\n{input_random_text2}\n\nResponse:\n{random_response2}\n\nInstruction:\n{instruction}\n\nInput:\n{input_text}\n\nResponse:\n"

def inference_prompt_4shot(prefix, random_example1, random_example2, random_example3, random_example4, sample, suffix=""):
    instruction = "Based on the provided academic paper, generate a corresponding scientific news report."
    input_random_text1 = random_example1.get("Paper_Body", "")
    random_response1 = random_example1.get("News_Body", "")
    input_random_text2 = random_example2.get("Paper_Body", "")
    random_response2 = random_example2.get("News_Body", "")
    input_random_text3 = random_example3.get("Paper_Body", "")
    random_response3 = random_example3.get("News_Body", "")
    input_random_text4 = random_example4.get("Paper_Body", "")
    random_response4 = random_example4.get("News_Body", "")

    input_text = sample.get("Paper_Body", "")
    return f"{prefix}Instruction:\n{instruction}\n\nInput:\n{input_random_text1}\n\nResponse:\n{random_response1}\n\nInstruction:\n{instruction}\n\nInput:\n{input_random_text2}\n\nResponse:\n{random_response2}Instruction:\n{instruction}\n\nInput:\n{input_random_text3}\n\nResponse:\n{random_response3}\n\nInstruction:\n{instruction}\n\nInput:\n{input_random_text4}\n\nResponse:\n{random_response4}\n\nInstruction:\n{instruction}\n\nInput:\n{input_text}\n\nResponse:\n"

def generate_answer_1shot(model, tokenizer, sample, max_length, beams):
    inputs_dict = tokenizer(inference_prompt_1shot("<s>", sample), truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = inputs_dict.input_ids.cuda()
    predicted_ids = model.generate(input_ids, do_sample=False, repetition_penalty=1.2, no_repeat_ngram_size=3, max_new_tokens = 768, num_beams=beams)[0]
    prediction = tokenizer.decode(predicted_ids, skip_special_tokens=False)
    sample["Prediction"] = prediction
    return sample

def generate_answer_2shot(model, tokenizer, sample, max_length, beams):
    inputs_dict = tokenizer(inference_prompt_2shot("<s>", sample), truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = inputs_dict.input_ids.cuda()
    predicted_ids = model.generate(input_ids, do_sample=False, repetition_penalty=1.2, no_repeat_ngram_size=3, max_new_tokens = 768, num_beams=beams)[0]
    prediction = tokenizer.decode(predicted_ids, skip_special_tokens=False)
    sample["Prediction"] = prediction
    return sample

def generate_answer_4shot(model, tokenizer,ex1, ex2, ex3, ex4, sample, max_length, beams):
    inputs_dict = tokenizer(inference_prompt_4shot("<s>", ex1, ex2,ex3,ex4,sample), truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = inputs_dict.input_ids.cuda()
    predicted_ids = model.generate(input_ids, do_sample=False, repetition_penalty=1.2, no_repeat_ngram_size=3, max_new_tokens = 768, num_beams=beams)[0]
    prediction = tokenizer.decode(predicted_ids, skip_special_tokens=False)
    sample["Prediction"] = prediction
    return sample