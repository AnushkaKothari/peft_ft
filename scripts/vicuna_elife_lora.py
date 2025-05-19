import torch
import wandb
from transformers import (AutoTokenizer)
from peft import LoraConfig
import logging
import sys
from accelerate import Accelerator

from functions.function_elife import (notebook_login, setup_environment, set_seed, load_and_prepare_dataset, 
                      model_and_tokenizer_setup, train_model, generate_answer,
                       postprocess_prediction, save_prediction_to_file)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("Start Hugging Face....................")
    hugging_face_token = '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'
    notebook_login(hugging_face_token)

    config = {
        "base_model": "lmsys/vicuna-7b-v1.5",
        "wandb_project": "Vicuna7b_elife",
        "random_seed": 2024,
        "huggingface_model": "Anushka1304/vicuna7b_v15_rank4_elife_lora_full", # TO CHANGE with Dataset Size
        "dataset_name": "Anushka1304/elife_with_full_paper",  # TO CHANGE with Dataset
        "dataset_sizes": {"train": 0, "validation": 0 ,"test": 241},  # TO CHANGE with Dataset Size
        "full": True, # TO CHANGE with Dataset Size
        "max_length": 16384,
        "num_beams" : 1,
        "method" : "lora",
        "rank" : 4,
        "size" : "full", # TO CHANGE with Dataset Size
        "result_json":"results_elife_rank4_lora_full.json", # TO CHANGE with Dataset Size
        "training_args": {
            "output_dir": "./vicuna_instruct_generation_results",
            "per_device_train_batch_size": 2,
            "num_train_epochs": 7,
            "save_strategy": "epoch",
            "evaluation_strategy": "epoch",
            "learning_rate": 2e-4,
            "report_to": "wandb",
            "bf16": True,
            "warmup_ratio":0.03,
            "gradient_accumulation_steps":4,
            "optim":"paged_adamw_8bit",
            "load_best_model_at_end":True,
            "greater_is_better":False,

        },
    }

    wandb.init(project=config["wandb_project"], config=config["training_args"])

    set_seed(config["random_seed"])
    setup_environment(config["wandb_project"])
    wandb.config.num_beams = config["num_beams"]
    wandb.config.method = config["method"]
    wandb.config.size = config["size"]
    wandb.config.rank = config["rank"]
    dataset = load_and_prepare_dataset(config["dataset_name"], config["dataset_sizes"],config["full"])
    if dataset is None:
        sys.exit("Failed to load dataset. Exiting...")


    peft_config = LoraConfig(r=config["rank"], use_rslora=True, lora_alpha=2*config["rank"], target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.1,  # Conventional
        task_type="CAUSAL_LM",)
    accelerator = Accelerator()
    model, tokenizer = model_and_tokenizer_setup(config["base_model"], peft_config)
    # Move model to multiple GPUs using accelerate
    model = accelerator.prepare(model)
    print("Start Model Parallelization....................")
    print(torch.cuda.device_count())

    if model is None or tokenizer is None:
        sys.exit("Failed to set up model. Exiting...")
    
    trainer = train_model(dataset, model, tokenizer, config["training_args"], config["max_length"])
    if trainer is None:
        sys.exit("Failed to set up trainer. Exiting...")

    try:
        trainer.train()
        logger.info("Training completed successfully.")
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")

    # Save the model and tokenizer
    try:
        model.push_to_hub(config["huggingface_model"],use_auth_token=True,
                  commit_message="thesis_training",
                  private=True)
        tokenizer.push_to_hub(config["huggingface_model"],use_auth_token=True,
                  commit_message="thesis_training",
                  private=True)
        model.save_pretrained(config["training_args"]["output_dir"] + '/final_trained_model')
        tokenizer.save_pretrained(config["training_args"]["output_dir"] + '/final_trained_tokenizer')
        logger.info("Model and tokenizer have been saved.")
    except Exception as e:
        logger.error(f"Failed to save the model and tokenizer: {e}")

    print("Starting inference...")
    print("Loading the model for inference...")
    model.to("cuda:0")
    try:
        model.eval()
        eval_tokenizer = AutoTokenizer.from_pretrained(config["training_args"]["output_dir"] + '/final_trained_tokenizer', add_bos_token=True, trust_remote_code=True)
        all_predictions = []
        with torch.no_grad():
            i=0
            for sample in dataset["test"]:
                prediction = generate_answer(model, eval_tokenizer, sample, config["max_length"], config["num_beams"])
                prediction = postprocess_prediction(prediction)
                all_predictions.append(prediction)
                save_prediction_to_file(all_predictions,file_path=config["result_json"])
                wandb.log({"Test_samples": i})
                i += 1
        print("Inference completed successfully.")
    except Exception as e:
        print(f"An error occurred during inference: {e}")

if __name__ == '__main__':
    main()