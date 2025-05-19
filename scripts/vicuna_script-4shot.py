import torch
import random
import wandb
from peft import AdaLoraConfig
import logging
import sys

from functions.function_elife import (notebook_login, setup_environment, set_seed, load_and_prepare_dataset, 
                      model_and_tokenizer_setup, generate_answer_4shot,
                       postprocess_prediction, save_prediction_to_file)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    print("Start Hugging Face....................")
    hugging_face_token = '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'
    notebook_login(hugging_face_token)
    
    print("Start Config....................")
    config = {
        "base_model": "lmsys/vicuna-7b-v1.5",
        "wandb_project": "Vicuna7b_elife",
        "random_seed": 2024,
        "huggingface_model": "Anushka1304/vicuna7b_v15_rank4_elife_adalora_4shot",
        "dataset_name": "Anushka1304/elife_with_full_paper",
        "dataset_sizes": {"train": 0, "validation": 0 ,"test": 241}, 
        "full": True,
        "max_length": 16384,
        "num_beams" : 1,
        "method" : "adalora", 
        "rank" : 4,
        "size" : "4shot",
        "result_json":"results_elife_rank4_adalora_4shot.json",
    }

    wandb.init(project=config["wandb_project"])
    print("Start Seeding....................")
    set_seed(config["random_seed"])
    setup_environment(config["wandb_project"])
    wandb.config.num_beams = config["num_beams"]
    wandb.config.method = config["method"]
    wandb.config.size = config["size"]
    wandb.config.rank = config["rank"]
    print("Start Data Preptation....................")
    dataset = load_and_prepare_dataset(config["dataset_name"], config["dataset_sizes"],config["full"])
    if dataset is None:
        sys.exit("Failed to load dataset. Exiting...")

    # TO CHANGE with ADAPTER
    print("Start AdaLoRA Config....................")
    peft_config = AdaLoraConfig(peft_type="ADALORA",r=config["rank"], lora_alpha=2*config["rank"],target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ], lora_dropout=0.1, task_type="CAUSAL_LM")
    model, tokenizer = model_and_tokenizer_setup(config["base_model"], peft_config)

    print("Start Model Parallelization....................")
    print(torch.cuda.device_count())
    if torch.cuda.device_count() > 1:
        print("Training on multiple GPUs...")
        model = model.to("cuda")  # Ensure model is on GPU
        model = torch.nn.DataParallel(model)  # Wrap model with DataParallel

    print("Start Tokenizer....................")
    if model is None or tokenizer is None:
        sys.exit("Failed to set up model. Exiting...")

    print("Starting inference...")
    print("Loading the model for inference...")

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)

    print("Start Prediction Storage....................")
    try:
        model.eval()
        all_predictions = []
        with torch.no_grad():
            i=0
            for sample in dataset["test"]:

                # Extract a random index from the train dataset
                random_index1 = random.randint(0, len(dataset["train"]) - 1)
                random_example1 = dataset["train"][random_index1]
                random_index2 = random.randint(0, len(dataset["train"]) - 1)
                random_example2 = dataset["train"][random_index2]
                random_index3 = random.randint(0, len(dataset["train"]) - 1)
                random_example3 = dataset["train"][random_index3]
                random_index4 = random.randint(0, len(dataset["train"]) - 1)
                random_example4 = dataset["train"][random_index4]
                
                #resume the test dataset
                prediction = generate_answer_4shot(model, tokenizer, random_example1, random_example2, random_example3, random_example4, sample, config["max_length"], config["num_beams"])
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