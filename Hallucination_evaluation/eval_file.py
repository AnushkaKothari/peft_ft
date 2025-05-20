from AlignScoreCS import AlignScoreCS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from unieval.utils import convert_to_json
from metric_unieval.evaluator import get_evaluator
from summac.model_summac import SummaCZS, SummaCConv
import json
import os
from huggingface_hub import HfApi, hf_hub_download, login


# Function to set the key mapping based on dataset name
def get_key_mapping(dataset):
    if dataset == "elife":
        return {'paper': 'document', 'summary': 'reference', 'Prediction': 'prediction'}
    elif dataset == "scinews":
        return {'Paper_Body': 'document', 'News_Body': 'reference', 'Prediction': 'prediction'}
    else:
        return {}


# Function to read and process JSON file with key mapping
def read_json_file(file_path, key_mapping):
    with open(file_path, 'r') as f:
        data = json.load(f)
    # Map keys based on key_mapping for uniformity
    processed_data = [{key_mapping.get(key, key): value for key, value in entry.items()} for entry in data]
    return processed_data


# Align Score Calculation
def align_score(context, claim):
    alignScoreCS = AlignScoreCS.from_pretrained("krotima1/AlignScoreCS")
    alignscore = alignScoreCS.score(context=context, claim=claim)
    return float(alignscore.item()) if isinstance(alignscore, torch.Tensor) else float(alignscore)


# FactKB Score Calculation
def fact_kb(summary, article):
    input = [[summary, article]]
    tokenizer = AutoTokenizer.from_pretrained("roberta-base", padding="max_length", truncation=True)
    model = AutoModelForSequenceClassification.from_pretrained("bunsenfeng/FactKB", num_labels=2)
    tokens = tokenizer(input, return_tensors="pt", padding="max_length", truncation=True)
    result = torch.softmax(model(**tokens).logits, dim=1)
    return float(result[0][1].item()) if isinstance(result[0][1], torch.Tensor) else float(result[0][1])


# UniEval Score Calculation
def uni_eval(output_list, src_list, ref_list):
    data = convert_to_json(output_list=output_list, src_list=src_list, ref_list=ref_list)
    evaluator = get_evaluator("summarization")
    eval_scores = evaluator.evaluate(data, print_result=True)
    return eval_scores[0]  # Return first dictionary as per output structure


# SummaC Score Calculation
def summac_score(document, summary, predicted_summary):
    model_zs = SummaCZS(granularity="sentence", model_name="vitc", device="cpu")
    model_conv = SummaCConv(models=["vitc"], bins="percentile", granularity="sentence", nli_labels="e", device="cpu", start_file="default", agg="mean")

    score_zs = float(model_zs.score([document], [summary])["scores"][0])
    score_conv = float(model_conv.score([document], [summary])["scores"][0])
    score_zs_pred = float(model_zs.score([document], [predicted_summary])["scores"][0])
    score_conv_pred = float(model_conv.score([document], [predicted_summary])["scores"][0])
    
    return score_zs, score_conv, score_zs_pred, score_conv_pred


# Function to process scores for each entry
def get_scores(data):
    align_scores, fact_kb_scores = [], []
    uni_eval_scores = {'coherence': [], 'consistency': [], 'fluency': [], 'relevance': [], 'overall': []}
    summac_scores = {"zs_summary": [], "conv_summary": [], "zs_pred": [], "conv_pred": []}
    counter = 0
    for entry in data:
        document = entry["document"]
        summary = entry["reference"]
        predicted_summary = entry["prediction"]

        # 1. Align Score
        alignscore = align_score(context=document, claim=predicted_summary)
        align_scores.append(alignscore)
        entry["align_score"] = alignscore

        # 2. FactKB Score
        fact_score = fact_kb(summary=predicted_summary, article=document)
        fact_kb_scores.append(fact_score)
        entry["fact_kb_score"] = fact_score

        # 3. SummaC Score
        score_zs, score_conv, score_zs_pred, score_conv_pred = summac_score(document=document, summary=summary, predicted_summary=predicted_summary)
        summac_scores["zs_summary"].append(score_zs)
        summac_scores["conv_summary"].append(score_conv)
        summac_scores["zs_pred"].append(score_zs_pred)
        summac_scores["conv_pred"].append(score_conv_pred)
        entry["summac_zs_summary_score"] = score_zs
        entry["summac_conv_summary_score"] = score_conv
        entry["summac_zs_pred_score"] = score_zs_pred
        entry["summac_conv_pred_score"] = score_conv_pred

        # 4. UniEval Scores
        eval_scores = uni_eval(output_list=[predicted_summary], src_list=[document], ref_list=[summary])
        for key, score in eval_scores.items():
            uni_eval_scores[key].append(score)
            entry[key] = score
        counter += 1
        print("THE SAMPLES {} DONE OUT OF THE TOTAL 241".format(counter))

    return align_scores, fact_kb_scores, summac_scores, uni_eval_scores


# Function to calculate and add corpus averages to data
def compute_average(data, align_scores, fact_kb_scores, summac_scores, uni_eval_scores):
    average_scores = {
        "align_score_avg": sum(align_scores) / len(align_scores),
        "fact_kb_score_avg": sum(fact_kb_scores) / len(fact_kb_scores),
        "summac_zs_summary_avg": sum(summac_scores["zs_summary"]) / len(summac_scores["zs_summary"]),
        "summac_conv_summary_avg": sum(summac_scores["conv_summary"]) / len(summac_scores["conv_summary"]),
        "summac_zs_pred_avg": sum(summac_scores["zs_pred"]) / len(summac_scores["zs_pred"]),
        "summac_conv_pred_avg": sum(summac_scores["conv_pred"]) / len(summac_scores["conv_pred"]),
    }

    for key in uni_eval_scores:
        average_scores[key + "_avg"] = sum(uni_eval_scores[key]) / len(uni_eval_scores[key])

    data.append({"average_corpus_scores": average_scores})
    return data


# Save data to JSON file
def write_to_json(data, path):
    with open(path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"Scoring complete. Scores saved to '{path}'")


# Function to process and update a specific file
def process_specific_file(repo_name, file_name):
    api = HfApi()

    # Check if `[hallucination_score]` version of the file already exists
    scored_file_name = file_name.replace(".json", "_[hallucination_score].json")
    repo_files = api.list_repo_files(repo_id=repo_name, repo_type="dataset")
    if scored_file_name in repo_files:
        print(f"File '{scored_file_name}' already exists. Skipping processing.")
        return

    # Download the file
    downloaded_file_path = hf_hub_download(repo_id=repo_name, filename=file_name, repo_type="dataset")
    dataset_name = file_name.split("/")[0]
    key_mapping = get_key_mapping(dataset_name)

    # Read and process the data
    data = read_json_file(downloaded_file_path, key_mapping)
    align_scores, fact_kb_scores, summac_scores, uni_eval_scores = get_scores(data)
    processed_data = compute_average(data, align_scores, fact_kb_scores, summac_scores, uni_eval_scores)

    # Save processed data locally
    scored_file_path = downloaded_file_path.replace(".json", "_scored.json")
    write_to_json(processed_data, scored_file_path)

    # Upload the processed file back to Hugging Face repository
    api.upload_file(
        path_or_fileobj=scored_file_path,
        path_in_repo=scored_file_name,
        repo_id=repo_name,
        repo_type="dataset"
    )
    print(f"Uploaded processed file as '{scored_file_name}'")

    # Clean up temporary files
    os.remove(downloaded_file_path)
    os.remove(scored_file_path)


def main():
    # Initialize Hugging Face API
    hugging_face_token = '*******************'
    login(token=hugging_face_token)

    repo_name = 'summarization_results'
    file_name = 'elife/adalora/rank4/adalora_elife_rank4_full.json'  # Specify the file name from Hugging Face repository
    # First part of the filename is the dataset name, second part is the peft technique, and third part is the rank
    # Process and update the specific file
    process_specific_file(repo_name, file_name)

if __name__ == '__main__':
    main()
