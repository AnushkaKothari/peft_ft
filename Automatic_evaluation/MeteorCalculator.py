import json
import os
import numpy as np
import evaluate
import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('omw-1.4', quiet=True)  # Open Multilingual Wordnet, often used with 'wordnet'

class MeteorCalculator:
    def __init__(self):
        self.meteor = evaluate.load('meteor')

    def compute_score(self, input_data):
        if not input_data:
            raise ValueError("Input data cannot be empty.")

        results = []
        for pair in input_data:
            try:
                reference = pair['reference'].strip()
                prediction = pair['prediction'].strip()
            except KeyError:
                raise ValueError("Each dictionary in input data should contain 'reference' and 'prediction' keys.")
                
            scores = self.meteor.compute(predictions=[prediction], references=[reference])

            # Round the results to 4 decimal places
            scores = {k: round(v , 4) for k, v in scores.items()}
            results.append(scores)

        average_scores = {k: round(np.mean([res[k] for res in results]), 4) for k in results[0]}
        results.append({'corpus_level': average_scores})
        return results

    def compute_from_file(self, input_json_path):
        if not os.path.isfile(input_json_path):
            raise FileNotFoundError(f"{input_json_path} does not exist.")

        with open(input_json_path, 'r') as f:
            data = json.load(f)
        results = self.compute_score(data)

        return results

    def compute_from_file_json(self, input_json_path, output_json_path):
        if not os.path.isfile(input_json_path):
            raise FileNotFoundError(f"{input_json_path} does not exist.")

        with open(input_json_path, 'r') as f:
            data = json.load(f)
        results = self.compute_score(data)

        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=4)
            
if __name__ == '__main__':

    meteor_calculator = MeteorCalculator()

    input_data = [
        {"reference": "The cat sat on the mat. It was happy.", "prediction": "A cat was sitting on the mat. The cat seemed pleased."},
        {"reference": "The quick brown fox jumps over the lazy dog. The dog did not react.", "prediction": "A fast brown fox jumps above the lazy dog. The dog seemed indifferent."}
    ]

    input_json_path = 'input.json'
    with open(input_json_path, 'w') as f:
        json.dump(input_data, f)

    output_json_path = 'output.json'
    meteor_calculator.compute_from_file(input_json_path, output_json_path)
