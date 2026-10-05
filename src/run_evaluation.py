import subprocess
from common import DATASETS, EXPERIMENTS

def run_batch_evaluation():
    print("Starting automated batch evaluation for all trained models...")
    
    for dataset_name in DATASETS.keys():
        for exp_key, exp_info in EXPERIMENTS.items():
            classifier_slug = exp_info["classifier"].lower().replace(" ", "_")
            
            # Reconstruct the expected model filename based on your convention
            model_filename = f"{dataset_name}_{exp_key}_{classifier_slug}.joblib"
            
            print(f"\nEvaluating -> Dataset: {dataset_name} | Experiment: {exp_key} | Model: {model_filename}")
            
            # Construct command for evaluate.py
            cmd = [
                "python", 
                "src/evaluate.py", 
                "--dataset", dataset_name, 
                "--experiment", exp_key, 
                "--model", model_filename
            ]
            
            try:
                result = subprocess.run(cmd, check=True, text=True, capture_output=True)
                print(f"Success: {dataset_name} {exp_key} evaluated and saved.")
            except subprocess.CalledProcessError as e:
                print(f"Error evaluating {dataset_name} {exp_key}:")
                print(e.stderr)

if __name__ == "__main__":
    run_batch_evaluation()