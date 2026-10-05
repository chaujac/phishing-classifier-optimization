import sys
import subprocess
from common import DATASETS, EXPERIMENTS

def run_batch_evaluation():
    print("Starting automated batch evaluation for all trained models...")
    
    for dataset_name in DATASETS.keys():
        for exp_key, exp_info in EXPERIMENTS.items():
            classifier_name = exp_info["classifier"]
            
            # Match your exact .joblib filename convention (e.g., random_forest or xgboost)
            if classifier_name == "RandomForest":
                classifier_slug = "random_forest"
            elif classifier_name == "XGBoost":
                classifier_slug = "xgboost"
            else:
                classifier_slug = classifier_name.lower().replace(" ", "_")
            
            model_filename = f"{dataset_name}_{exp_key}_{classifier_slug}.joblib"
            
            print(f"\nEvaluating -> Dataset: {dataset_name} | Experiment: {exp_key} | Model: {model_filename}")
            
            # Use sys.executable to guarantee the active virtual environment is used
            cmd = [
                sys.executable, 
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