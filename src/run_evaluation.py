import sys
import subprocess
from common import DATASETS, EXPERIMENTS, MODEL_DIR

def run_batch_evaluation():
    print("Starting automated batch evaluation for available trained models...")
    
    for dataset_name in DATASETS.keys():
        for exp_key, exp_info in EXPERIMENTS.items():
            classifier_name = exp_info["classifier"]
            
            if classifier_name == "RandomForest":
                classifier_slug = "random_forest"
            elif classifier_name == "XGBoost":
                classifier_slug = "xgboost"
            else:
                classifier_slug = classifier_name.lower().replace(" ", "_")
            
            model_filename = f"{dataset_name}_{exp_key}_{classifier_slug}.joblib"
            model_path = MODEL_DIR / model_filename
            
            # Check if the model file actually exists before running evaluation
            if not model_path.exists():
                print(f"Skipping -> Dataset: {dataset_name} | Experiment: {exp_key} (Model not found: {model_filename})")
                continue
            
            print(f"\nEvaluating -> Dataset: {dataset_name} | Experiment: {exp_key} | Model: {model_filename}")
            
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