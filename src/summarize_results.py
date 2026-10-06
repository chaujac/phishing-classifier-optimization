import json
from pathlib import Path
from common import RESULT_DIR, DATASETS, EXPERIMENTS

def summarize_evaluations():
    eval_root = RESULT_DIR / "evaluation" / "evaluation-result-1"
    
    print(f"{'Dataset':<10} | {'Exp':<5} | {'Accuracy':<10} | {'Macro F1':<10} | {'Mean Latency (ms)':<18}")
    print("-" * 65)
    
    folder_name_map = {
        "uci": "uci-evaluation-result",
        "web_page": "webpage-evaluation-result",
        "phiusil": "phiusil-evaluation-result",
        "zenodo": "zenodo-evaluation-result",
    }
    
    for dataset_name in DATASETS.keys():
        folder_name = folder_name_map.get(dataset_name, f"{dataset_name}-evaluation-result")
        dataset_eval_dir = eval_root / folder_name
        
        if not dataset_eval_dir.exists():
            continue
            
        for exp_key in EXPERIMENTS.keys():
            file_path = dataset_eval_dir / f"{dataset_name}_{exp_key}_evaluation.json"
            
            if not file_path.exists():
                continue
                
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            acc = data["classification"]["accuracy"]
            f1 = data["classification"]["macro_f1"]
            latency = data["inference"]["mean_ms"]
            
            print(f"{dataset_name:<10} | {exp_key:<5} | {acc:<10.4f} | {f1:<10.4f} | {latency:<18.4f}")

if __name__ == "__main__":
    summarize_evaluations()