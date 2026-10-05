import pandas as pd

def verify_dataset(name, df, expected_rows):
    print(f"=== {name} Verification ===")
    print(f"Loaded Shape: {df.shape} (Rows: {df.shape[0]}, Columns: {df.shape[1]})")
    
    # Check dimensions
    row_check = "MATCHES" if df.shape[0] == expected_rows else f"MISMATCH (Expected ~{expected_rows})"
    print(f"Row Count Status: {row_check}")
    
    # Print class distribution preview
    print("Class Distribution / Target Summary:")
    target_candidates = [col for col in ['Result', 'phishing', 'label', 'class', 'status', 'Type'] if col in df.columns]
    
    if target_candidates:
        target_col = target_candidates[0]
        print(df[target_col].value_counts(dropna=False))
    else:
        print("Target column name needs manual inspection.")
    print("-" * 50)

if __name__ == "__main__":
    # 1. UCI Phishing Websites Dataset (Expected: ~11,055 rows)
    try:
        df_uci = pd.read_csv('data/raw/uci-ml-phishing-dataset.csv')
        verify_dataset("UCI Phishing Websites", df_uci, 11055)
    except Exception as e:
        print(f"UCI Check Failed: {e}")

    # 2. Web Page Phishing Detection Dataset (Expected: ~11,430 rows)
    try:
        df_web = pd.read_csv('data/raw/mendeley-web-page-phishing-detection-dataset.csv')
        verify_dataset("Web Page Phishing", df_web, 11430)
    except Exception as e:
        print(f"Web Page Check Failed: {e}")

    # 3. PhiUSIIL Phishing URL Dataset (Expected: ~235,795 rows)
    try:
        df_phi = pd.read_csv('data/raw/phiusiil-phishing-url-dataset.csv')
        verify_dataset("PhiUSIIL Dataset", df_phi, 235795)
    except Exception as e:
        print(f"PhiUSIIL Check Failed: {e}")

    # 4. Zenodo Phishing Dataset (Expected total combined: ~10,395 rows)
    try:
        df_zen1 = pd.read_csv('data/raw/zenodo-phishing.csv')
        df_zen2 = pd.read_csv('data/raw/zenodo-not-phishing.csv')
        
        # Zenodo doesn't have native labels, so we assign them before concatenating
        if 'label' not in df_zen1.columns:
            df_zen1['label'] = 1  # Phishing
        if 'label' not in df_zen2.columns:
            df_zen2['label'] = 0  # Legitimate
            
        df_zen = pd.concat([df_zen1, df_zen2], ignore_index=True)
        verify_dataset("Zenodo Dataset (Combined)", df_zen, 10395)
    except Exception as e:
        print(f"Zenodo Check Failed: {e}")