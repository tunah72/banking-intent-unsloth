import os
import json
import pandas as pd
from datasets import load_dataset

def main():
    print("Loading BANKING77 dataset from HuggingFace...")
    dataset = load_dataset("PolyAI/banking77", trust_remote_code=True)
    
    # Convert to Pandas DataFrame for easier manipulation
    df_train = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()
    
    # Extract label names from dataset features
    label_feature = dataset['train'].features['label']
    label_names = label_feature.names
    
    # Save label mapping for inference
    label_mapping = {i: name for i, name in enumerate(label_names)}
    
    os.makedirs("sample_data", exist_ok=True)
    with open("sample_data/label_mapping.json", "w", encoding="utf-8") as f:
        json.dump(label_mapping, f, indent=4)
    print("Saved label mapping to sample_data/label_mapping.json")
    
    # Add human-readable label name column
    df_train['label_name'] = df_train['label'].apply(lambda x: label_names[x])
    df_test['label_name'] = df_test['label'].apply(lambda x: label_names[x])
    
    # ---------------------------------------------------------
    # Stratified Sampling: 50 Train / 5 Val / 10 Test per label
    # ---------------------------------------------------------
    print("Performing Stratified Sampling...")
    n_train_samples = 50
    n_val_samples = 5
    n_test_samples = 10
    
    sampled_train_list = []
    sampled_val_list = []
    
    # Split Train and Validation from the original train set
    for label_id in df_train['label'].unique():
        group = df_train[df_train['label'] == label_id]
        
        # Sample 55 total (50 for train, 5 for val)
        if len(group) < (n_train_samples + n_val_samples):
            samples = group.sample(frac=1, random_state=42)
        else:
            samples = group.sample(n=n_train_samples + n_val_samples, random_state=42)
            
        train_part = samples.iloc[:n_train_samples]
        val_part = samples.iloc[n_train_samples:n_train_samples+n_val_samples]
        
        sampled_train_list.append(train_part)
        sampled_val_list.append(val_part)
        
    sampled_train = pd.concat(sampled_train_list)
    sampled_val = pd.concat(sampled_val_list)
    
    # Sample Test from the original test set
    sampled_test = df_test.groupby('label', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), n_test_samples), random_state=42)
    )
    
    # Shuffle all DataFrames
    sampled_train = sampled_train.sample(frac=1, random_state=42).reset_index(drop=True)
    sampled_val = sampled_val.sample(frac=1, random_state=42).reset_index(drop=True)
    sampled_test = sampled_test.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Export to CSV
    sampled_train.to_csv("sample_data/train.csv", index=False)
    sampled_val.to_csv("sample_data/val.csv", index=False)
    sampled_test.to_csv("sample_data/test.csv", index=False)
    
    print(f"Saved sampled train data ({len(sampled_train)} rows) to sample_data/train.csv")
    print(f"Saved sampled validation data ({len(sampled_val)} rows) to sample_data/val.csv")
    print(f"Saved sampled test data ({len(sampled_test)} rows) to sample_data/test.csv")
    print("Data Preprocessing completed successfully!")

if __name__ == "__main__":
    main()
