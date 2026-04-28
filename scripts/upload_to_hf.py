import os
import sys
from huggingface_hub import HfApi, create_repo

def main():
    # Configuration
    repo_id = "tunah/banking-intent"
    
    # Choose the directory containing the model you want to upload
    # You can change this to "checkpoints/checkpoint-1200" or any other checkpoint
    folder_path = "checkpoints/final_best_model" 
    
    if not os.path.exists(folder_path):
        print(f"Error: Could not find directory '{folder_path}'. Please check the path.")
        sys.exit(1)

    print(f"Preparing to upload model from directory: {folder_path}")
    print(f"Destination on Hugging Face: https://huggingface.co/{repo_id}")
    
    api = HfApi()

    # 1. Create repository if it doesn't exist
    try:
        print("Checking / creating repository...")
        create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    except Exception as e:
        print(f"Could not create repo. Error: {e}")
        sys.exit(1)

    # 2. Upload the folder
    try:
        print("Uploading data to Hugging Face Hub (this may take a while depending on network speed)...")
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload best trained model"
        )
        print("\nUpload successful!")
        print(f"You can view your model at: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"\nError during upload: {e}")

if __name__ == "__main__":
    main()
