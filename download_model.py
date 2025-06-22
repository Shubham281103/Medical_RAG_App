from huggingface_hub import hf_hub_download
import os

def download_model():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Download the model from TheBloke's repository
    model_path = hf_hub_download(
        repo_id="TheBloke/meditron-7B-GGUF",
        filename="meditron-7b.Q4_K_M.gguf",
        local_dir="models"
    )
    print(f"Model downloaded to: {model_path}")

if __name__ == "__main__":
    download_model() 