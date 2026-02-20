from huggingface_hub import list_repo_files

try:
    files = list_repo_files("black-forest-labs/FLUX.2-dev-NVFP4")
    print("\n".join(files))
except Exception as e:
    print(f"Error: {e}")
