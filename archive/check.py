from huggingface_hub import HfApi
api = HfApi()
for f in ["requirements.txt", "predict.py"]:
    api.upload_file(
        path_or_fileobj=f,
        path_in_repo=f,
        repo_id="Pocket-Vakil/pocket-vakil-332m-ipc",
        repo_type="model",
    )