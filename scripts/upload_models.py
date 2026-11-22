"""
Upload the trained weights (models/) to the Hugging Face Hub.
Weights must NOT be committed to Git (requirement R7) - the app downloads them from here.

    huggingface-cli login            # or notebook_login() in Colab (needs a WRITE token)
    python scripts/upload_models.py --repo <your-hf-username>/chat-moderation-models

Then put the same repo id into MODEL_REPO in config.py and push that change to GitHub.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=config.MODEL_REPO, help="<username>/<repo-name>")
    args = parser.parse_args()
    if args.repo.startswith("YOUR_HF_USERNAME"):
        sys.exit("Pass --repo <your-hf-username>/chat-moderation-models")

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", exist_ok=True, private=False)
    api.upload_folder(folder_path=str(config.MODELS_DIR), repo_id=args.repo, repo_type="model",
                      ignore_patterns=["bert_onnx_fp32/*"])  # intermediate file, the app does not need it
    print(f"Uploaded {config.MODELS_DIR} to https://huggingface.co/{args.repo}")
    print(f'Now set MODEL_REPO = "{args.repo}" in config.py and push to GitHub.')


if __name__ == "__main__":
    main()
