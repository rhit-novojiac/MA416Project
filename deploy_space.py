"""
Deploys the local hf_space demo folder to a Hugging Face Space.
Usage:
    python deploy_space.py --repo-id <username>/<space_name> [--token <hf_token>]
"""

import argparse
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi, get_token


def deploy(repo_id: str, token: str = None):
    auth_token = token or os.environ.get("HF_TOKEN") or get_token()
    if not auth_token:
        print("Error: No Hugging Face token found.")
        print("Please provide --token <your_token> or set the HF_TOKEN environment variable.")
        sys.exit(1)

    api = HfApi(token=auth_token)

    # 1. Verify user identity
    try:
        user_info = api.whoami()
        username = user_info["name"]
        print(f"Authenticated as Hugging Face user: {username}")
    except Exception as e:
        print(f"Authentication failed: {e}")
        sys.exit(1)

    # If repo_id was provided without a slash, prepend username
    if "/" not in repo_id:
        repo_id = f"{username}/{repo_id}"

    # 2. Create the Space repo if it doesn't already exist
    print(f"Ensuring Space repository exists: {repo_id}...")
    try:
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
            private=False
        )
        print(f"Space repository confirmed: {repo_url}")
    except Exception as e:
        print(f"Notice during repository creation: {e}")

    # 3. Upload the hf_space folder
    folder_path = Path(__file__).resolve().parent / "hf_space"
    if not folder_path.exists():
        folder_path = Path.cwd() / "hf_space"

    print(f"Uploading files from {folder_path} to Space {repo_id}...")
    try:
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="space",
            commit_message="Deploy DeFungi classification Gradio demo"
        )
        print("\n" + "=" * 60)
        print("SUCCESS! Your Space has been deployed.")
        print(f"Space URL: https://huggingface.co/spaces/{repo_id}")
        print("=" * 60)
    except Exception as e:
        print(f"Error uploading files to Space: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy DeFungi Gradio demo to Hugging Face Spaces.")
    parser.add_argument("--repo-id", required=True, help="Space repository ID, e.g. 'username/defungi-classifier' or 'defungi-classifier'")
    parser.add_argument("--token", default=None, help="Hugging Face Access Token (with Write permission)")
    args = parser.parse_args()

    deploy(args.repo_id, args.token)

