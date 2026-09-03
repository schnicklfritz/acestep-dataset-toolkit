"""Push the dataset to Hugging Face (dataset.json + README)."""
import json
import os
import tempfile


def push_dataset(dataset, repo_id, token=None, private=False):
    """Create (if needed) the repo and upload ``dataset.json`` + ``README.md``.

    Returns ``repo_id`` on success. Requires ``huggingface_hub`` and a token
    (from ⚙ Settings → Model Manager or the ``HF_TOKEN`` environment variable).
    """
    from huggingface_hub import create_repo, upload_file

    create_repo(repo_id=repo_id, token=token, private=bool(private), exist_ok=True)

    tmp = tempfile.mkdtemp(prefix="hf_push_")
    json_path = os.path.join(tmp, "dataset.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    readme_path = os.path.join(tmp, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            f"# {repo_id.split('/')[-1]}\n\n"
            "Audio training dataset pushed from the ACE-Step Dataset Toolkit.\n"
            "See `dataset.json` for the manifest.\n"
        )

    upload_file(path_or_fileobj=json_path, path_in_repo="dataset.json",
                repo_id=repo_id, token=token)
    upload_file(path_or_fileobj=readme_path, path_in_repo="README.md",
                repo_id=repo_id, token=token)
    return repo_id