"""Assemble the self-contained bundle and deploy it to a Hugging Face Space.

    python deploy.py --assemble-only     # build ./_bundle, deploy nothing
    python deploy.py                      # build + create/update the Space

The bundle vendors the `platform_copilot` package source and the corpus so the Space
is self-contained. Requires a logged-in HF token with write access (`hf auth login`).
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SERVICE_ROOT = HERE.parents[1]  # platform-copilot/
# The base name `Hastika06/platform-copilot` hosts the free STATIC showcase
# (see ../hf-space-static). This interactive Gradio Space needs HF PRO, so it
# deploys under a distinct name to avoid overwriting the live static page.
REPO_ID = os.environ.get("SPACE_REPO_ID", "Hastika06/platform-copilot-live")


def assemble(bundle: Path) -> None:
    if bundle.exists():
        shutil.rmtree(bundle)
    bundle.mkdir(parents=True)
    for name in ("app.py", "requirements.txt", "README.md"):
        shutil.copy2(HERE / name, bundle / name)
    shutil.copytree(
        SERVICE_ROOT / "src" / "platform_copilot",
        bundle / "platform_copilot",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        SERVICE_ROOT / "corpus",
        bundle / "corpus",
        ignore=shutil.ignore_patterns("*.pdf"),
    )
    print(f"assembled bundle: {bundle}")


def main() -> None:
    bundle = HERE / "_bundle"
    assemble(bundle)
    if "--assemble-only" in sys.argv:
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(REPO_ID, repo_type="space", space_sdk="gradio", exist_ok=True, private=False)
    api.upload_folder(
        folder_path=str(bundle),
        repo_id=REPO_ID,
        repo_type="space",
        commit_message="Deploy Platform Copilot demo",
    )
    print(f"deployed: https://huggingface.co/spaces/{REPO_ID}")


if __name__ == "__main__":
    main()
