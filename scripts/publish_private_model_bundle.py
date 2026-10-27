"""Publish exactly SIX owner-trained RF joblib files to a PRIVATE HF model repo.

Owner-initiated local command only. NEVER run in CI, browser or Render;
never commit pickle artifacts, data, webcam media or Hugging Face tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from gym_pose.model_registry import KNOWN_TASKS, ModelRegistry
from gym_pose.private_model_bootstrap import EXPECTED, MANIFEST_NAME, make_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="HF owner/private-repository")
    parser.add_argument("--model-root", default=".", help="Local project root")
    parser.add_argument(
        "--trust-local-artifacts", action="store_true",
        help="Owner assertion: six LOCAL joblib files were created by this project",
    )
    args = parser.parse_args(argv)
    if not args.trust_local_artifacts:
        parser.error("Require --trust-local-artifacts; joblib can execute code")
    root = Path(args.model_root).resolve()
    manifest = make_manifest(root)

    # Must pass the full six source-contract checks BEFORE publishing. This
    # can execute local joblib (pickle): only run on your own trained files.
    from gym_pose.model_registry import DEFAULT_PATHS
    registry = ModelRegistry(paths={
        key: root / DEFAULT_PATHS[key] for key in KNOWN_TASKS
    })
    statuses = registry.load_local(trust_local_pickle_files=True)
    if set(statuses.values()) != {"loaded_experimental_only"} or len(statuses) != 6:
        raise RuntimeError("Do not publish: expected six trusted original-source models")

    from huggingface_hub import HfApi, get_token
    token = get_token()
    if not token:
        parser.error("Authenticate locally first: hf auth login (do not paste token into chat)")
    api = HfApi(token=token)
    # A preexisting public repo is NOT converted and is rejected below.
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=True, exist_ok=True)
    repo_info = api.repo_info(repo_id=args.repo_id, repo_type="model")
    if repo_info.private is not True:
        raise RuntimeError("Refusing to upload models: destination repository is NOT PRIVATE")

    with tempfile.TemporaryDirectory(prefix="gym-pose-private-publish-") as tmp:
        temp_root = Path(tmp)
        for relative in EXPECTED:
            output = temp_root / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / relative, output)
        manifest_bytes = (
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        (temp_root / MANIFEST_NAME).write_bytes(manifest_bytes)
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        response = api.upload_folder(
            folder_path=str(temp_root),
            repo_id=args.repo_id,
            repo_type="model",
            commit_message="Publish six trusted Gym Pose source-label RF models",
        )

    print("Private model repo:", args.repo_id)
    print("POSE_HF_REPO_ID=" + args.repo_id)
    print("POSE_HF_REVISION=" + response.oid)
    print("POSE_HF_MANIFEST_SHA256=" + manifest_digest)
    print("The Hugging Face read-only access token belongs in Render Secrets as HF_TOKEN.")
    print("Never copy the private token or model binary into GitHub or Vercel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
