from __future__ import annotations

import logging
import mimetypes
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ARGUS.storage")

try:
    import boto3
    from botocore.client import Config as _BotoConfig
    _BOTO_AVAILABLE = True
except Exception:  # noqa: BLE001 — optional dep, only needed by the server worker
    boto3 = None
    _BotoConfig = None
    _BOTO_AVAILABLE = False


S3_BUCKET = os.getenv("ARGUS_S3_BUCKET", "")
S3_ENDPOINT_URL = os.getenv("ARGUS_S3_ENDPOINT_URL", "")  # e.g. R2 account endpoint; unset for real AWS S3
S3_REGION = os.getenv("ARGUS_S3_REGION", "auto")
S3_ACCESS_KEY_ID = os.getenv("ARGUS_S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = os.getenv("ARGUS_S3_SECRET_ACCESS_KEY", "")
S3_PUBLIC_BASE_URL = os.getenv("ARGUS_S3_PUBLIC_BASE_URL", "").rstrip("/")
S3_PRESIGN_EXPIRES = int(os.getenv("ARGUS_S3_PRESIGN_EXPIRES", "604800"))  # 7 days, R2/S3 max


def is_configured() -> bool:
    return bool(_BOTO_AVAILABLE and S3_BUCKET and S3_ACCESS_KEY_ID and S3_SECRET_ACCESS_KEY)


_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    if not is_configured():
        raise RuntimeError(
            "core.storage is not configured — set ARGUS_S3_BUCKET, "
            "ARGUS_S3_ACCESS_KEY_ID, ARGUS_S3_SECRET_ACCESS_KEY "
            "(and ARGUS_S3_ENDPOINT_URL for R2) and install boto3."
        )
    _client = boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT_URL or None,
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY_ID,
        aws_secret_access_key=S3_SECRET_ACCESS_KEY,
        config=_BotoConfig(signature_version="s3v4"),
    )
    return _client


def upload(local_path: str | Path, key: str) -> str:
    """Upload one file to the configured bucket, return a URL to fetch it.

    Returns a public URL if ARGUS_S3_PUBLIC_BASE_URL is set (public bucket /
    custom domain in front of R2/S3), otherwise a presigned GET URL valid for
    ARGUS_S3_PRESIGN_EXPIRES seconds.
    """
    local_path = Path(local_path)
    client = _get_client()
    content_type = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"

    client.upload_file(
        str(local_path), S3_BUCKET, key,
        ExtraArgs={"ContentType": content_type},
    )

    if S3_PUBLIC_BASE_URL:
        return f"{S3_PUBLIC_BASE_URL}/{key}"

    return client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=S3_PRESIGN_EXPIRES,
    )


def upload_job_artifacts(final_dir: str | Path, run_id: str) -> dict[str, str]:
    """Upload every file in a finished job's out/final/<run_id>/ dir.

    Returns {filename: url}. Called by the worker after run_pipeline()
    succeeds, so job/asset records can store remote URLs instead of the
    worker's local (ephemeral) filesystem paths.
    """
    final_dir = Path(final_dir)
    if not final_dir.is_dir():
        raise FileNotFoundError(f"No such output directory: {final_dir}")

    urls: dict[str, str] = {}
    for path in sorted(final_dir.iterdir()):
        if not path.is_file():
            continue
        key = f"{run_id}/{path.name}"
        try:
            urls[path.name] = upload(path, key)
        except Exception as exc:  # noqa: BLE001 — one bad file shouldn't fail the whole job
            logger.warning("[STORAGE] upload failed for %s: %s", path.name, exc)

    return urls
