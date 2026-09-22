"""Check the recognition worker's MinIO credentials and snapshot uploads.

Run on the Pop!_OS server from the ai-recognition directory. This script does
not import recognition.py, so it does not load GPU models or start a worker.
"""

import argparse
import io
import os
import sys
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from dotenv import load_dotenv
from minio import Minio


SMOKE_OBJECT = "healthchecks/ai-recognition-upload-smoke.txt"
SMOKE_CONTENT = b"ai-recognition MinIO upload check\n"


def client_from_env():
    load_dotenv()
    bucket = os.getenv("MINIO_BUCKET", "recognition")
    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key = os.getenv("MINIO_ACCESS_KEY")
    secret_key = os.getenv("MINIO_SECRET_KEY")
    if not endpoint or not access_key or not secret_key:
        raise ValueError("MINIO_ENDPOINT, MINIO_ACCESS_KEY, atau MINIO_SECRET_KEY belum terisi")

    parsed = urlparse(endpoint if "://" in endpoint else f"http://{endpoint}")
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("MINIO_ENDPOINT tidak valid")
    client = Minio(
        parsed.netloc,
        access_key=access_key,
        secret_key=secret_key,
        secure=parsed.scheme == "https",
    )
    return client, bucket


def smoke(client, bucket):
    if not client.bucket_exists(bucket):
        raise RuntimeError(f"Bucket {bucket} tidak ditemukan")

    client.put_object(
        bucket,
        SMOKE_OBJECT,
        io.BytesIO(SMOKE_CONTENT),
        len(SMOKE_CONTENT),
        content_type="text/plain",
    )
    response = client.get_object(bucket, SMOKE_OBJECT)
    try:
        if response.read() != SMOKE_CONTENT:
            raise RuntimeError("Isi objek yang dibaca tidak sama dengan yang diunggah")
    finally:
        response.close()
        response.release_conn()
    print(f"MinIO upload/read OK: {bucket}/{SMOKE_OBJECT}")


def recent_snapshot(client, bucket, minutes):
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    latest = None
    for obj in client.list_objects(bucket, prefix="snapshots/", recursive=True):
        if obj.last_modified and (latest is None or obj.last_modified > latest.last_modified):
            latest = obj
    if latest is None or latest.last_modified < cutoff:
        raise RuntimeError(f"Tidak ada snapshot recognition dalam {minutes} menit terakhir")
    print(f"Snapshot recognition terbaru: {bucket}/{latest.object_name} ({latest.last_modified.isoformat()})")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("smoke", "recent"))
    parser.add_argument("--minutes", type=int, default=10, help="Jendela waktu untuk mode recent")
    args = parser.parse_args()
    if args.minutes < 1:
        parser.error("--minutes harus minimal 1")

    try:
        client, bucket = client_from_env()
        if args.mode == "smoke":
            smoke(client, bucket)
        else:
            recent_snapshot(client, bucket, args.minutes)
    except Exception as exc:
        print(f"MinIO check gagal: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
