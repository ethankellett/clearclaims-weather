# =============================================================================
#  Report storage — saves each generated PDF under an unguessable ID and keeps a
#  little metadata record, so reports can be re-opened and shared by link, and
#  listed in an admin history view.
#
#  Two backends, chosen automatically by environment variables:
#    * Local disk (default)  — set REPORTS_DIR (pair with a Render Disk so it
#                              survives restarts/redeploys).
#    * S3-compatible bucket  — set STORAGE_S3_BUCKET (+ keys). Works with AWS S3,
#                              Cloudflare R2, Backblaze B2, etc.
# =============================================================================

import os
import json
import time
import glob

LOCAL_DIR = os.environ.get("REPORTS_DIR", "/tmp/cc_reports")
S3_BUCKET = os.environ.get("STORAGE_S3_BUCKET", "").strip()
S3_ENDPOINT = os.environ.get("STORAGE_S3_ENDPOINT", "").strip() or None  # R2/B2 need this
S3_REGION = os.environ.get("STORAGE_S3_REGION", "auto")


def _s3():
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id=os.environ["STORAGE_S3_KEY"],
        aws_secret_access_key=os.environ["STORAGE_S3_SECRET"],
        region_name=S3_REGION,
    )


def using_s3() -> bool:
    return bool(S3_BUCKET)


def put_report(report_id: str, pdf_bytes: bytes, meta: dict) -> dict:
    """Persist the PDF + metadata. Returns the stored metadata record."""
    record = {**meta, "id": report_id, "created": time.time()}
    if S3_BUCKET:
        c = _s3()
        c.put_object(Bucket=S3_BUCKET, Key=f"reports/{report_id}.pdf",
                     Body=pdf_bytes, ContentType="application/pdf")
        c.put_object(Bucket=S3_BUCKET, Key=f"reports/{report_id}.json",
                     Body=json.dumps(record).encode(), ContentType="application/json")
    else:
        os.makedirs(LOCAL_DIR, exist_ok=True)
        with open(os.path.join(LOCAL_DIR, f"{report_id}.pdf"), "wb") as f:
            f.write(pdf_bytes)
        with open(os.path.join(LOCAL_DIR, f"{report_id}.json"), "w") as f:
            json.dump(record, f)
    return record


def get_report_pdf(report_id: str):
    """Return the PDF bytes for a report id, or None if it doesn't exist."""
    if not report_id or "/" in report_id or "\\" in report_id:
        return None
    if S3_BUCKET:
        c = _s3()
        try:
            obj = c.get_object(Bucket=S3_BUCKET, Key=f"reports/{report_id}.pdf")
            return obj["Body"].read()
        except Exception:
            return None
    path = os.path.join(LOCAL_DIR, f"{report_id}.pdf")
    return open(path, "rb").read() if os.path.isfile(path) else None


def get_report_meta(report_id: str):
    if S3_BUCKET:
        c = _s3()
        try:
            obj = c.get_object(Bucket=S3_BUCKET, Key=f"reports/{report_id}.json")
            return json.loads(obj["Body"].read())
        except Exception:
            return None
    path = os.path.join(LOCAL_DIR, f"{report_id}.json")
    return json.load(open(path)) if os.path.isfile(path) else None


def list_reports(limit: int = 50) -> list:
    """Most-recent-first metadata records (for the admin history page)."""
    metas = []
    if S3_BUCKET:
        c = _s3()
        resp = c.list_objects_v2(Bucket=S3_BUCKET, Prefix="reports/")
        for o in resp.get("Contents", []):
            if o["Key"].endswith(".json"):
                try:
                    metas.append(json.loads(c.get_object(
                        Bucket=S3_BUCKET, Key=o["Key"])["Body"].read()))
                except Exception:
                    pass
    else:
        for p in glob.glob(os.path.join(LOCAL_DIR, "*.json")):
            try:
                metas.append(json.load(open(p)))
            except Exception:
                pass
    metas.sort(key=lambda m: m.get("created", 0), reverse=True)
    return metas[:limit]
