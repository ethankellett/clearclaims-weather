# =============================================================================
#  ClearClaims Hail Report — Web API  (FastAPI)
#  Turns {address, date} into the branded PDF, stores it for re-opening/sharing,
#  and protects the endpoint with optional captcha / access-code / rate limit.
#
#  Endpoints:
#    GET  /health                  -> {"status":"ok"}
#    POST /generate                -> JSON {report_id, report_url, detected, ...}
#    GET  /report/{share_id}       -> application/pdf   (the shareable link)
#    GET  /reports                 -> JSON list (admin only; needs X-Admin-Key)
#
#  Environment variables (set in Render). All optional unless noted.
#   API_KEY              caller must send header X-API-Key (server-to-server)
#   ALLOWED_ORIGINS      comma-separated site origins for CORS (default "*")
#   DEFAULT_THRESHOLD_IN detection threshold in inches (default 0.75)
#   FONT_DIR             brand TTF folder (default /app/fonts)
#   RATE_PER_MIN         max requests per IP per minute (default 12)
#   PUBLIC_BASE_URL      this service's public URL, for building share links
#                        e.g. https://clearclaims-hail.onrender.com
#   --- the "gate" (turn on whichever you want) ---
#   ACCESS_CODE          if set, callers must supply a matching access code
#   TURNSTILE_SECRET     if set, a valid Cloudflare Turnstile token is required
#   ADMIN_KEY            protects the /reports history list
#   --- storage (see storage.py) ---
#   REPORTS_DIR  or  STORAGE_S3_BUCKET (+ keys)
# =============================================================================

import os
import time
import uuid
import datetime as dt
from collections import defaultdict, deque

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel

import pipeline
import perils
import storage
import auth
import emailer

# ---- configuration ----------------------------------------------------------
API_KEY = os.environ.get("API_KEY", "").strip()
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
DEFAULT_THRESHOLD_IN = float(os.environ.get("DEFAULT_THRESHOLD_IN", "0.75"))
FONT_DIR = os.environ.get("FONT_DIR", "/app/fonts")
RATE_PER_MIN = int(os.environ.get("RATE_PER_MIN", "12"))
PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "").rstrip("/")
ACCESS_CODE = os.environ.get("ACCESS_CODE", "").strip()
TURNSTILE_SECRET = os.environ.get("TURNSTILE_SECRET", "").strip()
ADMIN_KEY = os.environ.get("ADMIN_KEY", "").strip()

app = FastAPI(title="ClearClaims Hail Report API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---- simple in-memory rate limiter (per client IP, sliding 60s window) -------
_hits: dict[str, deque] = defaultdict(deque)
# dedupe map: same request within the process returns the same stored report
_dedupe: dict[str, str] = {}


def _rate_limit(ip: str):
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_PER_MIN:
        raise HTTPException(status_code=429,
                            detail="Too many requests — please wait a minute and try again.")
    q.append(now)


def _check_key(x_api_key):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


def verify_turnstile(token: str | None, remote_ip: str | None) -> bool:
    """Verify a Cloudflare Turnstile token server-side. Network call to Cloudflare."""
    if not token:
        return False
    import requests
    try:
        r = requests.post(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            data={"secret": TURNSTILE_SECRET, "response": token, "remoteip": remote_ip or ""},
            timeout=10,
        )
        return bool(r.json().get("success"))
    except Exception:
        return False


def _check_gate(req, remote_ip: str | None):
    """Enforce whichever gates are configured (honeypot, access code and/or captcha)."""
    # Honeypot: the "website" field is hidden from real users; only bots fill it.
    # If it arrives with anything in it, silently reject as a bot submission.
    if (getattr(req, "website", "") or "").strip():
        raise HTTPException(status_code=400, detail="Your submission could not be processed.")
    if ACCESS_CODE:
        if (req.access_code or "").strip() != ACCESS_CODE:
            raise HTTPException(status_code=403, detail="Incorrect or missing access code.")
    if TURNSTILE_SECRET:
        if not verify_turnstile(req.turnstile_token, remote_ip):
            raise HTTPException(status_code=403,
                                detail="Human verification failed — please complete the challenge and retry.")


class GenerateRequest(BaseModel):
    peril: str = "hail"                # "hail" | "wind" | "snow"
    address: str | None = None
    date: str                          # "YYYY-MM-DD"
    manual_lat: float | None = None
    manual_lon: float | None = None
    threshold_in: float | None = None
    claim_ref: str = ""
    access_code: str | None = None     # gate (optional)
    turnstile_token: str | None = None  # gate (optional)
    website: str | None = None         # honeypot: must stay empty (bots fill it)
    email_to: str | None = None        # auto-email the report link here (optional)


def _share_url(share_id: str) -> str:
    base = PUBLIC_BASE_URL or ""
    return f"{base}/report/{share_id}"


def _run(req: GenerateRequest, user: str = "") -> dict:
    try:
        date_of_loss = dt.date.fromisoformat(req.date)
    except Exception:
        raise HTTPException(status_code=400, detail="Date must be in YYYY-MM-DD format.")

    peril = (req.peril or "hail").lower()
    if peril not in perils.PERILS:
        raise HTTPException(status_code=400, detail="peril must be hail, wind, or snow.")

    loc_key = (f"{req.manual_lat},{req.manual_lon}"
               if req.manual_lat is not None and req.manual_lon is not None
               else (req.address or "").strip().lower())
    dedupe_key = f"{peril}|{loc_key}|{req.date}|{req.threshold_in}"

    if dedupe_key in _dedupe and storage.get_report_pdf(_dedupe[dedupe_key]) is not None:
        meta = storage.get_report_meta(_dedupe[dedupe_key]) or {}
        return {"share_id": _dedupe[dedupe_key], "meta": meta}

    try:
        result = perils.run_peril(
            peril, address=req.address, date_of_loss=date_of_loss,
            manual_lat=req.manual_lat, manual_lon=req.manual_lon,
            threshold=req.threshold_in, claim_ref=req.claim_ref,
            font_dir=FONT_DIR if os.path.isdir(FONT_DIR) else None)
    except ValueError as friendly:
        raise HTTPException(status_code=400, detail=str(friendly))

    with open(result["pdf_path"], "rb") as f:
        pdf_bytes = f.read()

    share_id = uuid.uuid4().hex
    meta = {
        "peril": result["peril"],
        "report_id": result["report_id"],
        "address": result["location"]["label"],
        "lat": result["location"]["lat"], "lon": result["location"]["lon"],
        "date_of_loss": req.date,
        "detected": result["detected"],
        "headline": result["headline"],
        "threshold": result["threshold"],
        "claim_ref": req.claim_ref or "",
        "user": user,
        "confidence": (result.get("confidence") or {}).get("level"),
        "data_source": result.get("data_source"),
        "n_reports": result.get("n_reports", 0),
        "metrics": result.get("metrics", {}),
    }
    storage.put_report(share_id, pdf_bytes, meta)
    _dedupe[dedupe_key] = share_id
    return {"share_id": share_id, "meta": storage.get_report_meta(share_id) or meta}


@app.get("/health")
def health():
    return {"status": "ok", "storage": "s3" if storage.using_s3() else "local"}


@app.post("/generate")
def generate(req: GenerateRequest, request: Request,
             x_api_key: str | None = Header(default=None),
             authorization: str | None = Header(default=None)):
    _check_key(x_api_key)
    ip = request.client.host if request.client else "unknown"
    _rate_limit(ip)

    # --- gate: Clerk login (if enabled) and/or access code / captcha ---
    user = ""
    if auth.clerk_enabled():
        token = authorization[7:] if (authorization or "").lower().startswith("bearer ") else None
        claims = auth.verify_clerk(token)
        if not claims:
            raise HTTPException(status_code=401, detail="Please sign in to generate a report.")
        user = auth.user_label(claims)
    _check_gate(req, ip)

    out = _run(req, user=user)
    m = out["meta"]

    # --- optional auto-email of the shareable link + PDF ---
    emailed = False
    email_error = None
    report_url = _share_url(out["share_id"])
    if req.email_to and emailer.email_enabled():
        pdf_bytes = storage.get_report_pdf(out["share_id"])
        ok, detail = emailer.send_report_email(
            req.email_to.strip(), m, report_url, pdf_bytes,
            filename=f"ClearClaims_Hail_Report_{m.get('report_id','report')}.pdf")
        emailed = ok
        if not ok:
            email_error = detail

    return {
        "ok": True,
        "peril": m.get("peril"),
        "report_id": m.get("report_id"),
        "share_id": out["share_id"],
        "report_url": report_url,
        "detected": m.get("detected"),
        "headline": m.get("headline"),
        "threshold": m.get("threshold"),
        "address": m.get("address"),
        "confidence": m.get("confidence"),
        "n_reports": m.get("n_reports"),
        "data_source": m.get("data_source"),
        "metrics": m.get("metrics"),
        "emailed": emailed,
        "email_error": email_error,
    }


@app.get("/report/{share_id}")
def get_report(share_id: str):
    pdf = storage.get_report_pdf(share_id)
    if pdf is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return Response(content=pdf, media_type="application/pdf",
                    headers={"Content-Disposition":
                             f'inline; filename="ClearClaims_Hail_Report_{share_id[:8]}.pdf"'})


@app.get("/reports")
def reports(limit: int = 50, x_admin_key: str | None = Header(default=None)):
    if not ADMIN_KEY:
        raise HTTPException(status_code=403, detail="Report history is disabled (no ADMIN_KEY set).")
    if x_admin_key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Invalid admin key.")
    items = storage.list_reports(limit=limit)
    for m in items:
        m["report_url"] = _share_url(m.get("id", ""))
    return {"count": len(items), "reports": items}


@app.exception_handler(HTTPException)
def http_exc_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
