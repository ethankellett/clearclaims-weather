# =============================================================================
#  Auto-email the finished report.
#
#  Primary: Resend (https://resend.com) — simplest API, free tier. Set:
#     RESEND_API_KEY, EMAIL_FROM   (e.g. "Clear Claims Co. <reports@clearclaimsco.co>")
#  Optional: a BCC archive copy via EMAIL_BCC.
#
#  Fallback: plain SMTP (e.g. Google Workspace) if RESEND_API_KEY is not set but
#  EMAIL_SMTP_HOST is. Set EMAIL_SMTP_HOST/PORT/USER/PASS + EMAIL_FROM.
#
#  send_report_email(...) never raises — it returns (ok: bool, detail: str) so a
#  mail hiccup can't break report generation.
# =============================================================================

import os
import base64

RESEND_API_KEY = os.environ.get("RESEND_API_KEY", "").strip()
EMAIL_FROM = os.environ.get("EMAIL_FROM", "").strip()
EMAIL_REPLY_TO = os.environ.get("EMAIL_REPLY_TO", "").strip()
EMAIL_BCC = os.environ.get("EMAIL_BCC", "").strip()
SMTP_HOST = os.environ.get("EMAIL_SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("EMAIL_SMTP_PORT", "587"))
SMTP_USER = os.environ.get("EMAIL_SMTP_USER", "").strip()
SMTP_PASS = os.environ.get("EMAIL_SMTP_PASS", "").strip()


def email_enabled() -> bool:
    return bool(EMAIL_FROM and (RESEND_API_KEY or SMTP_HOST))


def email_verdict(meta: dict):
    """Peril-aware verdict text + color for the email (F12, 2026-08-27).

    The old version hardcoded "Hail Detected"/"No Hail Detected" for every
    peril (wind and snow emails talked about hail) and showed a green
    "No Hail Detected" even when radar coverage was missing entirely.
    """
    peril = (meta.get("peril") or "hail").lower()
    badge = meta.get("badge") or ""
    coverage = meta.get("coverage")
    detected = meta.get("detected")

    if peril == "hail" and coverage == "none":
        return "Radar Coverage Unavailable", "#5a6b7e"
    if badge:
        color = {"Coverage Unavailable": "#5a6b7e"}.get(badge)
        if color is None:
            color = "#d94f3d" if detected else (
                "#e6a117" if badge in ("Trace / Indeterminate", "Hail Indicated")
                else "#28a678")
        return badge, color
    names = {"hail": "Hail", "wind": "Damaging Wind", "snow": "Significant Snow Load"}
    n = names.get(peril, "Weather")
    return (f"{n} Detected", "#d94f3d") if detected else (f"{n} Not Detected", "#28a678")


def build_email_html(meta: dict, report_url: str) -> str:
    verdict, color = email_verdict(meta)
    peril = (meta.get("peril") or "hail").title()
    addr = meta.get("address", "the property")
    dol = meta.get("date_of_loss", "")
    headline = meta.get("headline") or ""
    conf = meta.get("confidence")
    conf_line = f" &nbsp;&middot;&nbsp; Confidence: {conf}" if conf else ""
    return f"""\
<div style="font-family:Arial,Helvetica,sans-serif;max-width:560px;margin:auto;color:#1f2c3d;">
  <div style="background:#06101f;color:#fff;padding:18px 22px;border-radius:8px 8px 0 0;">
    <div style="font-size:20px;font-weight:bold;">Clear <span style="color:#4a9af5;">Claims</span> Co.</div>
    <div style="font-size:12px;color:#8aa0b8;font-style:italic;">Fairness in every claim</div>
  </div>
  <div style="border:1px solid #d9e4f0;border-top:0;border-radius:0 0 8px 8px;padding:22px;">
    <p style="margin:0 0 6px;">Your {peril} Verification Report is ready.</p>
    <p style="font-size:20px;font-weight:bold;color:{color};margin:10px 0 4px;">{verdict}</p>
    <p style="font-size:13px;color:#5a6b7e;margin:0 0 14px;">
      {addr}<br>Date of loss: {dol}<br>{headline}{conf_line}
    </p>
    <a href="{report_url}" style="display:inline-block;background:#2b7de9;color:#fff;
       text-decoration:none;font-weight:bold;padding:11px 20px;border-radius:7px;">View / download the PDF</a>
    <p style="font-size:11px;color:#8a99ab;margin:18px 0 0;line-height:1.5;">
      Derived from official NOAA data, for informational purposes only \u2014
      not a physical inspection or guarantee of damage. The PDF is also attached.
    </p>
  </div>
</div>"""


def send_report_email(to: str, meta: dict, report_url: str,
                      pdf_bytes: bytes | None = None, filename: str = "report.pdf"):
    """Send the report email. Returns (ok, detail). Never raises."""
    if not email_enabled():
        return False, "email not configured"
    if not to:
        return False, "no recipient"

    verdict, _ = email_verdict(meta)
    peril = (meta.get("peril") or "hail").title()
    subject = f"{peril} Report ({verdict}) — " + meta.get("address", "your property")
    html = build_email_html(meta, report_url)

    try:
        if RESEND_API_KEY:
            return _send_resend(to, subject, html, pdf_bytes, filename)
        return _send_smtp(to, subject, html, pdf_bytes, filename)
    except Exception as exc:                       # pragma: no cover
        return False, f"send failed: {exc}"


def _send_resend(to, subject, html, pdf_bytes, filename):
    import requests
    payload = {"from": EMAIL_FROM, "to": [to], "subject": subject, "html": html}
    if EMAIL_BCC:
        payload["bcc"] = [EMAIL_BCC]
    if EMAIL_REPLY_TO:
        payload["reply_to"] = EMAIL_REPLY_TO
    if pdf_bytes:
        payload["attachments"] = [{
            "filename": filename,
            "content": base64.b64encode(pdf_bytes).decode("ascii"),
        }]
    r = requests.post("https://api.resend.com/emails",
                      headers={"Authorization": f"Bearer {RESEND_API_KEY}"},
                      json=payload, timeout=20)
    if r.status_code in (200, 201):
        return True, "sent"
    return False, f"resend {r.status_code}: {r.text[:160]}"


def _send_smtp(to, subject, html, pdf_bytes, filename):
    import smtplib
    from email.message import EmailMessage
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = to
    if EMAIL_BCC:
        msg["Bcc"] = EMAIL_BCC
    if EMAIL_REPLY_TO:
        msg["Reply-To"] = EMAIL_REPLY_TO
    msg["Subject"] = subject
    msg.set_content("Your hail report is ready. Open the link in an HTML-capable client.")
    msg.add_alternative(html, subtype="html")
    if pdf_bytes:
        msg.add_attachment(pdf_bytes, maintype="application", subtype="pdf", filename=filename)
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        if SMTP_USER:
            s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)
    return True, "sent"
