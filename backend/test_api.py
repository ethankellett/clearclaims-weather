"""
Local smoke test for the FastAPI service (v2): JSON response + shareable link,
storage, abuse gate (access code + captcha), and the admin history list.
Uses the REAL app + pipeline but a synthetic GRIB2 instead of live S3.
"""
import os, sys, tempfile
import numpy as np
import eccodes as ec
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault("REPORTS_DIR", tempfile.mkdtemp())  # isolate storage
import pipeline
import storage
import app as appmod

PASS, FAIL = [], []
def check(n, c, extra=""):
    (PASS if c else FAIL).append(n)
    print(("  PASS " if c else "  FAIL ") + n + (f"  {extra}" if extra else ""))

# --- synthetic MRMS-format GRIB2 with a giant-hail blob ---------------------
NLAT, NLON = 120, 160
LAT_TOP, LAT_BOT, LON_L, LON_R = 35.10, 33.90, -101.80, -100.20
lats = np.linspace(LAT_TOP, LAT_BOT, NLAT); lons = np.linspace(LON_L, LON_R, NLON)
LON, LAT = np.meshgrid(lons, lats)
CY, CX = 34.50, -101.02
hail = np.clip(98.0 * np.exp(-(((LAT-CY)/0.10)**2 + ((LON-CX)/0.10)**2)), 0, None)
GRIB = os.path.join(tempfile.mkdtemp(), "synthetic.grib2")
gid = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
ec.codes_set(gid, "Ni", NLON); ec.codes_set(gid, "Nj", NLAT)
ec.codes_set(gid, "latitudeOfFirstGridPointInDegrees", float(LAT_TOP))
ec.codes_set(gid, "longitudeOfFirstGridPointInDegrees", float(LON_L % 360))
ec.codes_set(gid, "latitudeOfLastGridPointInDegrees", float(LAT_BOT))
ec.codes_set(gid, "longitudeOfLastGridPointInDegrees", float(LON_R % 360))
ec.codes_set(gid, "iDirectionIncrementInDegrees", float((LON_R-LON_L)/(NLON-1)))
ec.codes_set(gid, "jDirectionIncrementInDegrees", float((LAT_TOP-LAT_BOT)/(NLAT-1)))
ec.codes_set(gid, "jScansPositively", 0)
ec.codes_set_values(gid, hail.flatten())
with open(GRIB, "wb") as fh: ec.codes_write(gid, fh)
ec.codes_release(gid)

pipeline._fetch_grib_paths = lambda u0, u1, td, dol=None, max_files=5: ([GRIB], [], "TEST source")
# Inject a sample ground report so corroboration/confidence render deterministically.
pipeline.hc.fetch_storm_reports = lambda *a, **k: [
    {"source": "NWS LSR", "size_in": 1.75, "lat": 34.51, "lon": -101.0,
     "dist_mi": 2.1, "dir": "NE", "time": "2024-06-02T21:05Z", "city": "Tulia"}]
appmod.RATE_PER_MIN = 100000   # don't trip the rate limiter during the test run
client = TestClient(appmod.app)
PAY = {"manual_lat": CY, "manual_lon": CX, "date": "2024-06-02", "claim_ref": "T1"}

print("\n[1] /health")
r = client.get("/health")
check("health ok", r.status_code == 200 and r.json()["status"] == "ok")

print("\n[2] POST /generate -> JSON + share link")
r = client.post("/generate", json=PAY)
j = r.json()
check("200 + ok", r.status_code == 200 and j.get("ok") is True, str(r.status_code))
check("detected True", j.get("detected") is True)
check("has report_url", "/report/" in (j.get("report_url") or ""), j.get("report_url"))
check("at-property realistic", (j.get("metrics") or {}).get("at_property_in", 0) > 1.0, str(j.get("metrics")))
check("confidence High (radar + ground report)", j.get("confidence") == "High", str(j.get("confidence")))
check("ground report counted", j.get("n_reports") == 1, str(j.get("n_reports")))
share_id = j.get("share_id")

print("\n[3] GET /report/{id} -> PDF")
r = client.get(f"/report/{share_id}")
check("PDF served", r.status_code == 200 and r.content[:4] == b"%PDF", f"{len(r.content)} bytes")
r = client.get("/report/doesnotexist")
check("404 for unknown id", r.status_code == 404)

print("\n[4] dedupe returns same share id")
r2 = client.post("/generate", json=PAY)
check("same share_id on repeat", r2.json().get("share_id") == share_id)

print("\n[5] input validation")
check("bad date 400", client.post("/generate", json={**PAY, "date": "xx"}).status_code == 400)
check("no location 400", client.post("/generate", json={"date": "2024-06-02"}).status_code == 400)
check("pre-archive 400", client.post("/generate", json={**PAY, "date": "2010-01-01"}).status_code == 400)

print("\n[6] gate: access code")
appmod.ACCESS_CODE = "code123"
check("missing code 403", client.post("/generate", json=PAY).status_code == 403)
check("wrong code 403", client.post("/generate", json={**PAY, "access_code": "nope"}).status_code == 403)
check("correct code 200", client.post("/generate", json={**PAY, "access_code": "code123"}).status_code == 200)
appmod.ACCESS_CODE = ""

print("\n[7] gate: captcha (turnstile)")
appmod.TURNSTILE_SECRET = "secret"
appmod.verify_turnstile = lambda tok, ip: tok == "good"
check("bad token 403", client.post("/generate", json={**PAY, "turnstile_token": "bad"}).status_code == 403)
check("good token 200", client.post("/generate", json={**PAY, "turnstile_token": "good"}).status_code == 200)
appmod.TURNSTILE_SECRET = ""

print("\n[8] admin history")
check("disabled without ADMIN_KEY 403", client.get("/reports").status_code == 403)
appmod.ADMIN_KEY = "admin1"
check("wrong admin key 401", client.get("/reports", headers={"X-Admin-Key": "x"}).status_code == 401)
r = client.get("/reports", headers={"X-Admin-Key": "admin1"})
check("admin list ok + has entries", r.status_code == 200 and r.json()["count"] >= 1, str(r.json().get("count")))

print("\n[9] auto-email (Resend mocked)")
import emailer
sent = {}
emailer.email_enabled = lambda: True
def _fake_send(to, meta, url, pdf, filename="r.pdf"):
    sent["to"] = to; sent["url"] = url; sent["has_pdf"] = bool(pdf)
    return True, "sent"
emailer.send_report_email = _fake_send
r = client.post("/generate", json={**PAY, "date": "2024-06-02", "email_to": "client@example.com"})
check("emailed flag true", r.json().get("emailed") is True)
check("email went to recipient w/ PDF", sent.get("to") == "client@example.com" and sent.get("has_pdf"))
# failure path shouldn't break report generation
emailer.send_report_email = lambda *a, **k: (False, "resend 403")
r = client.post("/generate", json={"manual_lat": CY, "manual_lon": CX,
                                    "date": "2024-06-03", "email_to": "x@y.com"})
check("report still ok when email fails", r.status_code == 200 and r.json().get("emailed") is False)

print("\n[10] Clerk login gate (verification mocked)")
import auth
auth.clerk_enabled = lambda: True
appmod.auth.verify_clerk = lambda tok: {"email": "user@firm.com"} if tok == "goodjwt" else None
check("no token 401", client.post("/generate", json={**PAY, "date": "2024-06-02"}).status_code == 401)
check("bad token 401", client.post("/generate", json={**PAY, "date": "2024-06-02"},
      headers={"Authorization": "Bearer bad"}).status_code == 401)
r = client.post("/generate", json={**PAY, "date": "2024-06-02"},
                headers={"Authorization": "Bearer goodjwt"})
check("valid token 200", r.status_code == 200 and r.json().get("ok") is True)
auth.clerk_enabled = lambda: False

print("\n[11] storm-report parsers (real schemas)")
import hail_core as hc
# IEM LSR GeoJSON sample (hail report ~ at the property)
iem = {"features": [
    {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-101.02, 34.51]},
     "properties": {"type": "H", "typetext": "HAIL", "magnitude": 2.75,
                    "valid": "2024-06-02T21:05Z", "city": "Quitaque", "source": "TRAINED SPOTTER"}},
    {"type": "Feature", "geometry": {"type": "Point", "coordinates": [-95.0, 40.0]},
     "properties": {"type": "H", "magnitude": 1.0}},  # far away, filtered out
]}
iem_reports = hc.parse_iem_lsr_geojson(iem, 34.50, -101.02, radius_miles=12)
check("IEM parser keeps nearby hail only", len(iem_reports) == 1 and abs(iem_reports[0]["size_in"] - 2.75) < 1e-6,
      str(iem_reports))
# SPC CSV sample
spc_csv = ("Time,Size,Location,County,State,Lat,Lon,Comments\n"
           "2105,275,2 NE QUITAQUE,BRISCOE,TX,34.51,-101.02,trained spotter\n"
           "2110,100,FARAWAY,OTHER,KS,39.0,-98.0,nope\n")
spc_reports = hc.parse_spc_hail_csv(spc_csv, 34.50, -101.02, radius_miles=12)
check("SPC parser size hundredths->inches", len(spc_reports) == 1 and abs(spc_reports[0]["size_in"] - 2.75) < 1e-6,
      str(spc_reports))
# confidence logic
c_hi = hc.assess_confidence(3.8, 3.9, iem_reports, 0.75)
check("confidence High when corroborated", c_hi["level"] == "High")
c_mod = hc.assess_confidence(1.5, 1.6, [], 0.75)
check("confidence Moderate radar-only above thresh", c_mod["level"] == "Moderate")
c_clear = hc.assess_confidence(0.1, 0.1, [], 0.75)
check("confidence High for clean not-detected", c_clear["level"] == "High")

print("\n[12] WIND endpoint (fetchers mocked)")
import perils
perils.wc.gather_station_gusts = lambda lat, lon, us, ue, n=3: [
    {"id": "KTUL", "name": "TULIA MUNI", "lat": 34.54, "lon": -101.77, "dist_mi": 1.2,
     "gust_mph": 71.0, "dir": "NW"}]
perils.wc.fetch_wind_reports = lambda *a, **k: [
    {"source": "NWS LSR", "speed_mph": 70, "lat": 34.55, "lon": -101.70, "dist_mi": 3.0,
     "dir": "NE", "time": "2024-06-02T21:05Z", "kind": "TSTM WND GST"}]
r = client.post("/generate", json={"peril": "wind", "manual_lat": 34.537, "manual_lon": -101.764,
                                   "date": "2024-06-02"})
j = r.json()
check("wind 200 + peril", r.status_code == 200 and j.get("peril") == "wind", str(r.status_code))
check("wind detected + mph headline", j.get("detected") is True and "mph" in (j.get("headline") or ""), j.get("headline"))
rw = client.get(f"/report/{j['share_id']}")
check("wind PDF retrievable", rw.status_code == 200 and rw.content[:4] == b"%PDF")

print("\n[13] SNOW endpoint (SNODAS + station mocked)")
GEO = {"ncols": 80, "nrows": 80, "xll": -103.60, "yll": 43.80, "cell": 0.008333333333, "nodata": -9999}
CY2, CX2 = 44.06, -103.29
yur = GEO["yll"] + (GEO["nrows"]-1)*GEO["cell"]
rws = np.arange(GEO["nrows"]); cls = np.arange(GEO["ncols"])
LATg = yur - rws[:, None]*GEO["cell"]; LONg = GEO["xll"] + cls[None, :]*GEO["cell"]
d2 = ((LATg-CY2)/0.12)**2 + ((LONg-CX2)/0.12)**2
depth = np.clip(300.0*np.exp(-d2), 0, None).astype(">i2")
swe = np.clip(90.0*np.exp(-d2), 0, None).astype(">i2")
import snow_core as scmod
perils.sc.SNODAS = GEO
perils.sc.fetch_snodas_product = lambda date, code, tmp: (
    scmod.read_snodas_grid(depth.tobytes(), 80, 80) if code == scmod.SNODAS_DEPTH
    else scmod.read_snodas_grid(swe.tobytes(), 80, 80))
perils.sc.fetch_station_snowfall = lambda *a, **k: [
    {"source": "COOP/CoCoRaHS", "snow_in": 11.0, "lat": 44.08, "lon": -103.25,
     "dist_mi": 2.0, "dir": "NE", "name": "RAPID CITY"}]
r = client.post("/generate", json={"peril": "snow", "manual_lat": CY2, "manual_lon": CX2,
                                   "date": "2025-01-15"})
j = r.json()
check("snow 200 + peril", r.status_code == 200 and j.get("peril") == "snow", str(r.status_code))
check("snow detected + load headline", j.get("detected") is True and "psf" in (j.get("headline") or ""), j.get("headline"))
rs = client.get(f"/report/{j['share_id']}")
check("snow PDF retrievable", rs.status_code == 200 and rs.content[:4] == b"%PDF")

print("\n[14] bad peril rejected")
check("unknown peril 400", client.post("/generate", json={"peril": "flood", "manual_lat": 34.5,
      "manual_lon": -101.0, "date": "2024-06-02"}).status_code == 400)

print("\n================ SUMMARY ================")
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL: print("  FAILURES:", FAIL)
sys.exit(1 if FAIL else 0)
