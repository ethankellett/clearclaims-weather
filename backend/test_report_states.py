"""
Offline render test for the hail report (no network required).

Builds synthetic MRMS-format GRIB2 grids, pushes them through the real
pipeline, renders real PDFs with WeasyPrint, and asserts that each report
STATE says the right thing. Covers the acceptance checks for PR 1:

  * no radar coverage      -> states coverage unavailable, never 0.00" / NOT DETECTED
  * 0.00" with coverage    -> "none detected", explicitly a weak negative
  * 0.40" sub-severe       -> prints the number, does NOT call it verified hail
  * 0.60" indicated        -> prints the number, says below damage threshold
  * 0.95" at threshold     -> "probable", flags 1.00" as uncertain
  * 1.40" severe           -> "likely" at 1.00"
  * 3.10" significant      -> "likely - significant"
  * torture-length address -> still one page

Run:  python3 test_report_states.py
"""
import os, sys, tempfile, datetime as dt
import numpy as np
import eccodes as ec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hail_core as hc
import pipeline

PASS, FAIL = [], []
def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"   {extra}" if extra else ""))

NLAT, NLON = 90, 120
LAT_TOP, LAT_BOT, LON_L, LON_R = 44.60, 43.70, -103.60, -102.40   # Rapid City, SD
CY, CX = 44.08, -103.23
lats = np.linspace(LAT_TOP, LAT_BOT, NLAT)
lons = np.linspace(LON_L, LON_R, NLON)
LON, LAT = np.meshgrid(lons, lats)


def make_grib(field_mm, path):
    gid = ec.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    ec.codes_set(gid, "Ni", NLON); ec.codes_set(gid, "Nj", NLAT)
    ec.codes_set(gid, "latitudeOfFirstGridPointInDegrees", float(LAT_TOP))
    ec.codes_set(gid, "longitudeOfFirstGridPointInDegrees", float(LON_L % 360))
    ec.codes_set(gid, "latitudeOfLastGridPointInDegrees", float(LAT_BOT))
    ec.codes_set(gid, "longitudeOfLastGridPointInDegrees", float(LON_R % 360))
    ec.codes_set(gid, "iDirectionIncrementInDegrees", float((LON_R - LON_L) / (NLON - 1)))
    ec.codes_set(gid, "jDirectionIncrementInDegrees", float((LAT_TOP - LAT_BOT) / (NLAT - 1)))
    ec.codes_set(gid, "jScansPositively", 0)
    ec.codes_set_values(gid, np.asarray(field_mm, dtype="float64").flatten())
    with open(path, "wb") as fh:
        ec.codes_write(gid, fh)
    ec.codes_release(gid)
    return path


def blob(peak_in):
    """A hail core centred on the property, peaking at peak_in inches."""
    if peak_in <= 0:
        return np.zeros((NLAT, NLON))
    peak_mm = peak_in * hc.MM_PER_INCH
    return np.clip(peak_mm * np.exp(-(((LAT - CY) / 0.09) ** 2
                                      + ((LON - CX) / 0.09) ** 2)), 0, None)


def no_coverage():
    """MRMS sentinel for 'no radar coverage' across the whole domain."""
    return np.full((NLAT, NLON), -999.0)


TMP = tempfile.mkdtemp()
hc.fetch_storm_reports = lambda *a, **k: []          # no network in the sandbox
DOL = dt.date(2024, 6, 3)


def run(tag, field, address="1234 Mount Rushmore Rd, Rapid City, SD 57701", thr=0.75):
    g = make_grib(field, os.path.join(TMP, f"{tag}.grib2"))
    out = os.path.join(TMP, tag); os.makedirs(out, exist_ok=True)
    r = pipeline.generate_report(address=address, manual_lat=CY, manual_lon=CX,
                                 date_of_loss=DOL, threshold_in=thr,
                                 out_dir=out, _grib_paths=[g])
    html = open(os.path.join(out, "report.html"), "w")
    html.write(r["_html"]) if "_html" in r else None
    html.close()
    return r


def pdf_text(path):
    import subprocess
    return subprocess.run(["pdftotext", path, "-"], capture_output=True,
                          text=True).stdout


print("\n=== 1. ALL-SENTINEL GRID = QUIET DAY, NOT A COVERAGE FAILURE ===")
# Learned from the first live run (2026-08-27): MESH is a SPARSE field. On a
# quiet day almost every cell is a missing sentinel. That must read as 0.00 in,
# NOT as "coverage unavailable" - but the report must also stop claiming it has
# verified coverage, because MESH alone cannot tell a gap from a hail-free cell.
r = run("nocov", no_coverage())
t = pdf_text(r["pdf_path"])
check("coverage state is 'unknown' (honest)", r["coverage"]["state"] == "unknown",
      r["coverage"]["state"])
check("reads as 0.00 in, not a failure", r["classification"]["band"] == "none",
      r["classification"]["band"])
check("PDF prints 0.00", "0.00" in t)
check("PDF discloses coverage is unverified", "not independently verified" in t.lower())
check("negative still called weak", "weaker evidence" in t.lower())
check("confidence capped at Moderate", r["confidence"]["level"] == "Moderate",
      str(r["confidence"]["level"]))

print("\n=== 1b. GENUINE no-coverage state still renders correctly ===")
# Reached when a future coverage source (MRMS RQI, PR 2) reports a real gap.
cls = hc.classify_hail(None, None, 0.75, coverage_state="none")
check("band = no_coverage", cls["band"] == "no_coverage")
check("detected is None, not False", cls["detected"] is None)
check("verdict states coverage unavailable",
      "coverage was unavailable" in cls["verdict"].lower())
conf = hc.assess_confidence(None, None, [], 0.75, coverage_state="none")
check("no confidence level", conf["level"] is None)

print("\n=== 2. ZERO HAIL, GOOD COVERAGE ===")
r = run("zero", blob(0))
t = pdf_text(r["pdf_path"])
check("coverage state honest (unknown until RQI)", r["coverage"]["state"] == "unknown", r["coverage"]["state"])
check("badge = None Detected", r["classification"]["badge"] == "None Detected")
check("calls the negative weak", "weaker evidence" in t.lower())
check("confidence capped at Moderate", r["confidence"]["level"] == "Moderate",
      str(r["confidence"]["level"]))

print("\n=== 3. SUB-SEVERE 0.40in (must still print) ===")
r = run("sub", blob(0.40))
t = pdf_text(r["pdf_path"])
check("prints 0.40", "0.40" in t)
check("band = trace", r["classification"]["band"] == "trace", r["classification"]["band"])
check("says UNLIKELY", "unlikely" in t.lower())
check("explicitly disclaims verified hail", "not as verified hail" in t.lower())

print("\n=== 4. 0.60in HAIL INDICATED ===")
r = run("indicated", blob(0.60))
t = pdf_text(r["pdf_path"])
check("prints 0.60", "0.60" in t)
check("badge = Hail Indicated", r["classification"]["badge"] == "Hail Indicated")
check("mentions below threshold", "below the 0.75" in t)
check("detected False (below 0.75 threshold)", r["classification"]["detected"] is False)

print("\n=== 5. 0.95in AT THRESHOLD ===")
r = run("threshold", blob(0.95))
t = pdf_text(r["pdf_path"])
check("says PROBABLE", "probable" in t.lower())
check("flags 1.00in as uncertain", "uncertain" in t.lower())
check("detected True", r["classification"]["detected"] is True)

print("\n=== 6. 1.40in SEVERE LIKELY ===")
r = run("severe", blob(1.40))
t = pdf_text(r["pdf_path"])
check("says LIKELY", "likely" in t.lower())
check("cites the 1.14in proxy", "1.14" in t)
check("badge = Severe Hail Likely", r["classification"]["badge"] == "Severe Hail Likely")

print("\n=== 7. 3.10in SIGNIFICANT ===")
r = run("significant", blob(3.10))
t = pdf_text(r["pdf_path"])
check("band = significant", r["classification"]["band"] == "significant")

print("\n=== 8. UNIVERSAL COPY + LABEL RULES ===")
r = run("labels", blob(1.40))
t = pdf_text(r["pdf_path"])
check("MESH disclosure sentence present", "75th-percentile" in t)
check("nearest grid cell row present", "nearest grid cell" in t.lower())
check("half-mile row labelled as a peak", "Peak within" in t)
check("headline never says 'at this property' with a size",
      "hail at the property" not in t.lower())
check("methodology version in footer", "methodology v" in t.lower())
check("generated-at UTC in footer", "UTC" in t)

print("\n=== 9. STABLE REPORT IDs (D4) ===")
a = hc.stable_report_id("1234 Main St, Rapid City, SD", DOL)
b = hc.stable_report_id("1234 Main St, Rapid City, SD", DOL)
check("same input -> same id", a == b, a)
check("id has the CC-YYYY- shape", a.startswith("CC-2024-") and len(a) == 16, a)
import subprocess
sub = subprocess.run([sys.executable, "-c",
    "import sys;sys.path.insert(0,'.');import hail_core,datetime as dt;"
    "print(hail_core.stable_report_id('1234 Main St, Rapid City, SD', dt.date(2024,6,3)))"],
    capture_output=True, text=True, cwd=os.path.dirname(os.path.abspath(__file__)),
    env={**os.environ, "PYTHONHASHSEED": "12345"})
check("stable across a fresh process w/ different hash seed",
      sub.stdout.strip() == a, sub.stdout.strip() or sub.stderr[-200:])

print("\n=== 10. ONE-PAGE CONSTRAINT (incl. torture-length address) ===")
long_addr = ("12345 Northwest Extraordinarily Lengthy Commemorative Boulevard "
             "Southeast Apartment Complex Building Q Unit 1418, "
             "Rapid City, South Dakota 57701-9999")
r = run("longaddr", blob(1.40), address=long_addr)
n = subprocess.run(["pdfinfo", r["pdf_path"]], capture_output=True, text=True).stdout
pages = [l for l in n.splitlines() if l.startswith("Pages:")]
check("still one page", "1" in (pages[0] if pages else ""), pages[0] if pages else "?")

print(f"\n{'='*60}\n  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print(f"  PDFs in {TMP}\n{'='*60}")
sys.exit(1 if FAIL else 0)
