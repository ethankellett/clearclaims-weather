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


def ctx_sample(n_events=12, warned=True, gust=71.0, prior_ok=True):
    """A realistic PR3 context block for offline rendering."""
    import datetime as _d
    # Mimic the real shape: the BIGGEST stones all land on one early date (a big
    # hail day produces many spotter calls), while the MOST RECENT event is small.
    evs = [{"date": f"2022-{(i % 12) + 1:02d}-14", "size_in": round(2.5 - i * 0.12, 2),
            "dist_mi": 0.6 + i * 0.7, "dir": ["N", "NE", "E", "SE"][i % 4],
            "city": f"Test Locality {i}", "county": "PENNINGTON",
            "source": "Public", "lat": CY, "lon": CX} for i in range(n_events)]
    evs = sorted(evs, key=lambda e: -e["size_in"])
    return {
        "prior_hail": {
            "ok": prior_ok, "events": evs if prior_ok else [],
            "window": {"start": _d.date(2022, 6, 3), "end": _d.date(2024, 6, 3),
                       "months": 24},
            "summary": ({"inner_mi": 5.0, "outer_mi": 10.0, "min_size_in": 0.75,
                         "largest_in": 2.5, "largest_date": "2022-01-14",
                         "largest_dist_mi": 0.6, "count_inner": 6,
                         "count_outer": n_events, "days_since_severe": 141,
                         "last_severe_date": "2024-01-14", "last_any_date": "2022-12-14",
                         "last_any_size_in": 1.18, "last_any_dist_mi": 8.3,
                         "last_any_dir": "SE",
                         "n_raw": n_events + 4, "n_below_cutoff": 4, "months": 24}
                        if prior_ok else {}),
        },
        "warnings": {"ok": True, "warned": warned,
                     "warnings": ([{"name": "Severe Thunderstorm Warning",
                                    "issued": dt.datetime(2024, 6, 3, 22, 58,
                                                          tzinfo=dt.timezone.utc),
                                    "wfo": "UNR", "ugc": "SDC103"}] * 4) if warned else [],
                     "watches": []},
        "wind": {"ok": True, "peak_mph": gust, "station": "KRAP",
                 "dist_mi": 3.4, "n_stations": 3} if gust else
                {"ok": True, "peak_mph": None, "station": None, "dist_mi": None,
                 "n_stations": 3},
    }


def run(tag, field, address="1234 Mount Rushmore Rd, Rapid City, SD 57701", thr=0.75,
        rqi=0.92, context=None):
    """rqi: float 0..1, or None to simulate RQI being unavailable."""
    g = make_grib(field, os.path.join(TMP, f"{tag}.grib2"))
    out = os.path.join(TMP, tag); os.makedirs(out, exist_ok=True)
    r = pipeline.generate_report(address=address, manual_lat=CY, manual_lon=CX,
                                 date_of_loss=DOL, threshold_in=thr,
                                 out_dir=out, _grib_paths=[g],
                                 _rqi={"value": rqi, "n_files": 2 if rqi is not None else 0,
                                       "source": "test"},
                                 _context=context if context is not None else {})
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
r = run("nocov", no_coverage(), rqi=0.92)
t = pdf_text(r["pdf_path"])
check("sparse grid + good RQI -> coverage ok", r["coverage"]["state"] == "ok",
      r["coverage"]["state"])
check("reads as 0.00 in, not a failure", r["classification"]["band"] == "none",
      r["classification"]["band"])
check("PDF prints 0.00", "0.00" in t)
check("PDF prints the radar quality grade", "excellent" in t.lower())
check("negative still called weak", "weaker evidence" in t.lower())
check("clean view earns a High negative", r["confidence"]["level"] == "High",
      str(r["confidence"]["level"]))

print("\n=== 1b. GENUINE no-coverage state still renders correctly ===")
# Reached when a future coverage source (MRMS RQI, PR 2) reports a real gap.
cov = hc.assess_coverage(hc.grade_rqi(0.0))
check("RQI 0.0 -> coverage state 'none'", cov["state"] == "none", cov["state"])
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
check("coverage ok from RQI", r["coverage"]["state"] == "ok", r["coverage"]["state"])
check("badge = None Detected", r["classification"]["badge"] == "None Detected")
check("calls the negative weak", "weaker evidence" in t.lower())
check("clean view earns a High negative", r["confidence"]["level"] == "High",
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

print("\n=== 11. RQI GRADES AND THEIR EFFECT (PR 2) ===")
for val, want in [(0.95, "Excellent"), (0.62, "Good"), (0.31, "Fair"),
                  (0.08, "Poor"), (0.0, "No coverage"), (None, "Not assessed")]:
    g = hc.grade_rqi(val)
    check(f"RQI {val} -> {want}", g["grade"] == want, g["grade"])

r = run("poorrqi", blob(0), rqi=0.08)
check("Poor RQI drags a negative down to Low", r["confidence"]["level"] == "Low",
      str(r["confidence"]["level"]))
check("Poor RQI -> coverage 'partial'", r["coverage"]["state"] == "partial",
      r["coverage"]["state"])
t = pdf_text(r["pdf_path"])
check("PDF says degraded coverage", "degraded" in t.lower())

r = run("norqi", blob(0), rqi=None)
check("RQI unavailable -> coverage 'unknown'", r["coverage"]["state"] == "unknown",
      r["coverage"]["state"])
check("RQI unavailable caps a negative at Moderate",
      r["confidence"]["level"] == "Moderate", str(r["confidence"]["level"]))
t = pdf_text(r["pdf_path"])
check("PDF says quality not assessed", "not assessed" in t.lower())

r = run("hail_poorrqi", blob(1.40), rqi=0.08)
check("Poor RQI drags a POSITIVE down too", r["confidence"]["level"] == "Low",
      str(r["confidence"]["level"]))

print("\n=== 12. SPC 12Z-12Z CLOCK (defect D2) ===")
import datetime as _dt
cd = _dt.date(2024, 6, 3)
t0 = hc.spc_row_time_utc("2130", cd)
t1 = hc.spc_row_time_utc("0215", cd)
check("2130Z stays on the convective day", t0 and t0.date() == cd, str(t0))
check("0215Z rolls to the NEXT calendar day",
      t1 and t1.date() == cd + _dt.timedelta(days=1), str(t1))
check("pre-noon hour is later than evening hour", t1 > t0)
check("garbage time -> None", hc.spc_row_time_utc("xx", cd) is None)

csv = ("Time,Size,Location,County,State,Lat,Lon,Comments\n"
       "0215,175,QUITAQUE,BRISCOE,TX,44.081,-103.231,early morning\n"
       "2130,100,QUITAQUE,BRISCOE,TX,44.082,-103.232,evening\n")
win_s = _dt.datetime(2024, 6, 4, 0, 0, tzinfo=_dt.timezone.utc)
win_e = _dt.datetime(2024, 6, 4, 12, 0, tzinfo=_dt.timezone.utc)
got = hc.parse_spc_hail_csv(csv, 44.081, -103.231, 20.0,
                            convective_day=cd, utc_start=win_s, utc_end=win_e)
check("window filter keeps only the early-morning report",
      len(got) == 1 and abs(got[0]["size_in"] - 1.75) < 1e-6, str(got))

print("\n=== 13. GEOCODE PRECISION (T0-9) ===")
check("nominatim house -> rooftop",
      hc.classify_nominatim_precision({"class": "place", "type": "house"}) == "rooftop")
check("nominatim highway -> street",
      hc.classify_nominatim_precision({"class": "highway", "type": "residential"}) == "street")
check("nominatim town -> area",
      hc.classify_nominatim_precision({"class": "place", "type": "town"}) == "area")
loc = hc.resolve_location(None, 44.08, -103.23)
check("manual coords flagged as manual", loc["precision"] == "manual", loc["precision"])
r = run("geo", blob(1.40))
t = pdf_text(r["pdf_path"])
check("PDF prints the address-match class", "manual coordinates" in t.lower())

print("\n=== 14. STORM CONTEXT SUPPLEMENT (PR 3) ===")
r = run("context", blob(1.40), context=ctx_sample(n_events=16))
t = pdf_text(r["pdf_path"])
import subprocess as _sp
pg = _sp.run(["pdfinfo", r["pdf_path"]], capture_output=True, text=True).stdout
check("report is exactly 2 pages", "Pages:           2" in pg,
      [l for l in pg.splitlines() if l.startswith("Pages")])
check("page 1 footer says 1 of 2", "Page 1 of 2" in t)
check("page 2 footer says 2 of 2", "Page 2 of 2" in t)
check("supplement titled Storm History", "Storm History" in t)
check("prior-hail window in the heading",
      "jun 2022" in t.lower() and "jun 2024" in t.lower())
check("NWS warning card present", "severe warning" in t.lower())
check("warned pill shown", "warned" in t.lower())
check("measured gust shown", "71 mph" in t)
check("gust labelled a measurement not an estimate", "instrument" in t.lower())
check("largest prior report stat", "2.50" in t)
check("days-since stat", "141" in t)
check("event table capped with an overflow line",
      "further report(s) in this window are not listed" in t.lower())
# the overflow line must sit with the table, not after the disclaimer
_ti = t.lower().index("further report(s) in this window")
_di = t.lower().index("not evidence of damage to this roof")
check("overflow line stays above the disclaimer", _ti < _di, f"{_ti} vs {_di}")
check("history disclaimed as not damage evidence",
      "not evidence of damage to this roof" in t.lower())
check("rural undersampling stated", "nobody logs hail where nobody lives" in t.lower())
check("below-cutoff reports disclosed", "display cutoff" in t.lower())

print("\n=== 14b. CONTEXT DEGRADES HONESTLY ===")
r = run("ctx_none", blob(1.40), context=ctx_sample(n_events=0, warned=False, gust=None))
t = pdf_text(r["pdf_path"])
check("no-warning state stated", "no severe thunderstorm or tornado warning" in t.lower())
check("no-gust state stated", "no measurement" in t.lower())
check("downburst caveat on a null gust", "miss every station" in t.lower())
check("empty history stated explicitly", "no official hail report" in t.lower())

r = run("ctx_fail", blob(1.40),
        context={"prior_hail": {"ok": False, "error": "boom"},
                 "warnings": {"ok": False, "error": "boom"},
                 "wind": {"ok": False, "error": "boom"}})
t = pdf_text(r["pdf_path"])
check("failed lookup != clean history",
      "not the same as a clean history" in t.lower())
check("failed warning lookup says unavailable", "could not be retrieved" in t.lower())

print("\n=== 14c. NO CONTEXT -> STILL A CLEAN ONE-PAGER ===")
r = run("ctx_off", blob(1.40), context={})
pg = _sp.run(["pdfinfo", r["pdf_path"]], capture_output=True, text=True).stdout
t = pdf_text(r["pdf_path"])
check("falls back to a single page", "Pages:           1" in pg,
      [l for l in pg.splitlines() if l.startswith("Pages")])
check("footer says 1 of 1", "Page 1 of 1" in t)

print("\n=== 15. WIND PARITY (the D7-equivalent) ===")
import wind_core as wc
_r = lambda spd, d: {"speed_mph": spd, "dist_mi": d, "source": "NWS LSR", "dir": "NE"}
_st = lambda mph, d: {"id": "KRAP", "name": "Rapid City", "gust_mph": mph, "dist_mi": d}

# The bug: a below-threshold reading at a DISTANT station used to return High.
far = wc.assess_wind_confidence(40.0, 1, [], 58.0, nearest_dist_mi=27.0)
check("distant negative is no longer High", far["level"] != "High", far["level"])
check("distant negative is Low", far["level"] == "Low", far["level"])
check("distance is explained on the report", "miles away" in far["note"])
near = wc.assess_wind_confidence(40.0, 1, [], 58.0, nearest_dist_mi=3.0)
check("close negative still earns High", near["level"] == "High", near["level"])
mid = wc.assess_wind_confidence(40.0, 1, [], 58.0, nearest_dist_mi=9.0)
check("mid-range negative is Moderate", mid["level"] == "Moderate", mid["level"])
none = wc.assess_wind_confidence(None, 0, [], 58.0, nearest_dist_mi=None)
check("no measurement at all is Low", none["level"] == "Low", none["level"])
check("no-measurement wording is honest",
      "neither confirm nor rule out" in none["note"])
pos = wc.assess_wind_confidence(71.0, 3, [_r(70, 4)], 58.0, nearest_dist_mi=4.0)
check("corroborated positive is High", pos["level"] == "High", pos["level"])
for d, want in [(4.0, "Excellent"), (9.0, "Good"), (20.0, "Fair"), (40.0, "Poor"),
                (None, "No measurement")]:
    check(f"wind distance {d} -> {want}",
          wc.grade_wind_distance(d)["grade"] == want, wc.grade_wind_distance(d)["grade"])

wd = wc.build_wind_report_data(
    report_id="CC-W-TEST", address_label="1 Test St", lat=CY, lon=CX,
    date_of_loss=DOL, contact_url="x", contact_city="y", claim_ref="",
    threshold_mph=58.0, station_gusts=[_st(41.0, 22.0)], reports=[], map_data_uri="")
check("measured peak kept separate from reported", wd["_measured_peak_mph"] == 41.0)
check("nearest distance carried", wd["_nearest_dist_mi"] == 22.0)
check("headline says 'near this property', not 'at'",
      "near this property" in wd["findingHtml"] and "at this property" not in wd["findingHtml"])
check("sub states the reading is at the station",
      "not at the address" in wd["findingSubHtml"])
check("rows label the basis", any("Measured" in r["label"] for r in wd["rows"]))
check("measurement quality on the PDF meta", "Fair" in wd["measurementQuality"],
      wd["measurementQuality"])

wd2 = wc.build_wind_report_data(
    report_id="CC-W-TEST2", address_label="1 Test St", lat=CY, lon=CX,
    date_of_loss=DOL, contact_url="x", contact_city="y", claim_ref="",
    threshold_mph=58.0, station_gusts=[], reports=[_r(70.0, 12.0)], map_data_uri="")
check("a nearby REPORT is not presented as a measurement",
      "Reported nearby (not measured here)" in [r["label"] for r in wd2["rows"]])

print("\n=== 16. SNOW PARITY ===")
import snow_core as sc
# The bug: zero stations, yet the report claimed "nearby stations agree" at High.
z = sc.assess_snow_confidence(1.0, [], 6.0)
check("no-station negative is no longer High", z["level"] != "High", z["level"])
check("no longer claims stations agree", "stations agree" not in z["note"])
check("says plainly there was no station", "NO nearby station" in z["note"])
check("calls it an unverified model negative",
      "unverified model negative" in z["note"])
_sr = lambda snow, d: {"snow_in": snow, "dist_mi": d, "dir": "N", "source": "COOP"}
ok = sc.assess_snow_confidence(1.0, [_sr(0.5, 6.0)], 6.0)
check("close corroborated negative earns High", ok["level"] == "High", ok["level"])
far_s = sc.assess_snow_confidence(1.0, [_sr(0.5, 40.0)], 6.0)
check("distant corroborated negative is Moderate", far_s["level"] == "Moderate",
      far_s["level"])
conflict = sc.assess_snow_confidence(1.0, [_sr(9.0, 8.0)], 6.0)
check("conflicting station drops it to Low", conflict["level"] == "Low", conflict["level"])

sd = sc.build_snow_report_data(
    report_id="CC-S-TEST", address_label="1 Test St", lat=CY, lon=CX,
    date_of_loss=DOL, contact_url="x", contact_city="y", claim_ref="",
    threshold_in=6.0, depth_mm=254.0, swe_mm=40.0,
    station_reports=[_sr(0.2, 9.0)], map_data_uri="")
check("depth no longer called 'accumulation'",
      "accumulation" not in sd["findingHtml"].lower(), sd["findingHtml"][:80])
check("headline is about snow LOAD", "snow load" in sd["findingHtml"].lower())
check("states depth includes older snowpack",
      "includes" in sd["findingSubHtml"] and "older snowpack" in sd["findingSubHtml"])
check("explicitly not a claim that snow fell that day",
      "not a statement that snow fell on this date" in sd["findingSubHtml"])
check("new snowfall reported separately",
      any("NEW snowfall" in r["label"] for r in sd["rows"]))
check("depth row marked as modelled",
      any("modelled" in r["label"] for r in sd["rows"]))
check("section header no longer says 'accumulation'",
      "accumulation" not in sd["resultsTitle"].lower(), sd["resultsTitle"])

print("\n=== 17. STABLE IDs FOR WIND & SNOW (D4) ===")
import subprocess as _sp2, sys as _sys
_code = ("import sys;sys.path.insert(0,'.');import hashlib,datetime as dt;"
         "print(hashlib.sha1('1 Test St|2024-06-03|wind'.encode()).hexdigest()[:8].upper())")
_a = _sp2.run([_sys.executable, "-c", _code], capture_output=True, text=True,
              cwd=os.path.dirname(os.path.abspath(__file__)),
              env={**os.environ, "PYTHONHASHSEED": "1"}).stdout.strip()
_b = _sp2.run([_sys.executable, "-c", _code], capture_output=True, text=True,
              cwd=os.path.dirname(os.path.abspath(__file__)),
              env={**os.environ, "PYTHONHASHSEED": "999"}).stdout.strip()
check("wind/snow id seed is deterministic across hash seeds", _a == _b and len(_a) == 8, _a)

print("\n=== 18. ARCHIVE FLOOR + IEM FALLBACK (verified live 2026-08-27) ===")
# The IEM fallback matched ZERO files on older dates: IEM renamed the prefix
# from MRMS_Max_1440min_ to MESH_Max_1440min_ and the pattern only took one.
_modern = '<a href="MESH_Max_1440min_00.50_20230711-000000.grib2.gz">x</a>'
_older  = '<a href="MRMS_Max_1440min_00.50_20200620-122239.grib2.gz">x</a>'
u1 = hc.parse_iem_listing(_modern, "https://x/")
u2 = hc.parse_iem_listing(_older, "https://x/")
check("IEM listing: modern MESH_ prefix matches", len(u1) == 1, str(u1))
check("IEM listing: older MRMS_ prefix now matches too", len(u2) == 1, str(u2))
check("both parse to a usable timestamp",
      hc._parse_ts_from_key(u1[0]) is not None and hc._parse_ts_from_key(u2[0]) is not None)
check("de-dup still holds", len(hc.parse_iem_listing(_modern + _modern, "https://x/")) == 1)

# The floor was 2014-07-01, which we cannot serve: 2018/2019 hold no MESH on
# IEM at all. A pre-archive date must now be refused up front, clearly.
check("archive floor is the AWS start date",
      hc.IEM_ARCHIVE_START == hc.ARCHIVE_START, str(hc.IEM_ARCHIVE_START))
try:
    hc.validate_date_of_loss(dt.date(2018, 6, 20))
    check("2018 date refused up front", False, "no error raised")
except ValueError as e:
    check("2018 date refused up front", True)
    check("refusal explains the data does not exist", "does not exist" in str(e), str(e)[:90])
    check("refusal names the real start date", "2020" in str(e))
hc.validate_date_of_loss(dt.date(2023, 7, 11))   # must NOT raise
check("in-range date still accepted", True)

print("\n=== 19. PRIOR-HAIL TABLE READABILITY (reported by Ethan 2026-08-27) ===")
# A real report for 1834 Red Dale Dr filled all 8 rows with 2020-07-10 because
# the table sorted by SIZE. The most recent qualifying hail (2020-08-09) was in
# the stat box but nowhere in the table - the one thing the reader wanted.
import re as _re
r = run("recency", blob(1.40), context=ctx_sample(n_events=12))
t = pdf_text(r["pdf_path"])
# PDF text extraction letter-spaces small-caps headings, so normalise.
_n = lambda x: _re.sub(r"\s+", "", x).lower()
nt = _n(t)

_hdr = nt.index("howthenwsloggedthelocation")
# bound to the table itself: the page footer also carries a YYYY-MM-DD date
_tbl = t[t.index("SIZE (IN)"):]
_tbl = _tbl[:_tbl.index("Newest first")]
_rows = _re.findall(r"20\d\d-\d\d-\d\d", _tbl)
check("table is ordered newest-first", _rows == sorted(_rows, reverse=True),
      str(_rows[:6]))
check("the most recent event is the first row", _rows[0] == "2022-12-14", str(_rows[:3]))
check("every fixture date appears (cap raised to 12)", len(_rows) == 12, str(len(_rows)))
check("distance and direction are one field",
      _re.search(r"\d+\.\d mi (N|NE|E|SE|S|SW|W|NW)\b", t) is not None)
check("distance is anchored to the property", "fromthisproperty" in nt)
check("NWS wording labelled as the NWS's own", "howthenwsloggedthelocation" in nt)
check("caption says newest first", "newestfirst" in nt)
check("caption warns the last column is not relative to the property",
      "notrelativetothisproperty" in nt)
check("caption explains repeated dates",
      "manyreports" in nt and "samedate" in nt)
check("most-recent-hail stat present", "mostrecenthail" in nt)
check("most-recent stat shows the date + size + bearing",
      "2022-12-14" in t and "1.18" in t and "8.3 mi SE" in t)
check("most-recent >=1.00in stat shows its own date", "2024-01-14" in t)
check("still exactly 2 pages",
      "Pages:           2" in _sp.run(["pdfinfo", r["pdf_path"]],
                                      capture_output=True, text=True).stdout)

print(f"\n{'='*60}\n  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("  FAILED: " + ", ".join(FAIL))
print(f"  PDFs in {TMP}\n{'='*60}")
sys.exit(1 if FAIL else 0)
