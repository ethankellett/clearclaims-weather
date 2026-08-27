# =============================================================================
#  ClearClaims Hail Report — PIPELINE
#  One reusable function, generate_report(), that runs the whole job end-to-end
#  and returns the PDF bytes + the key numbers. The web app (app.py) calls this.
#
#  This is the SAME engine used by the Colab notebook (hail_core.py). The only
#  addition is a small "seam" (_fetch_grib_paths) so the S3 download step can be
#  swapped out in tests.
# =============================================================================

from __future__ import annotations

import os
import tempfile
import datetime as dt

import hail_core as hc
import storm_context as sctx

#  Bumped whenever the numbers on the report change meaning. Printed in the PDF
#  footer so any archived report can be traced to the logic that produced it.
METHODOLOGY_VERSION = "v2.2"


_GEOCODE_LABEL = {"rooftop": "Rooftop", "interpolated": "Address range",
                  "street": "Street centreline", "area": "Area centroid",
                  "manual": "Manual coordinates", "unknown": "Unspecified"}


def _geocode_label(loc: dict) -> str:
    """Short human label for how precisely the address was pinned (T0-9)."""
    return _GEOCODE_LABEL.get(loc.get("precision", "unknown"), "Unspecified")


def _coord_str(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.4f}° {ns}, {abs(lon):.4f}° {ew}"


def _fetch_grib_paths(utc_start, utc_end, tmpdir, date_of_loss=None, max_files=None):
    """Download the relevant MRMS MESH files (AWS first, then IEM archive).

    Returns (paths, keys, source_label). Kept as its own function so tests can
    monkeypatch it with a local synthetic file (no network needed).

    `max_files` controls how many MESH files are combined (cell-wise max) for the
    local day. MESH_Max_1440min is a running 24-hour maximum, so the single file
    at day-end already captures the day's peak; reading a few more guards against
    a missing/corrupt end-of-day file and timestamp edge cases. Memory peak is
    ~one CONUS grid regardless (each file is cropped and freed immediately), so on
    the 2GB tier we default to 3. Override with the MESH_MAX_FILES env var.
    """
    if max_files is None:
        try:
            max_files = max(1, int(os.environ.get("MESH_MAX_FILES", "3")))
        except ValueError:
            max_files = 3
    paths, source, keys = hc.fetch_mesh_paths(utc_start, utc_end, date_of_loss,
                                              tmpdir, max_files=max_files)
    if not paths:
        raise ValueError(
            "No MRMS MESH radar files were found for that date on either the AWS or "
            "IEM archive. Try a nearby date, or check spc.noaa.gov for the event.")
    return paths, keys, source


def generate_report(
    *,
    address: str | None = None,
    date_of_loss: dt.date,
    manual_lat: float | None = None,
    manual_lon: float | None = None,
    threshold_in: float = 0.75,
    claim_ref: str = "",
    contact_url: str = "clearclaimsco.co",
    contact_city: str = "Rapid City, SD",
    report_title: str = "Radar-Based Hail Estimate Report",
    band_label: str = "Weather Analysis",
    font_dir: str | None = None,
    out_dir: str | None = None,
    _grib_paths: list | None = None,   # test seam: skip S3 if provided
    _rqi: dict | None = None,          # test seam: inject an RQI reading
    _context: dict | None = None,      # test seam: inject the PR3 context block
    with_context: bool = True,         # set False to skip the supplement page
) -> dict:
    """Run geocode -> S3 fetch -> parse -> sample -> map -> PDF.

    Returns a dict: {pdf_path, map_path, report_id, detected, rings, location, ...}
    Raises ValueError with a friendly message for expected problems.
    """
    out_dir = out_dir or tempfile.mkdtemp()
    os.makedirs(out_dir, exist_ok=True)

    # 0. date within archive window
    hc.validate_date_of_loss(date_of_loss)

    # 1. resolve location
    loc = hc.resolve_location(address, manual_lat, manual_lon)

    # 2. UTC window for the local day
    utc_start, utc_end, tz_name = hc.local_day_utc_window(date_of_loss, loc["lat"], loc["lon"])

    # 3. get the GRIB2 files (AWS → IEM fallback, or injected for tests)
    tmpdir = tempfile.mkdtemp()
    if _grib_paths is not None:
        grib_paths, keys, source = _grib_paths, [], "injected (test)"
    else:
        grib_paths, keys, source = _fetch_grib_paths(utc_start, utc_end, tmpdir, date_of_loss)

    # 4. read + sample (nearest cell, plus 0.5/1/3/5-mile peaks)
    lats, lons, mesh_mm = hc.max_mesh_over_files(grib_paths, loc["lat"], loc["lon"], pad_deg=0.30)
    rings = hc.sample_rings(lats, lons, mesh_mm, loc["lat"], loc["lon"], rings=(0.5, 1, 3, 5))

    # 4a. Radar COVERAGE QUALITY from NOAA's RQI product — an independent field.
    # This is the only thing that can tell a radar gap from a hail-free cell;
    # the MESH grid cannot (it is sparse). Best-effort: if RQI can't be had, the
    # report says "not assessed" rather than guessing either way.
    if _rqi is not None:
        rqi = _rqi                                  # test seam
    else:
        try:
            rqi = hc.fetch_rqi_at_point(utc_start, utc_end, loc["lat"], loc["lon"], tmpdir)
        except Exception:
            rqi = {"value": None, "n_files": 0, "source": None}
    rqi_grade = hc.grade_rqi(rqi.get("value"))
    rqi_grade["source"] = rqi.get("source")
    _hc_cells, _tot_cells = hc.count_hail_cells(lats, lons, mesh_mm, loc["lat"], loc["lon"])
    coverage = hc.assess_coverage(rqi_grade, _hc_cells, _tot_cells)

    # Two DISTINCT readings, never conflated (defect D6 / T0-3):
    #   cell_in  = value at the single nearest ~1 km grid cell
    #   half_in  = PEAK anywhere within half a mile. This is the headline figure
    #              because the grid is ~1 km and street geocoding is routinely
    #              off by a few hundred feet — but it is a peak over an area and
    #              the report must say so, not call it "at the property".
    cell_in = rings["point"]["in"]
    half_in = rings[0.5]["in"]
    peak_in = max([v for v in (cell_in, half_in) if v is not None], default=None)

    classification = hc.classify_hail(peak_in, cell_in, threshold_in, coverage["state"])
    detected = bool(classification.get("detected"))
    at_property_in = peak_in          # kept for the API's existing metrics contract

    # 4b. ground-truth corroboration + confidence (best-effort; never fatal)
    try:
        reports = hc.fetch_storm_reports(loc["lat"], loc["lon"], utc_start, utc_end,
                                         date_of_loss, radius_miles=12.0)
    except Exception:
        reports = []
    confidence = hc.assess_confidence(peak_in, rings[1]["in"], reports, threshold_in,
                                      source, coverage_state=coverage["state"],
                                      quality_grade=coverage["quality_grade"])
    corrob_line = hc.corroboration_line(reports, 12.0, coverage["state"])

    # 4c. Storm context for the supplement page (PR 3). All three lookups are
    # best-effort and individually guarded: a slow or broken endpoint costs that
    # one card, never the report.
    if _context is not None:
        context = _context
    elif with_context:
        try:
            context = sctx.gather(loc["lat"], loc["lon"], date_of_loss,
                                  utc_start, utc_end)
        except Exception:
            context = {}
    else:
        context = {}

    # 5. footprint map
    map_path = os.path.join(out_dir, "hail_footprint.png")
    hc.make_footprint_map(lats, lons, mesh_mm, loc["lat"], loc["lon"], 1.0, map_path, brand=hc.BRAND)

    # 6. report ID + data dict + PDF
    report_id = hc.stable_report_id(loc["label"], date_of_loss)
    generated = dt.datetime.now(dt.timezone.utc)

    def fmt(d):
        return {"in": d["in"], "mm": d["mm"]}

    data = {
        "reportId": report_id,
        "dateGenerated": f"{generated:%B %d, %Y}",
        "generatedUtc": f"{generated:%Y-%m-%d %H:%M UTC}",
        "versionLine": f"methodology {METHODOLOGY_VERSION}",
        "dateOfLoss": f"{date_of_loss:%B %d, %Y}",
        "propertyAddress": loc["label"],
        "claimRef": claim_ref or "\u2014",
        "coordinates": _coord_str(loc["lat"], loc["lon"]),
        "radarQuality": (
            f'{coverage["quality_grade"]}'
            + (f' ({coverage["quality_value"]:.2f})'
               if coverage.get("quality_value") is not None else "")),
        "geocodeQuality": _geocode_label(loc),
        "contactUrl": contact_url, "contactCity": contact_city,
        "bandLabel": band_label, "reportTitle": report_title,
        "thresholdInches": threshold_in,
        "classification": classification,
        "coverage": coverage,
        "results": {"cell": fmt(rings["point"]), "half": fmt(rings[0.5]),
                    "mile1": fmt(rings[1]), "mile3": fmt(rings[3]),
                    "mile5": fmt(rings[5])},
        "mapDataUri": hc.png_to_data_uri(map_path),
        "mapCaption": (
            f"No radar coverage at this location on {date_of_loss:%B %d, %Y}; the map "
            f"shows the search area only."
            if coverage["state"] == "none" else
            f"Estimated hail footprint \u2014 NOAA MRMS MESH, {date_of_loss:%B %d, %Y}."),
        "confidenceLevel": confidence["level"],
        "confidenceColor": confidence["color"],
        "confidenceNote": confidence["note"],
        "corroborationLine": corrob_line,
        "context": context,
    }
    html = hc.build_report_html(data, font_dir=font_dir)
    pdf_path = os.path.join(out_dir, f"Clear_Claims_Hail_Report_{report_id}.pdf")
    hc.render_pdf_weasyprint(html, pdf_path)

    return {
        "pdf_path": pdf_path,
        "map_path": map_path,
        "report_id": report_id,
        "detected": detected,
        "at_property_in": at_property_in,
        "rings": rings,
        "location": loc,
        "tz_name": tz_name,
        "files_used": keys,
        "threshold_in": threshold_in,
        "cell_in": cell_in,
        "half_in": half_in,
        "coverage": coverage,
        "classification": classification,
        "rqi": rqi_grade,
        "context": context,
        "data_source": source,
        "confidence": confidence,
        "reports": reports,
    }
