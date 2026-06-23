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


def _coord_str(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):.4f}° {ns}, {abs(lon):.4f}° {ew}"


def _fetch_grib_paths(utc_start, utc_end, tmpdir, date_of_loss=None, max_files=5):
    """Download the relevant MRMS MESH files (AWS first, then IEM archive).

    Returns (paths, keys, source_label). Kept as its own function so tests can
    monkeypatch it with a local synthetic file (no network needed).
    """
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

    # 4. read + sample (At Property and 1/3/5-mile maxima)
    lats, lons, mesh_mm = hc.max_mesh_over_files(grib_paths, loc["lat"], loc["lon"], pad_deg=0.30)
    rings = hc.sample_rings(lats, lons, mesh_mm, loc["lat"], loc["lon"], rings=(1, 3, 5))
    point_in = rings["point"]["in"]
    detected = bool(point_in >= threshold_in)

    # 4b. ground-truth corroboration + confidence (best-effort; never fatal)
    try:
        reports = hc.fetch_storm_reports(loc["lat"], loc["lon"], utc_start, utc_end,
                                         date_of_loss, radius_miles=12.0)
    except Exception:
        reports = []
    confidence = hc.assess_confidence(point_in, rings[1]["in"], reports, threshold_in, source)
    corrob_line = hc.corroboration_line(reports, 12.0)

    # 5. footprint map
    map_path = os.path.join(out_dir, "hail_footprint.png")
    hc.make_footprint_map(lats, lons, mesh_mm, loc["lat"], loc["lon"], 1.0, map_path, brand=hc.BRAND)

    # 6. report ID + data dict + PDF
    report_id = f"CC-{date_of_loss:%Y}-{abs(hash((loc['label'], str(date_of_loss)))) % 100000:05d}"

    def fmt(d): return {"in": f"{d['in']:.2f}", "mm": f"{d['mm']:.0f}"}
    data = {
        "reportId": report_id,
        "dateGenerated": f"{dt.date.today():%B %d, %Y}",
        "dateOfLoss": f"{date_of_loss:%B %d, %Y}",
        "propertyAddress": loc["label"],
        "claimRef": claim_ref or "—",
        "coordinates": _coord_str(loc["lat"], loc["lon"]),
        "contactUrl": contact_url, "contactCity": contact_city,
        "bandLabel": band_label, "reportTitle": report_title,
        "detected": detected, "thresholdInches": threshold_in,
        "results": {"atProperty": fmt(rings["point"]), "mile1": fmt(rings[1]),
                    "mile3": fmt(rings[3]), "mile5": fmt(rings[5])},
        "mapDataUri": hc.png_to_data_uri(map_path),
        "mapCaption": f"Estimated hail footprint — NOAA MRMS MESH, {date_of_loss:%B %d, %Y}.",
        "confidenceLevel": confidence["level"],
        "confidenceColor": confidence["color"],
        "confidenceNote": confidence["note"],
        "corroborationLine": corrob_line,
    }
    html = hc.build_report_html(data, font_dir=font_dir)
    pdf_path = os.path.join(out_dir, f"ClearClaims_Hail_Report_{report_id}.pdf")
    hc.render_pdf_weasyprint(html, pdf_path)

    return {
        "pdf_path": pdf_path,
        "map_path": map_path,
        "report_id": report_id,
        "detected": detected,
        "rings": rings,
        "location": loc,
        "tz_name": tz_name,
        "files_used": keys,
        "threshold_in": threshold_in,
        "data_source": source,
        "confidence": confidence,
        "reports": reports,
    }
