# =============================================================================
#  Peril dispatcher — runs hail / wind / snow and returns a normalized result so
#  the web API (app.py) treats all three the same way.
#
#  Each peril produces: pdf_path, detected (bool), confidence {level,color,note},
#  a one-line headline, a metrics dict, the data source, and report count.
#  Network fetches live in hail_core / wind_core / snow_core (tests monkeypatch them).
# =============================================================================

from __future__ import annotations
import os
import tempfile
import datetime as dt

import hail_core as hc
import pipeline           # hail pipeline (reused)
import wind_core as wc
import snow_core as sc

PERILS = ("hail", "wind", "snow")
DEFAULT_THRESHOLDS = {"hail": 0.75, "wind": 58.0, "snow": 6.0}


def run_peril(peril, *, address, date_of_loss, manual_lat=None, manual_lon=None,
              threshold=None, claim_ref="", contact_url="clearclaimsco.co",
              contact_city="Rapid City, SD", font_dir=None, out_dir=None,
              _hail_grib_paths=None):
    peril = (peril or "hail").lower()
    if peril not in PERILS:
        raise ValueError(f"Unknown peril '{peril}'. Choose hail, wind, or snow.")
    thr = threshold if threshold is not None else DEFAULT_THRESHOLDS[peril]
    out_dir = out_dir or tempfile.mkdtemp()
    fd = font_dir if (font_dir and os.path.isdir(font_dir)) else None

    if peril == "hail":
        r = pipeline.generate_report(
            address=address, date_of_loss=date_of_loss, manual_lat=manual_lat,
            manual_lon=manual_lon, threshold_in=thr, claim_ref=claim_ref,
            contact_url=contact_url, contact_city=contact_city, font_dir=fd,
            out_dir=out_dir, _grib_paths=_hail_grib_paths)
        rings = r["rings"]
        return {
            "peril": "hail", "pdf_path": r["pdf_path"], "report_id": r["report_id"],
            "detected": r["detected"], "confidence": r["confidence"],
            "data_source": r.get("data_source"), "n_reports": len(r.get("reports") or []),
            "headline": f'Max hail {rings["point"]["in"]:.2f}" at property',
            "metrics": {"at_property_in": round(rings["point"]["in"], 2),
                        "mile1_in": round(rings[1]["in"], 2),
                        "mile3_in": round(rings[3]["in"], 2),
                        "mile5_in": round(rings[5]["in"], 2)},
            "location": r["location"], "threshold": thr,
        }

    # ---- shared setup for wind/snow ----
    hc.validate_date_of_loss(date_of_loss)
    loc = hc.resolve_location(address, manual_lat, manual_lon)
    us, ue, tz = hc.local_day_utc_window(date_of_loss, loc["lat"], loc["lon"])
    rid_seed = abs(hash((loc["label"], str(date_of_loss), peril))) % 100000

    if peril == "wind":
        stations = wc.gather_station_gusts(loc["lat"], loc["lon"], us, ue, n=3)
        reports = wc.fetch_wind_reports(loc["lat"], loc["lon"], us, ue, date_of_loss, radius_miles=15.0)
        mp = os.path.join(out_dir, "wind_map.png")
        wc.make_wind_map(loc["lat"], loc["lon"], stations, reports, mp, brand=hc.BRAND)
        rid = f"CC-W-{date_of_loss:%Y}-{rid_seed:05d}"
        data = wc.build_wind_report_data(
            report_id=rid, address_label=loc["label"], lat=loc["lat"], lon=loc["lon"],
            date_of_loss=date_of_loss, contact_url=contact_url, contact_city=contact_city,
            claim_ref=claim_ref, threshold_mph=thr, station_gusts=stations,
            reports=reports, map_data_uri=hc.png_to_data_uri(mp))
        pdf = os.path.join(out_dir, f"Clear_Claims_Wind_Report_{rid}.pdf")
        wc.render(data, pdf, font_dir=fd)
        peak = data["_peak_mph"]
        return {
            "peril": "wind", "pdf_path": pdf, "report_id": rid,
            "detected": data["_detected"], "confidence": data["_confidence"],
            "data_source": "NOAA ASOS + NWS/SPC reports", "n_reports": len(reports),
            "headline": (f"Peak gust {peak:.0f} mph" if peak is not None else "No measured gust"),
            "metrics": {"peak_mph": peak, "n_stations": len(stations)},
            "location": loc, "threshold": thr,
        }

    # snow
    tmp = tempfile.mkdtemp()
    arr_depth = sc.fetch_snodas_product(date_of_loss, sc.SNODAS_DEPTH, tmp)
    arr_swe = sc.fetch_snodas_product(date_of_loss, sc.SNODAS_SWE, tmp)
    depth_mm = sc.sample_snodas(arr_depth, loc["lat"], loc["lon"], sc.SNODAS, agg="point")
    swe_mm = sc.sample_snodas(arr_swe, loc["lat"], loc["lon"], sc.SNODAS, agg="point")
    stations = sc.fetch_station_snowfall(loc["lat"], loc["lon"], date_of_loss, radius_miles=25.0)
    mp = os.path.join(out_dir, "snow_map.png")
    sc.make_snow_map(arr_depth, loc["lat"], loc["lon"], sc.SNODAS, mp, brand=hc.BRAND)
    rid = f"CC-S-{date_of_loss:%Y}-{rid_seed:05d}"
    data = sc.build_snow_report_data(
        report_id=rid, address_label=loc["label"], lat=loc["lat"], lon=loc["lon"],
        date_of_loss=date_of_loss, contact_url=contact_url, contact_city=contact_city,
        claim_ref=claim_ref, threshold_in=thr, depth_mm=depth_mm, swe_mm=swe_mm,
        station_reports=stations, map_data_uri=hc.png_to_data_uri(mp))
    pdf = os.path.join(out_dir, f"Clear_Claims_Snow_Report_{rid}.pdf")
    sc.render(data, pdf, font_dir=fd)
    return {
        "peril": "snow", "pdf_path": pdf, "report_id": rid,
        "detected": data["_detected"], "confidence": data["_confidence"],
        "data_source": "NOAA SNODAS + station snowfall", "n_reports": len(stations),
        "headline": f'Snow depth {data["_depth_in"]:.1f}" · load {data["_load_psf"]:.0f} psf',
        "metrics": {"depth_in": data["_depth_in"], "load_psf": data["_load_psf"]},
        "location": loc, "threshold": thr,
    }
