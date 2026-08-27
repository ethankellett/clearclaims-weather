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

#  Shared with pipeline.py so all three perils stamp the same methodology version.
METHODOLOGY_VERSION = "v2.5"
_GEOCODE_LABEL = {"rooftop": "Rooftop", "interpolated": "Address range",
                  "street": "Street centreline", "area": "Area centroid",
                  "manual": "Manual coordinates", "unknown": "Unspecified"}
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
        cls = r.get("classification") or {}
        cov = r.get("coverage") or {}
        cell_in, half_in = r.get("cell_in"), r.get("half_in")

        def _r2(v):
            return round(v, 2) if v is not None else None

        # Headline must say what the number IS (defect D6). Never "at property".
        if cov.get("state") == "none" or half_in is None:
            headline = "Radar coverage unavailable \u2014 no hail size stated"
        else:
            headline = f'Peak MESH {half_in:.2f}" within \u00bd mi of geocoded location'

        return {
            "peril": "hail", "pdf_path": r["pdf_path"], "report_id": r["report_id"],
            "detected": r["detected"], "confidence": r["confidence"],
            "data_source": r.get("data_source"), "n_reports": len(r.get("reports") or []),
            "headline": headline,
            "coverage": cov.get("state"),
            "radar_quality": cov.get("quality_grade"),
            "prior_hail": ((r.get("context") or {}).get("prior_hail") or {}).get("summary"),
            "nws_warned": ((r.get("context") or {}).get("warnings") or {}).get("warned"),
            "same_day_gust_mph": ((r.get("context") or {}).get("wind") or {}).get("peak_mph"),
            "geocode_precision": (r["location"] or {}).get("precision"),
            "badge": cls.get("badge"),
            "likelihood": cls.get("likelihood"),
            "metrics": {"cell_in": _r2(cell_in),
                        "half_mile_in": _r2(half_in),
                        # kept for backward compatibility with stored reports
                        "at_property_in": _r2(half_in),
                        "mile1_in": _r2(rings[1]["in"]),
                        "mile3_in": _r2(rings[3]["in"]),
                        "mile5_in": _r2(rings[5]["in"])},
            "location": r["location"], "threshold": thr,
        }

    # ---- shared setup for wind/snow ----
    hc.validate_date_of_loss(date_of_loss, peril=peril)   # per-peril floors (F3)
    loc = hc.resolve_location(address, manual_lat, manual_lon)
    us, ue, tz = hc.local_day_utc_window(date_of_loss, loc["lat"], loc["lon"])
    # D4, same defect the hail report had: hash() is randomised per process, so
    # the same address and date produced a different id after every deploy.
    import hashlib as _h
    rid_seed = _h.sha1(f'{loc["label"]}|{date_of_loss}|{peril}'.encode()).hexdigest()[:8].upper()

    if peril == "wind":
        stations = wc.gather_station_gusts(loc["lat"], loc["lon"], us, ue, n=3)
        reports = wc.fetch_wind_reports(loc["lat"], loc["lon"], us, ue, date_of_loss, radius_miles=15.0)
        mp = os.path.join(out_dir, "wind_map.png")
        wc.make_wind_map(loc["lat"], loc["lon"], stations, reports, mp, brand=hc.BRAND)
        rid = f"CC-W-{date_of_loss:%Y}-{rid_seed}"
        _gp = _GEOCODE_LABEL.get(loc.get("precision", "unknown"), "Unspecified")
        data = wc.build_wind_report_data(
            report_id=rid, address_label=loc["label"], lat=loc["lat"], lon=loc["lon"],
            date_of_loss=date_of_loss, contact_url=contact_url, contact_city=contact_city,
            claim_ref=claim_ref, threshold_mph=thr, station_gusts=stations,
            reports=reports, map_data_uri=hc.png_to_data_uri(mp))
        data["geocodeQuality"] = _gp
        data["versionLine"] = f"methodology {METHODOLOGY_VERSION}"
        data["generatedUtc"] = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}"
        pdf = os.path.join(out_dir, f"Clear_Claims_Wind_Report_{rid}.pdf")
        wc.render(data, pdf, font_dir=fd)

        # F5: the old headline blended measured station gusts with spotter
        # REPORTS into one "Peak gust N mph", so an estimate 15 miles away
        # could read as an instrument reading. State the basis of every number.
        m_peak = data.get("_measured_peak_mph")
        r_peak = data.get("_reported_peak_mph")
        p_id = data.get("_peak_station_id") or "station"
        p_dist = data.get("_peak_station_dist_mi")
        if m_peak is not None:
            headline = (f"Measured {m_peak:.0f} mph at {p_id}"
                        + (f" ({p_dist:.1f} mi)" if p_dist is not None else ""))
            if r_peak is not None and r_peak > m_peak:
                headline += f"; {r_peak:.0f} mph reported nearby"
        elif r_peak is not None:
            headline = f"No station measurement; {r_peak:.0f} mph reported nearby"
        else:
            headline = "No wind measurement available near this property"

        return {
            "peril": "wind", "pdf_path": pdf, "report_id": rid,
            "detected": data["_detected"], "confidence": data["_confidence"],
            "data_source": "NOAA ASOS + NWS/SPC reports", "n_reports": len(reports),
            "headline": headline,
            "badge": data.get("statusText"),
            "radar_quality": data.get("measurementQuality"),
            "geocode_precision": loc.get("precision"),
            "metrics": {"peak_mph": data.get("_peak_mph"),
                        "measured_peak_mph": m_peak,
                        "reported_peak_mph": r_peak,
                        "peak_station": data.get("_peak_station_id"),
                        "peak_station_dist_mi": p_dist,
                        "nearest_station_mph": data.get("_nearest_mph"),
                        "nearest_station": data.get("_nearest_id"),
                        "nearest_dist_mi": data.get("_nearest_dist_mi"),
                        "n_stations": len(stations)},
            "location": loc, "threshold": thr,
        }

    # snow
    tmp = tempfile.mkdtemp()
    # SNODAS grids are an early-morning (~06Z) snapshot, so snow that falls
    # DURING the date of loss only shows up in the NEXT day's file. Sample both
    # days (next day best-effort) and keep whichever shows the deeper snow at
    # the property, so daytime/evening storms aren't under-reported.
    def _nz(v):
        return 0.0 if v is None or (isinstance(v, float) and v != v) else float(v)

    arr_depth = sc.fetch_snodas_product(date_of_loss, sc.SNODAS_DEPTH, tmp)
    arr_swe = sc.fetch_snodas_product(date_of_loss, sc.SNODAS_SWE, tmp)
    depth_mm = sc.sample_snodas(arr_depth, loc["lat"], loc["lon"], sc.SNODAS, agg="point")
    swe_mm = sc.sample_snodas(arr_swe, loc["lat"], loc["lon"], sc.SNODAS, agg="point")
    next_day = date_of_loss + dt.timedelta(days=1)
    if next_day <= dt.date.today():
        try:
            arr_depth2 = sc.fetch_snodas_product(next_day, sc.SNODAS_DEPTH, tmp)
            arr_swe2 = sc.fetch_snodas_product(next_day, sc.SNODAS_SWE, tmp)
            depth2 = sc.sample_snodas(arr_depth2, loc["lat"], loc["lon"], sc.SNODAS, agg="point")
            swe2 = sc.sample_snodas(arr_swe2, loc["lat"], loc["lon"], sc.SNODAS, agg="point")
            if _nz(depth2) > _nz(depth_mm):
                arr_depth, arr_swe = arr_depth2, arr_swe2
                depth_mm, swe_mm = depth2, swe2
        except Exception:
            pass  # next-day grid unavailable — the day-of grid still stands
    stations = sc.fetch_station_snowfall(loc["lat"], loc["lon"], date_of_loss, radius_miles=25.0)
    mp = os.path.join(out_dir, "snow_map.png")
    sc.make_snow_map(arr_depth, loc["lat"], loc["lon"], sc.SNODAS, mp, brand=hc.BRAND)
    rid = f"CC-S-{date_of_loss:%Y}-{rid_seed}"
    data = sc.build_snow_report_data(
        report_id=rid, address_label=loc["label"], lat=loc["lat"], lon=loc["lon"],
        date_of_loss=date_of_loss, contact_url=contact_url, contact_city=contact_city,
        claim_ref=claim_ref, threshold_in=thr, depth_mm=depth_mm, swe_mm=swe_mm,
        station_reports=stations, map_data_uri=hc.png_to_data_uri(mp))
    data["geocodeQuality"] = _GEOCODE_LABEL.get(loc.get("precision", "unknown"), "Unspecified")
    data["versionLine"] = f"methodology {METHODOLOGY_VERSION}"
    data["generatedUtc"] = f"{dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M UTC}"
    pdf = os.path.join(out_dir, f"Clear_Claims_Snow_Report_{rid}.pdf")
    sc.render(data, pdf, font_dir=fd)
    return {
        "peril": "snow", "pdf_path": pdf, "report_id": rid,
        "detected": data["_detected"], "confidence": data["_confidence"],
        "data_source": "NOAA SNODAS + NWS station reports", "n_reports": len(stations),
        "headline": (f'Modelled depth {data["_depth_in"]:.1f}" on ground '
                     f'· load {data["_load_psf"]:.0f} psf'),
        "badge": data.get("statusText"),
        "radar_quality": data.get("measurementQuality"),
        "geocode_precision": loc.get("precision"),
        "metrics": {"depth_in": data["_depth_in"], "load_psf": data["_load_psf"],
                    "new_snow_in": data.get("_new_snow_in"),
                    "station_depth_in": data.get("_station_depth_in")},
        "location": loc, "threshold": thr,
    }
