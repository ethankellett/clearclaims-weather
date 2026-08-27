# =============================================================================
#  Clear Claims Co. — STORM CONTEXT (PR 3)
#  Everything on the report that is NOT the date-of-loss radar reading:
#
#    * Prior hail near the property over the previous 24 months   (T0-11)
#    * Same-day measured wind gust                                 (T0-13)
#    * NWS warnings in force at the property that day              (T0-14)
#
#  All three are BEST-EFFORT and must never break a report: a network hiccup
#  yields an empty section that says so, not an exception.
#
#  Endpoints verified live on 2026-08-27 against Rapid City, SD:
#    LSR  : mesonet.agron.iastate.edu/geojson/lsr.geojson
#           A full 24-month bbox query returns in one request (~800 KB,
#           1,697 features, 116 of them hail).
#    VTEC : mesonet.agron.iastate.edu/json/vtec_events_bypoint.py
#           ~4 KB. For 2023-07-11 it returned a Severe Thunderstorm Watch plus
#           four Severe Thunderstorm Warnings from WFO UNR.
# =============================================================================

from __future__ import annotations

import datetime as dt

import hail_core as hc

LSR_URL = "https://mesonet.agron.iastate.edu/geojson/lsr.geojson"
VTEC_URL = "https://mesonet.agron.iastate.edu/json/vtec_events_bypoint.py"

#  Reports smaller than this are kept in the data but not printed: a 0.25 in
#  report two years ago is noise on a claim document.
PRIOR_MIN_SIZE_IN = 0.75
PRIOR_MONTHS = 24
PRIOR_RADII = (5.0, 10.0)
MAX_PRINTED_EVENTS = 8


# -----------------------------------------------------------------------------
#  T0-11  PRIOR HAIL  (Layer A — official ground reports only)
# -----------------------------------------------------------------------------
#  Deliberately LSR-only. SPC's daily CSVs would mean ~730 HTTP requests for a
#  24-month window, and the NCEI Storm Events bulk files are far too large to
#  pull per report. IEM's LSR service exposes the same NWS report stream with a
#  date range and a bounding box in a single call.
#
#  Honest limitation, stated on the report: these are *reported* hailstones.
#  Nobody logs a report where nobody lives, so a quiet history in open country
#  is not evidence that nothing fell.
# -----------------------------------------------------------------------------

def fetch_prior_hail(lat, lon, date_of_loss, months=PRIOR_MONTHS,
                     radii=PRIOR_RADII, min_size_in=PRIOR_MIN_SIZE_IN,
                     timeout=25):
    """Official hail reports near the property in the `months` before the loss.

    Returns a dict that is always safe to render:
      {ok, events, summary{...}, window{start,end,months}, error}
    `events` are sorted largest-first and exclude the date of loss itself.
    """
    import requests

    end = date_of_loss
    start = date_of_loss - dt.timedelta(days=int(round(months * 30.44)))
    outer = max(radii)
    pad = (outer / 69.0) * 1.6          # generous bbox; exact filtering is by distance

    out = {"ok": False, "events": [], "error": None,
           "window": {"start": start, "end": end, "months": months},
           "summary": {}}

    try:
        params = {
            "sts": f"{start:%Y-%m-%d}T00:00Z",
            "ets": f"{end:%Y-%m-%d}T00:00Z",
            "west": lon - pad, "east": lon + pad,
            "south": lat - pad, "north": lat + pad,
        }
        r = requests.get(LSR_URL, params=params, timeout=timeout)
        r.raise_for_status()
        feats = (r.json() or {}).get("features", []) or []
    except Exception as exc:
        out["error"] = str(exc)
        return out

    events = []
    for f in feats:
        p = f.get("properties", {}) or {}
        typ = (p.get("type") or "").upper()
        if typ != "H" and "HAIL" not in (p.get("typetext") or "").upper():
            continue
        coords = (f.get("geometry", {}) or {}).get("coordinates") or [None, None]
        rlon, rlat = coords[0], coords[1]
        if rlat is None or rlon is None:
            continue
        try:
            size_in = float(p.get("magnitude"))
        except (TypeError, ValueError):
            continue
        d = float(hc.haversine_miles(lat, lon, rlat, rlon))
        if d > outer:
            continue
        valid = (p.get("valid") or "")[:10]
        if valid == f"{date_of_loss:%Y-%m-%d}":
            continue                     # that's the loss itself, not history
        events.append({
            "date": valid, "size_in": size_in, "dist_mi": d,
            "dir": hc.compass_bearing(lat, lon, rlat, rlon),
            "city": p.get("city", ""), "county": p.get("county", ""),
            "source": p.get("source", "NWS LSR"),
            "lat": float(rlat), "lon": float(rlon),
        })

    # De-dup: the same stone often gets logged by several spotters.
    seen, deduped = set(), []
    for e in sorted(events, key=lambda x: (-x["size_in"], x["dist_mi"])):
        key = (e["date"], round(e["lat"], 2), round(e["lon"], 2), round(e["size_in"], 2))
        if key not in seen:
            seen.add(key)
            deduped.append(e)

    inner = min(radii)
    printable = [e for e in deduped if e["size_in"] >= min_size_in]
    within_inner = [e for e in printable if e["dist_mi"] <= inner]
    within_outer = printable

    largest = max(within_inner, key=lambda e: e["size_in"], default=None)
    severe_inner = [e for e in within_inner if e["size_in"] >= 1.00]

    days_since = None
    if severe_inner:
        try:
            latest = max(dt.date.fromisoformat(e["date"]) for e in severe_inner)
            days_since = (date_of_loss - latest).days
        except Exception:
            days_since = None

    out["ok"] = True
    out["events"] = printable
    out["summary"] = {
        "inner_mi": inner, "outer_mi": outer, "min_size_in": min_size_in,
        "largest_in": (largest or {}).get("size_in"),
        "largest_date": (largest or {}).get("date"),
        "largest_dist_mi": (largest or {}).get("dist_mi"),
        "count_inner": len(within_inner),
        "count_outer": len(within_outer),
        "days_since_severe": days_since,
        "n_raw": len(deduped),
        "n_below_cutoff": len(deduped) - len(printable),
    }
    return out


# -----------------------------------------------------------------------------
#  T0-14  NWS WARNINGS IN FORCE AT THE PROPERTY THAT DAY
# -----------------------------------------------------------------------------

def fetch_nws_warnings(lat, lon, utc_start, utc_end, timeout=20):
    """Severe-thunderstorm / tornado warnings and watches covering the point.

    Cheap, independent corroboration: a warning is a human forecaster deciding
    in real time that this county was in danger, which is evidence of a very
    different kind from a radar retrieval.
    """
    import requests
    out = {"ok": False, "warned": False, "warnings": [], "watches": [], "error": None}
    try:
        params = {"lat": round(float(lat), 4), "lon": round(float(lon), 4),
                  "sdate": f"{(utc_start - dt.timedelta(days=1)).date():%Y-%m-%d}",
                  "edate": f"{(utc_end + dt.timedelta(days=1)).date():%Y-%m-%d}"}
        r = requests.get(VTEC_URL, params=params, timeout=timeout)
        r.raise_for_status()
        events = (r.json() or {}).get("events", []) or []
    except Exception as exc:
        out["error"] = str(exc)
        return out

    for e in events:
        ph = (e.get("phenomena") or "").upper()
        sig = (e.get("significance") or "").upper()
        if ph not in ("SV", "TO"):        # severe thunderstorm / tornado only
            continue
        try:
            issued = dt.datetime.fromisoformat(
                (e.get("issue") or "").replace("Z", "+00:00"))
        except Exception:
            continue
        if not (utc_start <= issued <= utc_end):
            continue
        rec = {"name": e.get("name") or e.get("ph_name") or "",
               "issued": issued, "wfo": e.get("wfo", ""), "ugc": e.get("ugc", "")}
        if sig == "W":
            out["warnings"].append(rec)
        elif sig == "A":
            out["watches"].append(rec)

    out["warnings"].sort(key=lambda x: x["issued"])
    out["watches"].sort(key=lambda x: x["issued"])
    out["warned"] = bool(out["warnings"])
    out["ok"] = True
    return out


# -----------------------------------------------------------------------------
#  T0-13  SAME-DAY MEASURED WIND
# -----------------------------------------------------------------------------

def fetch_same_day_wind(lat, lon, utc_start, utc_end, n=3):
    """Peak measured ASOS/AWOS gust near the property on the date of loss.

    Hail-only versus thunderstorm-wind is a live dispute on these jobs, and the
    wind pipeline already exists — this just surfaces one row of it on the hail
    report. Unlike MESH this is an instrument reading, so it is stated as a
    measurement; the caveat is DISTANCE, since a downburst can miss every
    station in the county.
    """
    out = {"ok": False, "peak_mph": None, "station": None, "dist_mi": None,
           "n_stations": 0, "error": None}
    try:
        import wind_core as wc
        stations = wc.gather_station_gusts(lat, lon, utc_start, utc_end, n=n)
    except Exception as exc:
        out["error"] = str(exc)
        return out

    out["ok"] = True
    out["n_stations"] = len(stations)
    measured = [s for s in stations if s.get("gust_mph") is not None]
    if measured:
        best = max(measured, key=lambda s: s["gust_mph"])
        out["peak_mph"] = float(best["gust_mph"])
        out["station"] = best.get("id") or best.get("station") or best.get("sid") or ""
        out["dist_mi"] = best.get("dist_mi")
    return out


def gather(lat, lon, date_of_loss, utc_start, utc_end):
    """Run all three context lookups. Never raises."""
    ctx = {}
    for key, fn in (
        ("prior_hail", lambda: fetch_prior_hail(lat, lon, date_of_loss)),
        ("warnings", lambda: fetch_nws_warnings(lat, lon, utc_start, utc_end)),
        ("wind", lambda: fetch_same_day_wind(lat, lon, utc_start, utc_end)),
    ):
        try:
            ctx[key] = fn()
        except Exception as exc:
            ctx[key] = {"ok": False, "error": str(exc)}
    return ctx
