import json

import numpy as np
import pandas as pd
import pytest

from taxi_eta.features import haversine
from taxi_eta.traces import (MAX_POINTS, local_to_unix, map_columns, parse_trace_text, read_batch, resample_route,
                             to_polyline_json, validate_points)

GOOD = [[-8.6110, 41.1496], [-8.6105, 41.1498], [-8.6098, 41.1500], [-8.6090, 41.1503]]


def test_parse_json_pairs():
    r = parse_trace_text(json.dumps(GOOD))
    assert r.ok and r.pts.shape == (4, 2) and not r.errors


def test_parse_geojson_linestring_and_feature():
    for obj in ({"type": "LineString", "coordinates": GOOD}, {"type": "Feature", "geometry": {"type": "LineString", "coordinates": GOOD}}):
        r = parse_trace_text(json.dumps(obj))
        assert r.ok and r.pts.shape == (4, 2)


def test_parse_lines_with_header_and_separators():
    text = "lon,lat\n" + "\n".join(f"{a} ; {b}".replace(" ; ", ",") for a, b in GOOD)
    r = parse_trace_text(text)
    assert r.ok and r.pts.shape == (4, 2)
    r = parse_trace_text("\n".join(f"{a}\t{b}" for a, b in GOOD))
    assert r.ok and r.pts.shape == (4, 2)


def test_latlon_order_is_detected_and_swapped():
    swapped = [[lat, lon] for lon, lat in GOOD]
    r = parse_trace_text(json.dumps(swapped))
    assert r.ok and np.allclose(r.pts, np.array(GOOD)) and r.notes


def test_empty_and_garbage_are_errors():
    assert not parse_trace_text("").ok
    assert not parse_trace_text("   ").ok
    r = parse_trace_text("-8.6,41.1\nthis is not a point\n-8.5,41.2")
    assert not r.ok and r.errors


def test_validate_flags_outside_porto_and_jumps():
    far = np.array([[2.35, 48.85], [2.351, 48.851]])               # Paris
    _, w = validate_points(far)
    assert any("outside the Porto area" in x for x in w)
    jump = np.array([[-8.61, 41.15], [-8.40, 41.30]])              # about 25 km in one 15 s step
    _, w = validate_points(jump)
    assert any("GPS jumps" in x for x in w)


def test_validate_rejects_nan_range_and_too_long():
    e, _ = validate_points(np.array([[np.nan, 41.1]]))
    assert e
    e, _ = validate_points(np.array([[-8.6, 141.1]]))
    assert e
    e, _ = validate_points(np.tile([[-8.6, 41.1]], (MAX_POINTS + 1, 1)))
    assert e


def test_resample_route_spacing_and_endpoints():
    verts = [(-8.6110, 41.1496), (-8.5990, 41.1520), (-8.5855, 41.1486)]
    pts = resample_route(verts, 30.0)
    step_km = 30.0 * 15 / 3600
    d = haversine(pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1])
    assert d.max() <= step_km * 1.001                              # never longer than one step ...
    assert np.sum(~np.isclose(d, step_km, rtol=0.01)) <= 1         # ... and equal 15 s steps, except the one chord that cuts the bend
    assert np.allclose(pts[0], verts[0])
    total = haversine(np.array([verts[0][0], verts[1][0]]), np.array([verts[0][1], verts[1][1]]), np.array([verts[1][0], verts[2][0]]), np.array([verts[1][1], verts[2][1]])).sum()
    assert len(pts) == int(total // step_km) + 1


def test_resample_degenerate_inputs():
    assert len(resample_route([(-8.6, 41.1)], 25)) == 1
    assert len(resample_route([(-8.6, 41.1), (-8.6, 41.1)], 25)) == 1


def test_local_to_unix_handles_dst():
    summer = local_to_unix("2014-08-14 18:00")                     # Lisbon is UTC+1 in summer
    winter = local_to_unix("2014-01-15 18:00")                     # and UTC+0 in winter
    assert summer == int(pd.Timestamp("2014-08-14 17:00", tz="UTC").timestamp())
    assert winter == int(pd.Timestamp("2014-01-15 18:00", tz="UTC").timestamp())
    assert local_to_unix("2014-10-26 01:30") > 0                   # ambiguous hour (clocks go back) must not raise


def test_polyline_roundtrip():
    r = parse_trace_text(to_polyline_json(np.array(GOOD)))
    assert np.allclose(r.pts, np.array(GOOD), atol=1e-6)


def test_read_batch_aliases_and_row_errors():
    df = pd.DataFrame({"TRIP_ID": ["a", "b", "c"], "POLYLINE": [json.dumps(GOOD), "garbage", json.dumps(GOOD)],
                       "TIMESTAMP": [1407000000, 1407000000, "2014-08-14 18:00"], "CALL_TYPE": ["B", "A", "Z"],
                       "ORIGIN_STAND": [15, None, None], "TAXI_ID": [20000589, None, 1]})
    reqs, info = read_batch(df)
    assert reqs[0] is not None and reqs[0].call == 1 and reqs[0].stand == 15 and reqs[0].taxi == 20000589
    assert reqs[1] is None and info[1]["status"] == "error"
    assert reqs[2] is None and "unknown call type" in info[2]["message"]
    assert [i["id"] for i in info] == ["a", "b", "c"]               # one bad row must not stop the batch


def test_read_batch_milliseconds_and_optional_columns():
    df = pd.DataFrame({"polyline": [json.dumps(GOOD)], "start_time": [1407000000000]})
    reqs, info = read_batch(df)
    assert reqs[0].ts == 1407000000 and reqs[0].call == 2 and reqs[0].taxi is None
    assert "assumed C" in info[0]["message"]


def test_read_batch_requires_polyline_and_timestamp():
    with pytest.raises(ValueError, match="Required column"):
        read_batch(pd.DataFrame({"foo": [1]}))
    assert map_columns(pd.DataFrame(columns=["Trip ID", "Poly-Line", "Time"])) == {"id": "Trip ID", "polyline": "Poly-Line", "timestamp": "Time"}
