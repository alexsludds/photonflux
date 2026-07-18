"""The live schematic session bridge: /api/schematic + the photonflux.nb client.

Spins the real stdlib Handler on an ephemeral port with the simulation engine
stubbed out (the bridge is pure plumbing — no JAX in these tests), then
exercises the mirror endpoints raw and through the notebook client: push/pull
round-trips, revision bumps, optimistic-concurrency conflicts, server-sent
change events, run dispatch, and Builder-generated documents.
"""
import json
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "webapp"))

import server  # noqa: E402  (webapp/server.py — engine stays unloaded)
import session as session_mod  # noqa: E402
from photonflux import nb  # noqa: E402

CATALOG_STUB = {
    "cw_laser": {"params": [{"name": "power"}, {"name": "wavelength_nm"}]},
    "dir_coupler": {"params": [{"name": "coupling"}]},
}


def fake_run(payload: dict) -> dict:
    """Engine stub: echoes the analysis so client-side mapping is checkable."""
    a = payload.get("analysis") or {}
    if a.get("mode") == "explode":
        return {"ok": False, "error": "boom", "log": ["stack"]}
    if a.get("mode") == "dc":
        return {"ok": True, "kind": "op", "log": [],
                "rows": [{"name": "vout", "value": 1.8, "unit": "V",
                          "domain": "electrical"}]}
    return {"ok": True, "kind": "sweep", "x": [1.0, 2.0, 3.0],
            "xlabel": "x", "log": ["ran"], "echo_analysis": a,
            "traces": [
                {"name": "thru", "unit": "mW", "domain": "optical",
                 "values": [1.0, 2.0, 3.0]},
                {"name": "thru @ c=2", "unit": "mW", "domain": "optical",
                 "values": [4.0, 5.0, 6.0]},
            ]}


@pytest.fixture()
def srv(monkeypatch):
    """A live server on an ephemeral port with a fresh mirror + stub engine."""
    monkeypatch.setattr(server, "SESSION", session_mod.SchematicSession())
    monkeypatch.setattr(server, "simulate", SimpleNamespace(run=fake_run))
    monkeypatch.setattr(server, "CATALOG", CATALOG_STUB)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()


def _post(url: str, path: str, obj: dict) -> tuple[int, dict]:
    req = urllib.request.Request(url + path, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _get(url: str, path: str) -> dict:
    with urllib.request.urlopen(url + path, timeout=5) as r:
        return json.loads(r.read())


def _doc(**settings) -> dict:
    inst = {"type": "cw_laser", "x": 0, "y": 0, "rot": 0,
            "settings": {"power": 0.001, **settings}}
    return {"title": "t", "schematic": {"instances": {"LAS": inst},
                                        "wires": [], "probes": []},
            "analysis": {"mode": "dc"}, "selection": None}


# --- raw endpoint behaviour -------------------------------------------------
def test_mirror_starts_empty(srv):
    snap = _get(srv, "/api/schematic")
    assert snap == {"ok": True, "rev": 0, "source": "", "doc": None}


def test_put_get_roundtrip_and_rev(srv):
    code, res = _post(srv, "/api/schematic", {"source": "browser",
                                              "doc": _doc()})
    assert code == 200 and res == {"ok": True, "rev": 1}
    snap = _get(srv, "/api/schematic")
    assert snap["rev"] == 1 and snap["source"] == "browser"
    assert snap["doc"]["schematic"]["instances"]["LAS"]["settings"]["power"] \
        == 0.001
    code, res = _post(srv, "/api/schematic", {"source": "browser",
                                              "doc": _doc()})
    assert code == 200 and res["rev"] == 2


def test_put_requires_schematic(srv):
    code, res = _post(srv, "/api/schematic", {"source": "x",
                                              "doc": {"title": "no sch"}})
    assert code == 400 and not res["ok"]


def test_stale_base_rev_conflicts(srv):
    _post(srv, "/api/schematic", {"source": "browser", "doc": _doc()})
    _post(srv, "/api/schematic", {"source": "browser", "doc": _doc()})
    code, res = _post(srv, "/api/schematic",
                      {"source": "notebook", "doc": _doc(), "base_rev": 1})
    assert code == 409 and res["conflict"] and res["rev"] == 2
    # matching base_rev lands
    code, res = _post(srv, "/api/schematic",
                      {"source": "notebook", "doc": _doc(), "base_rev": 2})
    assert code == 200 and res["rev"] == 3


def test_malformed_base_rev_is_a_400_not_a_crash(srv):
    _post(srv, "/api/schematic", {"source": "browser", "doc": _doc()})
    for bad in ["abc", [1], 2.5, True]:
        code, res = _post(srv, "/api/schematic",
                          {"source": "notebook", "doc": _doc(),
                           "base_rev": bad})
        assert code == 400 and not res["ok"], f"base_rev={bad!r}"
    # the handler thread survived: a normal request still works
    assert _get(srv, "/api/schematic")["rev"] == 1


def test_bridge_disable_knob(srv, monkeypatch):
    monkeypatch.setattr(server, "_ENABLE_BRIDGE", False)
    code, res = _post(srv, "/api/schematic", {"source": "browser",
                                              "doc": _doc()})
    assert code == 403 and res["disabled"]
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(srv + "/api/schematic", timeout=5)
    assert ei.value.code == 403
    with pytest.raises(urllib.error.HTTPError) as ei:
        urllib.request.urlopen(srv + "/api/schematic/events", timeout=5)
    assert ei.value.code == 403


def test_sse_subscriber_cap(srv, monkeypatch):
    monkeypatch.setattr(server, "_SSE_LIMIT", 1)
    s = nb.Session(srv)
    events = s._sse("/api/schematic/events")
    with ThreadPoolExecutor(1) as ex:
        ex.submit(next, events).result(timeout=5)   # stream 1 is live
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(srv + "/api/schematic/events", timeout=5)
        assert ei.value.code == 503
    events.close()


def test_sse_stream_reports_changes(srv):
    s = nb.Session(srv)
    events = s._sse("/api/schematic/events")
    with ThreadPoolExecutor(1) as ex:
        first = ex.submit(next, events).result(timeout=5)
        assert first["rev"] == 0            # connect snapshot
        _post(srv, "/api/schematic", {"source": "browser", "doc": _doc()})
        ev = ex.submit(next, events).result(timeout=5)
        assert ev == {"rev": 1, "source": "browser"}
    events.close()


# --- the notebook client ----------------------------------------------------
def test_client_pull_empty_hints(srv):
    with pytest.raises(nb.SessionError, match="mirror is empty"):
        nb.Session(srv).pull()
    assert nb.Session(srv).pull(required=False) is None


def test_client_push_pull_set(srv):
    s = nb.Session(srv)
    b = nb.Builder()
    las = b.add("cw_laser", power="1m", wavelength_nm=1310)
    cpl = b.add("dir_coupler", coupling=0.088)
    b.wire(las.p1, cpl.p1)
    b.probe(cpl.p3, name="thru")
    rev = s.push(b, title="bench")
    assert rev == 1

    sch = s.pull()
    assert sch.title == "bench" and len(sch.instances) == 2
    assert sch["CWL1.power"] == 1e-3          # "1m" went through si()
    assert sch["CWL1"]["type"] == "cw_laser"
    # auto-layout spread the parts out
    xy = {(i["x"], i["y"]) for i in sch.instances.values()}
    assert len(xy) == 2

    s["DIR1.coupling"] = 0.3                  # read-modify-write via base_rev
    assert s["DIR1.coupling"] == 0.3
    assert s.pull().rev == 2

    with pytest.raises(KeyError, match="no parameter"):
        s["CWL1.wavelengthh_nm"] = 1310       # typo caught via catalog
    with pytest.raises(KeyError, match="no instance"):
        s["NOPE.power"] = 1


def test_client_run_and_result(srv):
    s = nb.Session(srv, progress=False)
    b = nb.Builder()
    b.add("cw_laser", power=1e-3)
    s.push(b)

    res = s.dcsweep("*", "wavelength_nm", 1304, "1316", points=11)
    assert res.ok and res.kind == "sweep"
    echoed = res.raw["echo_analysis"]
    assert echoed["mode"] == "dcsweep" and echoed["stop"] == 1316.0
    np.testing.assert_allclose(res["thru"], [1.0, 2.0, 3.0])
    assert set(res.family("thru")) == {"thru", "thru @ c=2"}
    with pytest.raises(KeyError, match="ambiguous"):
        res.trace("thr")

    op = s.dc()
    assert op.kind == "op" and op["vout"] == 1.8

    with pytest.raises(nb.RunError, match="boom") as ei:
        s.run({"mode": "explode"})
    assert ei.value.log == ["stack"]

    # transient helper maps its knobs into the analysis payload
    tr = s.transient(t_stop="20n", seeds=4, link="vout")
    a = tr.raw["echo_analysis"]
    assert a["t_stop"] == 2e-8
    assert a["noise"] == {"seeds": 4, "bw": 50e9}
    assert a["link"] == {"probe": "vout"}


def test_client_run_uses_mirror_analysis(srv):
    s = nb.Session(srv, progress=False)
    _post(srv, "/api/schematic", {"source": "browser", "doc": _doc()})
    res = s.run()                    # no args: the canvas Run button's twin
    assert res.kind == "op"          # _doc carries {"mode": "dc"}


def test_client_watch_yields_on_browser_edit(srv):
    s = nb.Session(srv)
    _post(srv, "/api/schematic", {"source": "browser", "doc": _doc()})
    w = s.watch()
    with ThreadPoolExecutor(1) as ex:
        fut = ex.submit(next, w)
        # watch() must skip our own pushes and yield only the browser's
        nb.Session(srv).push(_doc())
        _post(srv, "/api/schematic",
              {"source": "browser", "doc": _doc(wavelength_nm=1550.0)})
        sch = fut.result(timeout=5)
    assert sch.source == "browser"
    assert sch["LAS.wavelength_nm"] == 1550.0
    w.close()


def test_watch_initial_yields_own_push(srv):
    # the live-bench flow: push, then watch(initial=True) must yield the
    # current state even though its last writer is ourselves ("notebook")
    s = nb.Session(srv)
    s.push(_doc())
    w = s.watch(initial=True)
    with ThreadPoolExecutor(1) as ex:
        sch = ex.submit(next, w).result(timeout=5)
    assert sch.source == "notebook" and "LAS" in sch.instances
    w.close()


def test_string_settings_survive(srv):
    b = nb.Builder()
    b.add("cw_laser", ref="PRBS", mode="pam4", power="1m")
    d = b.doc()
    settings = d["schematic"]["instances"]["PRBS"]["settings"]
    assert settings["mode"] == "pam4" and settings["power"] == 1e-3
    sch = nb.Schematic(_doc())
    sch["LAS.note"] = "quadrature bias"      # non-numeric string: untouched
    assert sch["LAS.note"] == "quadrature bias"


def test_selected_roundtrip(srv):
    s = nb.Session(srv)
    doc = _doc()
    doc["selection"] = {"kind": "inst", "id": "LAS"}
    _post(srv, "/api/schematic", {"source": "browser", "doc": doc})
    ref, inst = s.selected()
    assert ref == "LAS" and inst["type"] == "cw_laser"
    doc["selection"] = None
    _post(srv, "/api/schematic", {"source": "browser", "doc": doc})
    assert s.selected() is None


def test_schematic_save_load(tmp_path):
    sch = nb.Schematic(_doc())
    p = sch.save(tmp_path / "snap")
    assert p.suffix == ".json"
    back = nb.Schematic.load(p)
    assert back["LAS.power"] == 0.001
    back["LAS.power"] = "2m"
    assert back["LAS.power"] == 0.002
    with pytest.raises(KeyError, match="no setting"):
        back["LAS.missing"]


def _spectrum_plot(probe: str, wl, db) -> dict:
    return {"x": list(wl), "xlabel": "wavelength [nm]", "xunit": "nm",
            "ydb": True, "yunit": "dB",
            "traces": [{"name": f"{probe} spectrum", "domain": "optical",
                        "unit": "dB", "values": list(db)}]}


def test_result_spectrum_helper():
    raw = {"ok": True, "kind": "transient", "x": [0.0, 1e-12],
           "traces": [{"name": "p_comb", "domain": "optical", "unit": "mW",
                       "values": [1.0, 1.0]}],
           "extra_plots": [_spectrum_plot("p_comb", [1309.9, 1310.0, 1310.1],
                                          [-40.0, 0.0, -38.0])]}
    res = nb.Result(raw)
    assert res.spectra == ["p_comb"]
    wl, db = res.spectrum()                      # sole spectrum: no name
    np.testing.assert_allclose(wl, [1309.9, 1310.0, 1310.1])
    np.testing.assert_allclose(db, [-40.0, 0.0, -38.0])
    wl2, _ = res.spectrum("p_comb")              # by probe name
    np.testing.assert_allclose(wl2, wl)
    # the merged .traces view still exists, but spectrum() is the safe path
    assert "p_comb spectrum" in res.names

    raw["extra_plots"].append(_spectrum_plot("p_drop", [1310.0], [0.0]))
    two = nb.Result(raw)
    assert two.spectra == ["p_comb", "p_drop"]
    with pytest.raises(KeyError, match="several spectrum probes"):
        two.spectrum()
    with pytest.raises(KeyError, match="no optical spectrum"):
        two.spectrum("p_nope")

    dry = nb.Result({"ok": True, "kind": "transient", "traces": []})
    assert dry.spectra == []
    with pytest.raises(KeyError, match="no optical spectra"):
        dry.spectrum()
