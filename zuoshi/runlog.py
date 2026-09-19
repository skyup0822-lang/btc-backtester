#!/usr/bin/env python3
"""runlog.py -- the four habits worth keeping, without the agent stack.

  1. manifest : every run leaves runs/<utc>_<name>/manifest.json -- params, script sha256,
                timing, structured summary. Written up-front and re-flushed as it fills, so
                a crash still leaves a readable record of what was running.
  2. monitor  : csv_append() appends one timestamped row per sample to a long-lived CSV,
                turning "no arbitrage right now" into a sampled time series.
                capture()/report() tee stdout into report.md with YAML front-matter.
  3. lock     : source()/source_glob() record sha256 digests of every input a run read,
                so a printed number traces back to the exact data that produced it.
  4. secrets  : secret() reads the environment only; every manifest/report/CSV write is
                redacted, so a wallet key cannot land in an artifact (or in an LLM context).

Stdlib only.  Self-check:  python runlog.py --selftest
"""
from __future__ import annotations

import contextlib
import csv
import glob as _glob
import hashlib
import io
import json
import os
import re
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
SCHEMA = 1

# anything whose *name* looks like this is treated as a secret: never written, always masked
SECRET_RE = re.compile(r"(PRIVATE|SECRET|API_?KEY|PASSPHRASE|MNEMONIC|SEED|TOKEN|PASSWORD)", re.I)
MASK = "***"


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


class _Tee:
    """stdout that also lands in the run's report (the 'persist output' step, minus the agent)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass


class Run:
    def __init__(self, name, params=None, root=None):
        self.name = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name)) or "run"
        self.started = _now()
        self.run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "_" + self.name
        self.dir = os.path.join(root or RUNS, self.run_id)
        os.makedirs(self.dir, exist_ok=True)
        # every secret-looking env value is masked everywhere, even if the script read it raw
        self._secrets = {k: v for k, v in os.environ.items() if v and SECRET_RE.search(k)}
        script = sys.argv[0] if sys.argv and sys.argv[0] else None
        self.manifest = {
            "schema": SCHEMA,
            "run_id": self.run_id,
            "name": self.name,
            "started_utc": self.started,
            "script": os.path.relpath(script, os.getcwd()) if script and os.path.isfile(script) else script,
            "script_sha256": sha256_file(script) if script and os.path.isfile(script) else None,
            "python": sys.version.split()[0],
            "cwd": os.getcwd(),
            "params": self.redact(params or {}),
            "sources": {},
            "artifacts": [],
            "secrets_used": sorted(self._secrets),
            "summary": None,
            "duration_s": None,
        }
        self._t0 = time.time()
        self._capture = None
        self._stdout = None
        self._flush()

    # ---- redaction ------------------------------------------------------------
    def redact(self, obj):
        if isinstance(obj, dict):
            return {k: (MASK if SECRET_RE.search(str(k)) else self.redact(v)) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self.redact(v) for v in obj]
        if isinstance(obj, str):
            for v in self._secrets.values():
                if v and v in obj:
                    obj = obj.replace(v, MASK)
            return obj
        return obj

    def secret(self, name, default=None):
        """Env-only secret accessor. The value is registered for masking and never stored."""
        v = os.environ.get(name, default)
        if v is not None:
            self._secrets.setdefault(name, v)
        return v

    def _flush(self):
        p = os.path.join(self.dir, "manifest.json")
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)
        os.replace(tmp, p)

    # ---- 3. hash lock ---------------------------------------------------------
    def source(self, path_or_url):
        key = str(path_or_url)
        if os.path.isfile(key):
            st = os.stat(key)
            rec = {"sha256": sha256_file(key), "bytes": st.st_size,
                   "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds")}
            key = os.path.relpath(key, os.getcwd())
        elif re.match(r"^https?://", key):
            rec = {"url": key, "read_utc": _now()}
        else:
            rec = {"missing": True, "read_utc": _now()}
        self.manifest["sources"][key] = rec
        self._flush()
        return rec

    def source_glob(self, pattern, root="."):
        """Lock a whole input set: one digest over (sha256, path) pairs + a sha256sum-style listing."""
        files = sorted(_glob.glob(os.path.join(root, pattern), recursive=True))
        lines, digest, total = [], hashlib.sha256(), 0
        for p in files:
            if not os.path.isfile(p):
                continue
            d, sz = sha256_file(p), os.path.getsize(p)
            rel = os.path.relpath(p, os.getcwd())
            total += sz
            lines.append("%s  %10d  %s" % (d, sz, rel))
            digest.update(("%s %s\n" % (d, rel)).encode())
        listing = os.path.join(self.dir, "sources.txt")
        with open(listing, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        rec = {"pattern": pattern, "n": len(lines), "bytes": total, "digest": digest.hexdigest()}
        self.manifest["sources"][pattern] = rec
        self.artifact(listing)
        return rec

    def artifact(self, path):
        if os.path.isfile(path):
            rel = os.path.relpath(path, os.getcwd())
            if rel not in self.manifest["artifacts"]:
                self.manifest["artifacts"].append(rel)
            self._flush()
        return path

    # ---- 2. monitor -----------------------------------------------------------
    def csv_append(self, path, row, fields=None):
        """Append one sample to a long-lived CSV (header written once)."""
        row = self.redact(dict(row))
        cols = list(fields or row.keys())
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)
        return self.artifact(path)

    def start_capture(self):
        if self._capture is not None:
            raise RuntimeError("capture already active")
        self._stdout = sys.stdout
        self._capture = io.StringIO()
        sys.stdout = _Tee(self._stdout, self._capture)
        return self._capture

    def stop_capture(self, title=None, **fm):
        buf, self._capture = self._capture, None
        if buf is None:
            return None
        sys.stdout = self._stdout
        return self.report(title or self.name, buf.getvalue(), **fm)

    @contextlib.contextmanager
    def capture(self, title=None, **fm):
        self.start_capture()
        try:
            yield
        finally:
            self.stop_capture(title, **fm)

    def report(self, title, body, **fm):
        """Markdown artifact with YAML front-matter; repeat calls append sections."""
        meta = {"run_id": self.run_id, "name": self.name, "started_utc": self.started}
        meta.update(self.redact(fm))
        p = os.path.join(self.dir, "report.md")
        first = not os.path.exists(p)
        with open(p, "a", encoding="utf-8") as f:
            if first:
                f.write("---\n")
                for k, v in meta.items():
                    f.write("%s: %s\n" % (k, json.dumps(v, ensure_ascii=False)))
                f.write("---\n")
            f.write("\n# %s\n\n%s\n" % (title, self.redact(body)))
        return self.artifact(p)

    # ---- 1. manifest ----------------------------------------------------------
    def finish(self, summary=None):
        self.manifest["duration_s"] = round(time.time() - self._t0, 3)
        self.manifest["finished_utc"] = _now()
        self.manifest["summary"] = self.redact(summary)
        self._flush()
        print("runlog: %s" % os.path.relpath(os.path.join(self.dir, "manifest.json"), os.getcwd()))
        return self.manifest


def _selftest():
    import shutil
    td = os.path.join(HERE, ".selftest_tmp")  # NOT tempfile.mkdtemp: the sandbox locks those dirs
    shutil.rmtree(td, ignore_errors=True)
    os.makedirs(td)
    try:
        os.environ["PM_DUMMY_PRIVATE_KEY"] = "0xdeadbeefcafe1234"
        run = Run("selftest", {"api_key": "0xdeadbeefcafe1234", "cfg": "not-a-secret", "n": 3}, root=td)
        assert run.dir.startswith(td), "run dir must live under root"

        # 3. hash lock
        src = os.path.join(td, "in.csv")
        with open(src, "w", newline="") as f:
            f.write("a,b\n1,2\n")
        rec = run.source(src)
        assert rec["sha256"] == sha256_file(src) and rec["bytes"] == os.path.getsize(src), rec
        assert run.source("https://example.invalid/x")["url"], "url source recorded"
        assert run.source_glob("in.csv", root=td)["n"] == 1
        assert os.path.exists(os.path.join(run.dir, "sources.txt"))

        # 2. monitor: header written once, one row per sample
        snap = os.path.join(td, "snap.csv")
        for i in (1, 2):
            run.csv_append(snap, {"ts": "t%d" % i, "hits": i, "token": "0xdeadbeefcafe1234"})
        body = open(snap).read()
        assert body.count("ts,hits,token") == 1, body
        assert body.count("\n") == 3, body
        assert "0xdeadbeefcafe1234" not in body, "secret leaked into CSV"

        # capture -> report with front-matter, second call appends
        with run.capture("sample output", n=1):
            print("visible line")
        assert run.report("second", "more")
        md = open(os.path.join(run.dir, "report.md")).read()
        assert md.startswith("---\n") and "run_id:" in md and "# sample output" in md and "# second" in md, md
        assert "visible line" in md, "stdout not captured"

        m = run.finish({"hits": 2, "api_key": "0xdeadbeefcafe1234"})["summary"]
        assert m["hits"] == 2 and m["api_key"] == MASK, m
        raw = open(os.path.join(run.dir, "manifest.json")).read()
        assert "0xdeadbeefcafe1234" not in raw, "secret leaked into manifest"
        assert "PM_DUMMY_PRIVATE_KEY" in raw, "secret *name* should be recorded"
        assert "not-a-secret" in raw, "non-secret env must not be masked"
        assert json.loads(raw)["duration_s"] is not None
    finally:
        shutil.rmtree(td, ignore_errors=True)
    print("runlog selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
