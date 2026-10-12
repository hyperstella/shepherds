"""Local server behind viz/box_labeler.html: reads and writes a fine-tuning dataset in place.

    serve("data/finetune")       # http://127.0.0.1:8765, Ctrl+C to stop

The dataset is the finetune.py layout. Review state lives in <dir>/review.json
({frame: {"reviewed": true, "time": ...}}), and pre-label confidences in
<dir>/prelabel.json ({frame: [conf per label line]}), which the page uses to
dash the boxes the model was unsure of until the frame is first saved.

It binds to localhost only and serves nothing outside the dataset and the page.
"""

import json
import re
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import finetune as FT

PAGE = Path(__file__).resolve().parent.parent / "viz" / "box_labeler.html"
NAME = re.compile(r"^[\w.\-]+$")  # frame names: no slashes, so no path traversal
MAX_BODY = 2_000_000


class Dataset:
    def __init__(self, root):
        self.root = Path(root)
        self.images = self.root / "images"
        self.labels = self.root / "labels"
        if not self.images.is_dir():
            raise FileNotFoundError(f"{self.images} not found; run finetune_detector.py prelabel first")
        self.labels.mkdir(exist_ok=True)
        self.review = _read_json(self.root / "review.json")
        self.prelabel = _read_json(self.root / "prelabel.json")

    def names(self):
        return sorted(p.stem for p in self.images.glob("*.jpg"))

    def _check(self, name):
        if not NAME.match(name) or not (self.images / f"{name}.jpg").exists():
            raise KeyError(name)

    def summary(self):
        out = []
        for n in self.names():
            counts = [0] * len(FT.CLASSES)
            for c, *_ in FT.read_labels(self.labels / f"{n}.txt"):
                if 0 <= c < len(counts):
                    counts[c] += 1
            out.append({"name": n, "video": FT.video_of(n), "reviewed": bool(self.review.get(n, {}).get("reviewed")),
                        "counts": counts})
        return {"classes": list(FT.CLASSES), "frames": out}

    def load(self, name):
        self._check(name)
        rec = self.review.get(name, {})
        # Confidences only line up with the label file until the labeler first rewrites it.
        conf = None if rec.get("saved") else self.prelabel.get(name)
        return {"name": name, "labels": [list(r) for r in FT.read_labels(self.labels / f"{name}.txt")],
                "conf": conf, "reviewed": bool(rec.get("reviewed"))}

    def save(self, name, labels, reviewed=None):
        self._check(name)
        lines = []
        for row in labels:
            c, cx, cy, w, h = row
            c = int(c)
            if not 0 <= c < len(FT.CLASSES):
                raise ValueError(f"class {c} out of range")
            vals = [min(max(float(v), 0.0), 1.0) for v in (cx, cy, w, h)]
            if vals[2] <= 0 or vals[3] <= 0:
                continue
            lines.append(f"{c} " + " ".join(f"{v:.6f}" for v in vals))
        (self.labels / f"{name}.txt").write_text("".join(l + "\n" for l in lines))
        rec = self.review.setdefault(name, {})
        rec["saved"] = True
        if reviewed is not None:
            rec["reviewed"] = bool(reviewed)
        rec["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = self.root / "review.json.tmp"
        tmp.write_text(json.dumps(self.review, indent=1, sort_keys=True))
        tmp.replace(self.root / "review.json")
        return {"ok": True, "boxes": len(lines), "reviewed": bool(rec.get("reviewed"))}


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def reviewed_names(root):
    """Frames marked reviewed in <root>/review.json, or None if the labeler was never used."""
    rev = _read_json(Path(root) / "review.json")
    return None if not rev else {n for n, r in rev.items() if r.get("reviewed")}


def make_handler(ds):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the terminal quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            try:
                if path in ("/", "/index.html"):
                    return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                if path == "/api/frames":
                    return self._send(200, ds.summary())
                if path.startswith("/api/frame/"):
                    return self._send(200, ds.load(path[len("/api/frame/"):]))
                if path.startswith("/img/") and path.endswith(".jpg"):
                    name = path[len("/img/"):-4]
                    ds._check(name)
                    return self._send(200, (ds.images / f"{name}.jpg").read_bytes(), "image/jpeg")
            except KeyError:
                return self._send(404, {"error": "no such frame"})
            self._send(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.split("?", 1)[0]
            if not path.startswith("/api/frame/"):
                return self._send(404, {"error": "not found"})
            # Same-origin only: the page posts JSON from this server; other sites cannot.
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                return self._send(415, {"error": "expected application/json"})
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._send(413, {"error": "too large"})
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
                return self._send(200, ds.save(path[len("/api/frame/"):], body.get("labels", []), body.get("reviewed")))
            except KeyError:
                return self._send(404, {"error": "no such frame"})
            except (ValueError, TypeError) as e:
                return self._send(400, {"error": str(e)})

    return Handler


def make_server(root, port=8765):
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(Dataset(root)))
