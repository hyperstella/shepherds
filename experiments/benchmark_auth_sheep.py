"""Score sheep detectors against AUTH-Sheep's hand-drawn boxes (Zenodo 13843376, CC BY 4.0).

    .venv/bin/python experiments/benchmark_auth_sheep.py DATA_DIR --videos 2 3 4 --every 15
    .venv/bin/python experiments/benchmark_auth_sheep.py DATA_DIR --configs icaerus:1280 yolo11x.pt:1280

DATA_DIR holds videoN.mp4 and videoN_annotations_hbb.json from the Zenodo record. AUTH-Sheep
labels sheep, goats and people from a drone; sheep and goats both count as flock animals to find,
and a box on a person counts as a false positive. Every `--every`-th frame is scored at IoU 0.5.
Writes results/benchmark_auth_sheep.json.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shepherd import video as V, weights  # noqa: E402

FLOCK = {"sheep", "goat"}


def load_gt(path):
    """{frame index (0-based): [(x1, y1, x2, y2, class name)]}, from COCO boxes."""
    d = json.loads(Path(path).read_text())
    names = {c["id"]: c["name"] for c in d["categories"]}
    frame = {im["id"]: int(im["file_name"].rsplit("_", 1)[1]) - 1 for im in d["images"]}
    gt = {f: [] for f in frame.values()}
    for a in d["annotations"]:
        x, y, w, h = a["bbox"]
        gt[frame[a["image_id"]]].append((x, y, x + w, y + h, names[a["category_id"]]))
    return gt


def iou(a, b):
    """IoU between (n, 4) and (m, 4) xyxy arrays."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(rb - lt, 0, None), axis=2)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def match(preds, gts, thr=0.5):
    """Greedy matching by confidence. preds: [(x1,y1,x2,y2,conf)]; gts: [(x1,y1,x2,y2,name)].

    Returns per-prediction (conf, hit, matched gt class or None) and the gt classes left unmatched.
    """
    preds = sorted(preds, key=lambda p: -p[4])
    if not preds:
        return [], [g[4] for g in gts]
    if not gts:
        return [(p[4], False, None) for p in preds], []
    m = iou(np.array([p[:4] for p in preds]), np.array([g[:4] for g in gts]))
    used, out = set(), []
    for i, p in enumerate(preds):
        order = np.argsort(-m[i])
        j = next((int(j) for j in order if m[i, j] >= thr and int(j) not in used), None)
        if j is None:
            out.append((p[4], False, None))
        else:
            used.add(j)
            out.append((p[4], gts[j][4] in FLOCK, gts[j][4]))
    return out, [g[4] for k, g in enumerate(gts) if k not in used]


def ap(scored, n_pos):
    """All-point interpolated average precision from (conf, hit) pairs."""
    if not scored or not n_pos:
        return 0.0
    s = sorted(scored, key=lambda x: -x[0])
    hits = np.array([h for _, h in s], float)
    tp, fp = np.cumsum(hits), np.cumsum(1 - hits)
    rec, prec = tp / n_pos, tp / np.maximum(tp + fp, 1e-9)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    r = np.concatenate([[0], rec])
    return float(np.sum((r[1:] - r[:-1]) * prec))


def frames_of(video, idx, cache):
    """Decode the sampled frames once and keep them as JPEGs."""
    import cv2

    cache.mkdir(parents=True, exist_ok=True)
    out, need = {}, [f for f in idx if not (cache / f"{f:05d}.jpg").exists()]
    if need:
        cap, f, want = cv2.VideoCapture(str(video)), 0, set(need)
        while want:
            ok, img = cap.read()
            if not ok:
                break
            if f in want:
                cv2.imwrite(str(cache / f"{f:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                want.discard(f)
            f += 1
        cap.release()
    for f in idx:
        if (cache / f"{f:05d}.jpg").exists():
            out[f] = cache / f"{f:05d}.jpg"
    return out


def run_config(model_name, imgsz, frames, gt):
    from ultralytics import YOLO

    model = YOLO(weights.resolve(model_name))
    keep = [k for k, kind in V.class_kinds(model.names).items() if kind == "sheep"]
    _, op_conf = weights.defaults(model_name)
    op_conf = op_conf or 0.25
    scored, n_pos, by_class, person_fp, t0 = [], 0, {}, 0, time.time()
    tp_op = fp_op = 0
    for vid, fr in frames.items():
        for f, path in fr.items():
            r = model.predict(str(path), conf=0.01, imgsz=imgsz, classes=keep, verbose=False)[0]
            preds = [(*b, c) for b, c in zip(r.boxes.xyxy.cpu().numpy().tolist(), r.boxes.conf.cpu().numpy().tolist())]
            g = gt[vid][f]
            res, _ = match(preds, g)
            scored += [(c, h) for c, h, _ in res]
            n_pos += sum(1 for x in g if x[4] in FLOCK)
            for c, h, cls in res:
                if c >= op_conf:
                    tp_op += h
                    fp_op += not h
                    person_fp += cls == "human"
                    if h:
                        by_class.setdefault(cls, [0, 0])[0] += 1
            for x in g:
                if x[4] in FLOCK:
                    by_class.setdefault(x[4], [0, 0])[1] += 1
    secs = (time.time() - t0) / max(1, sum(len(v) for v in frames.values()))
    return {
        "model": model_name, "imgsz": imgsz, "conf": op_conf, "frames": sum(len(v) for v in frames.values()),
        "flock_boxes": n_pos, "ap50": round(ap(scored, n_pos), 3),
        "precision": round(tp_op / max(1, tp_op + fp_op), 3), "recall": round(tp_op / max(1, n_pos), 3),
        "recall_by_class": {k: round(v[0] / max(1, v[1]), 3) for k, v in sorted(by_class.items())},
        "boxes_on_people": person_fp, "sec_per_frame": round(secs, 2),
    }


def main(argv=None):
    ap_ = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap_.add_argument("data")
    ap_.add_argument("--videos", type=int, nargs="+", default=[1, 2, 3, 4])
    ap_.add_argument("--every", type=int, default=15, help="score every Nth frame (15 = 2 per second)")
    ap_.add_argument("--configs", nargs="+", default=["icaerus:640", "icaerus:1280", "icaerus:1920", "icaerus-n:1280", "yolo11x.pt:1280"],
                     help="model:imgsz pairs")
    ap_.add_argument("--cache", default=str(ROOT / "results" / "video" / "auth_sheep_frames"))
    ap_.add_argument("--out", default=str(ROOT / "results" / "benchmark_auth_sheep.json"))
    a = ap_.parse_args(argv)
    data = Path(a.data)
    gt, frames = {}, {}
    for v in a.videos:
        mp4, ann = data / f"video{v}.mp4", data / f"video{v}_annotations_hbb.json"
        if not (mp4.exists() and ann.exists()):
            print(f"skipping video{v}: missing {mp4.name} or {ann.name}")
            continue
        gt[v] = load_gt(ann)
        idx = sorted(f for f in gt[v] if f % a.every == 0)
        frames[v] = frames_of(mp4, idx, Path(a.cache) / f"video{v}")
        print(f"video{v}: {len(frames[v])} frames, {sum(len(gt[v][f]) for f in frames[v])} boxes")
    results = []
    for cfg in a.configs:
        name, sz = cfg.rsplit(":", 1)
        res = run_config(name, int(sz), frames, gt)
        print(json.dumps(res), flush=True)
        results.append(res)
        # Written after every config, so a long run keeps what it has finished.
        Path(a.out).write_text(json.dumps({"videos": a.videos, "every": a.every, "results": results}, indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
