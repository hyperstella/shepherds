"""Label a herding video with a local detector: clean tracks, flock series and draft phases.

    .venv/bin/pip install -r requirements-video.txt      # once: ultralytics (YOLO + ByteTrack)
    .venv/bin/python experiments/label_video.py drive.mp4
    .venv/bin/python experiments/label_video.py drive.mp4 --px-per-m 18.5 --session 3 --offset 412.0
    .venv/bin/python experiments/label_video.py --from-detections results/video/drive_detections.csv

Writes to results/video/ (or --out):
    <name>_detections.csv  raw tracked boxes, so cleaning can be re-run without the model
    <name>_tracks.csv      cleaned tracks: one row per track per 0.1 s, metres and frame position
    <name>_flock.csv       flock centre, velocity, cohesion, sheep in view, dog and shepherd
    <name>_phases.csv      draft phase segments; open in viz/labeler.html with Import CSV to review

Units: --homography (a JSON 3x3 pixel -> metre matrix) for oblique footage, --px-per-m for
top-down footage, or neither to guess the scale from sheep box sizes (top-down only).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shepherd import video as V  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="video file")
    ap.add_argument("--from-detections", help="skip detection and clean this _detections.csv")
    ap.add_argument("--model", default="yolo11n.pt", help="YOLO weights (COCO-pretrained by default; a fine-tuned model must keep the sheep/dog/person class ids)")
    ap.add_argument("--conf", type=float, default=0.25, help="detection confidence threshold")
    ap.add_argument("--imgsz", type=int, default=1280, help="inference size; larger finds smaller sheep but runs slower")
    ap.add_argument("--stride", type=int, default=1, help="run the detector on every Nth frame")
    ap.add_argument("--device", default=None, help="e.g. cpu, 0 (GPU), mps")
    ap.add_argument("--px-per-m", type=float, default=None)
    ap.add_argument("--homography", help="JSON file with a 3x3 pixel -> ground-metre matrix")
    ap.add_argument("--min-track-s", type=float, default=1.0)
    ap.add_argument("--max-gap-s", type=float, default=1.0)
    ap.add_argument("--smooth-s", type=float, default=0.5)
    ap.add_argument("--session", default="", help="tracking session number, if this video has one")
    ap.add_argument("--offset", type=float, default=0.0, help="tracking time = video time + offset")
    ap.add_argument("--out", default=str(ROOT / "results" / "video"))
    a = ap.parse_args(argv)
    if not a.video and not a.from_detections:
        ap.error("give a video or --from-detections")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.from_detections:
        src = Path(a.from_detections)
        name = src.name.removesuffix("_detections.csv")
        det = V.read_detections(src)
        video_name = a.video and Path(a.video).name or name
    else:
        src = Path(a.video)
        name, video_name = src.stem, src.name
        print(f"detecting in {src} with {a.model}")
        det = V.detect(src, a.model, a.conf, a.imgsz, a.stride, device=a.device, progress=True)
        V.write_detections(det, out / f"{name}_detections.csv")
    kinds = {k: int((det.kind == k).sum()) for k in V.KINDS}
    print(f"{len(det.frame)} boxes in {len(set(det.track.tolist()))} tracks: {kinds}")

    H = json.loads(Path(a.homography).read_text()) if a.homography else None
    try:
        tracks = V.clean_tracks(det, a.px_per_m, H, min_track_s=a.min_track_s, max_gap_s=a.max_gap_s, smooth_s=a.smooth_s)
    except ValueError as e:
        sys.exit(f"cannot clean tracks: {e}")
    for note in tracks.notes:
        print("note:", note)
    flock = V.flock_series(tracks)
    lab, segs = V.label_phases(flock)

    V.write_tracks(tracks, out / f"{name}_tracks.csv", a.offset)
    V.write_flock(flock, out / f"{name}_flock.csv", a.offset)
    V.write_phases(segs, out / f"{name}_phases.csv", a.session, a.offset, video_name)
    n_kind = {k: tracks.kinds.count(k) for k in V.KINDS}
    print(f"{len(tracks.ids)} clean tracks {n_kind}, {len(segs)} segments over {tracks.t[-1]:.1f} s")
    print(f"wrote {out}/{name}_{{tracks,flock,phases}}.csv")


if __name__ == "__main__":
    main()
