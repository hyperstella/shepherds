"""Fine-tune a sheep/dog/person detector on your own herding footage.

Off-the-shelf models miss the dog in top-down video (see README, "Detector choice"), so the
practical route is a few hundred hand-checked frames and a short fine-tune from ICAERUS v4.

    # 1. sample frames and pre-draw sheep boxes with the base model
    .venv/bin/python experiments/finetune_detector.py prelabel drive1.mp4 drive2.mp4 --out data/finetune

    # 2. fix the sheep boxes and add every dog and person, saving as you go
    .venv/bin/python experiments/box_labeler.py data/finetune
    #    (or use CVAT "Ultralytics YOLO Detection" / Label Studio YOLO and export over labels/)

    # 3. train, holding out whole videos for validation
    .venv/bin/python experiments/finetune_detector.py train data/finetune --epochs 60

    # 4. compare models on the held-out videos, then label with the winner
    .venv/bin/python experiments/finetune_detector.py eval data/finetune --model icaerus
    .venv/bin/python experiments/finetune_detector.py eval data/finetune --model runs/finetune/sheepdog/weights/best.pt
    .venv/bin/python experiments/label_video.py drive3.mp4 --model runs/finetune/sheepdog/weights/best.pt

Labelling guide: box every sheep, dog and person that is at least half visible; label frames
where the dog is in shot first, since the dog class is what the base model lacks. Spread frames
across videos, light and altitude rather than taking many from one drive.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shepherd import box_labeler as BL, finetune as FT, video as V, weights  # noqa: E402


def prelabel(a):
    import cv2
    from ultralytics import YOLO

    imgsz, conf = weights.defaults(a.base)
    model = YOLO(weights.resolve(a.base))
    kinds = V.class_kinds(model.names)
    to_class = {k: FT.CLASSES.index({"shepherd": "person"}.get(kind, kind)) for k, kind in kinds.items()}
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("\n".join(FT.CLASSES) + "\n")
    total, confidences = 0, {}
    for video in a.videos:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, round(a.every_s * fps))
        frames = list(range(0, n, step))[: a.max_frames]
        for f in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            name = FT.frame_name(Path(video).stem, f)
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            r = model.predict(img, conf=a.conf or conf or 0.25, imgsz=a.imgsz or imgsz or 1280, verbose=False)[0]
            keep = [(to_class[int(c)], *xywh, float(p)) for xywh, c, p in
                    zip(r.boxes.xywh.cpu().numpy(), r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()) if int(c) in to_class]
            boxes = [k[:5] for k in keep]
            confidences[name] = [round(k[5], 3) for k in keep]
            (out / "labels" / f"{name}.txt").write_text("".join(l + "\n" for l in FT.label_lines(boxes, img.shape[1], img.shape[0])))
            total += 1
        cap.release()
        print(f"{video}: {len(frames)} frames every {step} frames")
    # Lets the box labeler dash the boxes the model was unsure of.
    old = BL._read_json(out / "prelabel.json")
    (out / "prelabel.json").write_text(json.dumps(old | confidences))
    print(f"wrote {total} frames to {out}; pre-drawn boxes: {FT.class_counts(out / 'labels')}")
    print(f"next: experiments/box_labeler.py {out}  (fix boxes, add dogs and people), then run `train`")


def _split(a):
    images = sorted((Path(a.dataset) / "images").glob("*.jpg"))
    if not images:
        sys.exit(f"no images in {a.dataset}/images")
    # Unchecked pre-labels teach the model its own mistakes (e.g. the dog boxed as a sheep),
    # so once the box labeler has been used only accepted frames are trained and scored on.
    reviewed = BL.reviewed_names(a.dataset)
    if reviewed is not None and not a.include_unreviewed:
        skipped = len(images)
        images = [p for p in images if p.stem in reviewed]
        print(f"using {len(images)} reviewed frames, skipping {skipped - len(images)} unreviewed (--include-unreviewed to keep them)")
        if not images:
            sys.exit("no reviewed frames yet: accept some in experiments/box_labeler.py")
    train, val = FT.split_by_video(images, a.val_frac, a.seed)
    if not val:
        sys.exit("need at least two frames to hold some out for validation")
    yaml = FT.write_data_yaml(a.dataset, train, val)
    print(f"{len(train)} train / {len(val)} val frames "
          f"({len({FT.video_of(p) for p in train})} / {len({FT.video_of(p) for p in val})} videos); "
          f"boxes {FT.class_counts(Path(a.dataset) / 'labels', {Path(p).stem for p in images})}")
    return yaml


def train(a):
    from ultralytics import YOLO

    yaml = _split(a)
    counts = FT.class_counts(Path(a.dataset) / "labels", {Path(p).stem for p in yaml.parent.joinpath("train.txt").read_text().split()})
    if counts["dog"] == 0:
        print("warning: no dog boxes yet, so the model will not learn to find the dog")
    imgsz, _ = weights.defaults(a.base)
    model = YOLO(weights.resolve(a.base))
    model.train(data=str(yaml), epochs=a.epochs, imgsz=a.imgsz or imgsz or 640, batch=a.batch, device=a.device,
                project=str(Path(a.project).resolve()), name=a.name, exist_ok=True, patience=a.patience, seed=a.seed,
                fliplr=0.5, flipud=0.5, degrees=180.0)  # nadir footage has no up: rotate and flip freely
    best = Path(a.project) / a.name / "weights" / "best.pt"
    print(f"best weights: {best}")
    print(f"next: experiments/finetune_detector.py eval {a.dataset} --model {best}")


def evaluate(a):
    from ultralytics import YOLO

    yaml = _split(a)
    imgsz, conf = weights.defaults(a.model)
    model = YOLO(weights.resolve(a.model))
    kinds = V.class_kinds(model.names)
    # A single-class sheep model is scored on sheep only; its missing dog class is reported, not hidden.
    m = model.val(data=str(yaml), split="val", imgsz=a.imgsz or imgsz or 640, conf=0.001, device=a.device,
                  plots=False, verbose=False, project=str(Path(a.project).resolve()), name="eval", exist_ok=True,
                  classes=list(kinds))
    print(f"\n{a.model} on held-out frames ({', '.join(f'{n}={k}' for n, k in model.names.items())})")
    print(f"{'class':8s} {'P':>6s} {'R':>6s} {'mAP50':>6s}")
    for i, c in enumerate(m.box.ap_class_index):
        p, r, ap50, _ = m.box.class_result(i)
        print(f"{model.names[int(c)]:8s} {p:6.3f} {r:6.3f} {ap50:6.3f}")
    unscored = [model.names[k] for k in kinds if k not in set(map(int, m.box.ap_class_index))]
    if unscored:
        print(f"no held-out boxes to score: {', '.join(unscored)} (hold out a video where they appear)")
    missing = [k for k in ("sheep", "dog", "shepherd") if k not in kinds.values()]
    if missing:
        print(f"not detectable by this model: {', '.join(missing)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prelabel", help="sample frames and pre-draw boxes with a base model")
    p.add_argument("videos", nargs="+")
    p.add_argument("--out", default=str(ROOT / "data" / "finetune"))
    p.add_argument("--base", default="icaerus", help="model that pre-draws boxes (shortcut or .pt)")
    p.add_argument("--every-s", type=float, default=2.0, help="seconds between sampled frames")
    p.add_argument("--max-frames", type=int, default=150, help="per video")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.set_defaults(fn=prelabel)
    for name, fn in (("train", train), ("eval", evaluate)):
        p = sub.add_parser(name)
        p.add_argument("dataset")
        p.add_argument("--val-frac", type=float, default=0.2, help="fraction of videos held out")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--imgsz", type=int, default=None)
        p.add_argument("--device", default=None, help="e.g. cpu, 0 (GPU), mps")
        p.add_argument("--project", default=str(ROOT / "runs" / "finetune"))
        p.add_argument("--include-unreviewed", action="store_true", help="also use frames not accepted in the box labeler")
        p.set_defaults(fn=fn)
        if name == "train":
            p.add_argument("--base", default="icaerus", help="starting weights (shortcut or .pt)")
            p.add_argument("--epochs", type=int, default=60)
            p.add_argument("--batch", type=int, default=8)
            p.add_argument("--patience", type=int, default=20)
            p.add_argument("--name", default="sheepdog")
        else:
            p.add_argument("--model", required=True, help="weights to score (shortcut or .pt)")
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
