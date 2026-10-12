"""Dataset helpers for fine-tuning a sheep/dog/person detector on your own footage.

The dataset uses the Ultralytics YOLO layout, which CVAT ("Ultralytics YOLO
Detection") and Label Studio (YOLO with classes.txt) both import and export:

    <dir>/images/<video>__f<frame>.jpg
    <dir>/labels/<video>__f<frame>.txt   one "class cx cy w h" line per box, normalised 0..1
    <dir>/classes.txt                    sheep, dog, person
    <dir>/data.yaml                      written by write_data_yaml for training

Frame names carry their video, so the train/validation split can keep whole
videos apart: frames from one drive are near-duplicates, and splitting them
across train and val would overstate accuracy.
"""

import random
from pathlib import Path

CLASSES = ("sheep", "dog", "person")  # video.NAMES maps these to sheep / dog / shepherd
SEP = "__f"


def frame_name(video_stem, frame):
    return f"{video_stem}{SEP}{frame:06d}"


def video_of(name):
    """Video stem from an image or label file name."""
    stem = Path(name).stem
    return stem.rsplit(SEP, 1)[0] if SEP in stem else stem


def label_lines(boxes, width, height):
    """YOLO label lines from (class_id, cx, cy, w, h) boxes in pixels."""
    out = []
    for c, x, y, w, h in boxes:
        cx, cy = min(max(x / width, 0.0), 1.0), min(max(y / height, 0.0), 1.0)
        out.append(f"{int(c)} {cx:.6f} {cy:.6f} {min(w / width, 1.0):.6f} {min(h / height, 1.0):.6f}")
    return out


def read_labels(path):
    """[(class_id, cx, cy, w, h)] normalised, from a YOLO label file (missing file = no boxes)."""
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            rows.append((int(parts[0]), *map(float, parts[1:5])))
    return rows


def split_by_video(images, val_frac=0.2, seed=0):
    """(train, val) image lists with every video wholly on one side.

    With a single video there is nothing to hold out, so frames are split by
    time instead (the last val_frac of frames), which is the least leaky option.
    """
    images = sorted(map(str, images))
    videos = sorted({video_of(p) for p in images})
    if len(videos) == 1:
        k = max(1, round(len(images) * (1 - val_frac))) if len(images) > 1 else len(images)
        return images[:k], images[k:]
    rng = random.Random(seed)
    rng.shuffle(videos)
    n_val = min(len(videos) - 1, max(1, round(len(videos) * val_frac)))
    val_videos = set(videos[:n_val])
    return [p for p in images if video_of(p) not in val_videos], [p for p in images if video_of(p) in val_videos]


def class_counts(label_dir):
    counts = {c: 0 for c in CLASSES}
    for f in Path(label_dir).glob("*.txt"):
        for c, *_ in read_labels(f):
            if 0 <= c < len(CLASSES):
                counts[CLASSES[c]] += 1
    return counts


def write_data_yaml(dataset, train, val):
    """train.txt, val.txt and data.yaml for ultralytics; returns the yaml path."""
    dataset = Path(dataset).resolve()
    (dataset / "train.txt").write_text("".join(f"{Path(p).resolve()}\n" for p in train))
    (dataset / "val.txt").write_text("".join(f"{Path(p).resolve()}\n" for p in val))
    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    path = dataset / "data.yaml"
    path.write_text(f"path: {dataset}\ntrain: train.txt\nval: val.txt\nnames:\n{names}\n")
    return path
