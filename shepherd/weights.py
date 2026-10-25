"""Named detector weights: pinned downloads, checksum and pickle checks.

    path = resolve("icaerus-n")   # downloads once into models/, verifies, returns the local path

YOLO .pt files are pickles, and unpickling can run code. Shortcut weights are
pinned to a commit and a SHA-256, and every external file is scanned (without
loading it) for imports outside torch/ultralytics before it is used.
"""

import hashlib
import pickletools
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_ICAERUS = "https://raw.githubusercontent.com/ICAERUS-EU/UC3_Livestock_Monitoring/064412963f112fc39b2c4ceac0f171229a6168ca/models/sheep_detection"


@dataclass(frozen=True)
class Weights:
    url: str
    sha256: str
    imgsz: int  # inference size, chosen on the AUTH-Sheep benchmark
    conf: float
    about: str
    train_imgsz: int = 640  # resolution the weights were trained at, used when fine-tuning


# ICAERUS UC3 Livestock Monitoring (Institut de l'Elevage, GPL-3.0): single-class "sheep"
# detectors for nadir drone video at 5-10 m. See README "Detector choice" for measured quality.
SHORTCUTS = {
    "icaerus": Weights(f"{_ICAERUS}/yolo8m_sheep_v4/yolov8m_all_finetune.pt",
                       "a292c078c27641ccd88e89635059c56ab8b7bef0611326f374e585fed4894600", 640, 0.5,
                       "ICAERUS YOLOv8m sheep v4 (their gold standard)"),
    "icaerus-n": Weights(f"{_ICAERUS}/yolo11n_sheep/yolo11n_all.pt",
                         "666fcf6d87c57ee87a5c001dac4ab35fc7f65be439ddd30a70714e9f4718e95a", 1280, 0.5,
                         "ICAERUS YOLO11n sheep (the default: best on AUTH-Sheep at 1280 px, and fastest)"),
}

# Globals a YOLO checkpoint legitimately needs. Anything else (os, subprocess, builtins.eval,
# ...) means the file can do more than rebuild a model, so it is refused.
_SAFE_PREFIXES = ("torch", "ultralytics", "collections OrderedDict", "__builtin__ set", "builtins set",
                  "numpy", "_codecs encode", "pathlib", "types SimpleNamespace", "argparse Namespace")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pickle_globals(path):
    """Every module.name a torch zip checkpoint imports when unpickled. Reads opcodes only."""
    found = set()
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".pkl"):
                continue
            strings = []
            for op, arg, _ in pickletools.genops(z.read(name)):
                if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINSTRING", "BINSTRING"):
                    strings.append(arg)
                elif op.name in ("GLOBAL", "INST"):
                    found.add(arg)
                elif op.name == "STACK_GLOBAL":
                    found.add(" ".join(strings[-2:]) if len(strings) >= 2 else "?")
    return found


def check_pickle(path):
    """Raise if the checkpoint imports anything outside _SAFE_PREFIXES."""
    try:
        bad = sorted(g for g in pickle_globals(path) if not g.startswith(_SAFE_PREFIXES))
    except zipfile.BadZipFile as e:
        raise ValueError(f"{path} is not a torch zip checkpoint") from e
    if bad:
        raise ValueError(f"refusing {path}: its pickle imports {bad}")


def resolve(model, models_dir=MODELS_DIR):
    """Local path for a shortcut name or a .pt path. Ultralytics' own names (yolo11n.pt) pass through."""
    if model in SHORTCUTS:
        w = SHORTCUTS[model]
        path = Path(models_dir) / f"{model}.pt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".part")
            print(f"downloading {w.about} to {path}")
            urllib.request.urlretrieve(w.url, tmp)
            if sha256(tmp) != w.sha256:
                tmp.unlink()
                raise ValueError(f"checksum mismatch for {w.url}")
            tmp.rename(path)
        elif sha256(path) != w.sha256:
            raise ValueError(f"{path} does not match the pinned checksum; delete it to re-download")
        check_pickle(path)
        return str(path)
    if Path(model).exists():
        check_pickle(model)
    return model


DEFAULT = "icaerus-n"


def defaults(model):
    """(inference imgsz, conf) for a shortcut, or (None, None) for unknown models."""
    w = SHORTCUTS.get(model)
    return (w.imgsz, w.conf) if w else (None, None)


def train_imgsz(model):
    """Resolution to fine-tune a shortcut at, or None for unknown models."""
    w = SHORTCUTS.get(model)
    return w.train_imgsz if w else None
