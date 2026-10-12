"""Open a fine-tuning dataset in the box labeler (viz/box_labeler.html) and save as you go.

    .venv/bin/python experiments/box_labeler.py data/finetune          # opens http://127.0.0.1:8765

The model's pre-drawn boxes are shown; fix them (1/2/3 sets sheep/dog/person, X deletes,
drag draws) and press Enter to accept a frame. Labels are written straight to
<dataset>/labels/, and accepted frames are recorded in <dataset>/review.json, which
finetune_detector.py train/eval use to skip frames nobody has checked. Ctrl+C stops it.
"""

import argparse
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from shepherd import box_labeler as BL  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", nargs="?", default=str(ROOT / "data" / "finetune"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args(argv)
    try:
        server = BL.make_server(a.dataset, a.port)
    except FileNotFoundError as e:
        sys.exit(str(e))
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"labelling {a.dataset} at {url}  (Ctrl+C to stop)")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped; labels are saved")


if __name__ == "__main__":
    main()
