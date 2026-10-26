import os
import pickle
import zipfile

import pytest

from shepherd import finetune as FT, weights as W


def test_label_lines_round_trip(tmp_path):
    lines = FT.label_lines([(0, 720, 540, 144, 108), (1, 1430, 5, 40, 30)], 1440, 1080)
    (tmp_path / "a.txt").write_text("\n".join(lines) + "\n")
    rows = FT.read_labels(tmp_path / "a.txt")
    assert rows[0] == (0, 0.5, 0.5, 0.1, 0.1)
    assert rows[1][0] == 1 and rows[1][1] == pytest.approx(1430 / 1440, abs=1e-6)
    assert FT.read_labels(tmp_path / "missing.txt") == []


def test_split_keeps_videos_apart():
    images = [f"images/{FT.frame_name(v, f)}.jpg" for v in "abcde" for f in range(0, 50, 10)]
    train, val = FT.split_by_video(images, 0.2, seed=1)
    assert len(train) + len(val) == len(images) and val
    assert not {FT.video_of(p) for p in train} & {FT.video_of(p) for p in val}


def test_split_single_video_holds_out_the_end():
    images = [f"{FT.frame_name('drive', f)}.jpg" for f in range(10)]
    train, val = FT.split_by_video(images, 0.2)
    assert train == images[:8] and val == images[8:]


def test_data_yaml(tmp_path):
    y = FT.write_data_yaml(tmp_path, ["x/a.jpg"], ["x/b.jpg"])
    text = y.read_text()
    assert "0: sheep" in text and "1: dog" in text and "2: person" in text
    assert (tmp_path / "val.txt").read_text().strip().endswith("b.jpg")


def _checkpoint(path, obj, protocol):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("best/data.pkl", pickle.dumps(obj, protocol=protocol))
    return path


@pytest.mark.parametrize("protocol", [2, 4])
def test_check_pickle_refuses_code(tmp_path, protocol):
    bad = _checkpoint(tmp_path / "bad.pt", os.system, protocol)
    with pytest.raises(ValueError, match="system"):
        W.check_pickle(bad)
    ok = _checkpoint(tmp_path / "ok.pt", {"epoch": 3, "names": {0: "sheep"}}, protocol)
    W.check_pickle(ok)


def test_check_pickle_rejects_non_zip(tmp_path):
    (tmp_path / "x.pt").write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="zip"):
        W.check_pickle(tmp_path / "x.pt")


def test_resolve_downloads_and_verifies(tmp_path, monkeypatch):
    src = _checkpoint(tmp_path / "src.pt", {"names": {0: "sheep"}}, 4)
    good = W.Weights(src.as_uri(), W.sha256(src), 640, 0.5, "test")
    monkeypatch.setitem(W.SHORTCUTS, "test-model", good)
    path = W.resolve("test-model", tmp_path / "models")
    assert W.sha256(path) == good.sha256 and W.defaults("test-model") == (640, 0.5)
    monkeypatch.setitem(W.SHORTCUTS, "test-bad", W.Weights(src.as_uri(), "0" * 64, 640, 0.5, "bad"))
    with pytest.raises(ValueError, match="checksum"):
        W.resolve("test-bad", tmp_path / "models")
    assert not (tmp_path / "models" / "test-bad.pt").exists()
    assert W.resolve("yolo11n.pt") == "yolo11n.pt" and W.defaults("yolo11n.pt") == (None, None)


def test_split_two_frames_and_counts(tmp_path):
    images = [f"{FT.frame_name('drive', f)}.jpg" for f in (0, 5)]
    assert FT.split_by_video(images, 0.2) == (images[:1], images[1:])
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")
    (tmp_path / "b.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    assert FT.class_counts(tmp_path) == {"sheep": 2, "dog": 1, "person": 0}
    assert FT.class_counts(tmp_path, {"b"}) == {"sheep": 1, "dog": 0, "person": 0}


def test_video_of_keeps_dots_in_video_names():
    name = FT.frame_name("23.11.23-15", 807)
    assert FT.video_of(name) == "23.11.23-15"
    assert FT.video_of(f"images/{name}.jpg") == "23.11.23-15"
    assert FT.video_of(f"{name}.txt") == "23.11.23-15"
    assert FT.video_of("plain.jpg") == "plain"
