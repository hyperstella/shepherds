import json
import threading
import urllib.error
import urllib.request

import pytest

from shepherd import box_labeler as BL, finetune as FT


@pytest.fixture
def dataset(tmp_path):
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    for f in (0, 10):
        name = FT.frame_name("drive", f)
        (tmp_path / "images" / f"{name}.jpg").write_bytes(b"\xff\xd8fake")
        (tmp_path / "labels" / f"{name}.txt").write_text("0 0.5 0.5 0.1 0.1\n0 0.2 0.2 0.1 0.1\n")
    (tmp_path / "prelabel.json").write_text(json.dumps({FT.frame_name("drive", 0): [0.9, 0.4]}))
    return tmp_path


@pytest.fixture
def server(dataset):
    srv = BL.make_server(dataset, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def get(url):
    with urllib.request.urlopen(url) as r:
        return r.status, r.read()


def post(url, body, ctype="application/json"):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": ctype})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, None


def test_list_load_save_review(server, dataset):
    name = FT.frame_name("drive", 0)
    status, page = get(server + "/")
    assert status == 200 and b"Box labeler" in page
    frames = json.loads(get(server + "/api/frames")[1])
    assert frames["classes"] == list(FT.CLASSES) and [f["counts"] for f in frames["frames"]] == [[2, 0, 0]] * 2
    d = json.loads(get(f"{server}/api/frame/{name}")[1])
    assert d["conf"] == [0.9, 0.4] and not d["reviewed"] and len(d["labels"]) == 2

    # Relabel one sheep as the dog, drop the other, accept the frame.
    status, r = post(f"{server}/api/frame/{name}", {"labels": [[1, 0.5, 0.5, 0.1, 0.1], [0, 0.3, 0.3, 0, 0.1]], "reviewed": True})
    assert status == 200 and r == {"ok": True, "boxes": 1, "reviewed": True}
    assert FT.read_labels(dataset / "labels" / f"{name}.txt") == [(1, 0.5, 0.5, 0.1, 0.1)]
    d = json.loads(get(f"{server}/api/frame/{name}")[1])
    assert d["reviewed"] and d["conf"] is None  # confidences no longer line up once saved
    assert BL.reviewed_names(dataset) == {name}
    assert get(f"{server}/img/{name}.jpg")[1].startswith(b"\xff\xd8")


def test_refuses_bad_requests(server, dataset):
    name = FT.frame_name("drive", 0)
    for path in ("/api/frame/..%2Fsecret", "/api/frame/nope", "/img/../review.jpg", "/etc/passwd"):
        with pytest.raises(urllib.error.HTTPError) as e:
            get(server + path)
        assert e.value.code == 404
    assert post(f"{server}/api/frame/{name}", {"labels": []}, ctype="text/plain")[0] == 415
    assert post(f"{server}/api/frame/{name}", {"labels": [[7, 0.5, 0.5, 0.1, 0.1]]})[0] == 400
    assert post(f"{server}/api/frame/../x", {"labels": []})[0] == 404
    assert BL.reviewed_names(dataset) is None  # nothing was saved


def test_missing_dataset(tmp_path):
    with pytest.raises(FileNotFoundError):
        BL.Dataset(tmp_path / "nothing")
