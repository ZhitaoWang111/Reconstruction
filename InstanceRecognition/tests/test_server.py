import json
from pathlib import Path

from web.server import create_app


SCANS = Path("/home/wzt/HDD/datasets/scannet_v2/scans")


def test_scene_and_frame_endpoints(tmp_path):
    app = create_app(SCANS, tmp_path)
    client = app.test_client()

    response = client.get("/api/scenes")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["count"] == 20
    assert payload["scenes"][0]["scene_id"] == "scene0000_00"

    response = client.get("/api/frame/scene0001_00/73.jpg")
    assert response.status_code == 200
    assert response.mimetype == "image/jpeg"
    assert response.data.startswith(b"\xff\xd8")


def test_api_rejects_unsafe_or_out_of_range_input(tmp_path):
    app = create_app(SCANS, tmp_path)
    client = app.test_client()
    assert client.get("/api/frame/not-a-scene/0.jpg").status_code == 400
    assert client.get("/api/frame/scene0001_00/999999.jpg").status_code == 400
    response = client.post("/api/infer", json={"scene_id": "scene0001_00"})
    assert response.status_code == 400


def test_existing_run_is_reported_as_cached(tmp_path):
    run_root = tmp_path / "results/scene0001_00/frame_000073"
    (run_root / "yoloe").mkdir(parents=True)
    (run_root / "sam31").mkdir()
    for relative in (
        "yoloe/annotated.jpg",
        "sam31/annotated.jpg",
        "sam31/instance_id_map.png",
    ):
        (run_root / relative).touch()
    summary = {
        "scene_id": "scene0001_00",
        "frame_index": 73,
        "cached": False,
        "wall_seconds": 12.0,
        "contract": {"text_prompt": None, "yoloe_masks_consumed": False},
        "yoloe": {
            "native_masks_forwarded": False,
            "instance_count": 0,
            "unique_classes": [],
            "timing": {},
            "instances": [],
            "artifacts": {"annotated_image": "annotated.jpg"},
        },
        "sam31": {
            "yoloe_masks_consumed": False,
            "instance_count": 0,
            "unique_classes": [],
            "timing": {},
            "instances": [],
            "artifacts": {
                "annotated_image": "annotated.jpg",
                "instance_id_map": "instance_id_map.png",
            },
        },
    }
    (run_root / "summary.json").write_text(json.dumps(summary))
    client = create_app(SCANS, tmp_path).test_client()
    response = client.get("/api/run/scene0001_00/73")
    assert response.status_code == 200
    assert response.get_json()["cached"] is True
