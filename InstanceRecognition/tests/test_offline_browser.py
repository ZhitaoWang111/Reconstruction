from pathlib import Path

from web.build_offline_browser import build_index


ROOT = Path(__file__).resolve().parent.parent


def test_offline_index_contains_every_completed_frame():
    results_root = ROOT / "runtime/results"
    index = build_index(results_root, ROOT / "offline")
    summary_count = len(list(results_root.glob("*/frame_*/summary.json")))

    assert index["frame_count"] == summary_count
    assert sum(scene["frame_count"] for scene in index["scenes"]) == summary_count
    assert all(scene["frames"] for scene in index["scenes"])
    assert all(
        frame["source"].startswith("../runtime/frames/")
        for scene in index["scenes"]
        for frame in scene["frames"]
    )


def test_offline_page_uses_only_relative_local_assets():
    html = (ROOT / "offline/index.html").read_text()

    assert 'href="styles.css"' in html
    assert 'src="data.js"' in html
    assert 'src="episode_stats.js"' in html
    assert 'src="app.js"' in html
    assert 'src="/assets/' not in html
    assert "fetch(" not in (ROOT / "offline/app.js").read_text()
    assert "整个 episode 识别汇总" in html
