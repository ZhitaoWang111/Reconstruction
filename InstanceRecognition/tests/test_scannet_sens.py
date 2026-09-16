from pathlib import Path

from PIL import Image

from src.scannet_sens import ScanNetSensReader, read_header


SCANS = Path("/home/wzt/HDD/datasets/scannet_v2/scans")


def test_reads_real_scannet_header():
    header = read_header(SCANS / "scene0000_00/scene0000_00.sens")
    assert header.version == 4
    assert header.frame_count == 5578
    assert (header.color_width, header.color_height) == (1296, 968)


def test_extracts_arbitrary_rgb_frame(tmp_path):
    reader = ScanNetSensReader(SCANS, tmp_path)
    frame = reader.frame_path("scene0001_00", 73)
    assert frame.is_file()
    with Image.open(frame) as image:
        assert image.format == "JPEG"
        assert image.size == (1296, 968)
    assert reader.frame_path("scene0001_00", 73) == frame


def test_rejects_invalid_scene_and_frame(tmp_path):
    reader = ScanNetSensReader(SCANS, tmp_path)
    try:
        reader.frame_path("../../etc", 0)
        raise AssertionError("invalid scene should fail")
    except ValueError:
        pass
    try:
        reader.frame_path("scene0001_00", 999999)
        raise AssertionError("out-of-range frame should fail")
    except IndexError:
        pass
