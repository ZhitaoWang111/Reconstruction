"""Small, dependency-light reader for RGB frames inside ScanNet v2 .sens files."""

from __future__ import annotations

import io
import os
import re
import struct
import threading
from dataclasses import asdict, dataclass
from pathlib import Path

from PIL import Image


SCENE_PATTERN = re.compile(r"scene\d{4}_\d{2}\Z")
SENS_VERSION = 4
JPEG_COMPRESSION = 2


@dataclass(frozen=True)
class SensHeader:
    scene_id: str
    sensor_name: str
    version: int
    color_compression: int
    depth_compression: int
    color_width: int
    color_height: int
    depth_width: int
    depth_height: int
    depth_shift: float
    frame_count: int
    frame_data_offset: int

    def public_dict(self) -> dict:
        value = asdict(self)
        value.pop("frame_data_offset")
        return value


def _read_exact(handle, size: int) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"Unexpected end of .sens file while reading {size} bytes")
    return data


def read_header(sens_path: Path) -> SensHeader:
    sens_path = Path(sens_path)
    with sens_path.open("rb") as handle:
        version = struct.unpack("<I", _read_exact(handle, 4))[0]
        name_length = struct.unpack("<Q", _read_exact(handle, 8))[0]
        if name_length > 4096:
            raise ValueError(f"Invalid sensor-name length in {sens_path}")
        sensor_name = _read_exact(handle, name_length).decode("utf-8", errors="replace")
        _read_exact(handle, 64 * 4)  # color/depth intrinsic and extrinsic matrices
        color_compression = struct.unpack("<i", _read_exact(handle, 4))[0]
        depth_compression = struct.unpack("<i", _read_exact(handle, 4))[0]
        color_width, color_height, depth_width, depth_height = struct.unpack(
            "<IIII", _read_exact(handle, 16)
        )
        depth_shift = struct.unpack("<f", _read_exact(handle, 4))[0]
        frame_count = struct.unpack("<Q", _read_exact(handle, 8))[0]
        frame_data_offset = handle.tell()
    if version != SENS_VERSION:
        raise ValueError(f"Unsupported ScanNet .sens version {version}: {sens_path}")
    if color_compression != JPEG_COMPRESSION:
        raise ValueError(
            f"Expected JPEG-compressed ScanNet RGB frames, got {color_compression}"
        )
    return SensHeader(
        scene_id=sens_path.stem,
        sensor_name=sensor_name,
        version=version,
        color_compression=color_compression,
        depth_compression=depth_compression,
        color_width=color_width,
        color_height=color_height,
        depth_width=depth_width,
        depth_height=depth_height,
        depth_shift=depth_shift,
        frame_count=frame_count,
        frame_data_offset=frame_data_offset,
    )


class ScanNetSensReader:
    def __init__(self, scans_root: Path, cache_root: Path):
        self.scans_root = Path(scans_root).expanduser().resolve()
        self.cache_root = Path(cache_root).expanduser().resolve()
        self._headers: dict[str, SensHeader] = {}
        self._cache_lock = threading.Lock()

    def _validate_scene_id(self, scene_id: str) -> None:
        if not SCENE_PATTERN.fullmatch(scene_id):
            raise ValueError(f"Invalid ScanNet scene id: {scene_id!r}")

    def sens_path(self, scene_id: str) -> Path:
        self._validate_scene_id(scene_id)
        path = self.scans_root / scene_id / f"{scene_id}.sens"
        if not path.is_file():
            raise FileNotFoundError(f"ScanNet scene is not available locally: {scene_id}")
        return path

    def header(self, scene_id: str) -> SensHeader:
        self._validate_scene_id(scene_id)
        if scene_id not in self._headers:
            self._headers[scene_id] = read_header(self.sens_path(scene_id))
        return self._headers[scene_id]

    def list_scenes(self) -> list[dict]:
        if not self.scans_root.is_dir():
            raise FileNotFoundError(f"ScanNet scans root not found: {self.scans_root}")
        scene_ids = sorted(
            path.name
            for path in self.scans_root.iterdir()
            if path.is_dir() and SCENE_PATTERN.fullmatch(path.name)
            and (path / f"{path.name}.sens").is_file()
        )
        return [self.header(scene_id).public_dict() for scene_id in scene_ids]

    def _read_color_bytes(self, scene_id: str, frame_index: int) -> bytes:
        header = self.header(scene_id)
        if frame_index < 0 or frame_index >= header.frame_count:
            raise IndexError(
                f"Frame {frame_index} is outside {scene_id}'s range "
                f"0..{header.frame_count - 1}"
            )
        with self.sens_path(scene_id).open("rb") as handle:
            handle.seek(header.frame_data_offset)
            for index in range(frame_index + 1):
                _read_exact(handle, 64 + 8 + 8)
                color_size, depth_size = struct.unpack("<QQ", _read_exact(handle, 16))
                if color_size > 100_000_000 or depth_size > 100_000_000:
                    raise ValueError(f"Implausible frame payload size in {scene_id}")
                if index == frame_index:
                    return _read_exact(handle, color_size)
                handle.seek(color_size + depth_size, 1)
        raise RuntimeError(f"Frame {frame_index} was not found in {scene_id}")

    def frame_path(self, scene_id: str, frame_index: int) -> Path:
        header = self.header(scene_id)
        if frame_index < 0 or frame_index >= header.frame_count:
            raise IndexError(
                f"Frame {frame_index} is outside {scene_id}'s range "
                f"0..{header.frame_count - 1}"
            )
        output = self.cache_root / "frames" / scene_id / f"frame_{frame_index:06d}.jpg"
        if output.is_file() and output.stat().st_size > 0:
            return output
        with self._cache_lock:
            if output.is_file() and output.stat().st_size > 0:
                return output
            jpeg = self._read_color_bytes(scene_id, frame_index)
            with Image.open(io.BytesIO(jpeg)) as image:
                if image.format != "JPEG" or image.size != (
                    header.color_width,
                    header.color_height,
                ):
                    raise ValueError(f"Invalid RGB payload for {scene_id} frame {frame_index}")
                image.verify()
            output.parent.mkdir(parents=True, exist_ok=True)
            temporary = output.with_suffix(".jpg.tmp")
            temporary.write_bytes(jpeg)
            os.replace(temporary, output)
        return output

    def frame_paths(self, scene_id: str, frame_indices: list[int]) -> dict[int, Path]:
        """Extract selected RGB frames in one sequential pass through a .sens file."""
        header = self.header(scene_id)
        indices = sorted(set(frame_indices))
        if not indices:
            return {}
        if indices[0] < 0 or indices[-1] >= header.frame_count:
            raise IndexError(
                f"Requested frame range is outside {scene_id}'s range "
                f"0..{header.frame_count - 1}"
            )
        outputs = {
            index: self.cache_root
            / "frames"
            / scene_id
            / f"frame_{index:06d}.jpg"
            for index in indices
        }
        missing = {index for index, path in outputs.items() if not path.is_file() or path.stat().st_size == 0}
        if not missing:
            return outputs

        with self._cache_lock, self.sens_path(scene_id).open("rb") as handle:
            handle.seek(header.frame_data_offset)
            for index in range(indices[-1] + 1):
                _read_exact(handle, 64 + 8 + 8)
                color_size, depth_size = struct.unpack("<QQ", _read_exact(handle, 16))
                if color_size > 100_000_000 or depth_size > 100_000_000:
                    raise ValueError(f"Implausible frame payload size in {scene_id}")
                if index not in missing:
                    handle.seek(color_size + depth_size, 1)
                    continue
                jpeg = _read_exact(handle, color_size)
                handle.seek(depth_size, 1)
                with Image.open(io.BytesIO(jpeg)) as image:
                    if image.format != "JPEG" or image.size != (
                        header.color_width,
                        header.color_height,
                    ):
                        raise ValueError(
                            f"Invalid RGB payload for {scene_id} frame {index}"
                        )
                    image.verify()
                output = outputs[index]
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix(".jpg.tmp")
                temporary.write_bytes(jpeg)
                os.replace(temporary, output)
        return outputs
