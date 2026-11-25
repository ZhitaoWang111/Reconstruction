"""Local browser point prompting for SAM3.1; export a mask only after confirmation."""
import argparse
from datetime import datetime
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import os
from pathlib import Path
import queue
import secrets
import sys
import tempfile
import threading
import time
import traceback
from urllib.parse import urlsplit, parse_qs
import uuid
import webbrowser


class ReviewError(ValueError):
    pass


def validate_points(points):
    if not isinstance(points, list) or len(points) > 100:
        raise ReviewError("最多支持 100 个点击点。")
    clean = []
    for point in points:
        if not isinstance(point, dict):
            raise ReviewError("点击点格式错误。")
        x, y, label = point.get("x"), point.get("y"), point.get("label")
        if (type(x) not in (int, float) or type(y) not in (int, float)
                or not math.isfinite(x) or not math.isfinite(y) or not 0 <= x <= 1 or not 0 <= y <= 1
                or type(label) is not int or label not in (0, 1)):
            raise ReviewError("点击坐标必须在图片内，标签必须为正点或负点。")
        clean.append({"x": float(x), "y": float(y), "label": label})
    if clean and not any(p["label"] == 1 for p in clean):
        raise ReviewError("请先在目标内部添加至少一个正点。")
    return clean


def image_bytes(image, *, thumbnail=False):
    image = image.copy()
    if thumbnail:
        image.thumbnail((1400, 1000))
    image.info.clear()  # Browser EXIF rotation must not change the point coordinate system.
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


class PointModel:
    def __init__(self, image, repo, checkpoint):
        import torch
        sys.path.insert(0, str(repo))
        from sam3.model_builder import build_sam3_multiplex_video_predictor
        self.image = image
        self.frames = tempfile.TemporaryDirectory(prefix="sam31_click_frames_")
        image.save(Path(self.frames.name) / "000000.jpg", quality=100, subsampling=0)
        self.predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=str(checkpoint), max_num_objects=16, multiplex_count=16,
            use_fa3=False, use_rope_real=False, compile=False, warm_up=False,
            async_loading_frames=False, default_output_prob_thresh=0.0,
        )
        # Use the same init_state workaround as the existing text-prompt adapter.
        self.state = self.predictor.model.init_state(
            resource_path=self.frames.name, offload_video_to_cpu=False, async_loading_frames=False)
        self.session = str(uuid.uuid4())
        now = time.time()
        self.predictor._all_inference_states[self.session] = {
            "state": self.state, "session_id": self.session, "start_time": now, "last_use_time": now}
        self.versions = {"python": sys.version, "torch": torch.__version__}

    def segment(self, points):
        import torch
        import numpy as np
        from PIL import Image
        # Rebuild the instance from the visible point set; undo/clear cannot retain hidden prompts.
        with torch.inference_mode():
            self.predictor.model.reset_state(self.state)
        # This checkout's _build_sam2_output drops a first point mask when frame 0
        # has no cached entry (its early return precedes merging the new mask).
        # Initialize an empty frame, so pure clicks work without a preceding text prompt.
        self.state.setdefault("cached_frame_outputs", {})[0] = {}
        outputs = self.predictor.handle_request({
            "type": "add_prompt", "session_id": self.session, "frame_index": 0,
            "points": [[p["x"], p["y"]] for p in points],
            "point_labels": [p["label"] for p in points], "obj_id": 1,
            "rel_coordinates": True, "clear_old_points": True, "output_prob_thresh": 0.0,
        })["outputs"]
        ids = np.asarray(outputs["out_obj_ids"]).reshape(-1)
        masks = np.asarray(outputs["out_binary_masks"], dtype=bool)
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        indices = np.flatnonzero(ids == 1)
        width, height = self.image.size
        if len(indices) != 1 or masks.shape != (len(ids), height, width):
            raise ReviewError(f"模型没有返回所选物体的有效 mask（ids={ids.tolist()}, shape={masks.shape}），请清空后重新点击。")
        mask = masks[int(indices[0])]
        if not mask.any():
            raise ReviewError("分割结果为空，请在物体内部补一个正点或重新选择。")
        return Image.fromarray(mask.astype(np.uint8) * 255)

    def close(self):
        self.predictor.handle_request({"type": "close_session", "session_id": self.session})
        self.frames.cleanup()


class Review:
    def __init__(self, image, output):
        self.image, self.output = image, output
        self.lock = threading.RLock()
        self.done = threading.Event()
        self.jobs = queue.Queue()
        self.phase = "loading"
        self.revision = 0
        self.points = []
        self.mask = self.overlay = None
        self.viewed_revision = None
        self.error = None
        self.history = []
        self.confirmed_at = None
        self.model_info = {}

    def public(self):
        with self.lock:
            return {"phase": self.phase, "revision": self.revision, "points": self.points,
                    "image_size": list(self.image.size), "error": self.error,
                    "has_preview": self.mask is not None and self.phase in ("preview", "confirmed"),
                    "can_confirm": self.phase == "preview" and self.viewed_revision == self.revision,
                    "confirmed_at": self.confirmed_at}

    def record(self, action):
        self.history.append({"action": action, "at": datetime.now().astimezone().isoformat(),
                             "revision": self.revision, "points": [dict(p) for p in self.points]})
        path = self.output / "interaction.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps({**self.public(), "history": self.history}, ensure_ascii=False, indent=2) + "\n")
        temp.replace(path)

    def submit(self, points, revision):
        points = validate_points(points)
        with self.lock:
            if self.done.is_set() or self.phase == "loading":
                raise ReviewError("当前不能修改点击点。")
            if type(revision) is not int or revision != self.revision:
                raise ReviewError("页面状态已过期，请等待最新状态后操作。")
            self.revision += 1
            self.points = points
            self.mask = self.overlay = None
            self.viewed_revision = None
            self.error = None
            self.phase = "segmenting" if points else "ready"
            self.record("points_changed")
            if points:
                self.jobs.put((self.revision, points))
            return self.public()

    def preview(self, revision):
        with self.lock:
            if self.phase != "preview" or revision != self.revision or self.overlay is None:
                raise ReviewError("预览已过期，请等待当前点击点的分割结果。")
            result = image_bytes(self.overlay, thumbnail=True)
            self.viewed_revision = revision
            return result

    def confirm(self, revision):
        with self.lock:
            if (type(revision) is not int or revision != self.revision
                    or not self.public()["can_confirm"] or self.mask is None):
                raise ReviewError("请先查看当前点击点的分割预览，再确认继续。")
            self.phase = "confirmed"
            self.confirmed_at = datetime.now().astimezone().isoformat()
            self.record("confirmed")
            self.done.set()
            return self.public()

    def cancel(self):
        with self.lock:
            if not self.done.is_set():
                self.phase = "cancelled"
                self.record("cancelled")
                self.done.set()
            return self.public()

    def work(self, engine_factory):
        engine = None
        try:
            started = time.monotonic()
            engine = engine_factory()
            with self.lock:
                self.model_info = {**getattr(engine, "versions", {}), "load_seconds": time.monotonic() - started}
                if self.done.is_set():
                    return
                self.phase = "ready"
                self.record("model_ready")
            while not self.done.is_set():
                try:
                    revision, points = self.jobs.get(timeout=0.2)
                except queue.Empty:
                    continue
                with self.lock:
                    if revision != self.revision:
                        continue
                try:
                    from PIL import Image
                    mask = engine.segment(points).convert("L")
                    if mask.size != self.image.size or mask.getbbox() is None:
                        raise ReviewError("分割 mask 尺寸不符或为空，请重新点击。")
                    overlay = Image.composite(
                        Image.blend(self.image, Image.new("RGB", self.image.size, (35, 210, 135)), 0.45),
                        self.image, mask)
                    with self.lock:
                        if revision != self.revision or self.done.is_set():
                            continue
                        self.mask, self.overlay = mask, overlay
                        self.phase = "preview"
                        self.record("preview_ready")
                except Exception as exc:
                    traceback.print_exc()
                    with self.lock:
                        if revision == self.revision and not self.done.is_set():
                            self.phase, self.error = "ready", str(exc)
                            self.record("segmentation_error")
        except Exception as exc:
            traceback.print_exc()
            with self.lock:
                if not self.done.is_set():
                    self.phase, self.error = "failed", str(exc)
                    self.record("model_error")
                    self.done.set()
        finally:
            if engine is not None:
                engine.close()


def make_server(review):
    token = secrets.token_urlsafe(24)
    prefix = f"/{token}/"
    html = Path(__file__).with_name("sam31_click.html").read_bytes()
    original = image_bytes(review.image, thumbnail=True)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, code, data, content_type="application/json"):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def route(self):
            path = urlsplit(self.path)
            if not path.path.startswith(prefix):
                self.reply(404, {"error": "Not found"})
                return None, path
            return path.path[len(prefix):], path

        def do_GET(self):
            route, path = self.route()
            if route is None:
                return
            try:
                if route == "":
                    self.reply(200, html, "text/html; charset=utf-8")
                elif route == "state":
                    self.reply(200, review.public())
                elif route == "image":
                    self.reply(200, original, "image/png")
                elif route == "preview":
                    revision = int(parse_qs(path.query).get("revision", ["-1"])[0])
                    self.reply(200, review.preview(revision), "image/png")
                else:
                    self.reply(404, {"error": "Not found"})
            except ValueError as exc:
                self.reply(409, {"error": str(exc)})

        def do_POST(self):
            route, _ = self.route()
            if route is None:
                return
            # No CORS and a per-run secret URL: other web pages cannot drive this session.
            origin = self.headers.get("Origin")
            if origin and origin != f"http://127.0.0.1:{self.server.server_port}":
                self.reply(403, {"error": "Origin rejected"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 65536:
                    raise ReviewError("请求大小无效。")
                data = json.loads(self.rfile.read(length))
                if not isinstance(data, dict):
                    raise ReviewError("请求格式无效。")
                if route == "points":
                    result = review.submit(data.get("points"), data.get("revision"))
                elif route == "confirm":
                    result = review.confirm(data.get("revision"))
                elif route == "cancel":
                    result = review.cancel()
                else:
                    self.reply(404, {"error": "Not found"})
                    return
                self.reply(200, result)
            except (ValueError, TypeError) as exc:
                self.reply(409, {"error": str(exc)})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    return server, f"http://127.0.0.1:{server.server_port}{prefix}"


def run_review(image, output, engine_factory, *, open_browser=True):
    review = Review(image, output)
    server, url = make_server(review)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    worker = threading.Thread(target=review.work, args=(engine_factory,), daemon=True)
    server_thread.start()
    review.record("opened")
    print("SAM31_INTERACTIVE_URL", url, flush=True)
    print("SAM31_INTERACTIVE_WAIT 点击图片选物，预览后确认；取消会停止本次流水线。", flush=True)
    try:
        if open_browser:
            try:
                webbrowser.open(url, new=2)
            except webbrowser.Error as exc:
                print("SAM31_INTERACTIVE_BROWSER 请手动打开上方链接：", exc, flush=True)
        worker.start()
        while not review.done.wait(0.2):
            pass
    except KeyboardInterrupt:
        review.cancel()
    finally:
        server.shutdown()
        server.server_close()
        if worker.ident is not None:
            worker.join()  # Release SAM3.1 before the parent can start SAM3D.
        server_thread.join()
    return review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "model": "SAM3.1 Multiplex", "mode": "points",
              "user_confirmed": False, "image": str(args.image.resolve()),
              "repo": str(args.repo.resolve()), "checkpoint": str(args.checkpoint.resolve())}
    start = time.monotonic()
    report_path = args.output / "segmentation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for path in (args.image, args.checkpoint, args.repo / "sam3/model_builder.py"):
            if not path.is_file():
                raise FileNotFoundError(path)
        from PIL import Image
        image = Image.open(args.image).convert("RGB")
        report.update(image_size=list(image.size), source_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest())
        review = run_review(image, args.output, lambda: PointModel(image, args.repo.resolve(), args.checkpoint.resolve()),
                            open_browser=os.environ.get("SAM31_NO_BROWSER") != "1")
        report.update(review.public(), **review.model_info)
        if review.phase == "cancelled":
            report.update(status="cancelled", error="User cancelled point selection")
        elif review.phase != "confirmed":
            raise RuntimeError(review.error or "Point selection was not confirmed")
        else:
            mask_path = args.output / "mask.png"
            mask_path.write_bytes(image_bytes(review.mask))
            (args.output / "mask_overlay.png").write_bytes(image_bytes(review.overlay))
            report.update(status="success", user_confirmed=True, mask=str(mask_path),
                          mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                          selected={"object_id": 1, "selection_policy": "human-confirmed point mask", "box_xyxy": list(review.mask.getbbox())})
            print("SAM31_INTERACTIVE_CONFIRMED", mask_path, flush=True)
    except Exception as exc:
        traceback.print_exc()
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["status"] == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
