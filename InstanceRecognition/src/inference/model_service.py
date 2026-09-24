"""Lifecycle manager for persistent YOLOE and SAM3.1 worker processes."""

from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import threading
import time
import uuid
from pathlib import Path

from src import config


class ModelServiceError(RuntimeError):
    pass


class PersistentWorker:
    def __init__(
        self,
        name: str,
        command: list[str],
        log_path: Path,
        request_timeout: float = 300.0,
    ) -> None:
        self.name = name
        self.command = command
        self.log_path = Path(log_path)
        self.request_timeout = request_timeout
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._responses: queue.Queue[str] = queue.Queue()
        self._reader: threading.Thread | None = None
        self._log = None
        self._loaded = False
        self._loading = False
        self._last_error: str | None = None
        self._load_seconds: float | None = None

    def _start_locked(self) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self._cleanup_locked()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("a", buffering=1)
        self._log.write(f"\n=== worker start {time.strftime('%F %T')} ===\n")
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        self._process = subprocess.Popen(
            self.command,
            cwd=config.ROOT,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            text=True,
            bufsize=1,
        )
        self._responses = queue.Queue()
        process = self._process

        def read_responses() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                self._responses.put(line)
            self._responses.put("")

        self._reader = threading.Thread(
            target=read_responses, name=f"{self.name}-responses", daemon=True
        )
        self._reader.start()
        self._loaded = False
        self._last_error = None
        self._load_seconds = None

    def request(self, command: str, **payload) -> dict:
        with self._lock:
            self._start_locked()
            assert self._process is not None and self._process.stdin is not None
            request_id = uuid.uuid4().hex
            message = {"id": request_id, "command": command, **payload}
            self._loading = command in {"load", "infer"} and not self._loaded
            try:
                self._process.stdin.write(json.dumps(message) + "\n")
                self._process.stdin.flush()
                deadline = time.monotonic() + self.request_timeout
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise ModelServiceError(
                            f"{self.name} timed out after {self.request_timeout:.0f}s"
                        )
                    try:
                        line = self._responses.get(timeout=remaining)
                    except queue.Empty as error:
                        raise ModelServiceError(f"{self.name} response timed out") from error
                    if not line:
                        code = self._process.poll()
                        raise ModelServiceError(
                            f"{self.name} exited before replying (exit code {code})"
                        )
                    try:
                        response = json.loads(line)
                    except json.JSONDecodeError:
                        if self._log:
                            self._log.write(f"unexpected stdout: {line}")
                        continue
                    if response.get("id") == request_id:
                        break
                if not response.get("ok"):
                    detail = response.get("traceback") or response.get("error")
                    raise ModelServiceError(f"{self.name} failed:\n{detail}")
                if command in {"load", "infer"}:
                    self._loaded = True
                    if response.get("load_seconds"):
                        self._load_seconds = float(response["load_seconds"])
                self._last_error = None
                return response
            except Exception as error:
                self._last_error = str(error)
                raise
            finally:
                self._loading = False

    def shutdown(self) -> None:
        with self._lock:
            process = self._process
            if process is None:
                return
            if process.poll() is None:
                try:
                    self.request("shutdown")
                    process.wait(timeout=10)
                except Exception:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
            self._cleanup_locked()

    def _cleanup_locked(self) -> None:
        process = self._process
        if process is not None:
            for stream in (process.stdin, process.stdout):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
        if self._log is not None:
            self._log.close()
        self._process = None
        self._log = None
        self._loaded = False
        self._loading = False
        self._load_seconds = None

    def status(self) -> dict:
        # This snapshot deliberately does not wait for the worker request lock:
        # status polling must stay responsive while a model is loading/inferencing.
        process = self._process
        running = process is not None and process.poll() is None
        return {
            "running": running,
            "loaded": running and self._loaded,
            "loading": running and self._loading,
            "pid": process.pid if running else None,
            "load_seconds": self._load_seconds,
            "last_error": self._last_error,
        }


class ModelServiceManager:
    def __init__(
        self,
        runtime_root: Path = config.RUNTIME_ROOT,
        heartbeat_timeout: float = 15.0,
        unload_grace: float = 5.0,
        monitor_interval: float = 1.0,
        workers: dict[str, PersistentWorker] | None = None,
        method: str = config.DEFAULT_METHOD,
    ) -> None:
        runtime_root = Path(runtime_root).resolve()
        if method not in {"gpt_sam31", "yoloe_sam31"}:
            raise ValueError("Unknown instance-recognition method: " + method)
        self.method = method
        self.workers = workers or {
            "yoloe": PersistentWorker(
                "YOLOE-26",
                [
                    str(config.YOLOE_PYTHON),
                    "-m",
                    "src.model_worker",
                    "--stage",
                    "yoloe",
                    "--checkpoint",
                    str(config.YOLOE_CHECKPOINT),
                    "--conf",
                    str(config.DEFAULT_CONFIDENCE),
                    "--imgsz",
                    str(config.DEFAULT_IMAGE_SIZE),
                    "--max-det",
                    str(config.DEFAULT_MAX_DETECTIONS),
                ],
                runtime_root / "logs/yoloe-worker.log",
            ),
            "sam31": PersistentWorker(
                "SAM3.1",
                [
                    str(config.SAM31_PYTHON),
                    "-m",
                    "src.model_worker",
                    "--stage",
                    "sam31",
                    "--checkpoint",
                    str(config.SAM31_CHECKPOINT),
                ],
                runtime_root / "logs/sam31-worker.log",
            ),
        }
        if workers is None and method == "gpt_sam31":
            self.workers = {"sam31_text": PersistentWorker(
                "SAM3.1 text prompts",
                [str(config.SAM31_PYTHON), "-m", "src.model_worker", "--stage", "sam31_text",
                 "--checkpoint", str(config.SAM31_CHECKPOINT), "--threshold", str(config.GPT_SAM_THRESHOLD)],
                runtime_root / "logs/sam31-text-worker.log",
            )}
        self.heartbeat_timeout = heartbeat_timeout
        self.unload_grace = unload_grace
        self.monitor_interval = monitor_interval
        self._lock = threading.RLock()
        self._clients: dict[str, float] = {}
        self._active_requests = 0
        self._zero_clients_since: float | None = None
        self._preload_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._monitor = threading.Thread(
            target=self._monitor_loop, name="model-lifecycle-monitor", daemon=True
        )
        self._monitor.start()
        atexit.register(self.shutdown)

    def open_client(self, client_id: str) -> dict:
        if not client_id:
            raise ValueError("client_id is required")
        with self._lock:
            self._clients[client_id] = time.monotonic()
            self._zero_clients_since = None
        self.preload_async()
        return self.status()

    def heartbeat(self, client_id: str) -> dict:
        if not client_id:
            raise ValueError("client_id is required")
        with self._lock:
            self._clients[client_id] = time.monotonic()
            self._zero_clients_since = None
        return self.status()

    def close_client(self, client_id: str) -> dict:
        with self._lock:
            self._clients.pop(client_id, None)
            if not self._clients:
                self._zero_clients_since = time.monotonic()
        return self.status()

    def preload_async(self) -> None:
        with self._lock:
            if self._preload_thread is not None and self._preload_thread.is_alive():
                return

            def preload_all() -> None:
                threads = [
                    threading.Thread(target=self._safe_load, args=(name,), daemon=True)
                    for name in self.workers
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()

            self._preload_thread = threading.Thread(
                target=preload_all, name="model-preloader", daemon=True
            )
            self._preload_thread.start()

    def _safe_load(self, name: str) -> None:
        try:
            self.workers[name].request("load")
        except Exception:
            # The status endpoint exposes the worker error; infer can retry/restart.
            return

    def infer_sam31_text(self, source: Path, labels: Path, output_dir: Path) -> dict:
        return self._infer("sam31_text", source=str(source), detections=str(labels), output_dir=str(output_dir))

    def infer_yoloe(self, source: Path, output_dir: Path) -> dict:
        return self._infer(
            "yoloe", source=str(source), output_dir=str(output_dir)
        )

    def infer_sam31(
        self, source: Path, detections: Path, output_dir: Path
    ) -> dict:
        return self._infer(
            "sam31",
            source=str(source),
            detections=str(detections),
            output_dir=str(output_dir),
        )

    def _infer(self, name: str, **payload) -> dict:
        with self._lock:
            self._active_requests += 1
            self._zero_clients_since = None
        try:
            return self.workers[name].request("infer", **payload)["result"]
        finally:
            with self._lock:
                self._active_requests -= 1
                if not self._clients and self._active_requests == 0:
                    self._zero_clients_since = time.monotonic()

    def unload_all(self) -> None:
        with self._lock:
            if self._active_requests:
                return
        for worker in self.workers.values():
            worker.shutdown()

    def status(self) -> dict:
        worker_status = {
            name: worker.status() for name, worker in self.workers.items()
        }
        with self._lock:
            now = time.monotonic()
            deadline = None
            if (
                any(item["running"] for item in worker_status.values())
                and self._zero_clients_since is not None
                and self._active_requests == 0
            ):
                deadline = max(
                    0.0, self.unload_grace - (now - self._zero_clients_since)
                )
            active_clients = len(self._clients)
            active_requests = self._active_requests
        return {
            "active_clients": active_clients,
            "active_requests": active_requests,
            "heartbeat_timeout_seconds": self.heartbeat_timeout,
            "unload_grace_seconds": self.unload_grace,
            "unload_in_seconds": None if deadline is None else round(deadline, 1),
            "workers": worker_status,
        }

    def _monitor_loop(self) -> None:
        while not self._stop.wait(self.monitor_interval):
            should_unload = False
            with self._lock:
                now = time.monotonic()
                expired = [
                    client_id
                    for client_id, last_seen in self._clients.items()
                    if now - last_seen > self.heartbeat_timeout
                ]
                for client_id in expired:
                    self._clients.pop(client_id, None)
                if not self._clients and self._zero_clients_since is None:
                    self._zero_clients_since = now
                if (
                    not self._clients
                    and self._active_requests == 0
                    and self._zero_clients_since is not None
                    and now - self._zero_clients_since >= self.unload_grace
                ):
                    should_unload = True
                    self._zero_clients_since = now
            if should_unload:
                self.unload_all()

    def shutdown(self) -> None:
        self._stop.set()
        self.unload_all()
