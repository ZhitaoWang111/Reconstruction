import time

from src.model_service import ModelServiceManager
from web.server import create_app


SCANS = "/home/wzt/HDD/datasets/scannet_v2/scans"


class FakeWorker:
    def __init__(self):
        self.running = False
        self.loaded = False
        self.shutdown_count = 0

    def request(self, command, **payload):
        self.running = True
        if command == "load":
            self.loaded = True
            return {"ok": True, "status": "loaded", "load_seconds": 0.01}
        if command == "infer":
            self.loaded = True
            return {"ok": True, "result": payload}
        raise AssertionError(command)

    def shutdown(self):
        self.running = False
        self.loaded = False
        self.shutdown_count += 1

    def status(self):
        return {
            "running": self.running,
            "loaded": self.loaded,
            "loading": False,
            "pid": 123 if self.running else None,
            "load_seconds": 0.01 if self.loaded else None,
            "last_error": None,
        }


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition did not become true")


def test_final_client_close_unloads_both_workers(tmp_path):
    workers = {"yoloe": FakeWorker(), "sam31": FakeWorker()}
    manager = ModelServiceManager(
        tmp_path,
        heartbeat_timeout=0.2,
        unload_grace=0.05,
        monitor_interval=0.01,
        workers=workers,
    )
    try:
        manager.open_client("tab-a")
        manager.open_client("tab-b")
        wait_until(lambda: all(worker.loaded for worker in workers.values()))
        manager.close_client("tab-a")
        time.sleep(0.08)
        assert all(worker.loaded for worker in workers.values())

        manager.close_client("tab-b")
        wait_until(lambda: all(not worker.running for worker in workers.values()))
        assert all(worker.shutdown_count >= 1 for worker in workers.values())
    finally:
        manager.shutdown()


def test_client_api_tracks_lease_and_exposes_model_status(tmp_path):
    workers = {"yoloe": FakeWorker(), "sam31": FakeWorker()}
    manager = ModelServiceManager(
        tmp_path,
        heartbeat_timeout=1.0,
        unload_grace=1.0,
        monitor_interval=0.05,
        workers=workers,
    )
    try:
        client = create_app(SCANS, tmp_path, manager).test_client()
        response = client.post("/api/client/open", json={"client_id": "browser-1"})
        assert response.status_code == 200
        assert response.get_json()["active_clients"] == 1
        assert client.get("/api/models/status").status_code == 200
        response = client.post(
            "/api/client/heartbeat", json={"client_id": "browser-1"}
        )
        assert response.get_json()["active_clients"] == 1
        response = client.post("/api/client/close", json={"client_id": "browser-1"})
        assert response.get_json()["active_clients"] == 0
    finally:
        manager.shutdown()
