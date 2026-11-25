"""Copy the finished jobs of a DualCraft batch from the server into local review copies.

``dualcraft_batch.py`` runs a batch on the server and keeps its state in
``<server batches>/<batch>/batch.json``.  For every object whose job has ended (``done``
or ``failed``) this pulls the job into ``output/dualcraft/<job>/``, laid out like the
earlier review copies::

    job.json  run.log  fetched.json
    inputs/        DualCraft's inputs: photos, anchors, blueprint (prompt.json)
    product/       model.urdf, object.usdc, compile_report.json, assets/meshes/...
    rev/           the record revision: cost.json, model.py, traces/, blender/ (renders,
                   diff ledger, joint plan)

Left on the server, being large and not needed for review or export: ``product/revision``
(a second copy of the record), the revision GLBs, ``scene.blend`` and ``source_assets``.

The transfer goes through ``ssh ... tar`` into a temporary directory that is renamed when
complete, and is retried, since the connection to the server drops now and then.

Usage::

    PYTHONPATH=src .venv-sam3d/bin/python -m objgen.tools.fetch_dualcraft_batch batch1_standalone
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ..config import ROOT
SERVER_ROOT = "/mnt/data/usr/wzt/submodules/dualcraft"
JOBS = f"{SERVER_ROOT}/dual-craft/output/demo"
BATCHES = f"{SERVER_ROOT}/batches"


def ssh(args, command: str, *, stdout=subprocess.PIPE, timeout: int = 120,
        tries: int | None = None) -> subprocess.CompletedProcess:
    """One ssh call, retried: the server resets connections during the handshake at times.
    Pass ``tries=1`` when stdout is a file, so a failed attempt never appends to it."""
    last = None
    tries = tries or args.retries
    for attempt in range(tries):
        try:
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-p", str(args.port),
                 f"root@{args.host}", command],
                stdout=stdout, stderr=subprocess.PIPE, timeout=timeout)
        except subprocess.TimeoutExpired:
            last = f"timeout after {timeout}s"
            continue
        if result.returncode == 0:
            return result
        last = result.stderr.decode(errors="replace").strip()[-300:]
        if attempt < tries - 1:
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"ssh failed {tries} time(s): {last}")


def fetch_job(args, job: str, dest: Path) -> None:
    """Stream the job as tar into ``<dest>.partial``, then rename it into place.  Each
    attempt starts from an empty directory; a failed job may lack product/ or a record,
    so only what exists is sent."""
    tmp = dest.with_name(dest.name + ".partial")
    # The record lives under articraft_data/records/<rec>/revisions/rev_000001; it is renamed
    # to rev/ on the way, so the local copy does not depend on the record id.
    # A finished run's record is under records/; a run that stopped short (max turns) never
    # published one, and its last revision is the staging directory under cache/runs/.
    command = (
        f"cd {JOBS}/{job} && R=$(ls -d articraft_data/records/*/revisions/rev_000001 2>/dev/null | head -1); "
        f"[ -n \"$R\" ] || R=$(ls -d articraft_data/cache/runs/*/staging/* 2>/dev/null | head -1); "
        f"L=''; for p in job.json run.log inputs product $R; do [ -e \"$p\" ] && L=\"$L $p\"; done; "
        f"tar cf - --exclude=product/revision "
        f"${{R:+--exclude=$R/blender/glb --exclude=$R/blender/scene.blend --exclude=$R/source_assets "
        f"--transform=s,^$R,rev,}} $L"
    )
    last = None
    for attempt in range(args.retries):
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        stream_path = tmp / ".stream.tar"
        try:
            with open(stream_path, "wb") as stream:
                ssh(args, command, stdout=stream, timeout=900, tries=1)
            untar = subprocess.run(["tar", "xf", str(stream_path), "-C", str(tmp)], capture_output=True)
            if untar.returncode == 0 and (tmp / "job.json").is_file():
                break
            last = f"tar stream did not unpack: {untar.stderr.decode(errors='replace')[-300:]}"
        except RuntimeError as exc:
            last = str(exc)
        time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError(f"{job}: fetch failed {args.retries} times: {last}")
    stream_path.unlink()
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("batch", help="batch directory name under the server's batches/")
    parser.add_argument("--dest", type=Path, default=ROOT / "output" / "dualcraft")
    parser.add_argument("--host", default="139.224.9.204")
    parser.add_argument("--port", type=int, default=1021)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--force", action="store_true", help="fetch again jobs already copied")
    args = parser.parse_args()

    state = json.loads(ssh(args, f"cat {BATCHES}/{args.batch}/batch.json").stdout)
    args.dest.mkdir(parents=True, exist_ok=True)
    fetched = []
    for oid in state["order"]:
        entry = state["objects"].get(oid) or {}
        job, stage = entry.get("job"), entry.get("stage")
        if not job or stage not in ("done", "failed"):
            print(f"{oid}: {stage or 'not started'}, not fetched")
            continue
        dest = args.dest / job
        marker = dest / "fetched.json"
        if marker.is_file() and not args.force and \
                json.loads(marker.read_text()).get("finished_at") == entry.get("finished_at"):
            print(f"{oid}: {job} already here")
            continue
        started = time.time()
        fetch_job(args, job, dest)
        (dest / "fetched.json").write_text(json.dumps({
            "object_id": oid, "batch": args.batch, "job": job, "stage": stage,
            "server": f"{args.host}:{JOBS}/{job}", "finished_at": entry.get("finished_at"),
            "fetched_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "batch_entry": entry,
        }, indent=2, ensure_ascii=False) + "\n")
        size = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) / 1e6
        print(f"{oid}: {job} ({stage}) -> {dest} ({size:.0f} MB, {time.time() - started:.0f}s)")
        fetched.append(oid)
    print(f"{len(fetched)} fetched")
    return 0


if __name__ == "__main__":
    sys.exit(main())
