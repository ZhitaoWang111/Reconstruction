"""Run stages 3, 4, 5 and both Isaac Sim checks in isolated subprocesses."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

from .layout import is_run, object_output_parent

from .config import (ROOT, ISAAC_ENV, PYTHON, SAM31_CHECKPOINT, SAM31_PYTHON, SAM31_REPO,
                     SAM3D_CONFIG)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--prompt", help="Text segmentation; omit prompt and mask for interactive clicks")
    source.add_argument("--mask", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument("--sequential-condition", action="store_true")
    parser.add_argument("--config", type=Path, default=SAM3D_CONFIG)
    parser.add_argument("--sam31-python", type=Path, default=SAM31_PYTHON)
    parser.add_argument("--sam31-repo", type=Path, default=SAM31_REPO)
    parser.add_argument("--sam31-checkpoint", type=Path, default=SAM31_CHECKPOINT)
    parser.add_argument("--sam31-threshold", type=float, default=0.5)
    parser.add_argument("--sam31-instance-index", type=int, default=0)
    size = parser.add_mutually_exclusive_group()
    size.add_argument("--height", type=float, help="Final height in meters")
    size.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--prim-name", default="Object")
    parser.add_argument("--asset-name")
    parser.add_argument("--mass", type=float, default=1.0, help="Mass in kg, set only in stage 5")
    parser.add_argument("--solver-position-iterations", type=int)
    parser.add_argument("--solver-velocity-iterations", type=int)
    parser.add_argument("--collision-approximation", choices=["convexHull", "convexDecomposition"], default="convexHull")
    parser.add_argument("--max-convex-hulls", type=int, default=128)
    parser.add_argument("--voxel-resolution", type=int, default=500000)
    parser.add_argument("--clearance-probe", action="store_true")
    parser.add_argument("--isaac-env", type=Path, default=ISAAC_ENV)
    parser.add_argument("--appearance-config", type=Path, help="Enable appearance refinement and portable verified delivery")
    parser.add_argument("--appearance-skip-feedback", action="store_true", help="Photo projection only; no GPT API calls")
    args = parser.parse_args(argv)
    if args.appearance_skip_feedback and args.appearance_config is None:
        parser.error("--appearance-skip-feedback requires --appearance-config")
    if args.prompt is not None and not args.prompt.strip():
        parser.error("--prompt must not be empty")
    if args.sequential_condition and not args.low_vram:
        parser.error("--sequential-condition requires --low-vram")
    for name in ("height", "scale", "mass"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name} must be finite and positive")
    if not math.isfinite(args.sam31_threshold) or not 0 <= args.sam31_threshold <= 1:
        parser.error("--sam31-threshold must be between 0 and 1")
    if args.sam31_instance_index < 0 or args.max_convex_hulls < 1 or args.voxel_resolution < 10000:
        parser.error("instance index must be >= 0, hull count >= 1, voxel resolution >= 10000")
    for name, minimum in (("solver_position_iterations", 1), ("solver_velocity_iterations", 0)):
        value = getattr(args, name)
        if value is not None and not minimum <= value <= 255:
            parser.error(f"--{name.replace('_', '-')} must be between {minimum} and 255")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.prim_name):
        parser.error("--prim-name must use ASCII letters, digits or underscores, and cannot start with a digit")
    # Preserve venv executable symlinks so Python selects the correct environment.
    for name in ("image", "mask", "output", "config", "sam31_python", "sam31_repo", "sam31_checkpoint", "isaac_env", "appearance_config"):
        path = getattr(args, name)
        if path is not None:
            setattr(args, name, path.expanduser().absolute())
    return args


def isaac_command(args, *command):
    # Positional shell arguments keep spaces, quotes and prompts out of shell code.
    return ["bash", "-c", 'set -e; source "$1"; shift; '
            'if [[ -n "${SIM_CPU_AFFINITY:-}" ]]; then '
            'exec taskset -c "$SIM_CPU_AFFINITY" python -u -X faulthandler "$@"; '
            'else exec python -u -X faulthandler "$@"; fi',
            "pipeline-isaac", str(args.isaac_env), *map(str, command)]


def preflight(args):
    paths = [args.image, args.config, args.isaac_env, PYTHON]
    if args.mask is not None:
        paths.append(args.mask)
    else:
        paths.extend([args.sam31_python, args.sam31_checkpoint, args.sam31_repo / "sam3/model_builder.py"])
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.appearance_config is not None:
        for path in (args.appearance_config, ROOT / ".venv-appearance/bin/python"):
            if not path.is_file(): raise FileNotFoundError(path)
        probe = subprocess.run([str(ROOT / ".venv-appearance/bin/python"), "-c",
            "import sys,yaml; from objgen.appearance.contracts import validate_config; "
            "from pathlib import Path; c=validate_config(yaml.safe_load(Path(sys.argv[1]).read_text())); "
            "assert Path(c['blender']).is_file(); "
            + ("" if args.appearance_skip_feedback else "from objgen.appearance.api_client import credentials; credentials(c['api_env_file']); "),
            str(args.appearance_config)],cwd=ROOT,env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
            capture_output=True,text=True)
        if probe.returncode: raise RuntimeError("Appearance configuration/environment preflight failed: " + probe.stderr[-1500:])
    result = subprocess.run(isaac_command(args, "-c",
        "import importlib.util; assert importlib.util.find_spec('isaacsim'), 'Isaac Sim is not installed'"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(f"Isaac Sim environment preflight failed:\n{result.stdout}")


def sam3d_command(args):
    command = [str(ROOT / "scripts/run_sam3d.sh")]
    for name in ("image", "prompt", "mask", "output", "seed", "config", "sam31_python", "sam31_repo",
                 "sam31_checkpoint", "sam31_threshold", "sam31_instance_index"):
        value = getattr(args, name)
        if value is not None:
            command.extend(["--" + name.replace("_", "-"), str(value)])
    for name in ("low_vram", "sequential_condition"):
        if getattr(args, name):
            command.append("--" + name.replace("_", "-"))
    return command


def usd_command(target, *arguments):
    """Match the SAM3D launcher's CPU workaround before starting its Python runtime.

    ``target`` is a dotted module name inside this package (run with ``-m``, so its
    relative imports resolve) or a path to a stand-alone script.
    """
    if isinstance(target, str) and "/" not in target and not target.endswith(".py"):
        entry = ["-m", target]
    else:
        entry = [str(target)]
    command = [str(PYTHON), "-u", "-X", "faulthandler", *entry, *map(str, arguments)]
    affinity = os.environ.get("SAM3D_CPU_AFFINITY", "0")
    return ["taskset", "-c", affinity, *command] if affinity else command


def remaining_stages(args, run):
    convert = usd_command("objgen.to_usd.mesh_to_usd", run,
                          "--prim-name", args.prim_name, "--asset-name", args.asset_name or args.prim_name)
    convert += ["--height", str(args.height)] if args.height is not None else ["--scale", str(args.scale)]
    physics = usd_command("objgen.to_usd.configure_usd_physics", run,
                          "--mass", args.mass, "--collision-approximation", args.collision_approximation,
                          "--max-convex-hulls", args.max_convex_hulls, "--voxel-resolution", args.voxel_resolution)
    for name in ("solver_position_iterations", "solver_velocity_iterations"):
        value = getattr(args, name)
        if value is not None:
            physics += ["--" + name.replace("_", "-"), str(value)]
    asset = run / "physics/object.usda"
    render = isaac_command(args, ROOT / "src/objgen/verify/check_usd_isaacsim.py", asset,
                           "--output", run / "physics", "--physics-preview")
    drop = isaac_command(args, ROOT / "src/objgen/verify/check_usd_physics_isaacsim.py", asset,
                         "--output", run / "physics")
    if args.clearance_probe:
        drop.append("--clearance-probe")
    return [
        ("4_convert", convert, "usd/run.log", "usd/object.conversion.json", "topology_valid", True,
         ["usd/object.usda"], None),
        ("5_physics", physics, "physics/configure.log", "physics/object.physics.json", "physics_configured", True,
         ["physics/object.usda"], None),
        ("6_1_render", render, "physics/isaacsim_render.log", "physics/isaacsim_verification.json", "passed", True,
         ["physics/object_isaacsim.png", "physics/object_view.usda"], "USD_ISAACSIM_VERIFIED"),
        ("6_2_drop", drop, "physics/isaacsim_physics.log", "physics/isaacsim_physics_verification.json", "passed", True,
         [], "USD_PHYSICS_VERIFIED"),
    ]


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.run_dir = None
        self.report = {"status": "running", "started_at": datetime.now().astimezone().isoformat(),
                       "parameters": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                       "stages": []}

    def save(self):
        if self.run_dir is not None:
            path = self.run_dir / "input/pipeline.json"
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(self.report, indent=2, ensure_ascii=False) + "\n")
            temporary.replace(path)

    def execute(self, name, command, log_name, report_name, key, expected, artifacts, marker):
        record = {"stage": name, "status": "running", "command": command}
        self.report["stages"].append(record)
        self.save()
        started = time.monotonic()
        print(f"PIPELINE_STAGE {name}", flush=True)
        env = os.environ.copy()
        if not name.startswith("6_"):
            env.update(OPENBLAS_CORETYPE="Haswell", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        marker_seen = marker is None
        try:
            if self.run_dir is not None:
                # Never accept a leftover report as evidence of this invocation succeeding.
                (self.run_dir / report_name).unlink(missing_ok=True)
            with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as log:
                try:
                    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, errors="replace", env=env, start_new_session=True) as process:
                        try:
                            for line in process.stdout:
                                print(line, end="", flush=True)
                                log.write(line)
                                if marker and line.startswith(marker + " "):
                                    marker_seen = True
                                if name == "3_sam3d" and self.run_dir is None and line.startswith("RUN_DIR "):
                                    run = Path(line.removeprefix("RUN_DIR ").strip()).resolve()
                                    if run.parent != object_output_parent(self.args.output, self.args.image) or not is_run(run):
                                        raise RuntimeError(f"Unexpected SAM3D run directory: {run}")
                                    self.run_dir = run
                                    self.report["run_dir"] = str(run)
                                    self.save()
                            record["exit_code"] = process.wait()
                        except BaseException:
                            if process.poll() is None:
                                os.killpg(process.pid, signal.SIGTERM)
                                try:
                                    process.wait(timeout=10)
                                except subprocess.TimeoutExpired:
                                    os.killpg(process.pid, signal.SIGKILL)
                                    process.wait()
                            raise
                finally:
                    if self.run_dir is not None:
                        log.seek(0)
                        log_path = self.run_dir / log_name
                        with log_path.open("w") as output:
                            shutil.copyfileobj(log, output)
                        record["log"] = str(log_path)
            if record["exit_code"] != 0:
                code = record["exit_code"]
                detail = f" ({signal.Signals(-code).name})" if code < 0 else ""
                raise RuntimeError(f"{name} exited with code {code}{detail}; see {record.get('log', 'console output')}")
            if self.run_dir is None:
                raise RuntimeError("SAM3D did not return RUN_DIR")
            report_path = self.run_dir / report_name
            if not marker_seen or not report_path.is_file():
                raise RuntimeError(f"{name} did not produce a success marker/report; see {record['log']}")
            metadata = json.loads(report_path.read_text())
            if type(metadata.get(key)) is not type(expected) or metadata.get(key) != expected or not marker_seen:
                raise RuntimeError(f"{name} did not pass verification: {report_path}")
            for relative in artifacts:
                path = self.run_dir / relative
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError(f"Missing or empty artifact: {path}")
            record.update(status="success", report=str(report_path))
        except BaseException as exc:
            record.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            self.save()

    def run(self):
        started = time.monotonic()
        try:
            self.execute("3_sam3d", sam3d_command(self.args), "sam3d/run.log", "sam3d/report.json",
                         "status", "success", ["sam3d/mesh.glb", "sam3d/splat.ply"], None)
            for stage in remaining_stages(self.args, self.run_dir):
                self.execute(*stage)
            asset = self.run_dir / "physics/object.usda"
            if self.args.appearance_config is not None:
                command=[str(ROOT/"scripts/refine_appearance.sh"),"--run-dir",str(self.run_dir),
                         "--config",str(self.args.appearance_config),"--isaac-env",str(self.args.isaac_env)]
                if self.args.appearance_skip_feedback: command.append("--skip-feedback")
                self.execute("7_appearance", command, "usd/appearance_pipeline.log", "usd/appearance/pipeline.json",
                             "passed", True, [], "APPEARANCE_SUCCESS")
                appearance=json.loads((self.run_dir/"usd/appearance/pipeline.json").read_text())
                asset=Path(appearance["asset"])
                if not asset.is_relative_to(self.run_dir/"physics/appearance") or not asset.is_file():
                    raise RuntimeError("Invalid final appearance asset")
                self.report["appearance_report"]=str(self.run_dir/"usd/appearance/pipeline.json")
                self.report["human_approved"]=False
            self.report.update(status="success", asset=str(asset))
            return 0
        except (Exception, KeyboardInterrupt) as exc:
            self.report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            print(f"PIPELINE_FAILED {self.report['error']}", file=sys.stderr, flush=True)
            return 130 if isinstance(exc, KeyboardInterrupt) else 1
        finally:
            self.report["elapsed_seconds"] = time.monotonic() - started
            self.save()
            if self.run_dir is not None:
                print("PIPELINE_REPORT", self.run_dir / "input/pipeline.json", flush=True)
            if self.report["status"] == "success":
                print("PIPELINE_SUCCESS", self.report["asset"], flush=True)


def main(argv=None):
    args = parse_args(argv)
    try:
        preflight(args)
    except Exception as exc:
        print(f"PIPELINE_PREFLIGHT_FAILED {exc}", file=sys.stderr)
        return 1
    return Pipeline(args).run()


if __name__ == "__main__":
    sys.exit(main())
