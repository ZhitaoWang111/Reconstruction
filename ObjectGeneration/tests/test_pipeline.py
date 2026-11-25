"""Exercise orchestration with real lightweight subprocesses, without loading models."""
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from objgen.layout import create_run
from objgen.run_pipeline import Pipeline, parse_args, remaining_stages, sam3d_command, usd_command, isaac_command


def child(run, report, contents, artifacts=(), marker=None, announce=False):
    """Build a child process that emits the chosen contract, including exit-zero failures."""
    code = "from pathlib import Path; import json\n"
    code += f"run = Path({str(run)!r})\n"
    if announce:
        code += "print('RUN_DIR', run, flush=True)\n"
    for artifact in artifacts:
        code += f"(run / {artifact!r}).write_text('artifact')\n"
    if contents is not None:
        code += f"(run / {report!r}).write_text({json.dumps(contents)!r})\n"
    if marker:
        code += f"print({marker!r} + ' {{}}')\n"
    return [sys.executable, "-u", "-c", code]


class PipelineTests(unittest.TestCase):
    def test_appearance_routing_preserves_default_sequence(self):
        baseline=parse_args(['--image','image.jpg'])
        self.assertIsNone(baseline.appearance_config)
        enabled=parse_args(['--image','image.jpg','--appearance-config','config with spaces.yaml','--appearance-skip-feedback'])
        self.assertTrue(enabled.appearance_skip_feedback)
        self.assertEqual(len(remaining_stages(enabled,Path('/tmp/run'))),4)
        self.assertNotIn('--appearance-config',sam3d_command(enabled))
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse_args(['--image','image.jpg','--appearance-skip-feedback'])

    def test_isaac_affinity_from_activation_script(self):
        with tempfile.TemporaryDirectory(prefix="isaac env ") as tmp:
            env = Path(tmp) / "activate test.sh"
            cpu = min(os.sched_getaffinity(0))
            env.write_text(f"export SIM_CPU_AFFINITY={cpu}\nexport PATH={str(Path(sys.executable).parent)!r}:\"$PATH\"\n")
            args = parse_args(["--image", "bottle.png", "--isaac-env", str(env)])
            result = subprocess.run(isaac_command(args, "-c", "import json,os; print(json.dumps(sorted(os.sched_getaffinity(0))))"),
                                    capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), [cpu])

    def test_usd_python_is_pinned_before_interpreter_start(self):
        with patch.dict(os.environ, {}, clear=True):
            args = parse_args(["--image", "bottle.png", "--prompt", "bottle"])
            stages = remaining_stages(args, Path("/tmp/run"))
            for stage in stages[:2]:
                self.assertEqual(stage[1][:3], ["taskset", "-c", "0"])
                self.assertIn("faulthandler", stage[1])
            for stage in stages[2:]:
                self.assertEqual(stage[1][0], "bash")
        with patch.dict(os.environ, {"SAM3D_CPU_AFFINITY": "1-2"}):
            self.assertEqual(usd_command("mesh_to_usd.py")[:3], ["taskset", "-c", "1-2"])
        with patch.dict(os.environ, {"SAM3D_CPU_AFFINITY": ""}):
            self.assertNotIn("taskset", usd_command("mesh_to_usd.py"))
        with tempfile.TemporaryDirectory() as tmp:
            probe = Path(tmp) / "affinity.py"
            probe.write_text("import json, os; print(json.dumps(sorted(os.sched_getaffinity(0))))")
            cpu = min(os.sched_getaffinity(0))
            with patch.dict(os.environ, {"SAM3D_CPU_AFFINITY": str(cpu)}):
                result = subprocess.run(usd_command(probe), capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout), [cpu])

    def test_signal_failure_names_signal_and_preserves_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parse_args(["--image", "bottle.png", "--prompt", "bottle", "--output", tmp])
            run, _ = create_run(tmp, args.image)
            pipeline = Pipeline(args)
            pipeline.run_dir = run
            command = [sys.executable, "-u", "-c",
                       "import os, signal; print('before termination', flush=True); os.kill(os.getpid(), signal.SIGTERM)"]
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "SIGTERM.*configure.log"):
                pipeline.execute("5_physics", command, "physics/configure.log", "physics/object.physics.json",
                                 "physics_configured", True, [], None)
            self.assertIn("before termination", (run / "physics/configure.log").read_text())
            self.assertEqual(pipeline.report["stages"][-1]["status"], "failed")

    def test_parameter_routing_and_paths_with_spaces(self):
        args = parse_args(["--image", "input with spaces.jpg", "--prompt", "white bottle's label",
                           "--height", "0.14", "--mass", "0.5", "--low-vram", "--sequential-condition",
                           "--solver-position-iterations", "64", "--solver-velocity-iterations", "8",
                           "--isaac-env", "environment with spaces.sh"])
        command = sam3d_command(args)
        self.assertEqual(command[command.index("--prompt") + 1], "white bottle's label")
        self.assertNotIn("--mass", command)
        stages = remaining_stages(args, Path("/tmp/run with spaces"))
        self.assertIn("--height", stages[0][1])
        self.assertIn("--mass", stages[1][1])
        self.assertEqual(stages[1][1][stages[1][1].index("--solver-position-iterations")+1], "64")
        self.assertEqual(stages[1][1][stages[1][1].index("--solver-velocity-iterations")+1], "8")
        self.assertNotIn("--solver-position-iterations", command)
        self.assertNotIn("--mass", stages[3][1])
        self.assertIn(str(args.isaac_env), stages[3][1])

    def test_reject_invalid_parameters_before_gpu_work(self):
        base = ["--image", "image.png", "--prompt", "bottle"]
        for invalid in [["--mass", "nan"], ["--height", "0"], ["--height", ".1", "--scale", "1"],
                        ["--mask", "mask.png"], ["--sequential-condition"], ["--sam31-threshold", "2"],
                        ["--solver-position-iterations", "0"], ["--solver-velocity-iterations", "256"]]:
            with self.subTest(invalid=invalid), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                parse_args(base + invalid)
            self.assertEqual(error.exception.code, 2)

    def exercise(self, mode):
        with tempfile.TemporaryDirectory(prefix="pipeline test ") as tmp:
            args = parse_args(["--image", str(Path(tmp) / "bottle.png"), "--prompt", "bottle", "--output", tmp])
            run, _ = create_run(args.output, args.image)
            first = child(run, "sam3d/report.json", {"status": "failed" if mode == "segmentation_failure" else "success"},
                          ["sam3d/mesh.glb", "sam3d/splat.ply"], announce=True)
            stages = []
            for stage in remaining_stages(args, run):
                name, _, log, report, key, expected, artifacts, marker = stage
                data = {key: expected}
                if name == "6_1_render":
                    if mode == "failed_report":
                        data = {key: False}
                    elif mode == "stale_report":
                        (run / report).write_text(json.dumps({key: True}))
                        data = None
                    elif mode == "missing_marker":
                        marker = None
                    elif mode == "missing_artifact":
                        artifacts = []
                command = child(run, report, data, artifacts, marker)
                stages.append((name, command, *stage[2:]))
            pipeline = Pipeline(args)
            with patch("objgen.run_pipeline.sam3d_command", return_value=first), \
                 patch("objgen.run_pipeline.remaining_stages", return_value=stages), \
                 contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                result = pipeline.run()
            summary = json.loads((run / "input/pipeline.json").read_text())
            self.assertEqual({p.name for p in run.iterdir()}, {"input", "sam3d", "usd", "physics"})
            self.assertTrue((run / "sam3d/run.log").is_file())
            if mode == "success":
                self.assertEqual(result, 0)
                self.assertEqual(summary["status"], "success")
                self.assertEqual(len(summary["stages"]), 5)
                self.assertTrue(all(s["status"] == "success" for s in summary["stages"]))
            else:
                self.assertEqual(result, 1)
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(summary["stages"][-1]["status"], "failed")
                self.assertEqual(summary["stages"][-1]["exit_code"], 0)
                self.assertFalse((run / "physics/isaacsim_physics_verification.json").exists())
                if mode == "segmentation_failure":
                    self.assertEqual(len(summary["stages"]), 1)
                    self.assertFalse((run / "usd/object.usda").exists())

    def test_full_sequence_and_summary(self):
        self.exercise("success")

    def test_appearance_final_asset_and_failure_propagation(self):
        for fail in (False,True):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as tmp:
                args=parse_args(['--image','bottle.png','--output',tmp,'--appearance-config','appearance.yaml'])
                run,_=create_run(args.output,args.image);(run/'usd/appearance').mkdir()
                final=run/'physics/appearance/v_test/m4_test/asset/object.usda';final.parent.mkdir(parents=True);final.write_text('final')
                first=child(run,'sam3d/report.json',{'status':'success'},['sam3d/mesh.glb','sam3d/splat.ply'],announce=True)
                pipeline=Pipeline(args);original=pipeline.execute
                def execute(name,command,*rest):
                    if name=='7_appearance':
                        command=child(run,'usd/appearance/pipeline.json',{'passed':not fail,'asset':str(final)},marker='APPEARANCE_SUCCESS')
                    return original(name,command,*rest)
                with patch('objgen.run_pipeline.sam3d_command',return_value=first),patch('objgen.run_pipeline.remaining_stages',return_value=[]),patch.object(pipeline,'execute',side_effect=execute),contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
                    result=pipeline.run()
                self.assertEqual(result,1 if fail else 0)
                if not fail:self.assertEqual(pipeline.report['asset'],str(final));self.assertFalse(pipeline.report['human_approved'])
                else:self.assertEqual(pipeline.report['stages'][-1]['status'],'failed')

    def test_failed_segmentation_stops_conversion(self):
        self.exercise("segmentation_failure")

    def test_exit_zero_is_not_proof_of_success(self):
        for mode in ("failed_report", "stale_report", "missing_marker", "missing_artifact"):
            with self.subTest(mode=mode):
                self.exercise(mode)


if __name__ == "__main__":
    unittest.main()
