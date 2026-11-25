"""M0 live API + isolated Blender + MCP visual/tool round-trip probe."""
import argparse
import asyncio
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import yaml

from .api_client import APIClient, image_content
from .mcp_client import blender_session, execute_template

from ..config import BLENDER_MCP, ROOT
INTENT = "验证 GPT-6、Blender MCP 与图片反馈闭环，不修改已有资产。"


def error_detail(exc):
    """Keep the actionable cause when MCP wraps failures in a task group."""
    children = getattr(exc, 'exceptions', None)
    if children:
        return '; '.join(error_detail(child) for child in children)
    return f'{type(exc).__name__}: {exc}'


def render_template(mesh, output):
    return f'''import bpy, math
from mathutils import Vector
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath={str(mesh)!r})
objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
assert objects
corners = [o.matrix_world @ Vector(c) for o in objects for c in o.bound_box]
lo = Vector(tuple(min(c[i] for c in corners) for i in range(3)))
hi = Vector(tuple(max(c[i] for c in corners) for i in range(3)))
center = (lo + hi) / 2
size = max(hi - lo)
for o in objects:
    material = bpy.data.materials.new('BaselineVertexColor')
    material.use_nodes = True
    nodes = material.node_tree.nodes
    attr = nodes.new('ShaderNodeVertexColor')
    if o.data.color_attributes:
        attr.layer_name = o.data.color_attributes[0].name
    material.node_tree.links.new(attr.outputs['Color'], nodes.get('Principled BSDF').inputs['Base Color'])
    o.data.materials.clear()
    o.data.materials.append(material)
bpy.ops.object.camera_add(location=center + Vector((0, -2.2, 0.6)) * size)
camera = bpy.context.object
camera.rotation_euler = (center - camera.location).to_track_quat('-Z', 'Y').to_euler()
camera.data.lens = 45
bpy.context.scene.camera = camera
bpy.ops.object.light_add(type='AREA', location=center + Vector((1, -2, 3)) * size)
bpy.context.object.data.energy = 500 * size * size
bpy.context.object.data.shape = 'DISK'
bpy.context.object.data.size = size * 2
bpy.context.object.rotation_euler = (center - bpy.context.object.location).to_track_quat('-Z', 'Y').to_euler()
scene = bpy.context.scene
scene.render.engine = 'BLENDER_EEVEE_NEXT'
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.resolution_percentage = 100
scene.render.filepath = {str(output / 'baseline.png')!r}
bpy.ops.wm.save_as_mainfile(filepath={str(output / 'probe.blend')!r})
bpy.ops.render.render(write_still=True)
print('APPEARANCE_RENDER_VERIFIED', len(objects), bpy.app.version_string)
'''


async def probe(config, run, output, report):
    rgb = next(iter(sorted((run / 'input').glob('rgb.*'))), None)
    mesh = run / 'sam3d/mesh.glb'
    if rgb is None or not mesh.is_file():
        raise ValueError("Run needs input/rgb.* and sam3d/mesh.glb")
    api = APIClient(config)
    report['stage'] = 'api_image_tool'
    messages = [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Inspect the product photo, then call render_asset exactly once. Do not answer before calling it.'},
        image_content(rgb)]}]
    tool = {'type': 'function', 'function': {'name': 'render_asset',
            'description': 'Render the existing reconstructed asset in isolated Blender.',
            'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}
    first = await asyncio.to_thread(api.chat, messages, tools=[tool],
                                    tool_choice={'type': 'function', 'function': {'name': 'render_asset'}})
    calls = first.get('tool_calls') or []
    if len(calls) != 1 or calls[0]['function']['name'] != 'render_asset' or json.loads(calls[0]['function']['arguments']) != {}:
        raise RuntimeError('API did not return the requested render tool')
    report['api_calls'] = api.records
    report['stage'] = 'blender_mcp_render'
    port = config['mcp_port']
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
    repo = BLENDER_MCP
    env = dict(os.environ, BLENDER_MCP_DISABLE_TELEMETRY='1')
    with (output / 'blender.log').open('w') as log:
        process = subprocess.Popen([config['blender'], '--factory-startup', '--python',
            str(Path(__file__).with_name('blender_bootstrap.py')), '--', str(repo / 'addon.py'), str(port)],
            stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            deadline = time.monotonic() + 60
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Isolated Blender exited; see blender.log')
                with socket.socket() as sock:
                    if sock.connect_ex(('127.0.0.1', port)) == 0:
                        break
                if time.monotonic() >= deadline:
                    raise TimeoutError('Isolated Blender startup timed out')
                await asyncio.sleep(.5)
            async with blender_session(repo, port) as session:
                result = await asyncio.wait_for(execute_template(session, render_template(mesh, output), INTENT), 170)
                if 'APPEARANCE_RENDER_VERIFIED' not in result:
                    raise RuntimeError('Missing Blender render marker')
                report['blender_result'] = result[-600:]
            from PIL import Image
            with Image.open(output / 'baseline.png') as img:
                assert img.size == (512, 512)
                img.verify()
            assert (output / 'probe.blend').stat().st_size > 0
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    report['stage'] = 'api_render_feedback'
    messages += [first, {'role': 'tool', 'tool_call_id': calls[0]['id'],
                          'content': 'Render completed successfully. Its image follows.'},
                 {'role': 'user', 'content': [image_content(output / 'baseline.png'),
                    {'type': 'text', 'text': 'Compare this render with the original photo. State two visible appearance differences. This is only a capability probe.'}]}]
    final = await asyncio.to_thread(api.chat, messages)
    if not isinstance(final.get('content'), str) or not final['content'].strip():
        raise RuntimeError('Missing visual feedback')
    report.update(api_calls=api.records, feedback=final['content'], passed=True, stage='complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/appearance/default.yaml')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    output = run / 'usd/appearance' / ('m0_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True)
    config = yaml.safe_load(args.config.read_text())
    report = {'schema_version': 1, 'milestone': 'M0', 'passed': False, 'stage': 'preflight',
              'run_dir': str(run), 'model': config['model'], 'protocol': 'chat_completions',
              'mcp_commit': subprocess.check_output(['git', '-C', str(BLENDER_MCP),
                                                     'rev-parse', 'HEAD'], text=True).strip()}
    try:
        asyncio.run(probe(config, run, output, report))
    except (Exception, KeyboardInterrupt) as exc:
        report.update(error_type=type(exc).__name__, error=error_detail(exc))
    (output / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print('APPEARANCE_PREFLIGHT_REPORT', output / 'report.json')
    print('APPEARANCE_PREFLIGHT_PASSED', report['passed'])
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
