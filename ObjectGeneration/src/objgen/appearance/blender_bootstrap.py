"""Load the checkout addon in an isolated GUI process without installing it."""
import importlib.util
import sys
from pathlib import Path

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
addon_path, port = Path(args[0]), int(args[1])
spec = importlib.util.spec_from_file_location("appearance_blender_addon", addon_path)
addon = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = addon
spec.loader.exec_module(addon)
# Register all command-routing properties, but suppress startup/update hooks
# in this process only. No addon installation or preference writes are needed.
addon._blendermcp_register_auto_start = lambda: None
addon._addon_update_check_on_startup = lambda: None
addon.register()
bpy.context.scene.blendermcp_auto_start_server = False
bpy.context.scene.blendermcp_port = port
bpy.types.blendermcp_server = addon.BlenderMCPServer(host="127.0.0.1", port=port)
bpy.types.blendermcp_server.start()
if not bpy.types.blendermcp_server.running:
    raise RuntimeError("Isolated Blender MCP server did not start")
print("APPEARANCE_BLENDER_READY", port, flush=True)
