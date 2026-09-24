"""Compatibility entry point; implementation lives in src.visualization.publish_gpt_viewer."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.visualization.publish_gpt_viewer", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("src.visualization.publish_gpt_viewer")
