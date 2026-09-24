"""Compatibility entry point; implementation lives in src.geometry.backproject_masks."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.geometry.backproject_masks", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("src.geometry.backproject_masks")
