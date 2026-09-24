"""Compatibility entry point; implementation lives in src.data.scannet_sens."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.data.scannet_sens", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("src.data.scannet_sens")
