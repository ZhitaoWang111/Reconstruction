"""Compatibility entry point; implementation lives in src.geometry.incremental_objects."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.geometry.incremental_objects", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("src.geometry.incremental_objects")
