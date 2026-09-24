"""Compatibility entry point; implementation lives in src.inference.yoloe_detect."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.inference.yoloe_detect", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("src.inference.yoloe_detect")
