"""Compatibility entry point; implementation lives in src.inference.model_service."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.inference.model_service", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("src.inference.model_service")
