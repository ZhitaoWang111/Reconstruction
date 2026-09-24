"""Compatibility entry point; implementation lives in src.inference.gpt_sam31."""
if __name__ == "__main__":
    import runpy
    runpy.run_module("src.inference.gpt_sam31", run_name="__main__")
else:
    import importlib
    import sys
    sys.modules[__name__] = importlib.import_module("src.inference.gpt_sam31")
