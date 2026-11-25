"""Opt-in CPU offload between SAM3D stages, keeping upstream code untouched."""
from contextlib import contextmanager
from functools import wraps
import gc
from types import MethodType

import torch


@contextmanager
def load_on_cpu():
    from sam3d_objects.pipeline.inference_pipeline import InferencePipeline
    original = InferencePipeline.instantiate_and_load_from_pretrained

    @wraps(original)
    def cpu_loader(self, *args, **kwargs):
        kwargs["device"] = "cpu"
        with torch.device("cpu"):
            return original(self, *args, **kwargs)

    InferencePipeline.instantiate_and_load_from_pretrained = cpu_loader
    try:
        yield
    finally:
        InferencePipeline.instantiate_and_load_from_pretrained = original


def enable_stage_offload(pipeline, sequential_condition=False):
    modules = dict(pipeline.models.items())
    modules.update(pipeline.condition_embedders)
    modules["depth"] = pipeline.depth_model.model

    def activate(names):
        for name, module in modules.items():
            if module is not None and name not in names:
                module.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        for name in names:
            if modules.get(name) is not None:
                modules[name].to(pipeline.device)
        print("GPU_STAGE", ",".join(names), flush=True)

    stages = {
        "compute_pointmap": ["depth"],
        "sample_sparse_structure": ["ss_condition_embedder", "ss_generator", "ss_decoder"],
        "sample_slat": ["slat_condition_embedder", "slat_generator"],
        "decode_slat": ["slat_decoder_gs", "slat_decoder_gs_4", "slat_decoder_mesh"],
        "postprocess_slat_output": [],
    }
    if sequential_condition:
        # Do not keep generators resident during the condition encoder peak.
        stages["sample_sparse_structure"] = []
        stages["sample_slat"] = []
        original_condition = pipeline.get_condition_input

        def sequential_get_condition(self, condition_embedder, *args, **kwargs):
            if condition_embedder is self.condition_embedders["ss_condition_embedder"]:
                encoder, following = "ss_condition_embedder", ["ss_generator", "ss_decoder"]
            elif condition_embedder is self.condition_embedders["slat_condition_embedder"]:
                encoder, following = "slat_condition_embedder", ["slat_generator"]
            else:
                return original_condition(condition_embedder, *args, **kwargs)
            activate([encoder])
            output = original_condition(condition_embedder, *args, **kwargs)
            activate(following)
            return output

        pipeline.get_condition_input = MethodType(sequential_get_condition, pipeline)

    for method, names in stages.items():
        original = getattr(pipeline, method)

        def wrapped(self, *args, _original=original, _names=names, **kwargs):
            activate(_names)
            return _original(*args, **kwargs)

        setattr(pipeline, method, MethodType(wrapped, pipeline))
    activate([])
