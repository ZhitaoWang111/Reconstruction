"""M6 batches: which objects a batch holds, which key frames it shares, and its life cycle.

    from batch import cut_batches, BatchSession
    plan = cut_batches(scene)                       # 11 batches for scene0000_00 (D02)
    s = BatchSession(plan, scene, structure, state, initial, observer, verifier, run_dir)
    pack = s.open(3)                                # opening material for batch 3
    s.set_order([...]); s.start(oid); ...; s.finish(oid)
    result = s.end()                                # batch-end check; may reopen objects once
    s.close()                                       # checkpoint

Rules (D02, D04, and M6 decided 2026-11-23):
  * every object's first key frame (lowest sequence number); objects sorted by it,
    ties by object id; five per batch; a frame shared across a batch boundary is
    used by both batches;
  * a batch shares the first key frames of its objects;
  * the agent sets the processing order first; the unprocessed part may be changed later;
  * up to 3 objects may be brought into a batch: an earlier one to fix it, or a later
    one that something here rests on (it then leaves its own batch);
  * an object brought in that no shared frame shows adds its own first key frame;
  * views are limited to the shared frames; holdout frames are never given;
  * at the end: global contact check and a 15 s physics run; objects that fail are
    reopened once, a second failure marks them failed;
  * each batch is a fresh agent conversation, opened with a short list of what is
    already placed (no images of earlier batches).
"""
from .plan import Batch, cut_batches
from .session import BatchSession

__all__ = ['Batch', 'cut_batches', 'BatchSession']
