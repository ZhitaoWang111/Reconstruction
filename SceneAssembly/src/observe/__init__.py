"""M5 observation: what the agent sees. Nothing here judges anything.

    from observe import Observer
    obs = Observer(scene, structure, state, initial, isaac, out_dir)
    pack = obs.batch_opening(batch_oids, batch_seqs)   # text cards, four-side views, boxed key frames
    img  = obs.stitched(oid, seq)                      # photo | Isaac render, same camera

Decided 2026-11-20 (D04, D05):
  * boxes on key frames come from the object's own scan points projected into the
    frame, kept only where the sensor depth says the point is actually visible;
    the object's own mask box is the fallback;
  * a small legend in the corner maps colours to object ids;
  * key frames are given at full size (1296x968); four-side views are four 210 px
    tiles in a strip; stitched images are two 648x484 halves side by side;
  * four-side views are rendered with the asset turned to its starting up axis;
  * a stitched image comes back automatically after every change that takes effect.
"""
from .observer import Observer

__all__ = ['Observer']
