"""M3 initial pose: where an object starts before the agent touches it.

    from initial import InitialPoses
    init = InitialPoses(scene, structure, run_dir)
    init.compute_all()                 # 52 objects, cached in <run>/initial_poses.json
    init.apply(state, 'obj_0037')      # writes it into the M2 state as version 1

The start is SAM3D's orientation plus one bounded registration against the
object's own scan points (D03), never a set of candidates:

  * up axis   the asset axis pointing most nearly to +Z. When that is the asset's
              -Z, +Z is used instead and the object is marked flipped: generated
              assets are upright, so -Z can only be SAM3D's whole rotation turned
              over (D03 M3-2). A side axis (-X, +Y, ...) is kept: those assets were
              generated lying down (the pillow, the guitar case).
  * yaw       from that same rotation; the flipped ones are marked unconfirmed.
  * tilt      0 (D03 M3-3).
  * position  the centre of the scan points' bounding box; the bottom at the scan's
              2% height, but never below the floor (D03 M3-1).
  * scale     (1, 1, 1); the handoff already sized the asset from the scan height.
  * refinement one bounded fit (yaw +-20 degrees, +-25 cm, height fixed) against the
              scan points. It may not end worse than where it started.

The support is not declared: the start sits at a height (`Support('height', ...)`),
so nothing claims the object rests on anything until the agent says so (D17).
"""
from .plan import InitialPoses, initial_pose

__all__ = ['InitialPoses', 'initial_pose']
