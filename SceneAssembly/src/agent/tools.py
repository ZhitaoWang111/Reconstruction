"""Tool definitions (strict JSON schemas) and the instructions the agent works under."""
from __future__ import annotations


def _obj(props: dict, required=None) -> dict:
    """Strict object: every property listed in required (optional ones are nullable)."""
    return {'type': 'object', 'additionalProperties': False, 'properties': props,
            'required': list(props) if required is None else required}


def _nullable(t) -> dict:
    t = dict(t)
    t['type'] = [t['type'], 'null'] if isinstance(t['type'], str) else t['type'] + ['null']
    return t


NUM = {'type': 'number'}
STR = {'type': 'string'}
OID = {'type': 'string', 'description': 'object id, e.g. obj_0037'}

SUPPORT = _obj({
    'kind': {'type': 'string', 'enum': ['floor', 'object', 'wall'],
             'description': 'floor: stands on the floor. object: rests on another object (give on). '
                            'wall: hangs on a wall (give height).'},
    'on': _nullable({'type': 'string', 'description': 'kind=object: the id of the object it rests on'}),
    'surface_z': _nullable({'type': 'number', 'description': 'kind=object: rough height (m) of the surface it rests on, '
                                                              'e.g. a lower shelf; null = from where its bottom is now'}),
    'height': _nullable({'type': 'number', 'description': 'kind=wall: height of its bottom above the floor (m)'}),
    'against': _nullable({'type': 'array', 'items': STR,
                          'description': 'pushed flush against a wall: ["auto"] for the nearest wall, or one or two '
                                         'wall ids from the plan view (two only in a corner). null = not against a wall'}),
})

POSE = {
    'x': {'type': 'number', 'description': 'object centre, m (scan frame)'},
    'y': {'type': 'number', 'description': 'object centre, m (scan frame)'},
    'yaw_deg': {'type': 'number', 'description': 'turn about vertical, counter-clockwise from +X'},
    'up': {'type': 'string', 'enum': ['+X', '-X', '+Y', '-Y', '+Z', '-Z'],
           'description': "which of the asset's own axes points to the sky (see the four-side view)"},
    'scale': {'type': 'array', 'items': NUM, 'minItems': 3, 'maxItems': 3,
              'description': 'factors on the upright asset\'s own length (x), width (y), height (z)'},
    'tilt_deg': {'type': 'number', 'description': '0 unless the object leans; tips its top over by this much'},
    'tilt_toward_deg': {'type': 'number', 'description': 'horizontal direction the top leans toward (same convention as yaw)'},
}

TOOLS = [
    {'type': 'function', 'name': 'set_order', 'strict': True,
     'description': 'Set the order you will process this batch\'s objects in. Required before begin. '
                    'Later calls may only reorder objects not processed yet.',
     'parameters': _obj({'order': {'type': 'array', 'items': STR}, 'reason': STR})},
    {'type': 'function', 'name': 'begin', 'strict': True,
     'description': 'Start working on an object. It enters the scene at its current pose; you get its stitched image.',
     'parameters': _obj({'object': OID})},
    {'type': 'function', 'name': 'set_pose', 'strict': True,
     'description': 'Move an object. Takes effect at once, then is checked: if it made anything worse (a new '
                    'penetration, into a wall, out of the room, off its support) it is undone automatically and you '
                    'are told why. Objects resting on it move with it. Answers with the check result and a stitched image.',
     'parameters': _obj({'object': OID, **POSE, 'support': SUPPORT, 'reason': STR})},
    {'type': 'function', 'name': 'undo', 'strict': True,
     'description': 'Take back this object\'s last change (what rests on it follows it back).',
     'parameters': _obj({'object': OID, 'reason': STR})},
    {'type': 'function', 'name': 'view', 'strict': True,
     'description': 'A stitched image of an object from another of this batch\'s key frames.',
     'parameters': _obj({'object': OID, 'frame': {'type': 'integer'}})},
    {'type': 'function', 'name': 'check_scene', 'strict': True,
     'description': 'Report every current problem in the scene (contacts, walls, room, supports) without changing anything.',
     'parameters': _obj({})},
    {'type': 'function', 'name': 'finish', 'strict': True,
     'description': 'Mark an object done. A 5-second gravity run checks it and what rests on it; if it falls over or '
                    'drops, its last change is undone and it stays open.',
     'parameters': _obj({'object': OID, 'note': STR})},
    {'type': 'function', 'name': 'give_up', 'strict': True,
     'description': 'Mark an object failed (e.g. the asset does not match the photo). It stays in the scene at its last pose.',
     'parameters': _obj({'object': OID, 'problem': STR})},
    {'type': 'function', 'name': 'bring_in', 'strict': True,
     'description': 'Bring an object from another batch into this one: an earlier one to fix it, or a later one that '
                    'something here rests on. At most 3 per batch. Its card, four-side view and a frame come with it.',
     'parameters': _obj({'object': OID, 'reason': STR})},
    {'type': 'function', 'name': 'end_batch', 'strict': True,
     'description': 'End the batch when every object is done or failed. A whole-batch check runs (contacts, walls, room, '
                    '15 s of gravity); objects with a problem are reopened once and you continue with them.',
     'parameters': _obj({'summary': STR})},
]

INSTRUCTIONS = """You assemble a 3-D copy of a scanned room from generated furniture assets, so that it matches the photos.
You work one batch of objects at a time; this conversation is one batch.

Frame and units. Scan frame, metres, +Z up. The floor's top is at z = {floor_z:.3f}; the room is {storey:.2f} m high.

A pose:
- x, y: where the object's centre goes.
- up: which of the asset's own axes points to the sky. The four-side view shows the asset already turned this
  way, with its axes drawn; a wrong up shows as an object lying on its side or upside down.
- scale: three factors on the upright asset's own length (x), width (y) and height (z), applied before the turn.
  Generated assets usually get the height right and the footprint wrong.
- yaw_deg: turn about the vertical, counter-clockwise from +X. Then tilt_deg / tilt_toward_deg if it leans (rare).
- You never give z. It follows from the support: floor (stands on the floor), object (rests on another object,
  give `on`; for a lower shelf give a rough surface_z), wall (hangs, give its bottom height). Add `against`
  to push it flush with a wall: ["auto"] for the nearest wall, or the wall id(s) from the plan view. Only
  the distance to the wall changes; your position along the wall is kept.
- A support must already be in the scene (processed, or in this batch). If it belongs to a later batch, bring it in.

Where things start. Each object starts at its scan position with SAM3D's orientation and scale 1; its height is
the bottom of its scan points and no support is declared. When the card says the orientation is NOT confirmed,
SAM3D had the object upside down: check which side faces where against the four-side view and the photos.
The first time you move an object you declare its support.

What you see.
- Key frames: the batch's photos, with this batch's objects boxed in colour; the legend names them.
- Four-side views: each asset alone from +X, +Y, -X, -Y.
- Plan view: the walls with their ids, and where each key frame's camera stands and looks.
- Stitched images: the photo on the left, a render from the same camera on the right. The render shows the
  object you are working on and the objects already processed, nothing else; walls and floor are there.
  Compare them: position, size, which side faces the camera, what it stands on.

Checks. Every change is checked right away. Order of importance: physically right (no penetration, not in a
wall, inside the room) > supports right (rests on what you declared, against the wall you named) > looks like
the photos > overlaps the scan points. A change that makes a physical or support problem worse is undone
automatically and you are told why; a problem that was already there and did not get worse is reported but kept.
Three automatic undos in a row mark the object failed. Scale far from 1 and a poorer fit to the scan are only reported.
When the batch ends, a stricter check uses absolute limits: anything still penetrating, in a wall or out of the
room is reopened once; if it is still wrong after that it is marked failed. So deal with problems the starting
pose brought, too.

Work. First set_order (supports before what rests on them). Then for each object: begin, compare, set_pose until
it matches, finish. Use view for another key frame, check_scene to see the whole picture, give_up when the asset
cannot be made to match. When every object is done or failed, end_batch. Keep reasons short."""
