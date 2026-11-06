"""Build and audit a window input before any agent actions occur."""
import json
from .client import image
from .observations import model_observation
from .spatial_context import build_spatial_context
from .state import save


def build_window_content(data, tools, first, cutoff, max_calls):
    state = tools.state
    new = [o for o in data.items.values() if first <= o['seq'] <= cutoff]
    tools.evidence.update(o['id'] for o in new)
    scene, candidates = build_spatial_context(data, state, new)
    for object_id in state['objects']:
        tools.representative(object_id)
    text = {'window': [first, cutoff], 'state': tools.listing(), 'max_calls': max_calls,
            'spatial_candidates_policy': ('candidates by gap (5th-pctile nearest distance); at most 8 + extras with gap<0.5 m, cap 15; '
                                          'each has gap_m and overlap (fraction within 10 cm); objects beyond 2 m are omitted')}
    content = [{'type': 'text', 'text': json.dumps(text, ensure_ascii=False)}]
    for seq in range(first, cutoff+1):
        observations = [{**model_observation(o), 'spatial_candidates': candidates[o['id']]} for o in new if o['seq'] == seq]
        content.extend([{'type': 'text', 'text': json.dumps({'sampled_sequence': seq, 'observations': observations}, ensure_ascii=False)},
                        {'type': 'text', 'text': f'Current sampled sequence {seq}: original RGB'}, image(data.rgb_view(seq)),
                        {'type': 'text', 'text': f'Current sampled sequence {seq}: full-frame colored mask overlay'}, image(data.view(seq))])
    tools.sent_images.update(str(p) for seq in range(first,cutoff+1) for p in (data.rgb_view(seq),data.view(seq)))
    audit_representatives = []
    for obj in tools.listing()['objects']:
        record = tools.representative(obj['id'])
        audit_representatives.append(record)
    content.append({'type':'text','text':json.dumps({
        'history_access':'Archived object records only; individual historical observations are not shown here. Call inspect_object(object_id) for representative metadata and one image (or all held observations when the object is full); call inspect_observations(ids) for additional views. The representative image is full-frame RGB with the original SAM3 mask bbox and Object ID.'},ensure_ascii=False)})
    save(data.out/'window_inputs'/f'{first:04d}_context.json',
         {'base_version': state['version'], 'window': [first, cutoff],
          'spatial_candidates': candidates, 'historical_representatives': audit_representatives})
    return content
