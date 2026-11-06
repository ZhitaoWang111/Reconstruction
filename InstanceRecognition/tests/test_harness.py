import json
from types import SimpleNamespace
import pytest
from src.harness.state import initial,propose
from src.harness.tools import Tools

OBS={'0000:001':{'seq':0},'0001:001':{'seq':1},'0005:001':{'seq':5}}
E={'0000:001','0001:001'}
def action(kind,**kw):return dict(action=kind,evidence=['0000:001'],reason='visible evidence',**kw)
def first():
 return propose(initial(),{'base_version':0,'actions':[action('create',object_id='Object1',object_label='toilet'),action('hold',object_id='Object1',observations=['0000:001'])]},OBS,0,E)['state']
def test_new_object_and_immutable_source():
 s=initial();new=first();assert new['assignments']['0000:001']['object_id']=='Object1';assert s==initial()
def test_future_and_missing_observations_rejected():
 for ids in [['0005:001'],[]]:
  with pytest.raises(ValueError):propose(initial(),{'base_version':0,'actions':[action('discard',observations=ids)]},OBS,0,E)
 with pytest.raises(ValueError):propose(first(),{'base_version':1,'actions':[action('discard',observations=['0000:001'])]},OBS,1,E)
def test_stale_and_invented_evidence():
 with pytest.raises(ValueError):propose(first(),{'base_version':0,'actions':[]},OBS,0,E)
 a=action('discard',observations=['0000:001']);a['evidence']=['hallucination']
 with pytest.raises(ValueError):propose(initial(),{'base_version':0,'actions':[a]},OBS,0,E)
def test_merge_then_reassign_keeps_old_state():
 s=first();p={'base_version':1,'actions':[action('create',object_id='Object2',object_label='base'),action('hold',object_id='Object2',observations=['0001:001'])]}
 s2=propose(s,p,OBS,1,E)['state'];s3=propose(s2,{'base_version':2,'actions':[action('merge',source_object='Object2',object_id='Object1')]},OBS,1,E)['state']
 assert s2['assignments']['0001:001']['object_id']=='Object2'
 assert s3['aliases']['Object2']=='Object1'
 checked=propose(s3,{'base_version':3,'actions':[action('create',object_id='Object3',object_label='bin'),action('hold',object_id='Object3',observations=['0001:001'])]},OBS,1,E)
 assert set(checked['affected'])=={'Object1','Object3'}
 assert checked['state']['assignments']['0001:001']['object_id']=='Object3'
def test_pending_and_repair_are_accounted():
 s=propose(initial(),{'base_version':0,'actions':[action('discard',observations=['0000:001']),action('repair',observations=['0001:001'])]},OBS,1,E)['state']
 assert all(a['object_id'] is None for a in s['assignments'].values())
def test_tools_block_future_and_unchecked_commit():
 class Data:items=OBS
 tools=Tools(Data(),initial(),0)
 with pytest.raises(ValueError):tools.observe(['0005:001'])
 with pytest.raises(ValueError):tools.call('commit_update',{'proposal_id':'guess'})
def test_duplicate_updates_rejected():
 with pytest.raises(ValueError,match='twice'):
  propose(initial(),{'base_version':0,'actions':[action('discard',observations=['0000:001']),action('repair',observations=['0000:001'])]},OBS,0,E)

def test_materialization_reassignment_rebuilds_from_original(tmp_path):
 import numpy as np
 from src.harness.publish import materialize
 class Data:
  def fused(self,state,oid,voxel):
   p=np.array([[int(k[0:4]),0,0] for k,a in state['assignments'].items() if a['object_id']==oid],dtype=np.float32)
   return p,np.zeros(p.shape,dtype=np.uint8)
 s=first()
 checked=propose(s,{'base_version':1,'actions':[action('hold',object_id='Object1',observations=['0001:001'])]},OBS,1,E)
 materialize(Data(),checked,tmp_path/'v2',None,.02)
 changed=propose(checked['state'],{'base_version':2,'actions':[action('create',object_id='Object2',object_label='new'),action('hold',object_id='Object2',observations=['0001:001'])]},OBS,1,E)
 materialize(Data(),changed,tmp_path/'v3',tmp_path/'v2',.02)
 with np.load(tmp_path/'v2/clouds/Object1.npz') as z:assert len(z['points'])==2
 with np.load(tmp_path/'v3/clouds/Object1.npz') as z:assert len(z['points'])==1
 with np.load(tmp_path/'v3/clouds/Object2.npz') as z:assert z['points'][0,0]==1

def test_stream_fragments_and_truncation():
 from src.harness.client import read_stream
 chunks=[{'id':'a','model':'test','choices':[{'delta':{'tool_calls':[{'index':0,'id':'c1','function':{'name':'propose_','arguments':'{'}}]}}]}, {'choices':[{'delta':{'tool_calls':[{'index':0,'function':{'name':'update','arguments':'}'}}]},'finish_reason':'tool_calls'}]}, {'usage':{'total_tokens':3},'choices':[]}]
 lines=[('data: '+json.dumps(c)+'\n').encode() for c in chunks]+[b'data: [DONE]\n']
 body=read_stream(lines);assert body['choices'][0]['message']['tool_calls'][0]['function']=={'name':'propose_update','arguments':'{}'}
 assert body['usage']['total_tokens']==3
 with pytest.raises(ValueError,match='Incomplete'):read_stream(lines[:-1])

def test_new_run_persistence_drops_retired_instance_field(tmp_path):
 from src.harness.state import save
 path=tmp_path/'nested.json'
 save(path, {'observations':[{'instance_id':7,'mask_id':'M07'}],
             'tool':{'result':{'mask_id':'M07','instance_id':7}},
             'arguments':json.dumps({'proposal_json':json.dumps({'mask_id':'M07','instance_id':7})})})
 text=path.read_text()
 assert 'mask_id' not in text
 saved=json.loads(text)
 assert saved['observations']==[{'instance_id':7}]
 assert saved['tool']=={'result':{'instance_id':7}}
 assert json.loads(saved['arguments'])=={'proposal_json':json.dumps({'instance_id':7},separators=(',', ':'))}

def test_new_run_persistence_normalizes_retired_token_in_text(tmp_path):
 from src.harness.state import dumps_sanitized, save
 path=tmp_path/'text.json'
 save(path, {'reason':'Compared the old mask_id field with mask_id candidates'})
 assert 'mask_id' not in path.read_text()
 assert json.loads(path.read_text())['reason']=='Compared the old instance_id field with instance_id candidates'
 assert 'mask_id' not in dumps_sanitized({'message':'mask_id'})


def test_full_frame_evidence_and_bbox(tmp_path):
 from PIL import Image
 import numpy as np
 from src.harness.observations import Observations
 source=tmp_path/'source';source.mkdir();out=tmp_path/'out';out.mkdir()
 Image.new('RGB',(120,80),(60,90,120)).save(source/'rgb.png')
 a=np.zeros((80,120),np.uint8);a[20:40,30:55]=255
 Image.fromarray(a).save(source/'mask.png')
 np.savez(source/'points.npz',points_world=np.array([[0.,0.,0.]]),rgb=np.array([[1,2,3]],np.uint8),mask_bits=np.array([3],np.uint32))
 (source/'manifest.json').write_text(json.dumps({'frames':[{'sequence_index':0,'source_frame_id':0,'npz':'points.npz','original_rgb_image':'rgb.png','instances':[{'instance_id':1,'class_name':'chair','mask_file':'mask.png'},{'instance_id':2,'class_name':'chair','mask_file':'mask.png'}]}]}))
 d=Observations(source,1,out)
 assert 'mask_id' not in (out/'evidence_policy.json').read_text()
 assert d.items['0000:001']['instance_id']==1
 assert 'mask_id' not in d.items['0000:001']
 assert d.items['0000:002']['instance_id']==2
 assert 'mask_id' not in d.items['0000:002']
 result2,_=Tools(d,initial(),0).observe(['0000:002'])
 assert result2['observations'][0]['instance_id']==2
 assert 'mask_id' not in result2['observations'][0]
 assert d.items['0000:001']['bbox_xyxy']==[30,20,55,40]
 assert d.items['0000:001']['image_size_wh']==[120,80]
 assert d.items['0000:001']['overlay_color_rgb']!=d.items['0000:002']['overlay_color_rgb']
 assert Image.open(d.view(0,'0000:001')).size==(120,80)
 result,images=Tools(d,initial(),0).observe(['0000:001'])
 assert images==[d.rgb_view(0),d.view(0,'0000:001')]
 assert 'original RGB' in result['image_order']
 assert result['observations'][0]['id']=='0000:001'
 assert result['observations'][0]['bbox_xyxy']==[30,20,55,40]
 assert not {'image','mask','point_file','mask_bit'} & result['observations'][0].keys()
 legacy=tmp_path/'legacy';legacy.mkdir();(legacy/'observations.json').write_text('{}')
 with pytest.raises(ValueError,match='Legacy'):Observations(source,1,legacy)


def test_explicit_label_schema():
 from src.harness.observations import MODEL_FIELDS, model_observation
 assert 'category_label' in MODEL_FIELDS and 'label' not in MODEL_FIELDS
 observation={key:None for key in MODEL_FIELDS};observation['category_label']='chair'
 assert model_observation(observation)['category_label']=='chair'
 s=first()
 assert s['objects']['Object1']['object_label']=='toilet'
 assert 'label' not in s['objects']['Object1']
 with pytest.raises(ValueError,match='object_label'):
  propose(initial(),{'base_version':0,'actions':[action('create',object_id='Object1',label='toilet')]},OBS,0,E)


@pytest.mark.parametrize('bad_id', ['O1', 'Object0', 'Object01', 'Object-1', 'Object1.0'])
def test_object_id_must_use_canonical_object_prefix(bad_id):
 with pytest.raises(ValueError):
  propose(initial(), {'base_version':0,'actions':[action('create',object_id=bad_id,object_label='toilet'),action('hold',object_id=bad_id,observations=['0000:001'])]}, OBS, 0, E)


def test_relabel_preserves_observations_identity_and_history(tmp_path):
 import copy
 import numpy as np
 from src.harness.publish import materialize
 from types import SimpleNamespace
 observations=copy.deepcopy(OBS)
 observations['0000:001']['category_label']='original category'
 before=first();original=copy.deepcopy(before);obs_before=copy.deepcopy(observations)
 t=Tools(SimpleNamespace(items=observations),before,0);t.evidence.update(E)
 proposal={'base_version':1,'actions':[action('relabel',object_id='Object1',object_label='basin')]}
 preview,_=t.call('propose_update',{'proposal_json':json.dumps(proposal)})
 assert before==original and observations==obs_before
 assert preview['affected']==[]
 ready,_=t.call('commit_update',{'proposal_id':preview['proposal_id']})
 assert ready['ready_to_commit']
 checked=t.pending;obj=checked['state']['objects']['Object1']
 assert obj['object_label']=='basin'
 assert obj['label_history']==[{'previous_label':'toilet','object_label':'basin','evidence':['0000:001'],'reason':'visible evidence','decided_at':0}]
 assert checked['state']['assignments']==before['assignments']
 previous=tmp_path/'previous';(previous/'clouds').mkdir(parents=True)
 cloud=previous/'clouds/Object1.npz';np.savez_compressed(cloud,points=np.array([[1.,2.,3.]]),rgb=np.array([[1,2,3]],dtype=np.uint8))
 class NoFusion:
  def fused(self,*args):raise AssertionError('relabel must reuse geometry')
 materialize(NoFusion(),checked,tmp_path/'next',previous,.02)
 assert cloud.stat().st_ino==(tmp_path/'next/clouds/Object1.npz').stat().st_ino
 saved=json.loads((tmp_path/'next/state.json').read_text())
 assert saved['objects']['Object1']==obj
 assert Tools(SimpleNamespace(items=observations),saved,0).listing()['objects'][0]['object_label']=='basin'


@pytest.mark.parametrize('changes', [
 {'object_id':'Object999'}, {'object_label':''}, {'object_label':'  '},
 {'object_label':None}, {'object_label':'toilet'},
 {'category_label':'basin'}, {'label':'basin'}, {'observations':['0000:001']},
 {'evidence':[]}, {'evidence':['unknown']}, {'reason':''}])
def test_relabel_rejects_invalid_or_ambiguous_updates(changes):
 a=action('relabel',object_id='Object1',object_label='basin');a.update(changes)
 with pytest.raises(ValueError):propose(first(),{'base_version':1,'actions':[a]},OBS,0,E)


def test_relabel_rejects_retired_object():
 s=first();s['aliases']['Object2']='Object1'
 with pytest.raises(ValueError,match='active'):
  propose(s,{'base_version':1,'actions':[action('relabel',object_id='Object2',object_label='basin')]},OBS,0,E)


def _window(count, start=0, seq=0):
    """A window with `count` observations; test_harness's own OBS has only three.

    `start` offsets the instance id so two windows do not collide, and `seq`
    sets the sequence index so the observations are visible at that cutoff.
    """
    obs = {f'{seq:04d}:{i:03d}': {'seq': seq} for i in range(start, start + count)}
    return obs, set(obs)


def test_retired_action_names_are_refused():
    """`assign` and `defer` are gone; the host names the action it does not know."""
    for retired in ('assign', 'defer'):
        payload = action(retired, observations=['0000:001'])
        if retired == 'assign':
            payload['object_id'] = 'Object1'
        with pytest.raises(ValueError, match='Unknown action'):
            propose(first(), {'base_version': 1, 'actions': [payload]}, OBS, 1, E)


def test_discard_is_per_observation():
    # A whole-window disposal under one reason used to pass: the window's own ids
    # are already in evidence, so it needed no inspection and no new image. That
    # is what turned 34 discarded observations in the P1 run into 535 in P3, with
    # single actions disposing of 41 at once. See DEVNOTES section 10.
    from src.harness.state import DISCARD_BATCH_LIMIT
    obs, evidence = _window(DISCARD_BATCH_LIMIT + 2)
    ids = sorted(obs)
    with pytest.raises(ValueError, match='Discard at most'):
        propose(initial(), {'base_version': 0,
                            'actions': [action('discard', observations=ids)]}, obs, 0, evidence)


def test_discard_at_the_limit_is_allowed():
    from src.harness.state import DISCARD_BATCH_LIMIT
    obs, evidence = _window(DISCARD_BATCH_LIMIT)
    ids = sorted(obs)
    checked = propose(initial(), {'base_version': 0,
                                  'actions': [action('discard', observations=ids)]}, obs, 0, evidence)
    assert all(checked['state']['assignments'][i]['status'] == 'discard' for i in ids)


def test_hold_at_limit_is_allowed():
    """An object can hold exactly HOLD_LIMIT observations without error."""
    from src.harness.state import HOLD_LIMIT
    obs, evidence = _window(HOLD_LIMIT)
    ids = sorted(obs)
    actions = [action('create', object_id='Object1', object_label='bin'),
               action('hold', object_id='Object1', observations=ids)]
    checked = propose(initial(), {'base_version': 0, 'actions': actions}, obs, 0, evidence)
    assert all(checked['state']['assignments'][i]['object_id'] == 'Object1' for i in ids)
    assert all(checked['state']['assignments'][i]['status'] == 'held' for i in ids)


def test_hold_over_limit_is_rejected():
    """Holding more than HOLD_LIMIT observations onto one object is an error."""
    from src.harness.state import HOLD_LIMIT
    obs, evidence = _window(HOLD_LIMIT + 1)
    ids = sorted(obs)
    actions = [action('create', object_id='Object1', object_label='bin'),
               action('hold', object_id='Object1', observations=ids)]
    with pytest.raises(ValueError, match='exceeding the limit'):
        propose(initial(), {'base_version': 0, 'actions': actions}, obs, 0, evidence)


def test_evict_makes_room_for_new_hold():
    """Evicting an existing held observation allows one new hold on a full object."""
    from src.harness.state import HOLD_LIMIT
    # Fill the object to the limit in window 0.
    obs1, ev1 = _window(HOLD_LIMIT, start=0, seq=0)
    ids1 = sorted(obs1)
    actions1 = [action('create', object_id='Object1', object_label='bin'),
                action('hold', object_id='Object1', observations=ids1)]
    s1 = propose(initial(), {'base_version': 0, 'actions': actions1}, obs1, 0, ev1)['state']
    # Window 1: one new observation, evict the first held one. `observations`
    # here is every observation the run has seen so far, not just this window's,
    # because `visible` is derived from it by cutoff.
    obs2, _ev2 = _window(1, start=0, seq=1)
    new_obs = sorted(obs2)[0]
    all_obs = {**obs1, **obs2}
    evidence = set(all_obs) | {ids1[0]}
    hold_act = action('hold', object_id='Object1', observations=[new_obs])
    hold_act['evict'] = ids1[0]
    hold_act['evidence'] = [new_obs, ids1[0]]
    checked = propose(s1, {'base_version': 1, 'actions': [hold_act]}, all_obs, 1, evidence)
    state = checked['state']
    assert state['assignments'][new_obs]['status'] == 'held'
    assert state['assignments'][ids1[0]]['status'] == 'discard'
    held = [k for k, a in state['assignments'].items()
            if a.get('object_id') == 'Object1' and a['status'] in ('held', 'assigned')]
    assert len(held) == HOLD_LIMIT


def test_listing_hides_discards_but_state_keeps_them():
 """A discard is a closed decision, not a queue: the listing omits it, the
 state still records it so audit and coverage keep working.

 Across the 500-frame run no discarded observation was ever reclaimed, because
 its mask never appears in a later window. Broadcasting 550 of them every
 window cost 149 KB of the first message and bought nothing.
 """
 s=propose(initial(),{'base_version':0,'actions':[action('discard',observations=['0000:001']),
     action('exclude',observations=['0001:001'])]},OBS,1,E)['state']
 listing=Tools(SimpleNamespace(items=OBS),s,0).listing()
 assert set(listing)=={'version','objects','aliases'}
 assert '0000:001' not in json.dumps(listing)
 assert s['assignments']['0000:001']['status']=='discard'
 assert s['assignments']['0001:001']['status']=='exclude'
