"""Check lazy visual evidence delivery without an API call."""
from types import SimpleNamespace
import json
import numpy as np
from PIL import Image
from src.harness.observations import MODEL_FIELDS
from src.harness.tools import Tools
from src.harness.inputs import build_window_content


def fixture(tmp_path):
    rgb=tmp_path/'rgb.png';overlay=tmp_path/'overlay.png'
    Image.fromarray(np.zeros((40,60,3),dtype=np.uint8)).save(rgb)
    Image.fromarray(np.full((40,60,3),100,dtype=np.uint8)).save(overlay)
    items={}
    for seq in (0,1,2):
        oid=f'{seq:04d}:001'
        items[oid]={k:None for k in MODEL_FIELDS}
        items[oid].update(id=oid,seq=seq,bbox_xyxy=[2,2,30,30],category_label='chair')
    data=SimpleNamespace(items=items,out=tmp_path/'output',rgb_view=lambda seq:rgb,view=lambda *args:overlay)
    state={'version':1,'cutoff':0,'objects':{'Object1':{'object_label':'chair'}},
           'assignments':{'0000:001':{'object_id':'Object1'}},'aliases':{}}
    return data,Tools(data,state,1)


def test_initial_input_has_no_historical_images_or_evidence(tmp_path,monkeypatch):
    data,tools=fixture(tmp_path)
    monkeypatch.setattr('src.harness.inputs.build_spatial_context',lambda *args:({}, {'0001:001':[]}))
    content=build_window_content(data,tools,1,1,14)
    assert sum(part['type']=='image_url' for part in content)==2
    assert tools.evidence=={'0001:001'}
    payloads=[json.loads(part['text']) for part in content
              if part['type']=='text' and part['text'].startswith('{')]
    assert all('historical_representatives' not in payload for payload in payloads)
    assert payloads[0]['state']['objects'][0]['object_label']=='chair'
    assert 'global_object_geometry' in payloads[0]
    assert payloads[1]['observations'][0]['id']=='0001:001'
    assert 'inspect_object' in payloads[-1]['history_access']
    assert not (data.out/'evidence').exists()
    audit=json.loads((data.out/'window_inputs/0001_context.json').read_text())
    assert audit['historical_representatives'][0]['observation']['id']=='0000:001'
    result,images=tools.call('inspect_object',{'object_id':'Object1'})
    assert result['observation']['id']=='0000:001'
    assert result['observation']['category_label']=='chair'
    assert len(images)==1
    assert 'selection' not in result and 'image_description' not in result


def test_object_inspection_one_image_then_reuse_and_other_format(tmp_path):
    data,tools=fixture(tmp_path)
    tools.listing()
    assert tools.evidence==set()
    result,images=tools.call('inspect_object',{'object_id':'Object1'})
    assert len(images)==len(result['visual_images'])==1
    assert result['visual_images'][0]['kind']=='historical_rgb_bbox'
    assert tools.evidence=={'0000:001'}
    repeated,images=tools.call('inspect_object',{'object_id':'Object1'})
    assert not images and len(repeated['reused_images'])==1
    extra,images=tools.observe(['0000:001'])
    assert len(images)==2  # Other visual formats remain inspectable.
    repeated,images=tools.observe(['0000:001'])
    assert not images and len(repeated['reused_images'])==2


def test_representative_cannot_use_uncommitted_observation(tmp_path):
    data,tools=fixture(tmp_path)
    tools.state['assignments']['0001:001']={'object_id':'Object1'}
    assert tools.representative('Object1')['observation']['seq']==0
