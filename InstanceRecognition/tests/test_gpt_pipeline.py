"""Primary-method routing and cache isolation, without models or API requests."""
import json
from pathlib import Path
import pytest
from src import config
from src.pipeline import InstanceRecognitionPipeline,InferenceFailure
from src.model_service import ModelServiceManager
from web.server import public_summary

class Reader: pass
class Service: pass

def test_primary_result_cache_does_not_reuse_yoloe(tmp_path):
    gpt=InstanceRecognitionPipeline(Reader(),tmp_path,Service(),method='gpt_sam31')
    old=InstanceRecognitionPipeline(Reader(),tmp_path,Service(),method='yoloe_sam31')
    assert gpt.result_root('scene0000_00',22)==tmp_path/'results_gpt/scene0000_00/frame_000022'
    assert old.result_root('scene0000_00',22)==tmp_path/'results/scene0000_00/frame_000022'

def test_primary_service_does_not_start_yoloe(tmp_path):
    manager=ModelServiceManager(tmp_path,method='gpt_sam31')
    try:
        assert set(manager.workers)=={'sam31_text'}
        assert not any(w.status()['running'] for w in manager.workers.values())
    finally:manager.shutdown()

def test_gpt_summary_allows_empty_text_prompt_result(tmp_path):
    sam={'instance_count':0,'instances':[],'raw_candidates':[],
         'unique_classes':[],'timing':{},'policy':{'yoloe_input_consumed':False},
         'artifacts':{'annotated_image':'annotated.jpg','instance_id_map':'instance_id_map.png'},
         'prompts':[{'raw_candidate_ids':[]}]}
    summary={'method':'gpt_sam31','scene_id':'scene0000_00','frame_index':22,'wall_seconds':1.,'contract':{},
             'gpt':{'requested_model':'openai/gpt-5.6-terra','elapsed_seconds':.8,
                    'labels':{'categories':[{'category':'chair'}],'unresolved_regions':[]}},'sam31':sam}
    for name in ('gpt/labels.json','sam31/predictions.json','sam31/annotated.jpg','sam31/instance_id_map.png'):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.touch()
    InstanceRecognitionPipeline._validate(summary,tmp_path)
    public=public_summary(summary)
    assert public['sam31']['instance_count']==0
    assert public['sam31']['empty_prompt_count']==1
    assert len(public['gpt']['categories'])==1
    assert 'yoloe' not in public
    (tmp_path/'gpt/labels.json').unlink()
    with pytest.raises(InferenceFailure):InstanceRecognitionPipeline._validate(summary,tmp_path)

def test_primary_rejects_consumed_yoloe_input(tmp_path):
    summary={'method':'gpt_sam31','sam31':{'instance_count':0,'instances':[],
              'policy':{'yoloe_input_consumed':True}}}
    with pytest.raises(InferenceFailure):InstanceRecognitionPipeline._validate(summary,tmp_path)


def test_primary_runs_text_stage_and_reuses_completed_cache(tmp_path,monkeypatch):
    from src import gpt_labels
    source=tmp_path/'source.jpg';source.write_bytes(b'test RGB')
    labels={'requested_model':'openai/gpt-5.6-terra','elapsed_seconds':1,
            'labels':{'categories':[],'unresolved_regions':[]}}
    calls=[]
    def label(source_path,scene,fid,output,force=False):
        calls.append('gpt');output.mkdir(parents=True,exist_ok=True)
        (output/'labels.json').write_text(json.dumps(labels));return labels
    class TestReader:
        def frame_path(self,scene,fid):return source
    class TextService:
        def infer_sam31_text(self,source_path,labels_path,output):
            calls.append('sam_text');assert json.loads(labels_path.read_text())==labels
            output.mkdir(parents=True,exist_ok=True)
            result={'instance_count':0,'instances':[],'raw_candidates':[],
                    'policy':{'yoloe_input_consumed':False},
                    'artifacts':{'annotated_image':'annotated.jpg','instance_id_map':'instance_id_map.png'}}
            for name in ('annotated.jpg','instance_id_map.png'):(output/name).touch()
            (output/'predictions.json').write_text(json.dumps(result));return result
    monkeypatch.setattr(gpt_labels,'label_frame',label)
    pipeline=InstanceRecognitionPipeline(TestReader(),tmp_path,TextService(),method='gpt_sam31')
    monkeypatch.setattr(pipeline,'_assert_dependencies',lambda:None)
    result=pipeline.run('scene0000_00',0)
    assert result['method']=='gpt_sam31' and not result['cached']
    assert pipeline.run('scene0000_00',0)['cached']
    assert calls==['gpt','sam_text']
