"""Adapter to the audited API client stored in the user's API directory."""
from __future__ import annotations
import hashlib, importlib.util, json
from functools import lru_cache
from pathlib import Path
from src import config

@lru_cache(maxsize=1)
def client_module():
    path=config.GPT_API_DIR/'label_frames.py'
    spec=importlib.util.spec_from_file_location('scannet_gpt_label_client',path)
    if spec is None or spec.loader is None: raise RuntimeError('Cannot load GPT API client')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def label_frame(source: Path, scene_id: str, frame_index: int, output: Path, force=False) -> dict:
    client=client_module(); settings=client.settings()
    key=settings.get('VIBEG_API_KEY','')
    if not key: raise RuntimeError('Missing VIBEG_API_KEY in /home/seer/wzt/API/.env or environment')
    base=settings.get('VIBEG_OPENAI_BASE_URL','https://vibeg.seer-robotics.cn/openai/v1').rstrip('/')
    prompt=(config.GPT_API_DIR/'prompt_en.txt').read_text()
    record={'scene_id':scene_id,'sequence_index':frame_index,'source_frame_id':frame_index,
        'source':str(source.resolve()),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    request_config={'model':config.GPT_MODEL,'base_url':base,'schema':client.SCHEMA,
        'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'source_sha256':record['source_sha256'],
        'reasoning_effort':'none','image_detail':'high','max_completion_tokens':4096}
    config_path=output/'request_config.json'
    if config_path.is_file() and json.loads(config_path.read_text())!=request_config and not force:
        raise ValueError('Cached GPT request configuration changed; use force to regenerate')
    client.save(config_path,request_config)
    # Online requests use the raw frame ID as their local cache key; no sampling is implied.
    result=client.label_one(record,output,config.GPT_MODEL,base,key,prompt,force=force)
    if 'error' in result: raise RuntimeError('GPT labeling failed: '+str(result['error']))
    result['input_policy']='single original RGB; no YOLOE boxes, labels, masks, or temporal history'
    result['prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
    client.save(output/'labels.json',result)
    return result
