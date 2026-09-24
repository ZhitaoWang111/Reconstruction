"""Reuse exported per-mask points and preserve an immutable input audit."""
import colorsys
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from src.geometry.incremental_objects import voxelize
from .state import save

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT/'runtime/mask_pointclouds/scene0000_00/amb3r_gt_raw'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


MODEL_FIELDS = ('id', 'seq', 'source_frame_id', 'category_label', 'bbox_xyxy',
                'image_size_wh', 'overlay_color_rgb', 'mask_id', 'point_count', 'centroid', 'bounds')


def model_observation(observation):
    """Keep storage/provenance fields in the audit, outside model context."""
    return {key: observation[key] for key in MODEL_FIELDS}


class Observations:
    def __init__(self, source, count, out):
        self.source, self.out = Path(source).resolve(), Path(out)
        policy = {'version': 4, 'overlay_text': 'frame-local mask ID (M01, M02, ...)', 'images': 'full-frame RGB + instance-colored overlay', 'bbox': 'original pixels xyxy exclusive maximum'}
        policy_path = self.out/'evidence_policy.json'
        if policy_path.exists():
            if json.loads(policy_path.read_text()) != policy: raise ValueError('Evidence policy changed; use new output')
        elif (self.out/'observations.json').exists():
            raise ValueError('Legacy evidence directory; use a new output to preserve history')
        save(policy_path, policy)
        self.manifest=json.loads((self.source/'manifest.json').read_text())
        self.frames=self.manifest['frames'][:count]
        if len(self.frames)!=count or [f['sequence_index'] for f in self.frames]!=list(range(count)):
            raise ValueError('Expected contiguous sampled sequences starting at zero')
        self.items={};self.points={};self.colors={};self.views={}
        self.audit={'manifest_sha256':digest(self.source/'manifest.json'),'files':{}}
        for seq,frame in enumerate(self.frames):
            path=(self.source/frame['npz']).resolve()
            self.audit['files'][str(path)]=digest(path)
            rgb=(self.source/frame['original_rgb_image']).resolve()
            self.audit['files'][str(rgb)]=digest(rgb)
            with np.load(path) as z:
                points=z['points_world'];colors=z['rgb'];bits=z['mask_bits']
                if points.shape!=colors.shape or points.shape[1]!=3 or len(points)!=len(bits):
                    raise ValueError('Invalid point export')
            with Image.open(rgb) as rgb_image: image_size = list(rgb_image.size)
            if len(frame['instances'])>32:raise ValueError('mask_bits capacity exceeded')
            for i,item in enumerate(frame['instances']):
                oid=f'{seq:04d}:{item["instance_id"]:03d}'
                mask=(self.source/item['mask_file']).resolve()
                self.audit['files'][str(mask)]=digest(mask)
                with Image.open(mask) as mask_image:
                    if list(mask_image.size) != image_size: raise ValueError('RGB/mask size mismatch')
                    bbox = mask_image.convert('L').getbbox()
                color = [round(v*255) for v in colorsys.hsv_to_rgb((i*.61803398875)%1, .75, .95)]
                keep=((bits & np.uint32(1<<i))!=0)&np.isfinite(points).all(1)
                self.points[oid]=points[keep].copy();self.colors[oid]=colors[keep].copy()
                p=self.points[oid]
                self.items[oid]={'id':oid,'seq':seq,'source_frame_id':frame['source_frame_id'],
                    'instance_id':item['instance_id'],'category_label':item['class_name'],'image':str(rgb),'mask':str(mask),
                    'bbox_xyxy':list(bbox) if bbox else None,'image_size_wh':image_size,
                    'bbox_convention':'original RGB pixels; origin top-left; xmax/ymax exclusive',
                    'overlay_color_rgb':color,
                    'mask_id':f'M{i+1:02d}',
                    'point_file':str(path),'mask_bit':i,'point_count':len(p),
                    'centroid':p.mean(0).tolist() if len(p) else None,
                    'bounds':[p.min(0).tolist(),p.max(0).tolist()] if len(p) else None}
        audit_path=self.out/'input_audit.json'
        if audit_path.exists() and json.loads(audit_path.read_text()) != self.audit:
            raise ValueError('Input changed; cannot resume')
        save(self.out/'observations.json',self.items);save(audit_path,self.audit)

    def rgb_view(self, seq):
        return (self.source/self.frames[seq]['original_rgb_image']).resolve()

    def view(self, seq, observation=None):
        key=observation or f'frame_{seq:04d}'
        if key in self.views:return self.views[key]
        frame=self.frames[seq]
        im=Image.open(self.source/frame['original_rgb_image']).convert('RGB')
        ids=[k for k,o in self.items.items() if o['seq']==seq and (observation is None or k==observation)]
        overlay=Image.new('RGBA',im.size);draw=ImageDraw.Draw(overlay)
        font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',max(17,im.width//45))
        for oid in ids:
            o=self.items[oid];mask=Image.open(o['mask']).convert('L')
            if mask.size!=im.size:raise ValueError('RGB/mask size mismatch')
            box=mask.getbbox()
            if box is None:continue
            tint=Image.new('RGBA',im.size,(*o['overlay_color_rgb'],0));tint.putalpha(mask.point(lambda x:65 if x else 0));overlay=Image.alpha_composite(overlay,tint);draw=ImageDraw.Draw(overlay)
            draw.rectangle(box,outline=(*o['overlay_color_rgb'],255),width=4)
            text=o["mask_id"]
            x,y=box[:2];bounds=draw.textbbox((x,y),text,font=font);draw.rectangle(bounds,fill=(0,0,0,225));draw.text((x,y),text,font=font,fill='white')
        im=Image.alpha_composite(im.convert('RGBA'),overlay).convert('RGB')
        path=self.out/'evidence'/f'{key.replace(":","_")}.jpg';path.parent.mkdir(exist_ok=True);im.save(path,quality=87)
        self.views[key]=path;return path

    def fused(self,state,oid,voxel):
        members=[k for k,a in state['assignments'].items() if a['object_id']==oid]
        p=np.concatenate([self.points[k] for k in members]);c=np.concatenate([self.colors[k] for k in members])
        if len(p)==0:return np.empty((0,3),np.float32),np.empty((0,3),np.uint8)
        _,p,c=voxelize(p,c,voxel)
        return p,c.astype(np.uint8)
