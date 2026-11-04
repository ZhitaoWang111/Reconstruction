"""Reuse exported per-mask points and preserve an immutable input audit."""
import colorsys
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont
from src import config
from src.geometry.incremental_objects import voxelize
from .state import save

ROOT = Path(__file__).resolve().parents[2]
# The AMB3R-lifted source is the only geometry this tree builds, so it is the
# default and there is no fallback to a retired policy: a missing source must fail
# loudly rather than silently run against geometry nobody chose.
DEFAULT_SOURCE = config.ARTIFACTS_ROOT / 'sources/scene0000_00/scene_amb3r_v1'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


MODEL_FIELDS = ('id', 'seq', 'source_frame_id', 'category_label', 'bbox_xyxy',
                'image_size_wh', 'overlay_color_rgb', 'instance_id', 'point_count', 'centroid', 'bounds')


def model_observation(observation):
    """Keep storage/provenance fields in the audit, outside model context."""
    return {key: observation[key] for key in MODEL_FIELDS}


class Observations:
    def __init__(self, source, count, out):
        self.source, self.out = Path(source).resolve(), Path(out)
        policy = {'version': 5, 'overlay_text': 'frame-local instance_id label (e.g. 007); only instance_id is exposed to the agent', 'images': 'full-frame RGB + instance-colored overlay', 'bbox': 'original pixels xyxy exclusive maximum', 'overlay_style': 'title bar, translucent masks, bold contours/bboxes, large black-white instance labels, collision-aware leader lines'}
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
        self.items={};self.points={};self.colors={};self.views={};self.scene_point_ids={}
        self.audit={'manifest_sha256':digest(self.source/'manifest.json'),'files':{}}
        for seq,frame in enumerate(self.frames):
            path=(self.source/frame['npz']).resolve()
            self.audit['files'][str(path)]=digest(path)
            rgb=(self.source/frame['original_rgb_image']).resolve()
            self.audit['files'][str(rgb)]=digest(rgb)
            with np.load(path) as z:
                points=z['points_world'];colors=z['rgb'];bits=z['mask_bits']
                scene_ids=z['scene_point_ids'] if 'scene_point_ids' in z else None
                if scene_ids is not None and (scene_ids.shape != bits.shape or not np.issubdtype(scene_ids.dtype,np.integer) or np.any(scene_ids<0)):
                    raise ValueError('Invalid scene point IDs')
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
                if scene_ids is not None:self.scene_point_ids[oid]=scene_ids[keep].copy()
                p=self.points[oid]
                self.items[oid]={'id':oid,'seq':seq,'source_frame_id':frame['source_frame_id'],
                    'instance_id':item['instance_id'],'category_label':item['class_name'],'image':str(rgb),'mask':str(mask),
                    'bbox_xyxy':list(bbox) if bbox else None,'image_size_wh':image_size,
                    'bbox_convention':'original RGB pixels; origin top-left; xmax/ymax exclusive',
                    'overlay_color_rgb':color,
                    'point_file':str(path),'mask_bit':i,'point_count':len(p),
                    'centroid':p.mean(0).tolist() if len(p) else None,
                    'bounds':[p.min(0).tolist(),p.max(0).tolist()] if len(p) else None}
        audit_path=self.out/'input_audit.json'
        if audit_path.exists() and json.loads(audit_path.read_text()) != self.audit:
            raise ValueError('Input changed; cannot resume')
        save(self.out/'observations.json',self.items);save(audit_path,self.audit)

    def rgb_view(self, seq):
        return (self.source/self.frames[seq]['original_rgb_image']).resolve()

    @staticmethod
    def _rect_overlap(a, b, pad=7):
        ax0, ay0, ax1, ay1 = a
        bx0, by0, bx1, by1 = b
        return not (ax1 + pad <= bx0 or bx1 + pad <= ax0 or ay1 + pad <= by0 or by1 + pad <= ay0)

    @staticmethod
    def _clamp_label(x, y, w, h, image_w, image_h, title_h):
        x = max(6, min(int(x), image_w - w - 6))
        y = max(title_h + 5, min(int(y), image_h - h - 6))
        return x, y

    def _label_candidates(self, box, label_w, label_h, image_w, image_h, title_h):
        x0, y0, x1, y1 = box
        gap = 9
        # Prefer positions outside the box; the later candidates fall back inside it.
        raw = [
            (x0, y0 - label_h - gap),
            (x1 - label_w, y0 - label_h - gap),
            (x0 - label_w - gap, y0),
            (x1 + gap, y0),
            (x0, y1 + gap),
            (x1 - label_w, y1 + gap),
            (x0 + 8, y0 + 8),
            (x1 - label_w - 8, y0 + 8),
            (x0 + (x1 - x0 - label_w) / 2, y0 + (y1 - y0 - label_h) / 2),
        ]
        return [self._clamp_label(x, y, label_w, label_h, image_w, image_h, title_h) for x, y in raw]

    def view(self, seq, observation=None):
        key=observation or f'frame_{seq:04d}'
        if key in self.views:return self.views[key]
        frame=self.frames[seq]
        im=Image.open(self.source/frame['original_rgb_image']).convert('RGB')
        ids=[k for k,o in self.items.items() if o['seq']==seq and (observation is None or k==observation)]
        image_w, image_h = im.size
        title_h = max(42, image_h // 23)
        title_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', max(18, image_w // 59))
        label_font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', max(22, image_w // 50))
        overlay=Image.new('RGBA', im.size, (0, 0, 0, 0))
        draw=ImageDraw.Draw(overlay)
        placed=[]
        records=[]
        for oid in ids:
            o=self.items[oid]
            mask=Image.open(o['mask']).convert('L')
            if mask.size!=im.size:raise ValueError('RGB/mask size mismatch')
            box=mask.getbbox()
            if box is None:continue
            arr=np.asarray(mask)
            ys, xs = np.nonzero(arr > 0)
            anchor=(int(xs.mean()), int(ys.mean())) if len(xs) else ((box[0]+box[2])//2, (box[1]+box[3])//2)
            text=f'{int(o["instance_id"]):03d}'
            text_box=draw.textbbox((0, 0), text, font=label_font, stroke_width=0)
            label_w=text_box[2]-text_box[0]+18
            label_h=text_box[3]-text_box[1]+12
            area=max(1, (box[2]-box[0])*(box[3]-box[1]))
            records.append((area, oid, o, mask, box, anchor, text, label_w, label_h))
        # Place large/important regions first so small masks are moved around them.
        records.sort(key=lambda row: (-row[0], row[1]))
        for _, oid, o, mask, box, anchor, text, label_w, label_h in records:
            pos=None
            for candidate in self._label_candidates(box, label_w, label_h, image_w, image_h, title_h):
                if not any(self._rect_overlap((candidate[0], candidate[1], candidate[0]+label_w, candidate[1]+label_h), r) for r in placed):
                    pos=candidate; break
            if pos is None:
                pos=self._label_candidates(box, label_w, label_h, image_w, image_h, title_h)[0]
            lx,ly=pos
            placed.append((lx,ly,lx+label_w,ly+label_h))
            # draw the translucent mask, bbox and a clean binary contour
            color=tuple(o['overlay_color_rgb'])
            tint=Image.new('RGBA', im.size, (*color, 0))
            tint.putalpha(mask.point(lambda value: 78 if value else 0))
            overlay=Image.alpha_composite(overlay, tint)
            draw=ImageDraw.Draw(overlay)
            draw.rectangle(box, outline=(*color, 255), width=4)
            binary=mask.point(lambda value: 255 if value else 0)
            dilated=binary.filter(ImageFilter.MaxFilter(5))
            contour=ImageChops.subtract(dilated, binary)
            contour=ImageChops.lighter(contour, ImageChops.subtract(binary, binary.filter(ImageFilter.MinFilter(5))))
            contour_layer=Image.new('RGBA', im.size, (*color, 255))
            contour_layer.putalpha(contour.point(lambda value: 220 if value else 0))
            overlay=Image.alpha_composite(overlay, contour_layer)
            draw=ImageDraw.Draw(overlay)
            # Connect the label edge to a point inside the mask.
            sx=min(max(anchor[0], lx), lx+label_w)
            sy=min(max(anchor[1], ly), ly+label_h)
            if lx <= anchor[0] <= lx+label_w and ly <= anchor[1] <= ly+label_h:
                distances=[(abs(anchor[0]-lx), (lx, anchor[1])), (abs(anchor[0]-(lx+label_w)), (lx+label_w, anchor[1])), (abs(anchor[1]-ly), (anchor[0], ly)), (abs(anchor[1]-(ly+label_h)), (anchor[0], ly+label_h))]
                _, (sx,sy)=min(distances, key=lambda item:item[0])
            draw.line((sx, sy, anchor[0], anchor[1]), fill=(*color, 255), width=3)
            draw.ellipse((anchor[0]-3, anchor[1]-3, anchor[0]+3, anchor[1]+3), fill=(*color, 255))
            draw.rounded_rectangle((lx, ly, lx+label_w, ly+label_h), radius=4, fill=(0,0,0,235), outline=(*color,255), width=3)
            draw.text((lx+9, ly+5), text, font=label_font, fill=(255,255,255,255), stroke_width=1, stroke_fill=(0,0,0,255))
        rendered=Image.alpha_composite(im.convert('RGBA'), overlay)
        draw=ImageDraw.Draw(rendered)
        source_frame=frame.get('source_frame_id', seq)
        title=f'PROPOSED OVERLAY · frame {int(source_frame):04d} · instance IDs + contours + leader lines'
        draw.rectangle((0,0,image_w,title_h), fill=(5,10,18,255))
        draw.text((12, (title_h-title_font.size)//2-1), title, font=title_font, fill=(245,248,252,255))
        rendered=rendered.convert('RGB')
        path=self.out/'evidence'/f'{key.replace(":","_")}.jpg';path.parent.mkdir(exist_ok=True);rendered.save(path,quality=90,subsampling=0)
        self.views[key]=path;return path

    def fused(self,state,oid,voxel):
        members=[k for k,a in state['assignments'].items() if a['object_id']==oid]
        p=np.concatenate([self.points[k] for k in members]);c=np.concatenate([self.colors[k] for k in members])
        if len(p)==0:return np.empty((0,3),np.float32),np.empty((0,3),np.uint8)
        if all(k in self.scene_point_ids for k in members):
            ids=np.concatenate([self.scene_point_ids[k] for k in members])
            _,unique=np.unique(ids,return_index=True)
            p,c=p[unique],c[unique]
        _,p,c=voxelize(p,c,voxel)
        return p,c.astype(np.uint8)
