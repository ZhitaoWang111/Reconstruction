"""Fit each known-angle photograph without changing canonical geometry."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image
from camera_fit import fit, project
folder=Path(sys.argv[1]); views=json.loads((folder/'views.json').read_text())
arrays=np.load(folder/'canonical_mesh.npz')
for view in views:
    ref=folder/view['folder']; mask=np.array(Image.open(ref/'reference_mask.png').convert('L'))>0
    angle=view['angle']; yaw=float(angle) if angle not in ('up','down') else 0.
    elevation=90. if angle=='up' else -90. if angle=='down' else None
    camera=fit(arrays['points'],mask,yaw=yaw,elevation=elevation)
    h=512; w=round(h*camera['aspect']); target=cv2.resize(mask.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST)>0
    xy=np.rint(project(arrays['points'],camera)*[w-1,-(h-1)]+[0,h-1]).astype(np.int32)
    predicted=np.zeros((h,w),np.uint8)
    for tri in xy[arrays['indices'].reshape(-1,3)]: cv2.fillConvexPoly(predicted,tri,1)
    predicted=predicted>0
    camera.update(silhouette_iou=float((predicted&target).sum()/max(1,(predicted|target).sum())),pose_source='filename angle plus silhouette fit')
    camera['alignment_accepted']=camera['silhouette_iou']>=.90
    (ref/'camera.json').write_text(json.dumps(camera,indent=2))
    rgb=np.array(Image.open(ref/'reference_crop.png').convert('RGB').resize((w,h)))
    edge=cv2.morphologyEx(predicted.astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
    rgb[edge]=[0,255,255]; Image.fromarray(rgb).save(ref/'alignment.png')
    print(view['angle'],camera['silhouette_iou'],flush=True)
    if not camera['alignment_accepted']: raise RuntimeError(f"View {angle} needs alignment: IoU {camera['silhouette_iou']:.4f}")
