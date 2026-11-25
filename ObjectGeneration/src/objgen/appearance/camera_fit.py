"""Orthographic silhouette initializer; camera pose can be overridden explicitly."""
import cv2
import numpy as np
from scipy.optimize import minimize


def basis(elevation, roll=0, yaw=0):
    e, r, y = np.deg2rad([elevation, roll, yaw])
    right = np.array([np.cos(y), np.sin(y), 0.])
    toward = np.array([np.sin(y)*np.cos(e), -np.cos(y)*np.cos(e), np.sin(e)])
    up = np.cross(toward, right)
    return np.stack([right*np.cos(r)+up*np.sin(r), up*np.cos(r)-right*np.sin(r), toward])


def project(points, camera):
    q = (points-np.array(camera['center'])) @ np.array(camera['basis']).T
    uv = q[:, :2] / camera['height_m']
    uv[:, 0] /= camera['aspect']
    return uv + np.array(camera['offset'])


def apply_overrides(camera, overrides):
    """Accept only a complete camera or bounded explicit pose/framing fields."""
    allowed = set(camera) | {'elevation_degrees', 'roll_degrees', 'yaw_degrees', 'scale_factor',
        'silhouette_iou','alignment_accepted','pose_source','contour_distance_normalized'}
    if not isinstance(overrides, dict) or set(overrides)-allowed:
        raise ValueError('Unknown camera fields')
    out = dict(camera)
    out.update({k:v for k,v in overrides.items() if k not in {'silhouette_iou','alignment_accepted','pose_source','contour_distance_normalized'}})
    out.setdefault('elevation_degrees',float(np.rad2deg(np.arcsin(np.clip(np.asarray(out['basis'])[2,2],-1,1)))))
    out.setdefault('roll_degrees',float(np.rad2deg(np.arctan2(np.asarray(out['basis'])[0,2],np.asarray(out['basis'])[1,2]))))
    out.setdefault('yaw_degrees',0)
    for key, bounds in {'elevation_degrees':(-60,60), 'roll_degrees':(-30,30),
                        'yaw_degrees':(-180,180), 'scale_factor':(.7,1.5)}.items():
        value = out.get(key, 1 if key=='scale_factor' else 0)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not bounds[0]<=value<=bounds[1]:
            raise ValueError('Camera field out of range: '+key)
    if 'basis' not in overrides or all(k in overrides for k in ('elevation_degrees','roll_degrees','yaw_degrees')):
        out['basis']=basis(out['elevation_degrees'],out['roll_degrees'],out['yaw_degrees']).tolist()
    out['height_m'] *= overrides.get('scale_factor',1)
    out.pop('scale_factor',None)
    for key,shape in [('basis',(3,3)),('center',(3,)),('offset',(2,))]:
        arr=np.asarray(out[key],dtype=float)
        if arr.shape!=shape or not np.isfinite(arr).all(): raise ValueError('Invalid camera '+key)
    B=np.asarray(out['basis'])
    if not np.allclose(B@B.T,np.eye(3),atol=1e-6) or not np.isclose(np.linalg.det(B),1,atol=1e-6):
        raise ValueError('Camera basis must be right-handed orthonormal')
    for key in ('height_m','aspect'):
        if isinstance(out[key],bool) or not np.isfinite(out[key]) or out[key]<=0: raise ValueError('Invalid '+key)
    if out['aspect']!=camera['aspect'] or out['center']!=camera['center']:
        raise ValueError('Image aspect and canonical center cannot be overridden')
    if not .5*camera['height_m']<=out['height_m']<=2*camera['height_m'] or not all(.2<=v<=.8 for v in out['offset']):
        raise ValueError('Camera framing outside permitted range')
    return out


def fit(points, mask, overrides=None, yaw=0, elevation=None):
    h, w = mask.shape
    target = cv2.resize(mask.astype(np.uint8), (128, 256), interpolation=cv2.INTER_NEAREST) > 0
    center = (points.min(axis=0)+points.max(axis=0))/2
    size = np.ptp(points, axis=0).max()
    # At the poles the projected diameter, not object height, sets framing.
    if elevation is not None and abs(elevation) > 60:
        projected = (points-center) @ basis(elevation, yaw=yaw).T
        ys, xs = np.nonzero(mask)
        if not len(xs): raise ValueError('Empty reference mask')
        coverage_x = max((xs.max()-xs.min()+1)/w, .01)
        coverage_y = max((ys.max()-ys.min()+1)/h, .01)
        size = max(np.ptp(projected[:,0])/(w/h)/coverage_x,
                   np.ptp(projected[:,1])/coverage_y)
    sample = points[::max(1, len(points)//60000)]
    def camera(x):
        return {'type': 'orthographic', 'center': center.tolist(), 'basis': basis(x[0], x[1], yaw).tolist(),
                'height_m': float(size*x[2]), 'aspect': w/h, 'offset': [float(x[3]), float(x[4])]}
    def score(x):
        uv = project(sample, camera(x))
        xy = np.rint(np.column_stack([uv[:, 0]*127, (1-uv[:, 1])*255])).astype(int)
        keep = (xy[:, 0]>=0)&(xy[:, 0]<128)&(xy[:, 1]>=0)&(xy[:, 1]<256)
        predicted = np.zeros((256,128),np.uint8)
        predicted[xy[keep,1],xy[keep,0]]=1
        predicted = cv2.morphologyEx(predicted, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))>0
        return 1-(predicted & target).sum()/max(1,(predicted | target).sum())
    initial = [5 if elevation is None else elevation, 0, 1.03, .5, .5]
    result = minimize(score, initial, method='Powell', bounds=[(-15,25) if elevation is None else (elevation-0.001,elevation+0.001),(-12,12),(.85,1.35),(.4,.6),(.4,.6)],
                      options={'maxiter': 12, 'xtol': .002, 'ftol': .001})
    out = camera(result.x)
    out.update(initializer_iou=float(1-score(result.x)), method='vertex-occupancy silhouette initializer',
               limitations=['orthographic approximation', 'yaw is ambiguous for symmetric objects'], yaw_degrees=yaw, elevation_degrees=float(result.x[0]), roll_degrees=float(result.x[1]))
    if overrides is not None: out=apply_overrides(out,overrides)
    return out


def main():
    import json
    import sys
    from pathlib import Path
    from PIL import Image
    folder=Path(sys.argv[1]); arrays=np.load(folder/'canonical_mesh.npz')
    mask=np.array(Image.open(folder/'reference_mask.png').convert('L'))>0
    overrides=json.loads(Path(sys.argv[2]).read_text()) if len(sys.argv)>2 else None
    camera=fit(arrays['points'],mask,overrides)
    # Rasterize actual triangles for acceptance; the initializer score is only approximate.
    h=512; w=round(h*camera['aspect']); target=cv2.resize(mask.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST)>0
    xy=np.rint(project(arrays['points'],camera)*[w-1,-(h-1)]+[0,h-1]).astype(np.int32)
    predicted=np.zeros((h,w),np.uint8)
    for tri in xy[arrays['indices'].reshape(-1,3)]: cv2.fillConvexPoly(predicted,tri,1)
    predicted=predicted>0
    camera['silhouette_iou']=float((predicted&target).sum()/max(1,(predicted|target).sum()))
    # Symmetric normalized contour distance is diagnostic, not evidence of label orientation.
    pe=cv2.morphologyEx(predicted.astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
    te=cv2.morphologyEx(target.astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
    pd=cv2.distanceTransform((~pe).astype('uint8'),cv2.DIST_L2,3)
    td=cv2.distanceTransform((~te).astype('uint8'),cv2.DIST_L2,3)
    camera['contour_distance_normalized']=float((pd[te].mean()+td[pe].mean())/(2*h)) if pe.any() and te.any() else 1.0
    camera['alignment_accepted']=camera['silhouette_iou']>=.90
    camera['pose_source']='explicit override' if overrides else 'silhouette initializer; yaw fixed, manual review required'
    (folder/'camera.json').write_text(json.dumps(camera,indent=2))
    rgb=np.array(Image.open(folder/'reference_crop.png').convert('RGB').resize((w,h)))
    edge=cv2.morphologyEx(predicted.astype('uint8'),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
    rgb[edge]=[0,255,255]; Image.fromarray(rgb).save(folder/'renders/alignment.png')
    if not camera['alignment_accepted']: raise RuntimeError(f"needs_alignment: IoU={camera['silhouette_iou']:.4f}; supply --camera JSON")


if __name__=='__main__': main()
