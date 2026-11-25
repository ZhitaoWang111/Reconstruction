"""Assemble the input package for one assembly batch of scene0000_00.

Reads both handoffs through handoff.load() and writes images + a session.json
holding every number the page quotes, so make_page.py reads nothing else.
"""
import io, json, os, sys, time, zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import handoff
from usdmesh import load_visual, srgb_faces
from render import render_asset, persp_camera
from icp import refine, yaw_of, Rz

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs', 'assets', 'scene-assembly-input-example')
DS = 2  # render at half resolution, display at that size

BATCH = ['obj_0037', 'obj_0048', 'obj_0049', 'obj_0062', 'obj_0061', 'obj_0016']

os.makedirs(OUT + '/kf', exist_ok=True)
os.makedirs(OUT + '/crop', exist_ok=True)
os.makedirs(OUT + '/surround', exist_ok=True)
os.makedirs(OUT + '/cand', exist_ok=True)

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def font(sz, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, sz)


def mask_bbox(m, pad=0):
    ys, xs = np.nonzero(m)
    return xs.min() - pad, ys.min() - pad, xs.max() + 1 + pad, ys.max() + 1 + pad


def place(verts, yaw, t):
    """Asset local -> scene: yaw about Z about the XY bbox centre, then translate."""
    lo, hi = verts.min(0), verts.max(0)
    off = np.array([(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, lo[2]])
    return (verts - off) @ Rz(yaw).T + t


def silhouette_iou(a, b):
    inter = (a & b).sum()
    uni = (a | b).sum()
    return float(inter/uni) if uni else 0.0


def outline(m):
    """Boundary pixels of a boolean mask."""
    e = np.zeros_like(m)
    e[1:, :] |= m[1:, :] & ~m[:-1, :]
    e[:-1, :] |= m[:-1, :] & ~m[1:, :]
    e[:, 1:] |= m[:, 1:] & ~m[:, :-1]
    e[:, :-1] |= m[:, :-1] & ~m[:, 1:]
    return e


def nice(ob):
    return 'object (unlabelled)' if ob.generic else ob.label


def bottom_z(points):
    """2nd-percentile height: the lowest scanned surface, robust to stray points."""
    return float(np.percentile(points[:, 2], 2))


def catalog(scene, scans):
    """One row per asset: what is left to place, without opening an image."""
    rows = []
    for oid, ob in scene.objects.items():
        pts = scans[oid] if oid in scans else ob.points()
        rows.append(dict(oid=oid, label=ob.label, extent=[round(float(x), 2) for x in ob.extent],
                         bottom_z=round(bottom_z(pts), 2), body=ob.body_type, collider=ob.collider,
                         seq=ob.primary.seq, coverage=ob.primary.visible_pixels / (scene.camera.width * scene.camera.height),
                         key_frames=len(ob.key_frames)))
    return rows


def covisibility(scene):
    """Share of each batch object in every key frame of the batch (ScanNet 2-D instance masks)."""
    if scene.instance_2d is None:
        raise SystemExit('co-visibility needs the ScanNet 2-D instance masks; this handoff has none')
    seqs = sorted({f.seq for oid in BATCH for f in scene.objects[oid].key_frames})
    area = scene.camera.width * scene.camera.height
    table = {}
    with zipfile.ZipFile(scene.instance_2d) as z:
        for s in seqs:
            im = np.array(Image.open(io.BytesIO(z.read(f'instance/{scene.frame_ids[s]}.png'))))
            # pixel value = ScanNet objectId + 1, and obj_NNNN carries that objectId
            table[s] = {oid: float((im == int(oid[4:]) + 1).sum()) / area for oid in BATCH}
    return table


def proj(P, C2W, fx, fy, cx, cy):
    """Project scene points into image coords. Returns (u,v,depth)."""
    Pinv = np.linalg.inv(C2W)
    Pc = (np.c_[P, np.ones(len(P))] @ Pinv.T)[:, :3]
    z = Pc[:, 2]
    u = fx*Pc[:, 0]/z + cx
    v = fy*Pc[:, 1]/z + cy
    return u, v, z


def main():
    for sub in ('kf', 'crop', 'surround', 'cand'):
        os.makedirs(f'{OUT}/{sub}', exist_ok=True)
    scene = handoff.load()
    cam = scene.camera
    FX, FY, CX, CY, W, H = cam.fx, cam.fy, cam.cx, cam.cy, cam.width, cam.height
    session = {'batch': 1, 'objects': BATCH, 'scene_id': scene.scene_id, 'floor_z': scene.floor_z,
               'camera': {'fx': FX, 'fy': FY, 'cx': CX, 'cy': CY, 'W': W, 'H': H},
               'objects_meta': {}}
    scans = {oid: scene.objects[oid].points() for oid in BATCH}
    session['catalog'] = catalog(scene, scans)
    session['covisibility'] = covisibility(scene)

    # ---------- shared per-batch evidence: support contacts ----------
    contacts = {}
    for oid in BATCH:
        v = scans[oid]
        bot = bottom_z(v)
        lo = v[:, :2].min(0) - 0.04
        hi = v[:, :2].max(0) + 0.04
        sup = []
        for oid2 in BATCH:
            if oid2 == oid:
                continue
            s2 = scans[oid2]
            band = (s2[:, 2] > bot - 0.05) & (s2[:, 2] < bot + 0.01)
            inbox = (s2[:, 0] > lo[0]) & (s2[:, 0] < hi[0]) & (s2[:, 1] > lo[1]) & (s2[:, 1] < hi[1])
            sel = band & inbox
            if sel.sum() >= 5:
                sup.append({'support': oid2, 'points': int(sel.sum()),
                            'z': [round(float(s2[sel, 2].min()), 3), round(float(s2[sel, 2].max()), 3)]})
        sup.sort(key=lambda d: -d['points'])
        contacts[oid] = {'bottom_p2': round(bot, 3), 'list': sup}

    for oid in BATCH:
        ob = scene.objects[oid]
        name = nice(ob)
        t0 = time.time()
        verts, faces, col, interp = load_visual(ob.usd)
        fcol = srgb_faces(col, interp, len(faces), faces)
        tr = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        tr.ray  # build BVH
        scan = scans[oid]
        primary = ob.primary
        seq = primary.seq
        C2W = primary.c2w
        photo = np.array(Image.open(primary.rgb).convert('RGB'))
        gt_mask = np.array(Image.open(primary.mask)) > 0
        vis_px = int(gt_mask.sum())
        cov = vis_px/(W*H)

        # --- SAM3D up / yaw ---
        Rw = ob.rotation_world
        up_vec = Rw @ np.array([0, 0, 1.])
        up_err = float(np.degrees(np.arccos(np.clip(up_vec[2], -1, 1))))
        yaw_sam3d = yaw_of(Rw)

        # --- ICP from 4 yaw starts + SAM3D yaw ---
        starts = [0.0, np.pi/2, np.pi, 3*np.pi/2]
        cands = []
        for st in starts:
            r = refine(verts, scan, st)
            r['source'] = 'icp'
            cands.append(r)

        # --- render everything at the exact keyframe camera ---
        lo, hi = scan.min(0), scan.max(0)
        t_scan = np.array([(lo[0]+hi[0])/2, (lo[1]+hi[1])/2, lo[2]])

        bx = mask_bbox(gt_mask)
        win = (bx[0]-0.15*(bx[2]-bx[0])-40, bx[1]-0.15*(bx[3]-bx[1])-40,
               bx[2]+0.15*(bx[2]-bx[0])+40, bx[3]+0.15*(bx[3]-bx[1])+40)
        win = (int(max(0, win[0])), int(max(0, win[1])), int(min(W, win[2])), int(min(H, win[3])))

        tiles = []
        best_iou = None
        for c in cands:
            t = t_scan.copy()
            t[:2] = c['t'][:2]
            t[2] = lo[2]
            Pv = place(verts, c['yaw'], t)
            trc = trimesh.Trimesh(vertices=Pv, faces=faces, process=False)
            rgb, m, dep = render_asset(trc, fcol, C2W, FX, FY, CX, CY, W, H, window=win, ambient=0.42, gain=0.95)
            iou = silhouette_iou(m, gt_mask)
            c2 = dict(kind='icp', yaw=float(np.degrees(c['yaw'])) % 360, residual=round(c['residual'], 4),
                      residual_start=round(c['residual_start'], 4), diverged=bool(c['diverged']),
                      inlier=round(c['inlier'], 3), iou=round(iou, 3), t=[round(float(x), 3) for x in t],
                      start=round(float(np.degrees(c['start'])) % 360))
            tiles.append((c2, rgb, m))
            if best_iou is None or iou > best_iou[1]:
                best_iou = (c2, iou, rgb, m)
        # SAM3D candidate
        rs = refine(verts, scan, yaw_sam3d)
        t_s = t_scan.copy(); t_s[:2] = rs['t'][:2]; t_s[2] = lo[2]
        Pv = place(verts, rs['yaw'], t_s)
        trc = trimesh.Trimesh(vertices=Pv, faces=faces, process=False)
        rgb_s, m_s, _ = render_asset(trc, fcol, C2W, FX, FY, CX, CY, W, H, window=win, ambient=0.42, gain=0.95)
        iou_s = silhouette_iou(m_s, gt_mask)
        csam = dict(kind='sam3d', yaw=round(float(np.degrees(yaw_sam3d)) % 360, 1),
                    up_err=round(up_err, 1), iou=round(iou_s, 3),
                    residual=round(rs['residual'], 4), yaw_refined=round(float(np.degrees(rs['yaw'])) % 360, 1),
                    t=[round(float(x), 3) for x in t_s])
        session['objects_meta'][oid] = dict(
            label=ob.label, nice=name, seq=int(seq), visible_pixels=vis_px, coverage=round(cov, 4),
            bbox=[int(v) for v in bx], dims=[round(float(x), 3) for x in (hi-lo)],
            asset_dims=[round(float(x), 3) for x in (verts.max(0)-verts.min(0))],
            height_m=ob.height_m, scale_factor=ob.scale_factor, size_source=ob.size_source,
            body=ob.body_type, collider=ob.collider,
            up_err=round(up_err, 1), yaw_sam3d=round(float(np.degrees(yaw_sam3d)) % 360, 1),
            candidates=[t[0] for t in tiles], sam3d=csam,
            best_iou=best_iou[0], contacts=contacts[oid],
        )

        # ---------- 1. keyframe with mask outline ----------
        kf = Image.fromarray(photo).convert('RGB')
        ol = outline(gt_mask)
        arr = np.array(kf)
        arr[ol] = [40, 220, 90]
        kf = Image.fromarray(arr)
        d = ImageDraw.Draw(kf)
        d.rectangle(bx, outline=(60, 235, 120), width=2)
        d.rectangle([6, 6, 520, 46], fill=(12, 22, 32))
        d.text((14, 12), f'{oid}  {name}   frame {seq}   mask {cov*100:.2f}%', font=font(24, True),
               fill=(120, 240, 160))
        kf.resize((W//DS, H//DS), Image.LANCZOS).save(f'{OUT}/kf/{oid}.jpg', quality=90)

        # ---------- 2. crop for small objects only (design rule: <5% of the frame) ----------
        bw, bh = bx[2]-bx[0], bx[3]-bx[1]
        side = int(max(bw, bh)*2)
        cx_, cy_ = (bx[0]+bx[2])//2, (bx[1]+bx[3])//2
        c0 = [cx_-side//2, cy_-side//2, cx_+side//2, cy_+side//2]
        c0[0], c0[1] = max(0, c0[0]), max(0, c0[1])
        c0[2], c0[3] = min(W, c0[2]), min(H, c0[3])
        if cov < 0.05:
            im_c = kf.crop(c0)
            side_px = c0[2]-c0[0]
            im_c = im_c.resize((512, int(512*im_c.size[1]/im_c.size[0])), Image.LANCZOS)
            im_c.save(f'{OUT}/crop/{oid}.jpg', quality=92)
            session['objects_meta'][oid]['crop_box'] = c0
            session['objects_meta'][oid]['crop_side'] = side_px
        else:
            p_crop = f'{OUT}/crop/{oid}.jpg'
            if os.path.exists(p_crop):
                os.remove(p_crop)

        # ---------- 3. surround ----------
        ctr = (verts.max(0)+verts.min(0))/2
        rad = np.linalg.norm(verts.max(0)-verts.min(0))/2
        dist = rad*2.7
        views = [('+X', (ctr[0]+dist, ctr[1], ctr[2]+rad*0.25)), ('+Y', (ctr[0], ctr[1]+dist, ctr[2]+rad*0.25)),
                 ('top', (ctr[0], ctr[1]-0.001, ctr[2]+dist)), ('iso', (ctr[0]+dist*0.7, ctr[1]-dist*0.7, ctr[2]+dist*0.55))]
        T = 380
        sheet = Image.new('RGB', (T*2, T*2), (247, 250, 252))
        for i, (nm, cam) in enumerate(views):
            up = (0, 0, 1) if nm != 'top' else (0, 1, 0)
            C = persp_camera(cam, ctr, up)
            f = T*1.15
            rgbv, mv, _ = render_asset(tr, fcol, C, f, f, T/2, T/2, T, T, ambient=0.42, gain=0.95)
            im = np.where(mv[..., None], rgbv, np.array([247, 250, 252])/255)
            im = Image.fromarray((im*255).astype(np.uint8))
            dd = ImageDraw.Draw(im)
            dd.text((10, 8), f'view {nm}', font=font(18, True), fill=(30, 50, 70))
            # world-axis triad at the asset origin
            o3 = np.array([ctr[0], ctr[1], verts.min(0)[2]])
            L = rad*0.9
            cols = {'X': (220, 60, 60), 'Y': (40, 160, 70), 'Z': (50, 100, 220)}
            for k, ax in enumerate(np.eye(3)):
                p0 = o3
                p1 = o3 + ax*L
                u0, v0, z0 = proj(np.array([p0]), C, f, f, T/2, T/2)
                u1, v1, z1 = proj(np.array([p1]), C, f, f, T/2, T/2)
                if z0[0] > 0 and z1[0] > 0:
                    dd.line([float(u0[0]), float(v0[0]), float(u1[0]), float(v1[0])],
                            fill=cols['XYZ'[k]], width=3)
                    dd.text((float(u1[0])+4, float(v1[0])-8), 'XYZ'[k], font=font(16, True), fill=cols['XYZ'[k]])
            sheet.paste(im, ((i % 2)*T, (i//2)*T))
        sheet.save(f'{OUT}/surround/{oid}.jpg', quality=92)

        # ---------- 4. candidate sheet 1x5 ----------
        order = [csam] + [t[0] for t in tiles]
        imgs = [('sam3d', rgb_s, m_s)] + [('icp', t[1], t[2]) for t in tiles]
        tw = int(min(300, (win[2]-win[0])//DS))
        th = int(tw * (win[3]-win[1]) / (win[2]-win[0]))
        crop_photo = Image.fromarray(photo).crop(win).resize((tw, th), Image.LANCZOS)
        cp = np.array(crop_photo).astype(np.float32)
        cp = (cp*0.55 + 255*0.45)
        mfull = np.zeros((H, W), bool)
        mfull[win[1]:win[3], win[0]:win[2]] = gt_mask[win[1]:win[3], win[0]:win[2]]
        ol2 = outline(np.array(Image.fromarray(mfull[win[1]:win[3], win[0]:win[2]]).resize((tw, th), Image.NEAREST)))
        pad = 8
        SHEET = Image.new('RGB', (tw*5 + pad*6, th + 76), (247, 250, 252))
        meta = session['objects_meta'][oid]
        for i, (kind, rgb_c, m_c) in enumerate(imgs):
            sub = np.array(Image.fromarray((np.clip(rgb_c[win[1]:win[3], win[0]:win[2]], 0, 1)*255).astype(np.uint8))
                           .resize((tw, th), Image.BILINEAR)).astype(np.float32)
            msub = np.array(Image.fromarray((m_c[win[1]:win[3], win[0]:win[2]]*255).astype(np.uint8))
                            .resize((tw, th), Image.NEAREST)) > 127
            base = np.array(crop_photo).astype(np.float32)
            a = 0.82
            base[msub] = base[msub]*(1-a) + sub[msub]*a
            im = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
            ia = np.array(im)
            ia[ol2] = [60, 240, 120]
            im = Image.fromarray(ia)
            dd = ImageDraw.Draw(im)
            c = order[i]
            if c['kind'] == 'sam3d':
                l1 = f"SAM3D  yaw {c['yaw']:.0f}°"
                l2 = f"IoU {c['iou']:.2f} · res {c['residual']*100:.1f}cm · up {c['up_err']:.0f}°"
            else:
                l1 = f"ICP {i}  yaw {c['yaw']:.0f}°"
                l2 = f"IoU {c['iou']:.2f} · res {c['residual']*100:.1f}cm"
            dd.rectangle([0, 0, tw, 44], fill=(12, 22, 32))
            dd.text((6, 4), l1, font=font(15, True), fill=(255, 255, 255))
            dd.text((6, 23), l2, font=font(13), fill=(190, 215, 235))
            SHEET.paste(im, (pad + i*(tw+pad), 6))
        dd = ImageDraw.Draw(SHEET)
        dd.text((pad, th+14), f'{oid} {name} · frame {seq} · same exact keyframe camera · green = scanned outline',
                font=font(17, True), fill=(30, 50, 70))
        dd.text((pad, th+40), f'tile 1 = SAM3D yaw (+ upright check). tiles 2-5 = yaw-locked ICP from 0/90/180/270°.',
                font=font(14), fill=(90, 110, 130))
        SHEET.save(f'{OUT}/cand/{oid}.jpg', quality=92)
        print(f'{oid}: {time.time()-t0:.1f}s  cov={cov*100:.2f}%  sam3d yaw={meta["yaw_sam3d"]} up={up_err:.1f}  best={meta["best_iou"]["yaw"]}° res={meta["best_iou"]["residual"]*100:.1f}cm iou={meta["best_iou"]["iou"]}')

    json.dump(session, open(f'{OUT}/session.json', 'w'), indent=1,
              default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else str(o))
    print('done')


if __name__ == '__main__':
    main()
