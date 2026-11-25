"""Images the agent looks at. Every one is rendered under a real key-frame camera.

The agent sees the photo, the green outline of the object's mask in that photo,
and the asset rendered on top under the same intrinsics and pose, together with
everything already placed (drawn grey) so occlusion is visible. Images are kept
small on purpose: the gateway does not cache prompts, so every image is paid
again on every later turn.
"""
from __future__ import annotations

import io

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

import handoff
from pose import Pose
from render import render_asset

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
GREEN = np.array([40, 225, 90], np.uint8)


def font(size, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, size)


def outline(m):
    e = np.zeros_like(m)
    e[1:] |= m[1:] & ~m[:-1]
    e[:-1] |= m[:-1] & ~m[1:]
    e[:, 1:] |= m[:, 1:] & ~m[:, :-1]
    e[:, :-1] |= m[:, :-1] & ~m[:, 1:]
    return e


def thick(e, r=1):
    out = e.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(e, dy, 0), dx, 1)
    return out


def jpeg(im: Image.Image, quality=85) -> bytes:
    buf = io.BytesIO()
    im.convert('RGB').save(buf, 'JPEG', quality=quality)
    return buf.getvalue()


def window(mask, cam, pad=0.35, min_side=220):
    """Crop box around a mask, padded, clipped to the image; returned as ints."""
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    w, h = x1 - x0, y1 - y0
    side = max(w, h) * (1 + 2 * pad)
    side = max(side, min_side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
    box = [int(max(0, box[0])), int(max(0, box[1])), int(min(cam.width, box[2])), int(min(cam.height, box[3]))]
    return box


class Views:
    def __init__(self, scene):
        self.s = scene
        self.cam = scene.handoff.camera
        self._photo = {}

    def frame(self, oid, which):
        """KeyFrame of `oid` by role name or seq; raises with the valid choices."""
        frames = self.s.assets[oid].ob.key_frames
        for f in frames:
            if which in (f.role, str(f.seq), f.seq):
                return f
        raise ValueError(f'{oid} key frames are ' + ', '.join(f'{f.role} (seq {f.seq})' for f in frames))

    def photo(self, f: handoff.KeyFrame):
        if f.seq not in self._photo:
            self._photo[f.seq] = np.asarray(Image.open(f.rgb).convert('RGB'))
        return self._photo[f.seq]

    def _render(self, oid, pose, C2W, win, include_placed=True):
        """Colour render of `oid` at `pose` composited with a grey pass of the other placed objects.

        Returns the composite colour, the composite mask, and the pixels where
        `oid` itself is the nearest surface (what its silhouette IoU is scored on).
        """
        c = self.cam
        a = self.s.assets[oid]
        args = (c.fx, c.fy, c.cx, c.cy, c.width, c.height)
        rgb, own, depth = render_asset(a.mesh(pose), a.colours, C2W, *args, window=win, ambient=0.42, gain=0.95)
        others = [self.s.assets[o].mesh(p.pose) for o, p in self.s.placed.items() if o != oid] if include_placed else []
        if not others:
            return rgb, own, own
        om = trimesh.util.concatenate(others)
        orgb, omask, odepth = render_asset(om, np.full((len(om.faces), 3), 0.62), C2W, *args,
                                           window=win, ambient=0.42, gain=0.95)
        front = omask & (odepth < depth)          # a neighbour hides the object here
        rgb = np.where(front[..., None], orgb, rgb)
        return rgb, own | omask, own & ~front

    def tile(self, oid, pose, f, win, label, size=300, include_placed=True):
        """Photo crop with the asset blended in and the photo mask outlined in green."""
        photo = self.photo(f)
        gt = np.asarray(Image.open(f.mask)) > 0
        rgb, mask, own = self._render(oid, pose, f.c2w, win, include_placed)
        x0, y0, x1, y1 = win
        base = photo[y0:y1, x0:x1].astype(np.float32)
        r = (np.clip(rgb[y0:y1, x0:x1], 0, 1) * 255).astype(np.float32)
        m = mask[y0:y1, x0:x1]
        base[m] = base[m] * 0.2 + r[m] * 0.8
        im = Image.fromarray(base.clip(0, 255).astype(np.uint8))
        scale = size / max(im.size)
        tw, th = max(1, round(im.size[0] * scale)), max(1, round(im.size[1] * scale))
        im = im.resize((tw, th), Image.LANCZOS)
        e = thick(outline(np.asarray(Image.fromarray(gt[y0:y1, x0:x1]).resize((tw, th), Image.NEAREST))))
        arr = np.asarray(im).copy()
        arr[e] = GREEN
        im = Image.fromarray(arr)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, tw, 20], fill=(12, 22, 32))
        d.text((5, 3), label, font=font(13, True), fill=(255, 255, 255))
        o = own & gt
        iou = float(o.sum() / max(1, (own | gt).sum()))
        return im, iou

    def compare(self, oid, pose: Pose, which='primary'):
        """Before/after style single view: the photo, and the proposal rendered over it."""
        f = self.frame(oid, which)
        gt = np.asarray(Image.open(f.mask)) > 0
        win = window(gt, self.cam)
        x0, y0, x1, y1 = win
        photo = Image.fromarray(self.photo(f)[y0:y1, x0:x1])
        size = 380
        s = size / max(photo.size)
        photo = photo.resize((max(1, round(photo.size[0] * s)), max(1, round(photo.size[1] * s))), Image.LANCZOS)
        arr = np.asarray(photo).copy()
        e = thick(outline(np.asarray(Image.fromarray(gt[y0:y1, x0:x1]).resize(photo.size, Image.NEAREST))))
        arr[e] = GREEN
        photo = Image.fromarray(arr)
        ImageDraw.Draw(photo).text((5, 3), f'photo seq {f.seq}', font=font(13, True), fill=(255, 255, 255))
        rend, iou = self.tile(oid, pose, f, win, f'proposal · seq {f.seq}', size)
        sheet = Image.new('RGB', (photo.size[0] + rend.size[0] + 6, max(photo.size[1], rend.size[1])), (247, 250, 252))
        sheet.paste(photo, (0, 0))
        sheet.paste(rend, (photo.size[0] + 6, 0))
        return sheet, iou, f.seq

    def candidate_sheet(self, oid, which='primary'):
        """SAM3D and the four registrations side by side under one key-frame camera."""
        f = self.frame(oid, which)
        gt = np.asarray(Image.open(f.mask)) > 0
        win = window(gt, self.cam)
        tiles, ious = [], []
        for c in self.s.candidates(oid):
            label = f"{c['name']} yaw {c['pose'].yaw_deg:.0f} up {c['pose'].up}"
            im, iou = self.tile(oid, c['pose'], f, win, label, 230)
            tiles.append(im)
            ious.append(round(iou, 2))
        w = sum(t.size[0] for t in tiles) + 6 * (len(tiles) - 1)
        h = max(t.size[1] for t in tiles)
        sheet = Image.new('RGB', (w, h), (247, 250, 252))
        x = 0
        for t in tiles:
            sheet.paste(t, (x, 0))
            x += t.size[0] + 6
        return sheet, ious, f.seq

    def turntable(self, oid, up='+Z'):
        """The asset alone from four sides, with its local axes, to judge which face is the front."""
        from render import persp_camera
        a = self.s.assets[oid]
        v = a.mesh(Pose(0, 0, 0, 0, up)).vertices
        m = trimesh.Trimesh(v, a.faces, process=False)
        lo, hi = v.min(0), v.max(0)
        ctr = (lo + hi) / 2
        rad = float(np.linalg.norm(hi - lo) / 2)
        dist = rad * 2.8
        T = 210
        views = [('from +X', (ctr[0] + dist, ctr[1], ctr[2] + rad * 0.5)),
                 ('from +Y', (ctr[0], ctr[1] + dist, ctr[2] + rad * 0.5)),
                 ('from -X', (ctr[0] - dist, ctr[1], ctr[2] + rad * 0.5)),
                 ('from -Y', (ctr[0], ctr[1] - dist, ctr[2] + rad * 0.5))]
        sheet = Image.new('RGB', (T * 4 + 18, T), (247, 250, 252))
        for i, (name, eye) in enumerate(views):
            C = persp_camera(eye, ctr)
            fpx = T * 1.25
            rgb, mask, _ = render_asset(m, a.colours, C, fpx, fpx, T / 2, T / 2, T, T, ambient=0.42, gain=0.95)
            im = np.where(mask[..., None], rgb, np.array([247, 250, 252]) / 255)
            im = Image.fromarray((im * 255).astype(np.uint8))
            d = ImageDraw.Draw(im)
            d.text((6, 4), name, font=font(13, True), fill=(30, 50, 70))
            # local axes at the footprint centre, on the floor of the asset
            o = np.array([ctr[0], ctr[1], lo[2]])
            for k, (col, ax) in enumerate(zip([(220, 60, 60), (40, 160, 70), (50, 100, 220)], np.eye(3))):
                p = np.array([o, o + ax * rad * 0.8])
                Pc = (np.c_[p, np.ones(2)] @ np.linalg.inv(C).T)[:, :3]
                if (Pc[:, 2] > 0).all():
                    u = fpx * Pc[:, 0] / Pc[:, 2] + T / 2
                    w = fpx * Pc[:, 1] / Pc[:, 2] + T / 2
                    d.line([u[0], w[0], u[1], w[1]], fill=col, width=3)
                    d.text((u[1] + 3, w[1] - 7), 'XYZ'[k], font=font(12, True), fill=col)
            sheet.paste(im, (i * (T + 6), 0))
        return sheet
