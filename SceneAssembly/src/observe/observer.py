"""The four kinds of things the agent looks at, in one place."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

from inputs.usdmesh import load_visual, srgb_faces
from state import Pose

from .boxes import Sensor, box_from_mask, box_from_points
from .render import look_at, render_mesh

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
# distinct, readable on photos; up to 10 objects per batch (5 + extended)
COLOURS = [(230, 25, 75), (60, 180, 75), (255, 200, 25), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212)]
TILE = 210                      # four-side view tile
HALF = (648, 484)               # each half of a stitched image
BG = (247, 249, 251)


def font(size):
    return ImageFont.truetype(FONT, size)


def png(im: Image.Image) -> bytes:
    b = io.BytesIO()
    im.convert('RGB').save(b, 'PNG')
    return b.getvalue()


class Observer:
    def __init__(self, scene, structure, state, initial, isaac, out_dir: Path):
        self.scene = scene              # inputs.handoff.Scene
        self.structure = structure
        self.state = state              # state.SceneState
        self.initial = initial          # initial.InitialPoses
        self.isaac = isaac              # isaac.client.IsaacService (None: no stitched images)
        self.cam = scene.camera
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.sensor = Sensor(self.cam)
        self._colour = {}
        # numbering continues across resumed runs, so a resume never overwrites earlier images
        done = [int(f.name[:5]) for f in (self.out / 'stitched').glob('*.png') if f.name[:5].isdigit()] \
            if (self.out / 'stitched').exists() else []
        self._n = max(done, default=0)

    # ------------------------------------------------------------ frames

    def frame(self, seq):
        """KeyFrame object for a sequence number (any object's key frame)."""
        for o in self.scene.objects.values():
            for f in o.key_frames:
                if f.seq == seq:
                    return f
        raise KeyError(f'no key frame {seq}')

    def photo(self, seq) -> Image.Image:
        return Image.open(self.frame(seq).rgb).convert('RGB').resize((self.cam.width, self.cam.height))

    def colour(self, oid):
        return self._colour.get(oid, (128, 128, 128))

    def set_batch_colours(self, oids):
        self._colour = {o: COLOURS[i % len(COLOURS)] for i, o in enumerate(oids)}

    # ------------------------------------------------------------ 1. text card

    def card(self, oid) -> dict:
        o = self.scene.objects[oid]
        st = self.state.objects.get(oid)
        init = self.initial.card(oid)
        d = {
            'object': oid, 'label': o.label + (' (generic label)' if o.generic else ''),
            'colour': '#%02x%02x%02x' % self.colour(oid),
            'asset_size_m': [round(float(x), 3) for x in o.dims],
            'key_frames': [f.seq for f in o.key_frames],
            'start': {'pose': init['pose'], 'note': init['note'],
                      'orientation_confirmed': init['orientation_confirmed'],
                      'flipped_from_sam3d': init['flipped_from_sam3d']},
        }
        if st is not None:
            d['now'] = {'pose': st.pose.to_dict(), 'support': st.support.to_dict(),
                        'bottom_z': round(st.bottom_z, 3), 'status': st.status}
            if st.walls:
                d['now']['walls'] = st.walls
        return d

    def card_text(self, oid) -> str:
        c = self.card(oid)
        s = c['start']
        lines = [f"{c['object']}  {c['label']}  colour {c['colour']}",
                 f"  asset size (upright, m): {c['asset_size_m']}   key frames: {c['key_frames']}",
                 f"  start: x {s['pose']['x']:.2f}, y {s['pose']['y']:.2f}, yaw {s['pose']['yaw_deg']:.0f}, "
                 f"up {s['pose']['up']}  — {s['note']}"]
        if not s['orientation_confirmed']:
            lines.append('  orientation NOT confirmed: SAM3D had it upside down; check the facing against the four-side view')
        if 'now' in c:
            n = c['now']
            lines.append(f"  now: {n['status']}, support {n['support']}, bottom z {n['bottom_z']}")
        return '\n'.join(lines)

    # ------------------------------------------------------------ 2. four-side view

    def four_sides(self, oid) -> Image.Image:
        """The asset alone from +X, +Y, -X, -Y, turned to its starting up axis, with its axes drawn."""
        o = self.scene.objects[oid]
        v, f, c, it = load_visual(o.usd)
        col = srgb_faces(c, it, len(f), f)
        up = self.initial[oid]['pose']['up']
        u = Pose(0, 0, up=up).upright(np.asarray(v, float))
        m = trimesh.Trimesh(u, f, process=False)
        lo, hi = u.min(0), u.max(0)
        ctr = (lo + hi) / 2
        rad = float(np.linalg.norm(hi - lo) / 2)
        dist = rad * 2.8
        views = [('+X', (ctr[0] + dist, ctr[1], ctr[2] + rad * 0.5)), ('+Y', (ctr[0], ctr[1] + dist, ctr[2] + rad * 0.5)),
                 ('-X', (ctr[0] - dist, ctr[1], ctr[2] + rad * 0.5)), ('-Y', (ctr[0], ctr[1] - dist, ctr[2] + rad * 0.5))]
        T = TILE
        sheet = Image.new('RGB', (T * 4 + 18, T + 22), BG)
        dr = ImageDraw.Draw(sheet)
        dr.text((4, 2), f'{oid} {o.label} · up {up} · seen from:', font=font(13), fill=(30, 50, 70))
        for i, (name, eye) in enumerate(views):
            C = look_at(eye, ctr)
            fpx = T * 1.25
            rgb, mask = render_mesh(m, col, C, fpx, fpx, T / 2, T / 2, T, T)
            im = Image.fromarray((np.where(mask[..., None], rgb, np.array(BG) / 255) * 255).astype(np.uint8))
            d = ImageDraw.Draw(im)
            d.text((6, 4), name, font=font(14), fill=(30, 50, 70))
            o3 = np.array([ctr[0], ctr[1], lo[2]])
            for k, (cc, ax) in enumerate(zip([(220, 60, 60), (40, 160, 70), (50, 100, 220)], np.eye(3))):
                p = np.array([o3, o3 + ax * rad * 0.8])
                Pc = (np.c_[p, np.ones(2)] @ np.linalg.inv(C).T)[:, :3]
                if (Pc[:, 2] > 0).all():
                    uu = fpx * Pc[:, 0] / Pc[:, 2] + T / 2
                    vv = fpx * Pc[:, 1] / Pc[:, 2] + T / 2
                    d.line([uu[0], vv[0], uu[1], vv[1]], fill=cc, width=3)
                    d.text((uu[1] + 3, vv[1] - 8), 'XYZ'[k], font=font(12), fill=cc)
            sheet.paste(im, (i * (T + 6), 22))
        return sheet

    # ------------------------------------------------------------ 3. boxed key frame

    def boxes(self, oids, seq) -> dict:
        """oid -> (x0, y0, x1, y1, source) for the batch objects visible in this frame."""
        f = self.frame(seq)
        depth = self.sensor.depth(seq)
        out = {}
        for oid in oids:
            o = self.scene.objects[oid]
            b = box_from_points(o.points(), f.c2w, self.cam, depth)
            if b is not None:
                out[oid] = (*b[:4], 'scan')
                continue
            own = next((k for k in o.key_frames if k.seq == seq), None)
            if own is not None:
                mb = box_from_mask(own.mask)
                if mb is not None:
                    out[oid] = (*mb[:4], 'mask')
        return out

    def boxed_frame(self, oids, seq) -> tuple:
        """The key frame at full size with a coloured box per visible batch object and a legend."""
        im = self.photo(seq)
        d = ImageDraw.Draw(im)
        boxes = self.boxes(oids, seq)
        for oid, (x0, y0, x1, y1, _) in boxes.items():
            c = self.colour(oid)
            d.rectangle([x0, y0, x1, y1], outline=c, width=5)
            d.rectangle([x0, y0, x0 + 96, y0 + 24], fill=c)
            d.text((x0 + 4, y0 + 1), oid[4:], font=font(18), fill=(0, 0, 0))
        # legend, in the corner that covers the least of the boxes (it must not hide one)
        rows = [o for o in oids]
        lw, lh = 330, 30 * len(rows) + 34
        W, H = im.size
        corners = [(8, 8), (W - lw - 8, 8), (8, H - lh - 8), (W - lw - 8, H - lh - 8)]

        def covered(x, y):
            tot = 0
            for (x0, y0, x1, y1, _) in boxes.values():
                ix = max(0, min(x1, x + lw) - max(x0, x)); iy = max(0, min(y1, y + lh) - max(y0, y))
                tot += ix * iy / max(1, (x1 - x0) * (y1 - y0))      # share of that box hidden
            return tot
        lx, ly = min(corners, key=lambda c: covered(*c))
        d.rectangle([lx, ly, lx + lw, ly + lh], fill=(255, 255, 255))
        d.text((lx + 8, ly + 4), f'key frame {seq}', font=font(18), fill=(20, 30, 40))
        for i, oid in enumerate(rows):
            y = ly + 32 + 30 * i
            c = self.colour(oid)
            d.rectangle([lx + 8, y, lx + 32, y + 22], fill=c)
            seen = '' if oid in boxes else '  (not visible here)'
            d.text((lx + 40, y), f'{oid}  {self.scene.objects[oid].label}{seen}', font=font(16), fill=(20, 30, 40))
        return im, boxes

    # ------------------------------------------------------------ 4. stitched image

    def stitched(self, oid, seq, include=()) -> tuple:
        """Photo | Isaac render through the same camera. In the render: this object and
        everything already in the scene (M2 decides), nothing else."""
        if self.isaac is None:
            raise RuntimeError('stitched images need the Isaac service')
        f = self.frame(seq)
        self._n += 1
        out = self.out / 'renders' / f'{self._n:05d}_{oid}_{seq:04d}'
        objects = self.state.isaac_objects(include=[oid, *include])
        r = self.isaac.render(objects, [{'name': f'seq_{seq:04d}', 'c2w': f.c2w}], self.cam, out)
        render = Image.open(r['views'][0]['rgb']).convert('RGB')
        photo = self.photo(seq)
        W, H = HALF
        img = Image.new('RGB', (W * 2 + 6, H + 30), (255, 255, 255))
        img.paste(photo.resize((W, H)), (0, 30))
        img.paste(render.resize((W, H)), (W + 6, 30))
        d = ImageDraw.Draw(img)
        d.text((6, 4), f'key frame {seq} · photo', font=font(16), fill=(20, 30, 40))
        d.text((W + 12, 4), f'Isaac Sim, same camera · {oid} + objects already placed', font=font(16), fill=(20, 30, 40))
        path = self.out / 'stitched' / f'{self._n:05d}_{oid}_{seq:04d}.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(path)
        return img, path

    def first_key_frame(self, oid) -> int:
        """The object's first key frame: its lowest sequence number (D02), which is the batch's shared frame.
        Not `primary` - that is the frame the asset was generated from, often a later one."""
        return min(f.seq for f in self.scene.objects[oid].key_frames)

    def after_change(self, oid) -> tuple:
        """The image that comes back after every change that took effect: this object's first key frame."""
        return self.stitched(oid, self.first_key_frame(oid))

    # ------------------------------------------------------------ batch opening

    def batch_opening(self, oids, seqs) -> dict:
        """Everything given to the agent at the start of a batch, written to disk too."""
        self.set_batch_colours(oids)
        root = self.out / 'opening'
        root.mkdir(parents=True, exist_ok=True)
        cards = {o: self.card(o) for o in oids}
        text = '\n\n'.join(self.card_text(o) for o in oids)
        (root / 'cards.json').write_text(json.dumps(cards, indent=1, ensure_ascii=False))
        (root / 'cards.txt').write_text(text)
        sides, frames = {}, {}
        for o in oids:
            im = self.four_sides(o)
            p = root / f'four_sides_{o}.png'
            im.save(p)
            sides[o] = p
        boxes_all = {}
        for s in seqs:
            im, boxes = self.boxed_frame(oids, s)
            p = root / f'key_frame_{s:04d}.png'
            im.save(p)
            frames[s] = p
            boxes_all[s] = {o: list(b) for o, b in boxes.items()}
        (root / 'boxes.json').write_text(json.dumps(boxes_all, indent=1))
        return {'cards': cards, 'cards_text': text, 'four_sides': sides, 'key_frames': frames, 'boxes': boxes_all}
