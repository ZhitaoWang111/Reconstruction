# Interactive point-cloud comparison

Run the builder, then open the generated [`output/visualization/index.html`](../../output/visualization/index.html) directly in a WebGL-capable browser. The page works from `file://`; it does not require a local server, network access, Three.js, or another JavaScript dependency.

```bash
/usr/bin/google-chrome file:///home/wzt/wzt/CVPR_2027/Video2PointCloud/output/visualization/index.html
```

Controls:

- Left-drag rotates; right-drag or Shift-drag pans; the wheel zooms.
- Perspective, top, and side presets affect all three viewers.
- `同步视角` propagates manual camera changes between viewers.
- RGB and height coloring can be switched without reloading the clouds.
- Each method card can enter browser fullscreen mode.

The page compares three application categories: GT-corrected AMB3R-VO,
original AMB3R-VO with its RGB-only predicted scale, and DA3. The GT-corrected
variant uses all 500 ScanNet camera-pose correspondences to fit a Sim(3), so it
is explicitly labeled as a GT-assisted/oracle result. AMB3R colors are sampled
from the stored input images using the same confidence mask and deterministic
point selection as its evaluation. Positions are deterministically quantized to
unsigned 16-bit values within each cloud's recorded bounding box; colors remain
8-bit RGB. All three viewers decode those coordinates through one shared ScanNet
scene bound, so synchronized views preserve their real relative sizes instead
of fitting each cloud independently to its card.

Rebuild the browser assets after replacing any source PLY:

```bash
/home/wzt/.conda/envs/amb3r/bin/python \
  /home/wzt/wzt/CVPR_2027/Video2PointCloud/src/visualization/build_pointcloud_assets.py
```

Source paths, source hashes, point counts, bounds, and generated-asset hashes are recorded in [`output/visualization/assets/manifest.json`](../../output/visualization/assets/manifest.json). The builder also copies `index.html`, `styles.css`, and `viewer.js` from this source directory into the output bundle.
