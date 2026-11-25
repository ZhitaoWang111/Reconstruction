"""Render four views of a GLB mesh with its vertex colors."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    AmbientLights, BlendParams, FoVPerspectiveCameras, MeshRasterizer,
    MeshRenderer, RasterizationSettings, SoftPhongShader, TexturesVertex,
    look_at_view_transform,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mesh = trimesh.load(args.mesh, force="mesh", process=False)
    vertices = mesh.vertices.copy()
    vertices -= mesh.bounds.mean(axis=0)
    vertices /= np.ptp(mesh.vertices, axis=0).max()
    device = torch.device("cuda")
    verts = torch.as_tensor(vertices, dtype=torch.float32, device=device)
    faces = torch.as_tensor(mesh.faces.copy(), dtype=torch.int64, device=device)
    colors = torch.as_tensor(np.asarray(mesh.visual.vertex_colors[:, :3]).copy(), dtype=torch.float32, device=device) / 255
    meshes = Meshes(verts=[verts], faces=[faces], textures=TexturesVertex(verts_features=[colors]))
    lights = AmbientLights(device=device)
    settings = RasterizationSettings(image_size=512, blur_radius=0, faces_per_pixel=1, bin_size=0)
    sheet = Image.new("RGB", (1024, 1064), "white")
    for i, azimuth in enumerate([0, 90, 180, 270]):
        rotation, translation = look_at_view_transform(dist=2.1, elev=0, azim=azimuth, device=device)
        cameras = FoVPerspectiveCameras(device=device, R=rotation, T=translation, fov=32)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=settings),
            shader=SoftPhongShader(device=device, cameras=cameras, lights=lights, blend_params=BlendParams(background_color=(1, 1, 1))),
        )
        with torch.no_grad():
            rgba = renderer(meshes)[0].clamp(0, 1).cpu().numpy()
        im = Image.fromarray((rgba[:, :, :3] * 255).astype(np.uint8))
        x, y = (i % 2) * 512, (i // 2) * 532
        sheet.paste(im, (x, y + 20))
        ImageDraw.Draw(sheet).text((x + 12, y + 4), f"Azimuth {azimuth} degrees", fill="black")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
