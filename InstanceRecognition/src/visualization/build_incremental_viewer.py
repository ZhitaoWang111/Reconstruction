"""Reuse the existing point renderer and raw-mask viewer for incremental replay."""
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
def build(out,frames=30):
    template=ROOT/'web/incremental_objects'
    for name in ('index.html','incremental.js','incremental.css'):shutil.copy2(template/name,out/name)
    shutil.copy2(ROOT/'web/mask_pointcloud/styles.css',out/'styles.css')
    script=(ROOT/'web/mask_pointcloud/viewer.js').read_text()
    assert script.rstrip().endswith('main();')
    (out/'viewer.js').write_text(script.rstrip()[:-len('main();')])

if __name__=='__main__':
    import sys
    build(Path(sys.argv[1]))
