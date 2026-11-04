"""Assemble the renderer bundle that both pages build on.

The result page is not self-contained: it needs the point renderer and the shared
stylesheet that the incremental viewer and the raw-mask viewer were built around.
This module copies that bundle into a site directory so the result page can load
it. It deliberately lives in `pagegen/` because every byte it writes is page
output, not pipeline output.
"""
from pathlib import Path
import shutil

from src import config

TEMPLATES = config.WEB_ROOT
SHARED = TEMPLATES / 'shared'


def build(out, frames=30):
    """Copy the shared renderer bundle into `out` (a site directory)."""
    del frames  # kept for call-site compatibility; the bundle is window-agnostic
    for name in ('index.html', 'incremental.js', 'incremental.css'):
        shutil.copy2(SHARED / name, out / name)
    shutil.copy2(SHARED / 'styles.css', out / 'styles.css')
    script = (SHARED / 'viewer.js').read_text()
    assert script.rstrip().endswith('main();'), (
        'viewer.js no longer ends with main(); the result page cannot strip it')
    (out / 'viewer.js').write_text(script.rstrip()[:-len('main();')])


if __name__ == '__main__':
    import sys
    build(Path(sys.argv[1]))
