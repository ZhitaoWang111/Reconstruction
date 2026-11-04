"""Shared names for per-experiment result, records and translation artifacts."""
from pathlib import Path
import re

# The directory name is the public experiment ID. Old absolute evidence paths
# remain readable through filesystem aliases; they are not rewritten in logs.
LEGACY_RUN_IDS = {
    'harness_first30_v1': 'harness-v1-gpt56-terra-30frames-20260921',
    'harness_first30_v2_sol_20260924': 'harness-v2-gpt6-sol-30frames-20260924',
    'harness_all500_v1': 'harness-v2-gpt6-sol-500frames-20260924',
    'harness_first30windows_v2_sol_20260929': 'harness-v2-gpt6-sol-150frames-30windows-20260929',
}
# Minute precision: `-YYYYMMDD-HHMM`. Two experiments started the same day would
# otherwise collide, and the run name is the directory name on both output trees.
# The time part is optional so the pre-refactor runs (date only) still validate.
RUN_ID = re.compile(
    r'harness-v\d+-[a-z0-9]+(?:-[a-z0-9]+)*-\d+frames(?:-\d+windows)?-\d{8}(?:-\d{4})?(?:-r[1-9]\d*)?')

# Both artifacts live INSIDE the experiment directory, beside the result page.
# The experiment ID is already carried by that directory name, so the inner names
# are deliberately constant — repeating the ID there bought nothing and made the
# page/asset pairing depend on two independent derivations from the same string.
RECORDS_PAGE = 'records.html'
RECORDS_ASSETS = 'records'


def canonical_run_id(run):
    """Accept a directory name, not a path; keep legacy CLI invocations usable."""
    if not run or Path(run).name != run or run in ('.', '..'):
        raise ValueError('Use an experiment directory name for --run, not a path')
    return LEGACY_RUN_IDS.get(run, run)


def records_page_name(run):
    """Records page filename within the experiment directory."""
    run = canonical_run_id(run)
    if not RUN_ID.fullmatch(run):
        raise ValueError('Use harness-vN-model-Nframes[-Nwindows]-YYYYMMDD-HHMM[-rN] '
                         'or explicitly supply --page for a legacy experiment')
    return RECORDS_PAGE


def records_asset_name(page):
    """Translation and publication must resolve to the same resource directory.

    The page argument is still validated: an explicit `--page` must remain a bare
    filename, because the resource directory is resolved relative to the page.
    """
    page = Path(page)
    if page.name != str(page) or page.suffix != '.html':
        raise ValueError('--page must be an HTML filename within the experiment directory')
    return RECORDS_ASSETS
