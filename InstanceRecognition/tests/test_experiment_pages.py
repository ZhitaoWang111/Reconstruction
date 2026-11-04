"""Publishing different experiments must not overwrite one shared report.

Pipeline records and generated pages live in two different trees, so this test
pins BOTH: that the page and its assets land under the site root, and that the
links the page emits back into the records tree still resolve.
"""
import json
import sys
from pagegen import build_records as builder
from src import config as app_config
from src.visualization.page_naming import records_asset_name, records_page_name


def test_two_runs_get_separate_pages_assets_and_unchanged_logs(tmp_path, monkeypatch):
    run_root = tmp_path / 'artifacts/experiments'
    site_root = tmp_path / 'site/experiments'
    ids = ['harness-v2-gpt6-sol-30frames-20260924',
           'harness-v2-gpt6-sol-150frames-30windows-20260929']
    first_page = None
    for run in ids:
        folder = run_root / run
        (folder / 'calls').mkdir(parents=True)
        (folder / 'config.json').write_text(json.dumps({'model':'openai/gpt-6-sol', 'frames':30, 'window':5}))
        log = folder / 'calls/w0000_123_call00_request.json'
        raw = b'{"messages": [{"content": "unaltered original evidence"}]}\n'
        log.write_bytes(raw)
        monkeypatch.setattr(sys, 'argv', ['build_records', '--run', run,
                                          '--run-dir', str(folder),
                                          '--site-dir', str(site_root / run)])
        builder.main()
        # The page and its resources live under the SITE root; the logs stay under
        # the artifacts root. Asserting both locations is what lets this test catch
        # a layout drift — a filename-only assertion would keep passing.
        site = site_root / run
        page = site / 'records.html'
        assets = site / 'records'
        assert page.is_file() and (assets / 'records.js').is_file()
        assert page.parent == site
        assert assets.parent == site, 'records assets must sit inside the site directory'
        assert (folder / 'calls').is_dir() and not (site / 'calls').exists(), (
            'pipeline logs must not be duplicated into the site tree')
        assert 'href="index.html"' in page.read_text()
        assert 'experiment-navigation.js' in page.read_text()
        # Per-window data and log links are relative to the page, so they point back
        # across into the records tree from wherever the site tree happens to be.
        records_js = (assets / 'records.js').read_text()
        assert '"data_url":"records/window-0000.js"' in records_js
        assert '../artifacts/experiments/' in records_js, (
            'log links must be computed relative to the page, not assumed to be siblings')
        assert log.read_bytes() == raw
        # The page and its resources must resolve to the same directory.
        assert assets.name == records_asset_name(records_page_name(run))
        if first_page is None:
            first_page = (page, page.read_bytes())
        else:
            assert first_page[0].read_bytes() == first_page[1]


def test_records_names_are_position_independent(tmp_path):
    """Two different experiments must not borrow each other's page or assets."""
    a = records_page_name('harness-v2-gpt6-sol-30frames-20260924')
    b = records_page_name('harness-v2-gpt6-sol-500frames-20260924')
    assert a == b == 'records.html', (
        'inner names are constant on purpose: the experiment directory carries the '
        'identity, so the pairing cannot drift between two derivations of one string')
    assert records_asset_name(a) == records_asset_name('anything-else.html') == 'records'


def test_records_default_directories_follow_the_two_roots(tmp_path, monkeypatch):
    """With no explicit overrides, the page goes to <site> and the logs are read
    from <artifacts> — never the same tree."""
    monkeypatch.setattr(app_config, 'ARTIFACTS_ROOT', tmp_path / 'artifacts')
    monkeypatch.setattr(app_config, 'SITE_ROOT', tmp_path / 'site')
    run = 'harness-v2-gpt6-sol-30frames-20260924'
    folder = app_config.ARTIFACTS_ROOT / 'experiments' / run
    (folder / 'calls').mkdir(parents=True)
    (folder / 'config.json').write_text(json.dumps({'model': 'openai/gpt-6-sol', 'frames': 30, 'window': 5}))
    (folder / 'calls/w0000_1_call00_request.json').write_text('{}')
    monkeypatch.setattr(sys, 'argv', ['build_records', '--run', run])
    builder.main()
    assert (app_config.SITE_ROOT / 'experiments' / run / 'records.html').is_file()
    assert not (folder / 'records.html').exists(), (
        'the records page must not be written next to the pipeline logs')


def test_both_templates_resolve_shared_navigation_by_placeholder():
    """Neither page may bake in a depth for the shared navigation assets.

    The two pages live at the same depth but under independently-relocatable
    roots, so a hardcoded `../../..` would break as soon as one root moves.
    """
    from src import config as app_config

    result = (app_config.WEB_ROOT / 'result' / 'harness_result.html').read_text()
    records = (app_config.WEB_ROOT / 'records' / 'page.html').read_text()
    assert result.count('@@NAV@@/experiment-navigation.css') == 1
    assert result.count('@@NAV@@/experiment-navigation.js') == 1
    assert '@@NAV@@/experiment-navigation.css' in records
    assert '@@NAV@@/experiment-navigation.js' in records


def test_records_template_command_matches_the_portable_entry_point():
    """The page tells the reader how to regenerate it; a stale path misleads."""
    from src import config as app_config

    records = (app_config.WEB_ROOT / 'records' / 'page.html').read_text()
    assert 'pagegen/build_records.py --run @@RUN@@' in records
    assert 'InstanceRecognition/src/visualization' not in records
