"""The harness gate must accept exactly the source policy this tree can build.

`run.ACCEPTED_LIFTING_POLICIES` is exact-matched on purpose: an exact check is what
stops a mis-pointed `--source` from silently running unintended geometry. These
tests pin the accepted set -- one entry, not a family -- and guard the places that
would silently misdescribe a source if a second policy were ever reintroduced.

The count is asserted rather than the containment alone, because adding a policy
without deciding to is exactly the failure this file exists to catch.

Run-level behaviour under the accepted policy is covered in `test_harness_run.py`,
where the shared `runner` fixture lives.
"""

import inspect

from src.geometry.amb3r_mask_source import POLICY as AMB3R_FRAME_POLICY
from src.harness import run


def test_accepted_set_is_the_amb3r_lift_and_nothing_else():
    assert AMB3R_FRAME_POLICY == 'amb3r_per_frame_v1'
    assert run.ACCEPTED_LIFTING_POLICIES == (AMB3R_FRAME_POLICY,), (
        'Widening the accepted set must be a deliberate act; a second entry here '
        'means some source policy was added without a decision.')


def test_retired_policies_are_gone_from_the_tree():
    """A retired policy must not linger as an accepted input.

    The cloud and mesh lifts were deleted rather than merely retired: keeping a
    policy the tree can no longer build would accept a source nobody can rebuild.
    """
    for policy in ('independent_scene_visibility_v1', 'cluster_pruned_v1',
                   'mesh_first_surface_v1'):
        assert policy not in run.ACCEPTED_LIFTING_POLICIES


def test_publish_label_branch_names_every_accepted_policy():
    """publish.py refuses a policy it has no wording for.

    A source policy missing from that check would be published under whatever
    label the template happened to carry, describing geometry it was not built
    from.
    """
    from pagegen import build_site as publish_module

    source = ' '.join(inspect.getsource(publish_module.publish).split())
    assert 'policy = data.manifest.get' in source, (
        'publish() no longer reads the source lifting policy')
    assert 'AMB3R_FRAME_POLICY' in source
    assert publish_module.AMB3R_FRAME_POLICY == AMB3R_FRAME_POLICY
    assert 'AMB3R 逐帧深度' in source


def test_result_page_discloses_the_geometry_it_was_built_from():
    """The result page overwrites index.html, so publish.py's note is discarded.

    The geometry disclosure therefore has to be reapplied in build_result; without
    it the source is described on its own page by the template's legacy wording —
    geometry it was not built from.
    """
    from pagegen import build_result as result_module

    source = ' '.join(inspect.getsource(result_module.build).split())
    assert 'AMB3R_FRAME_POLICY' in source
    assert 'RESULT_NOTE_AMB3R' in source
    assert 'Sim(3)' in result_module.RESULT_NOTE_AMB3R
    assert 'AMB3R' in result_module.RESULT_NOTE_AMB3R
    assert result_module.RESULT_NOTE_AMB3R != result_module.RESULT_NOTE, (
        'the disclosure must differ from the template note it replaces, or the '
        'replace is a no-op')


def test_result_template_still_carries_the_note_being_replaced():
    """The disclosure is a string replace; a reworded template must fail loudly.

    build() raises when the anchor text is absent, so a silent no-op is already
    impossible — this pins the template side of that contract.
    """
    from src import config as app_config
    from pagegen.build_result import RESULT_NOTE

    template = (app_config.WEB_ROOT / 'result' / 'harness_result.html').read_text()
    assert template.count(RESULT_NOTE) == 1
