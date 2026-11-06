"""The re-identification gate: which assigns need the archive looked at first.

The three scenarios are the ones the 150-frame run actually produced, reduced to
their coordinates.  They are written by hand rather than read from a fixture so
the test states the intended behaviour instead of restating observed output.
"""
import unittest
from src.harness.reid_gate import (DEFAULT_FLOOR_M, check_assign, self_scale,
                                   threshold_for, verify_reidentification)


def around(origin, offsets):
    """Centroids a few centimetres apart, the way a tracked object's are."""
    x, y, z = origin
    return [[x + dx, y + dy, z] for dx, dy in offsets]


# Eight archived observations of a small bin, spread under four centimetres.
BIN_ARCHIVE = around((3.70, 1.23, 0.21),
                     [(0, 0), (.01, .01), (-.01, .02), (.02, -.01),
                      (0, .03), (-.02, -.02), (.01, 0), (0, -.01)])


def archive(centroids, object_id='Object25'):
    return ({f'_{index:04d}': {'object_id': object_id}
             for index in range(len(centroids))},
            {f'_{index:04d}': point for index, point in enumerate(centroids)})


class SelfScale(unittest.TestCase):

    def test_small_spread_object_uses_the_floor(self):
        limit, scale = threshold_for(BIN_ARCHIVE)
        self.assertLess(scale, 0.05)
        self.assertEqual(limit, DEFAULT_FLOOR_M)

    def test_large_object_raises_its_own_threshold(self):
        # A rug's observations land on different parts of its surface.
        rug = [[3.8 + 0.25 * i, 1.5 + 0.1 * i, 0.09] for i in range(10)]
        limit, scale = threshold_for(rug)
        self.assertGreater(scale, 0.2)
        self.assertGreater(limit, DEFAULT_FLOOR_M)

    def test_short_history_has_no_estimate(self):
        self.assertIsNone(self_scale(BIN_ARCHIVE[:4]))
        self.assertEqual(threshold_for(BIN_ARCHIVE[:4])[0], DEFAULT_FLOOR_M)

    def test_duplicate_centroids_do_not_create_negative_scale(self):
        twins = [[0.0, 0.0, 0.0]] * 8
        self.assertEqual(self_scale(twins), 0.0)


class NeedsVerification(unittest.TestCase):

    def setUp(self):
        self.assignments, self.centroids = archive(BIN_ARCHIVE)

    def screen(self, observations, **kwargs):
        action = {'action': 'hold', 'object_id': 'Object25',
                  'observations': observations}
        return check_assign(0, action, self.assignments, self.centroids, **kwargs)

    def test_distant_reassignment_is_held(self):
        # W25: eight masks of a bin in another room, 52 frames after the object
        # was last seen.  Each is over five metres from anywhere it has been.
        self.centroids['0123:008'] = [6.77, 5.96, 0.52]
        check = self.screen(['0123:008'])
        self.assertIsNotNone(check)
        self.assertEqual(check['object_id'], 'Object25')
        self.assertGreater(check['worst_distance_m'], 5.0)
        self.assertEqual(check['threshold_m'], DEFAULT_FLOOR_M)
        self.assertEqual(check['history_count'], len(BIN_ARCHIVE))

    def test_nearby_assignment_passes_silently(self):
        # The ordinary case: three centimetres from the object's last sighting.
        self.centroids['0070:004'] = [3.705, 1.225, 0.21]
        self.assertIsNone(self.screen(['0070:004']))

    def test_only_the_offending_observations_are_counted(self):
        self.centroids['0123:008'] = [6.77, 5.96, 0.52]
        self.centroids['0070:004'] = [3.71, 1.22, 0.21]
        check = self.screen(['0070:004', '0123:008'])
        self.assertEqual(check['assigned_count'], 2)
        self.assertEqual(check['offending_count'], 1)
        self.assertEqual(check['nearest_archived_observation'][0], '_')

    def test_named_history_observation_is_reported(self):
        assignments = {'0069:005': {'object_id': 'Object25'},
                       '0068:006': {'object_id': 'Object25'}}
        centroids = {'0069:005': [3.65, 1.32, 0.19],
                     '0068:006': [3.67, 1.32, 0.20],
                     '0123:008': [6.77, 5.96, 0.52]}
        check = check_assign(0, {'action': 'hold', 'object_id': 'Object25',
                                 'observations': ['0123:008']},
                             assignments, centroids)
        self.assertIn(check['nearest_archived_observation'], ('0069:005', '0068:006'))

    def test_new_object_is_never_held(self):
        # An object created in this same proposal has no archive to compare to.
        action = {'action': 'hold', 'object_id': 'Object73',
                  'observations': ['0123:008']}
        self.centroids['0123:008'] = [6.77, 5.96, 0.52]
        self.assertIsNone(check_assign(0, action, self.assignments, self.centroids))

    def test_observation_without_centroid_is_skipped(self):
        # Two observations in the run had empty masks and no centroid at all.
        action = {'action': 'hold', 'object_id': 'Object25',
                  'observations': ['0073:004']}
        self.assertIsNone(check_assign(0, action, self.assignments, self.centroids))

    def test_large_object_absorbs_a_real_drift(self):
        # W18: a rug observation 1.08 m from the archive is legitimate.
        rug = [[3.8 + 0.25 * i, 1.5 + 0.1 * i, 0.09] for i in range(10)]
        assignments, centroids = archive(rug, 'Object26')
        centroids['0085:009'] = [5.53, 2.39, 0.11]
        action = {'action': 'hold', 'object_id': 'Object26',
                  'observations': ['0085:009']}
        self.assertIsNone(check_assign(0, action, assignments, centroids))


class ProposalLevel(unittest.TestCase):

    def test_only_assigns_are_screened(self):
        centroids = {'a': [0.0, 0.0, 0.0], 'b': [9.0, 9.0, 9.0]}
        assignments = {'a': {'object_id': 'Object25'}}
        actions = [
            {'action': 'create', 'object_id': 'Object73', 'object_label': 'bin'},
            {'action': 'hold', 'object_id': 'Object73', 'observations': ['b']},
            {'action': 'discard', 'observations': ['a']},
        ]
        result = verify_reidentification(actions, assignments, centroids)
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(result['checks'], [])

    def test_each_held_assign_reports_its_own_index(self):
        # W25 committed an Object25 and an Object54 assign in one proposal.
        centroids = {'a': [0.0, 0.0, 0.0], 'b': [1.0, 0.0, 0.0],
                     'c': [9.0, 9.0, 9.0], 'd': [0.0, 9.0, 9.0]}
        assignments = {'a': {'object_id': 'Object25'},
                       'b': {'object_id': 'Object54'}}
        actions = [
            {'action': 'hold', 'object_id': 'Object25', 'observations': ['c']},
            {'action': 'hold', 'object_id': 'Object54', 'observations': ['d']},
        ]
        result = verify_reidentification(actions, assignments, centroids)
        self.assertEqual(result['status'], 'verification_required')
        self.assertEqual([c['action_index'] for c in result['checks']], [0, 1])
        self.assertEqual([c['object_id'] for c in result['checks']],
                         ['Object25', 'Object54'])

    def test_verdict_is_stable_across_the_threshold_band(self):
        # The run's held cases sit 2 m and beyond; everything legitimate is under
        # 1.1 m.  Any floor in this band must classify identically, which is what
        # makes 1.0 a choice about margin rather than about outcomes.
        centroids = {'a': [0.0, 0.0, 0.0], 'near': [0.0, 0.5, 0.0],
                     'far': [6.0, 0.0, 0.0]}
        assignments = {'a': {'object_id': 'Object25'}}
        actions = [{'action': 'hold', 'object_id': 'Object25',
                    'observations': ['near']},
                   {'action': 'hold', 'object_id': 'Object25',
                    'observations': ['far']}]
        for floor in (0.8, 1.0, 1.2):
            result = verify_reidentification(actions, assignments, centroids,
                                             floor_m=floor)
            self.assertEqual([c['action_index'] for c in result['checks']], [1],
                             msg=f'floor={floor}')


if __name__ == '__main__':
    unittest.main()
