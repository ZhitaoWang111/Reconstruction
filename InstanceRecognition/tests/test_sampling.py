from src.sampling import uniform_sample_indices


def test_uniform_sampling_reduces_scene0000_to_exactly_500_frames():
    indices = uniform_sample_indices(5578, target_frames=500)

    assert len(indices) == 500
    assert indices[0] == 0
    assert indices[-1] == 5577
    assert indices == sorted(set(indices))
    assert max(b - a for a, b in zip(indices, indices[1:])) <= 12


def test_uniform_sampling_keeps_every_frame_in_short_sequences():
    assert uniform_sample_indices(4, target_frames=500) == [0, 1, 2, 3]


def test_uniform_sampling_rejects_invalid_sizes():
    for frame_count, target_frames in ((0, 500), (10, 0)):
        try:
            uniform_sample_indices(frame_count, target_frames)
            raise AssertionError("invalid sampling input should fail")
        except ValueError:
            pass
