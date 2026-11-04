"""Regression checks for the packaged SAM3/SAM3.1 session compatibility patch."""
import pytest
from sam3.model.sam3_base_predictor import Sam3BasePredictor


class MultiplexModel:
    def init_state(self, resource_path, offload_video_to_cpu=False):
        return (resource_path, offload_video_to_cpu)


class Sam3Model:
    def init_state(self, resource_path, offload_video_to_cpu=False, offload_state_to_cpu=False):
        return (resource_path, offload_video_to_cpu, offload_state_to_cpu)


class FlexibleModel:
    def init_state(self, **kwargs):
        return kwargs


@pytest.mark.parametrize('model', [MultiplexModel(), Sam3Model(), FlexibleModel()])
def test_public_start_session(model):
    predictor = Sam3BasePredictor()
    predictor.model = model
    result = predictor.handle_request({'type': 'start_session', 'resource_path': 'frames'})
    assert result['session_id'] in predictor._all_inference_states


def test_supported_offloading_passed_through():
    predictor = Sam3BasePredictor()
    predictor.model = Sam3Model()
    result = predictor.handle_request({'type': 'start_session', 'resource_path': 'frames',
                                       'offload_state_to_cpu': True})
    assert predictor._all_inference_states[result['session_id']]['state'] == ('frames', False, True)


def test_unsupported_offloading_is_not_silently_ignored():
    predictor = Sam3BasePredictor()
    predictor.model = MultiplexModel()
    with pytest.raises(ValueError, match='does not support offload_state_to_cpu'):
        predictor.handle_request({'type': 'start_session', 'resource_path': 'frames',
                                  'offload_state_to_cpu': True})
    assert not predictor._all_inference_states
