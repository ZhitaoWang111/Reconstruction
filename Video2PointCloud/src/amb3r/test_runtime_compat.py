"""Numerical checks for the optional PTv3 attention adapter."""
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from runtime_compat import varlen_qkv_sdpa


class PackedAttentionTest(unittest.TestCase):
    def check_reference(self, lengths, device, dtype):
        torch.manual_seed(7)
        packed = torch.randn(sum(lengths), 3, 2, 16, device=device, dtype=dtype)
        offsets = torch.tensor([0] + torch.tensor(lengths).cumsum(0).tolist(),
                               device=device, dtype=torch.int32)
        actual = varlen_qkv_sdpa(packed, offsets, max(lengths), softmax_scale=0.3)
        expected = []
        for segment in packed.split(lengths):
            q, k, v = segment.double().permute(1, 2, 0, 3).unbind(0)
            expected.append(((q @ k.transpose(-1, -2) * 0.3).softmax(-1) @ v).transpose(0, 1))
        tolerance = 0.002 if dtype == torch.float16 else 1e-6
        torch.testing.assert_close(actual.double(), torch.cat(expected),
                                   atol=tolerance, rtol=tolerance)
        self.assertEqual(actual.dtype, dtype)

    def test_equal_length_patches(self):
        self.check_reference([4, 4, 4], 'cpu', torch.float32)

    def test_variable_length_patches(self):
        self.check_reference([3, 5, 1], 'cpu', torch.float32)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_half_precision(self):
        self.check_reference([4, 4, 4], 'cuda', torch.float16)
        self.check_reference([3, 5, 1], 'cuda', torch.float16)

    def test_invalid_offsets_rejected(self):
        with self.assertRaises(ValueError):
            varlen_qkv_sdpa(torch.zeros(3, 3, 2, 4), torch.tensor([0, 2]), 2)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_original_interpolation_chunk_invariance(self):
        repo = Path(__file__).resolve().parents[3] / 'Submodule/amb3r'
        sys.path[:0] = [str(repo), str(repo / 'thirdparty')]
        from amb3r.backend import BackEnd
        torch.manual_seed(3)
        points = SimpleNamespace(
            coord=torch.randn(11, 3, device='cuda'),
            feat=torch.randn(11, 8, device='cuda'),
            batch=torch.zeros(11, dtype=torch.long, device='cuda'),
        )
        query = torch.randn(1, 10, 3, device='cuda')
        backend = SimpleNamespace(k_neighbors=4)
        small = BackEnd.voxel_to_point_interpolation(backend, points, query, chunk_size=2)
        original = BackEnd.voxel_to_point_interpolation(backend, points, query, chunk_size=50_000)
        torch.testing.assert_close(small, original, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
