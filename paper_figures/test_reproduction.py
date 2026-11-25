"""Regression checks for paper selection, GLS and checkpoint-aware caches.

Run: python -m unittest discover -s paper_figures -p 'test_*.py'
No training data or model weights are needed.
"""
import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from common import validate_paper_cnn
from fig4_5_parameter_response import (CHANGES, PARAMS, cache_identity, pooled_moments,
                                      responses, setting_paths, summarize)


class ReproductionTests(unittest.TestCase):
    def test_fixed_class_composition_variance(self):
        mean, variance = pooled_moments([np.array([[-1.], [1.]]), np.array([[9.], [11.]])])
        np.testing.assert_allclose(mean, [5.])
        # Within-class mean variances are both one, with fixed weights one half.
        np.testing.assert_allclose(variance, [.5])

    def test_gls_uses_free_intercept_and_heteroscedastic_weights(self):
        x = np.array([-.2, -.1, .1, .2])
        y = np.array([.30, .07, -.08, -.31])
        independent_variances = np.array([1., 2., 4., 8.])
        weights = 1 / independent_variances
        xm = np.average(x, weights=weights); ym = np.average(y, weights=weights)
        expected = np.sum(weights * (x - xm) * (y - ym)) / np.sum(weights * (x - xm)**2)
        means = np.full((8, 5, 1), -5.)
        means[:, [0, 1, 3, 4], 0] += y
        variance = np.ones_like(means)
        variance[:, [0, 1, 3, 4], 0] = independent_variances
        for common_variance in [.01, 10.]:
            variance[:, 2, 0] = common_variance
            delta, slopes = responses(means, np.array([True]), variance)
            np.testing.assert_allclose(slopes, expected, atol=1e-13)
            np.testing.assert_array_equal(delta[:, 2], 0.)
        self.assertGreater(abs(expected - np.polyfit(x, y, 1)[0]), .01)

    def test_weak_response_exclusion_is_only_for_cnn_alpha_summary(self):
        data = {}
        means = np.broadcast_to(np.array(CHANGES)[None, :, None] * [.02, 1.], (8, 5, 2)).copy()
        for model in ['cnn', 'transformer']:
            data[f'mean_lnvar_{model}'] = means
            data[f'variance_mean_{model}'] = np.full_like(means, .01)
            data[f'active_{model}'] = np.array([True, True])
        with contextlib.redirect_stdout(io.StringIO()):
            result = summarize(data)
        self.assertEqual(result['cnn']['summary'][0][2], 1)
        self.assertEqual(result['cnn']['summary'][1][2], 2)
        self.assertEqual(result['transformer']['summary'][0][2], 2)
        self.assertEqual(result['cnn']['delta'].shape[2], 2)
        np.testing.assert_allclose(result['cnn']['summary'][0][:2], [1., 0.])

    def test_cache_detects_checkpoint_replacement_at_same_path(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = SimpleNamespace(scan_dir=root, cnn_checkpoint=root/'cnn.pth',
                                transformer_checkpoint=root/'transformer.pth')
            for checkpoint in [a.cnn_checkpoint, a.transformer_checkpoint]:
                checkpoint.write_bytes(b'old!')
            for param, _, _ in PARAMS:
                for change in CHANGES:
                    for path in setting_paths(root, param, change):
                        path.parent.mkdir(parents=True, exist_ok=True); path.touch()
            before = cache_identity(a)
            a.cnn_checkpoint.write_bytes(b'new!')
            self.assertNotEqual(before, cache_identity(a))

    def test_old_cnn_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'beta=2e-6'):
            validate_paper_cnn(dict(latent_dim=32, img_size=32, n_channels=1, beta=.001,
                                   recon_loss_weight=1000, normalization='event_pt', output_mode='event_pt'))


if __name__ == '__main__':
    unittest.main()
