import unittest

import numpy as np
import torch

from jetvae.transformer.model import ParticleTransformerShapeEMDVAE
from jetvae.data.pointset import event_to_z_eta_phi as _e2z
from jetvae.transformer.shape_ot import (
    jet_axis_from_shape,
    shape_sinkhorn_emd_loss,
)


def event_to_z_eta_phi(event, max_particles=None):
    return _e2z(event, max_particles=max_particles)[0]


def _massless_event(rows):
    """rows: list of (pid, px, py, pz, E) -- E is intentionally arbitrary/wrong
    to make sure preprocessing never reads it."""
    return np.asarray(rows, dtype=np.float32)


class EventToZEtaPhiTests(unittest.TestCase):
    def test_z_sums_to_one(self):
        event = _massless_event(
            [
                [22, 4.0, 0.0, 1.0, 999.0],
                [22, 0.0, 2.0, -0.5, -5.0],
                [22, -1.0, 1.0, 0.3, 0.0],
            ]
        )
        shape = event_to_z_eta_phi(event)
        self.assertAlmostEqual(float(shape[:, 0].sum()), 1.0, places=5)

    def test_phi_axis_is_exactly_centered(self):
        # Azimuthal recentering is a rigid 2D rotation, so it is exact for any
        # spread, not just small-angle jets.
        event = _massless_event(
            [
                [22, 4.0, 0.0, 1.0, 999.0],
                [22, 0.0, 2.0, -0.5, -5.0],
                [22, -1.0, 1.0, 0.3, 0.0],
                [22, 0.5, -0.5, -0.1, 0.0],
            ]
        )
        shape = event_to_z_eta_phi(event)
        z, eta, phi = shape[:, 0], shape[:, 1], shape[:, 2]
        px = (z * np.cos(phi)).sum()
        py = (z * np.sin(phi)).sum()
        self.assertAlmostEqual(float(np.arctan2(py, px)), 0.0, places=5)

    def test_eta_shift_matches_original_axis_exactly(self):
        # eta_i' must equal eta_i minus the *original* jet's own pseudorapidity
        # -- this is the shift's definition, and holds exactly regardless of
        # angular spread (see METHOD.md for why the *recentered* axis is only
        # approximately, not exactly, zero for a spread-out jet).
        rows = [
            [22, 4.0, 0.0, 1.0, 999.0],
            [22, 0.0, 2.0, -0.5, -5.0],
            [22, -1.0, 1.0, 0.3, 0.0],
            [22, 0.5, -0.5, -0.1, 0.0],
        ]
        event = _massless_event(rows)
        px = np.array([r[1] for r in rows])
        py = np.array([r[2] for r in rows])
        pz = np.array([r[3] for r in rows])
        pt = np.hypot(px, py)
        eta = np.arcsinh(pz / pt)
        eta_jet = np.arcsinh(pz.sum() / np.hypot(px.sum(), py.sum()))

        shape = event_to_z_eta_phi(event)
        order = np.argsort(-pt, kind="stable")
        np.testing.assert_allclose(shape[:, 1], (eta - eta_jet)[order], atol=1e-5)

    def test_recentered_axis_is_small_for_a_collimated_jet(self):
        # For a realistically collimated jet (spread << 1 rad, as expected
        # inside an R=0.4 cone) the recentered axis lands close to, though not
        # exactly at, the origin -- see METHOD.md's "What this shift is exact
        # for, and what it isn't" section for why an exact zero isn't possible
        # for a jet with nonzero invariant mass.
        rng = np.random.default_rng(0)
        n = 20
        eta = rng.normal(scale=0.15, size=n)
        phi = rng.normal(scale=0.15, size=n)
        pt = rng.uniform(0.5, 5.0, size=n)
        px = pt * np.cos(phi)
        py = pt * np.sin(phi)
        pz = pt * np.sinh(eta)
        rows = [[22, px[i], py[i], pz[i], 0.0] for i in range(n)]
        event = _massless_event(rows)

        shape = event_to_z_eta_phi(event)
        z, eta_s, phi_s = shape[:, 0], shape[:, 1], shape[:, 2]
        eta_axis, phi_axis = jet_axis_from_shape(
            torch.from_numpy(z)[None], torch.from_numpy(eta_s)[None],
            torch.from_numpy(phi_s)[None],
        )
        self.assertLess(float(eta_axis.abs()), 0.05)
        self.assertAlmostEqual(float(phi_axis), 0.0, places=4)

    def test_energy_column_is_ignored(self):
        base = [
            [22, 4.0, 0.0, 1.0, 999.0],
            [22, 0.0, 2.0, -0.5, -5.0],
        ]
        perturbed = [[row[0], row[1], row[2], row[3], row[4] + 123.0] for row in base]
        shape_a = event_to_z_eta_phi(_massless_event(base))
        shape_b = event_to_z_eta_phi(_massless_event(perturbed))
        np.testing.assert_allclose(shape_a, shape_b, atol=1e-6)

    def test_pt_ranked_order(self):
        event = _massless_event(
            [
                [22, 0.5, 0.0, 0.0, 0.5],
                [22, 4.0, 0.0, 0.0, 4.0],
                [22, 1.0, 0.0, 0.0, 1.0],
            ]
        )
        shape = event_to_z_eta_phi(event)
        self.assertTrue(np.all(np.diff(shape[:, 0]) <= 0))

    def test_truncation_keeps_invariants(self):
        rows = [[22, float(i + 1), 0.1, 0.05 * i, 0.0] for i in range(10)]
        shape = event_to_z_eta_phi(_massless_event(rows), max_particles=4)
        self.assertEqual(shape.shape[0], 4)
        self.assertAlmostEqual(float(shape[:, 0].sum()), 1.0, places=5)


class DecoderConstraintTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = ParticleTransformerShapeEMDVAE(
            max_particles=6,
            latent_dim=3,
            embed_dim=8,
            depth=1,
            num_heads=2,
            dropout=0.0,
            sinkhorn_n_iters=20,
        )

    def test_decode_sums_z_to_one(self):
        z = torch.randn(5, self.model.latent_dim)
        z_frac, _eta, _phi = self.model.decode(z)
        totals = z_frac.sum(dim=1)
        torch.testing.assert_close(totals, torch.ones_like(totals), atol=1e-5, rtol=1e-5)

    def test_decode_axis_is_origin(self):
        z = torch.randn(5, self.model.latent_dim)
        z_frac, eta, phi = self.model.decode(z)
        eta_axis, phi_axis = jet_axis_from_shape(z_frac, eta, phi)
        torch.testing.assert_close(
            eta_axis, torch.zeros_like(eta_axis), atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            phi_axis, torch.zeros_like(phi_axis), atol=1e-4, rtol=1e-4
        )

    def test_sample_satisfies_constraints(self):
        z_frac, eta, phi = self.model.sample(4, torch.device("cpu"))
        totals = z_frac.sum(dim=1)
        torch.testing.assert_close(totals, torch.ones_like(totals), atol=1e-5, rtol=1e-5)
        eta_axis, phi_axis = jet_axis_from_shape(z_frac, eta, phi)
        self.assertTrue(torch.allclose(eta_axis, torch.zeros_like(eta_axis), atol=1e-4))
        self.assertTrue(torch.allclose(phi_axis, torch.zeros_like(phi_axis), atol=1e-4))

    def test_backward_is_finite(self):
        particles = torch.randn(2, 6, 3) * 0.1
        mask = torch.ones(2, 6, dtype=torch.bool)
        losses = self.model.loss_function(
            *self.model(particles, mask, particles), M_N=1.0
        )
        losses["loss"].backward()
        self.assertTrue(torch.isfinite(losses["loss"]))
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all())


class ShapeSinkhornLossTests(unittest.TestCase):
    def test_identical_jets_have_near_zero_debiased_loss(self):
        z = torch.tensor([[0.7, 0.3, 0.0]])
        eta = torch.tensor([[0.1, -0.2, 0.0]])
        phi = torch.tensor([[0.05, -0.1, 0.0]])
        mask = torch.tensor([[True, True, False]])
        loss = shape_sinkhorn_emd_loss(
            z, eta, phi, mask, z.clone(), eta.clone(), phi.clone(), n_iters=100
        )
        self.assertLess(float(loss), 2e-4)

    def test_angular_scale_rescales_cost(self):
        z = torch.tensor([[1.0]])
        eta = torch.tensor([[0.0]])
        phi = torch.tensor([[0.0]])
        mask = torch.tensor([[True]])
        pred_eta = torch.tensor([[0.4]])
        loss_r1 = shape_sinkhorn_emd_loss(
            z, eta, phi, mask, z, pred_eta, phi, angular_scale=1.0,
            debias=False, n_iters=50,
        )
        loss_r04 = shape_sinkhorn_emd_loss(
            z, eta, phi, mask, z, pred_eta, phi, angular_scale=0.4,
            debias=False, n_iters=50,
        )
        self.assertGreater(float(loss_r04), float(loss_r1))


if __name__ == "__main__":
    unittest.main()
