"""End-to-end on synthetic data: .dat -> preprocess -> train (2 epochs, CPU) -> encode."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from jetvae.data.dat_io import load_events_from_dat
from jetvae.data.preprocess import main as preprocess_main
from jetvae.transformer import encode as encode_mod
from jetvae.transformer import train as train_mod

PIDS = [22, 211, -211, 2112, 130, 11, 13, 321, 2212]


def write_dat(path, n_jets, rng, mean_mult):
    lines = []
    for i in range(n_jets):
        lines.append(f"<event_begin>:{i}")
        for _ in range(max(3, rng.poisson(mean_mult))):
            px, py = rng.normal(0, 2, 2)
            pz = rng.normal(0, 5)
            e = float(np.sqrt(px * px + py * py + pz * pz))
            lines.append(f"{rng.choice(PIDS)} {px:.5f} {py:.5f} {pz:.5f} {e:.5f}")
        lines.append(f"<event_end>:{i}")
    Path(path).write_text("\n".join(lines) + "\n")


class PipelineTest(unittest.TestCase):
    def test_end_to_end(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_dat(d / "u.dat", 60, rng, 8)
            write_dat(d / "g.dat", 60, rng, 14)
            self.assertEqual(len(load_events_from_dat(d / "u.dat")), 60)

            cache = d / "cache"
            preprocess_main(["--u-jet", str(d / "u.dat"), "--g-jet", str(d / "g.dat"),
                             "--cache-dir", str(cache), "--max-particles", "16"])
            meta = json.loads((cache / "meta.json").read_text())
            self.assertEqual(meta["train_events"] + meta["val_events"], 120)
            par = np.load(cache / "train_particles.npy")
            msk = np.load(cache / "train_mask.npy")
            lab = np.load(cache / "train_labels.npy")
            self.assertEqual(par.shape[1:], (16, 3))
            np.testing.assert_allclose((par[..., 0] * msk).sum(1), 1.0, atol=1e-5)
            self.assertEqual(set(np.unique(lab)), {0, 1})
            self.assertAlmostEqual(float(lab.mean()), 0.5, delta=0.02)  # stratified

            out = d / "run" / "best_model.pth"
            train_mod.main_with_args([
                "--cache-dir", str(cache), "--output", str(out), "--device", "cpu",
                "--epochs", "2", "--batch-size", "32", "--embed-dim", "16",
                "--depth", "1", "--num-heads", "2", "--latent-dim", "4",
                "--sinkhorn-epsilon-anneal-epochs", "1", "--sinkhorn-n-iters", "5",
                "--sinkhorn-n-iters-final", "8", "--n-generate", "4"])
            self.assertTrue(out.exists())
            self.assertTrue((d / "run" / "best_model_history.json").exists())

            latents = d / "lat.npz"
            encode_mod.main(["--checkpoint", str(out), "--cache-dir", str(cache),
                             "--split", "val", "--out", str(latents)])
            z = np.load(latents)
            self.assertEqual(z["mu"].shape, (meta["val_events"], 4))
            self.assertTrue(np.isfinite(z["logvar"]).all())

    def test_unknown_pid_raises(self):
        from jetvae.data.pointset import event_to_z_eta_phi
        with self.assertRaises(ValueError):
            event_to_z_eta_phi(np.array([[9999, 1, 0, 0, 1]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
