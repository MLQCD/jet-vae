"""CNN scripts accept --config YAML; flags override it; bad keys / values are rejected; and a tiny
.dat -> preprocess -> train run works when driven entirely by YAML files."""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "cnn" / "code_training"
CFG = ROOT / "cnn" / "configs"
sys.path.insert(0, str(CODE))
import argparse  # noqa: E402

from config_utils import add_config_argument, parse_args_with_config  # noqa: E402


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--hidden-dims", type=str, default="32,64")
    p.add_argument("--loss-type", choices=["H", "B", "A"], default="B")
    p.add_argument("--no-pid", action="store_true")
    add_config_argument(p)
    return p


class ConfigParsingTests(unittest.TestCase):
    def parse(self, yaml_text, argv=()):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "c.yaml"; f.write_text(yaml_text)
            return parse_args_with_config(make_parser(), ["--config", str(f), *argv])

    def test_yaml_values_applied(self):
        a = self.parse("epochs: 7\nlatent_dim: 16\nhidden-dims: [8, 16]\nloss-type: A\nno-pid: true\n")
        self.assertEqual((a.epochs, a.latent_dim, a.hidden_dims, a.loss_type, a.no_pid), (7, 16, "8,16", "A", True))

    def test_command_line_overrides_yaml(self):
        a = self.parse("epochs: 7\nloss-type: A\n", ["--epochs", "3"])
        self.assertEqual((a.epochs, a.loss_type), (3, "A"))

    def test_unknown_key_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse("epochz: 7\n")

    def test_bad_choice_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse("loss-type: Z\n")

    def test_no_config_unchanged(self):
        a = parse_args_with_config(make_parser(), [])
        self.assertEqual((a.epochs, a.loss_type, a.no_pid), (200, "B", False))


class ShippedConfigTests(unittest.TestCase):
    def test_default_yaml_equals_code_defaults(self):
        """configs/betavae_default.yaml must not drift from the defaults of train_betavae.py."""
        import importlib.util, runpy  # noqa: F401
        out = subprocess.run([sys.executable, str(CODE / "train_betavae.py"), "--help"],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("--config", out.stdout)
        # parse both ways inside a child process and compare the namespaces
        code = (
            "import sys; sys.path.insert(0, %r); sys.argv=['x']\n"
            "import argparse, runpy\n"
            "captured = {}\n"
            "orig = argparse.ArgumentParser.parse_args\n"
            "def spy(self, args=None, namespace=None):\n"
            "    ns = orig(self, args, namespace); captured['ns'] = vars(ns).copy(); raise SystemExit(0)\n"
            "argparse.ArgumentParser.parse_args = spy\n"
            "import json\n"
            "def run(argv):\n"
            "    sys.argv = ['train_betavae.py'] + argv\n"
            "    try: runpy.run_path(%r, run_name='__main__')\n"
            "    except SystemExit: pass\n"
            "    return captured['ns']\n"
            "a = run(['--cache-dir', 'x'])\n"
            "b = run(['--cache-dir', 'x', '--config', %r])\n"
            "a.pop('config'); b.pop('config')\n"
            "print(json.dumps({k: [a[k], b[k]] for k in a if a[k] != b[k]}))\n"
        ) % (str(CODE), str(CODE / "train_betavae.py"), str(CFG / "betavae_default.yaml"))
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stdout.strip().splitlines()[-1], "{}", "yaml differs from code defaults")


def write_dat(path, n, rng):
    lines = []
    for i in range(n):
        lines.append(f"<event_begin>:{i}")
        for _ in range(max(5, rng.poisson(15))):
            px, py, pz = rng.normal(0, 1, 3)
            lines.append(f"{rng.choice([22, 211, -211, 2112])} {px:.4f} {py:.4f} {pz:.4f} {np.sqrt(px*px+py*py+pz*pz):.4f}")
        lines.append(f"<event_end>:{i}")
    Path(path).write_text("\n".join(lines) + "\n")


class EndToEndYamlTests(unittest.TestCase):
    def test_preprocess_and_train_from_yaml(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            write_dat(d / "u.dat", 60, rng); write_dat(d / "g.dat", 60, rng)
            cache, out = d / "cache", d / "out" / "m.pth"
            (d / "small.yaml").write_text(
                "epochs: 1\nbatch-size: 16\nlatent-dim: 8\nhidden-dims: [8, 16]\nloss-type: H\nbeta: 1.0\nn-generate: 4\nimg-size: 8\n")
            (d / "pre.yaml").write_text("image-size: 8\neta-max: 0.5\nphi-max: 0.5\n")
            env = dict(os.environ, OMP_NUM_THREADS="2")
            r = subprocess.run([sys.executable, str(CODE / "preprocess_jet_images.py"), "--config", str(d / "pre.yaml"),
                                "--g-jet", str(d / "g.dat"), "--u-jet", str(d / "u.dat"), "--cache-dir", str(cache)],
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-800:])
            r = subprocess.run([sys.executable, str(CODE / "train_betavae.py"), "--config", str(d / "small.yaml"),
                                "--cache-dir", str(cache), "--output", str(out)],
                               capture_output=True, text=True, env=env)
            self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-800:])
            self.assertTrue(out.exists())


if __name__ == "__main__":
    unittest.main()
