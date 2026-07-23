import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from psd_ml.vae import PulseVAE, RealVAEConfig, _encode, vae_loss


class PulseVAETest(unittest.TestCase):
    def test_forward_shapes_and_finite_loss(self):
        torch.manual_seed(1)
        model = PulseVAE()
        batch = torch.rand(16, 144)
        reconstruction, mu, logvar = model(batch)
        self.assertEqual(reconstruction.shape, (16, 144))
        self.assertEqual(mu.shape, (16, 3))
        self.assertEqual(logvar.shape, (16, 3))
        total, reconstruction_loss, kl = vae_loss(
            reconstruction, batch, mu, logvar, beta=0.01
        )
        self.assertTrue(torch.isfinite(total))
        self.assertGreater(float(reconstruction_loss.detach()), 0)
        self.assertGreaterEqual(float(kl.detach()), 0)

    def test_deterministic_mu_encoding(self):
        torch.manual_seed(2)
        model = PulseVAE()
        array = np.random.default_rng(3).normal(size=(20, 144)).astype(np.float32)
        first = _encode(model, array, batch_size=7)
        second = _encode(model, array, batch_size=9)
        np.testing.assert_allclose(first["mu"], second["mu"], atol=3e-8, rtol=1e-6)
        np.testing.assert_allclose(
            first["reconstruction"], second["reconstruction"], atol=3e-8, rtol=1e-6
        )

    def test_checkpoint_state_dict_roundtrip(self):
        torch.manual_seed(4)
        model = PulseVAE()
        batch = torch.rand(5, 144)
        expected = model.decode(model.encode(batch)[0])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save({"state_dict": model.state_dict()}, path)
            restored = PulseVAE()
            restored.load_state_dict(torch.load(path)["state_dict"])
            actual = restored.decode(restored.encode(batch)[0])
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)

    def test_config_requires_three_latents_and_seeds(self):
        with self.assertRaises(ValueError):
            RealVAEConfig(latent_dim=2)
        with self.assertRaises(ValueError):
            RealVAEConfig(model_seeds=(1,))


if __name__ == "__main__":
    unittest.main()
