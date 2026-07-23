import unittest

import numpy as np

from psd_ml.vae_interpretation import (
    _crossing_time,
    _qlong_adjusted_spearman,
    _rankdata,
    _spearman,
)


class VAEInterpretationTest(unittest.TestCase):
    def test_rankdata_uses_average_tie_ranks(self) -> None:
        ranks = _rankdata(np.array([3.0, 1.0, 1.0, 2.0]))
        np.testing.assert_allclose(ranks, [4.0, 1.5, 1.5, 3.0])

    def test_qlong_adjustment_removes_monotonic_energy_trend(self) -> None:
        qlong = np.linspace(1.0, 100.0, 500)
        direction = np.log(qlong)
        metric = np.sqrt(qlong)
        self.assertGreater(_spearman(direction, metric), 0.99)
        self.assertTrue(
            np.isnan(_qlong_adjusted_spearman(direction, metric, qlong))
        )

    def test_waveform_crossing_times_are_subsample(self) -> None:
        pulse = np.array([
            [0.0, 0.2, 0.6, 1.0, 0.8, 0.4, 0.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ])
        rise = _crossing_time(pulse, 0.5, rising=True)
        decay = _crossing_time(pulse, 0.5, rising=False)
        np.testing.assert_allclose(rise[:1], [1.75])
        np.testing.assert_allclose(decay[:1], [4.75])
        self.assertTrue(np.isnan(rise[1]))
        self.assertTrue(np.isnan(decay[1]))


if __name__ == "__main__":
    unittest.main()
