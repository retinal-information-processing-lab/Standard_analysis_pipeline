"""
Unit tests for the receptive-field ellipse fit (utils.sta).

Regression for the gaussian2D grid bug: the model grid was built with
np.linspace(0, N, N), whose step is N/(N-1), so every fitted centre and sigma came out
scaled by N/(N-1) (~0.8 pixel off, i.e. ~150 µm with 12x12 checks). The fit must return
coordinates in pixel indices, the same convention as imshow and np.argmax.

Run just these tests:
    python -m unittest tests.test_sta_fit
"""

import unittest

import numpy as np

import utils


class TestEllipseFitCentre(unittest.TestCase):
    """A Gaussian placed at a known pixel position must be recovered at that position."""

    def _fit_blob(self, n_pixels, true_row, true_col, sigma):
        rows, cols = np.indices((n_pixels, n_pixels))
        blob = np.exp(
            -((rows - true_row) ** 2 + (cols - true_col) ** 2) / (2 * sigma**2)
        )
        (amp, x0, y0, sigma_x, sigma_y, angle), _ = utils.double_gaussian_fit(blob)
        return x0, y0, sigma_x, sigma_y

    def test_centre_recovered_on_coarse_and_fine_grids(self):
        # 12 x 12 = big checks (where the bug was visible), 80 x 80 = usual STA size.
        for n_pixels in (12, 80):
            true_row, true_col, sigma = (
                0.40 * n_pixels,
                0.65 * n_pixels,
                0.08 * n_pixels,
            )
            x0, y0, _, _ = self._fit_blob(n_pixels, true_row, true_col, sigma)
            with self.subTest(n_pixels=n_pixels):
                self.assertAlmostEqual(x0, true_col, delta=0.1)  # x = column
                self.assertAlmostEqual(y0, true_row, delta=0.1)  # y = row

    def test_sigma_recovered(self):
        sigma = 1.5
        _, _, sigma_x, sigma_y = self._fit_blob(12, 4.8, 7.8, sigma)
        self.assertAlmostEqual(sigma_x, sigma, delta=0.05)
        self.assertAlmostEqual(sigma_y, sigma, delta=0.05)

    def test_model_peaks_at_its_own_centre(self):
        # gaussian2D evaluated at an integer centre must peak exactly on that pixel.
        model = utils.gaussian2D((12, 12), 1.0, 7, 4, 1.0, 1.0, 0)
        self.assertEqual(np.unravel_index(np.argmax(model), model.shape), (4, 7))
        self.assertAlmostEqual(model[4, 7], 1.0)


if __name__ == "__main__":
    unittest.main()
