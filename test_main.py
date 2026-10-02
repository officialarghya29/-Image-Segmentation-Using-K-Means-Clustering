"""Small regression suite for the image-segmentation core."""
import io
import unittest

import numpy as np
from PIL import Image

import main


class KMeansTests(unittest.TestCase):
    def test_clusters_separable_colours(self):
        pixels = np.array([[0.02, 0.02, 0.02], [0.08, 0.08, 0.08],
                           [0.92, 0.92, 0.92], [0.98, 0.98, 0.98]])
        labels, centers, inertia = main.kmeans(pixels, k=2, seed=7)

        self.assertEqual(set(labels), {0, 1})
        self.assertEqual(centers.shape, (2, 3))
        self.assertGreaterEqual(inertia, 0)
        self.assertLess(inertia, 0.02)

    def test_seed_makes_results_repeatable(self):
        pixels = np.random.default_rng(4).random((40, 3))
        first = main.kmeans(pixels, k=4, seed=12)
        second = main.kmeans(pixels, k=4, seed=12)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_allclose(first[1], second[1])
        self.assertEqual(first[2], second[2])

    def test_rejects_invalid_cluster_count_and_pixels(self):
        with self.assertRaisesRegex(ValueError, "k must be"):
            main.kmeans(np.zeros((2, 3)), k=3)
        with self.assertRaisesRegex(ValueError, "finite"):
            main.kmeans(np.array([[np.nan, 0, 0]]), k=1)

    def test_segment_image_returns_uint8_rgb_and_error_stats(self):
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        image[:, 2:] = 255
        result, centers, inertia, pixel_count = main.segment_image(image, k=2)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(centers.shape, (2, 3))
        self.assertEqual(pixel_count, 20)
        self.assertLess(inertia, 0.01)
        self.assertGreaterEqual(main.error_stats(inertia, pixel_count)["rmse_255"], 0)

    def test_load_resizes_and_rejects_non_images(self):
        buffer = io.BytesIO()
        Image.new("RGB", (40, 20), "navy").save(buffer, format="PNG")
        image = main.load_image_from_bytes(buffer.getvalue(), max_size=10)
        self.assertEqual(image.shape, (5, 10, 3))
        with self.assertRaises(ValueError):
            main.load_image_from_bytes(b"not an image")

    def test_elbow_returns_ordered_k_and_png_is_decodable(self):
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[:, 4:] = 255
        ks, errors = main.compute_elbow(image, 2, 4, sample_pixels=64)
        self.assertEqual(ks, [2, 3, 4])
        self.assertEqual(len(errors), len(ks))
        self.assertTrue(all(np.isfinite(errors)))

        encoded = main.image_to_base64_png(image)
        self.assertTrue(encoded)
        self.assertEqual(Image.open(io.BytesIO(__import__("base64").b64decode(encoded))).size, (8, 8))


if __name__ == "__main__":
    unittest.main()
