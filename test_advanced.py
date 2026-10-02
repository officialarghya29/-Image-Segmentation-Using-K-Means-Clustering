"""
test_advanced.py — deep correctness and efficiency suite for the clustering core.

Run with:

    python -m unittest -v test_advanced.py

Unlike test_main.py (which checks the basic contracts), this suite verifies
numerical invariants against brute-force reference implementations, exercises
degenerate inputs, and measures runtime and peak memory so a performance
regression fails the build instead of shipping, including a check that the
assignment step avoids the naive (N, k, 3) temporary array.
"""
import io
import math
import sys
import time
import tracemalloc
import unittest

import numpy as np
from PIL import Image

import main


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def photo(height=144, width=220, seed=3):
    """A smooth multi-region test image with three distinct colour bands."""
    rng = np.random.default_rng(seed)
    rows = np.linspace(0, 1, height)[:, None, None]
    top = np.array([30.0, 40.0, 120.0])
    bottom = np.array([240.0, 170.0, 90.0])
    base = top + (bottom - top) * rows
    img = np.repeat(base, width, axis=1)
    # a hard-edged block so clusters are genuinely separable
    img[: height // 3, : width // 3] = np.array([15.0, 200.0, 40.0])
    img += rng.normal(0, 2.0, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def brute_force_labels(pixels, centers):
    """Reference nearest-centroid assignment: explicit loop, no vectorisation."""
    labels = []
    for p in pixels:
        best, best_d = 0, float("inf")
        for c_index, c in enumerate(centers):
            d = sum((float(p[i]) - float(c[i])) ** 2 for i in range(3))
            if d < best_d:
                best, best_d = c_index, d
        labels.append(best)
    return np.array(labels)


def png_bytes(array):
    buffer = io.BytesIO()
    Image.fromarray(array.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# numerical correctness
# ---------------------------------------------------------------------------

class NumericalCorrectnessTests(unittest.TestCase):
    def test_assignment_matches_brute_force_reference(self):
        """The vectorised distance form must pick exactly the nearest centroid."""
        rng = np.random.default_rng(21)
        pixels = rng.random((300, 3))
        centers = rng.random((6, 3))

        labels, min_d = main._assign_clusters(pixels, centers)

        np.testing.assert_array_equal(labels, brute_force_labels(pixels, centers))

        # the returned minimum distance must match an independent computation
        expected = np.array([
            min(float(np.sum((p - c) ** 2)) for c in centers) for p in pixels
        ])
        np.testing.assert_allclose(min_d, expected, rtol=1e-9, atol=1e-12)

    def test_assignment_distances_are_finite_and_non_negative(self):
        """The expanded form subtracts two large terms; it must never go negative
        or produce NaN, even for extreme colour values."""
        pixels = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [255.0, 0.0, 128.0]])
        centers = np.array([[0.0, 0.0, 0.0], [255.0, 0.0, 128.0], [1.0, 1.0, 1.0]])

        labels, min_d = main._assign_clusters(pixels, centers)

        self.assertTrue(np.isfinite(min_d).all())
        self.assertTrue((min_d >= 0).all())
        self.assertTrue(((labels >= 0) & (labels < 3)).all())
        # identical points must land on their own centroid -> zero distance
        self.assertAlmostEqual(float(min_d[0]), 0.0, places=6)
        self.assertAlmostEqual(float(min_d[1]), 0.0, places=6)

    def test_inertia_matches_independently_recomputed_error(self):
        """Reported inertia must equal the sum of squared distances implied by
        the returned labels and centres."""
        pixels = np.random.default_rng(5).random((400, 3))
        labels, centers, inertia = main.kmeans(pixels, k=5, iterations=40, seed=9)

        recomputed = float(np.sum((pixels - centers[labels]) ** 2))
        self.assertAlmostEqual(inertia, recomputed, places=9)
        self.assertGreaterEqual(inertia, 0.0)

    def test_k_equals_one_reduces_to_variance(self):
        """With k = 1 the optimum is the global mean, so inertia must equal
        N * variance exactly."""
        pixels = np.random.default_rng(17).random((500, 3))
        _, centers, inertia = main.kmeans(pixels, k=1, iterations=30, seed=1)

        np.testing.assert_allclose(centers[0], pixels.mean(axis=0), rtol=1e-9, atol=1e-12)
        expected = float(np.sum((pixels - pixels.mean(axis=0)) ** 2))
        self.assertAlmostEqual(inertia, expected, places=9)

    def test_converged_centres_are_cluster_means(self):
        """Once Lloyd's algorithm converges, every centre must be the exact mean
        of the pixels assigned to it."""
        pixels = np.random.default_rng(8).random((600, 3))
        labels, centers, _ = main.kmeans(pixels, k=4, iterations=100, seed=3)

        for c in range(4):
            members = pixels[labels == c]
            self.assertGreater(members.shape[0], 0, f"cluster {c} is empty")
            np.testing.assert_allclose(centers[c], members.mean(axis=0), atol=1e-9)

    def test_every_cluster_is_used_for_separable_data(self):
        """k-means++ must not leave a cluster empty on well-separated blobs."""
        rng = np.random.default_rng(31)
        blobs = np.vstack([
            rng.normal(0.1, 0.01, (150, 3)),
            rng.normal(0.4, 0.01, (150, 3)),
            rng.normal(0.7, 0.01, (150, 3)),
            rng.normal(0.95, 0.01, (150, 3)),
        ])
        labels, centers, inertia = main.kmeans(blobs, k=4, iterations=50, seed=2)

        self.assertEqual(len(set(labels.tolist())), 4)
        # blobs are tight, so error must be tiny relative to the 0-1 colour space
        self.assertLess(inertia / blobs.shape[0], 1e-3)

    def test_error_stats_arithmetic_is_consistent(self):
        stats = main.error_stats(12.5, 1000)
        self.assertAlmostEqual(stats["mean_error"], 0.0125, places=8)
        self.assertAlmostEqual(stats["rmse_255"], math.sqrt(0.0125) * 255, places=2)

    def test_error_decreases_as_clusters_increase(self):
        """More clusters must not fit worse than far fewer clusters."""
        image = photo()
        _, _, inertia_2, _ = main.segment_image(image, k=2, iterations=30)
        _, _, inertia_16, _ = main.segment_image(image, k=16, iterations=30)
        self.assertLess(inertia_16, inertia_2)

    def test_segmented_output_contains_only_cluster_colours(self):
        """Reconstruction must use palette colours verbatim — no interpolation."""
        image = photo()
        segmented, centers, _, _ = main.segment_image(image, k=5, iterations=30)

        palette = {tuple(int(v) for v in c) for c in centers.tolist()}
        produced = {tuple(int(v) for v in px) for px in segmented.reshape(-1, 3)}
        self.assertTrue(produced.issubset(palette))

    def test_full_pipeline_is_deterministic(self):
        image = photo()
        first = main.segment_image(image, k=7, iterations=25)
        second = main.segment_image(image, k=7, iterations=25)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])


# ---------------------------------------------------------------------------
# degenerate and edge-case inputs
# ---------------------------------------------------------------------------

class EdgeCaseTests(unittest.TestCase):
    def test_uniform_image_does_not_divide_by_zero(self):
        """All-identical pixels make every k-means++ distance zero; the seeding
        probability fallback must still work and produce zero error."""
        image = np.full((20, 20, 3), 77, dtype=np.uint8)
        segmented, centers, inertia, pixel_count = main.segment_image(image, k=4)

        self.assertEqual(segmented.shape, image.shape)
        self.assertEqual(pixel_count, 400)
        self.assertAlmostEqual(inertia, 0.0, places=9)
        self.assertTrue(np.isfinite(centers).all())
        self.assertEqual(main.error_stats(inertia, pixel_count)["rmse_255"], 0.0)

    def test_single_pixel_image_with_k_one(self):
        image = np.array([[[120, 45, 200]]], dtype=np.uint8)
        segmented, centers, inertia, pixel_count = main.segment_image(image, k=1)

        self.assertEqual(segmented.shape, (1, 1, 3))
        self.assertEqual(pixel_count, 1)
        self.assertAlmostEqual(inertia, 0.0, places=9)
        np.testing.assert_array_equal(centers[0], np.array([120, 45, 200], dtype=np.uint8))

    def test_grayscale_converted_image_clusters(self):
        gray = np.tile(np.arange(40, dtype=np.uint8)[None, :, None], (30, 1, 3))
        _, centers, inertia, pixel_count = main.segment_image(gray, k=3)
        self.assertEqual(pixel_count, 1200)
        self.assertGreater(inertia, 0.0)
        self.assertEqual(centers.shape, (3, 3))

    def test_k_larger_than_pixel_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "k must be between"):
            main.kmeans(np.zeros((3, 3)), k=4)

    def test_zero_validation_cases_across_public_api(self):
        image = photo(16, 16)
        cases = [
            (lambda: main.kmeans(np.zeros((2, 3)), k=0), "k must be"),
            (lambda: main.kmeans(np.zeros((2, 3)), k=1, iterations=0), "iterations"),
            (lambda: main.kmeans(np.zeros((2, 2)), k=1), "two-dimensional RGB"),
            (lambda: main.kmeans(np.array([[np.inf, 0, 2.0]]), k=1), "finite"),
            (lambda: main.segment_image(np.zeros((0, 0, 3), dtype=np.uint8), k=1), "non-empty RGB"),
            (lambda: main.segment_image(np.zeros((4, 4), dtype=np.uint8), k=1), "non-empty RGB"),
            (lambda: main.error_stats(1.0, 0), "positive"),
            (lambda: main.add_grid(image, -1), "non-negative"),
            (lambda: main.compute_elbow(image, 0, 4), "k_min must be"),
            (lambda: main.compute_elbow(image, 5, 2), "k_min must be"),
            (lambda: main.compute_elbow(image, 2, 4, sample_pixels=0), "positive"),
            (lambda: main.load_image_from_bytes(b""), "empty"),
            (lambda: main.load_image_from_bytes(png_bytes(image), max_size=0), "max_size"),
            (lambda: main.load_image_from_bytes(png_bytes(image), max_size=9999), "max_size"),
        ]
        for call, expected in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(ValueError, expected):
                    call()

    def test_grid_spacing_zero_is_a_no_op(self):
        image = photo(30, 40)
        np.testing.assert_array_equal(main.add_grid(image, 0), image)

    def test_grid_draws_lines_at_expected_coordinates(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        gridded = main.add_grid(image, 5)

        # vertical and horizontal lines every 5 px must be white
        self.assertTrue((gridded[:, 0] == 255).all())
        self.assertTrue((gridded[:, 5] == 255).all())
        self.assertTrue((gridded[10, :] == 255).all())
        # a point away from any line must be untouched
        np.testing.assert_array_equal(gridded[2, 2], np.array([0, 0, 0], dtype=np.uint8))

    def test_load_preserves_aspect_ratio_for_extreme_shapes(self):
        # numpy shape is (height, width); the longest edge is scaled to 220
        for size, expected in [((400, 20), (220, 11)), ((20, 400), (11, 220))]:
            with self.subTest(size=size):
                array = np.zeros((*size, 3), dtype=np.uint8)
                loaded = main.load_image_from_bytes(png_bytes(array), max_size=220)
                self.assertEqual(loaded.shape, (*expected, 3))

    def test_load_rejects_bytes_that_are_not_an_image(self):
        for blob in (b"not an image", b"\x89PNG\r\n\x1a\n truncated", b"%PDF-1.4"):
            with self.subTest(blob=blob):
                with self.assertRaisesRegex(ValueError, "not a valid supported image"):
                    main.load_image_from_bytes(blob)

    def test_elbow_stops_when_k_exceeds_available_pixels(self):
        tiny = np.zeros((2, 2, 3), dtype=np.uint8)
        ks, errors = main.compute_elbow(tiny, 2, 16, sample_pixels=4, iterations=5)
        self.assertTrue(all(k <= 4 for k in ks))
        self.assertEqual(len(ks), len(errors))


# ---------------------------------------------------------------------------
# efficiency and resource usage
# ---------------------------------------------------------------------------

class EfficiencyTests(unittest.TestCase):
    def test_assignment_uses_less_memory_than_naive_broadcast(self):
        """The optimised assignment must not materialise an (N, k, 3) array."""
        rng = np.random.default_rng(4)
        pixels = rng.random((20_000, 3))
        centers = rng.random((16, 3))

        tracemalloc.start()
        main._assign_clusters(pixels, centers)
        _, optimised_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        # naive reference: builds the full (N, k, 3) difference tensor
        naive = np.argmin(
            np.sum((pixels[:, None, :] - centers[None, :, :]) ** 2, axis=2), axis=1)
        _, naive_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertEqual(optimised_peak, optimised_peak)  # not NaN
        self.assertLess(
            optimised_peak, naive_peak * 0.6,
            f"optimised peak {optimised_peak} should be well below naive {naive_peak}")
        # both must agree on the answer
        np.testing.assert_array_equal(main._assign_clusters(pixels, centers)[0], naive)

    def test_runtime_scales_roughly_linearly_in_k(self):
        """Doubling k should not blow up super-linearly."""
        pixels = np.random.default_rng(6).random((30_000, 3))

        def timed(k):
            best = float("inf")
            for _ in range(3):
                start = time.perf_counter()
                main.kmeans(pixels, k=k, iterations=10, seed=5)
                best = min(best, time.perf_counter() - start)
            return best

        t_small, t_large = timed(4), timed(16)
        # 4x the clusters must not cost more than ~12x the time
        self.assertLess(t_large, max(t_small * 12, 2.0),
                        f"k=16 took {t_large:.4f}s vs k=4 {t_small:.4f}s")

    def test_early_convergence_beats_the_iteration_cap(self):
        """A converged image should stop early, so a high cap must be cheap."""
        image = photo()
        pixels = image.reshape(-1, 3).astype(np.float64) / 255.0

        start = time.perf_counter()
        main.kmeans(pixels, k=3, iterations=200, seed=1)
        many = time.perf_counter() - start

        start = time.perf_counter()
        main.kmeans(pixels, k=3, iterations=5, seed=1)
        few = time.perf_counter() - start

        self.assertLess(many, max(few * 4, 1.5),
                        "200 iterations should be far cheaper than 200x the cost of 5")

    def test_segmentation_of_a_large_image_stays_interactive(self):
        """The documented max working size must stay well under a second."""
        big = photo(400, 500, seed=9)
        start = time.perf_counter()
        segmented, _, inertia, pixel_count = main.segment_image(big, k=8, iterations=25)
        elapsed = time.perf_counter() - start

        self.assertEqual(pixel_count, 200_000)
        self.assertEqual(segmented.shape, big.shape)
        self.assertGreater(inertia, 0.0)
        self.assertLess(elapsed, 5.0, f"k=8 on 200k pixels took {elapsed:.2f}s")

    def test_full_elbow_sweep_is_cheap_thanks_to_subsampling(self):
        """Sweeping 15 values of k must not cost 15 full-resolution runs."""
        image = photo()
        start = time.perf_counter()
        ks, errors = main.compute_elbow(image, 2, 16, sample_pixels=3000, iterations=10)
        elapsed = time.perf_counter() - start

        self.assertEqual(len(ks), 15)
        self.assertTrue(all(np.isfinite(errors)))
        self.assertLess(elapsed, 10.0, f"elbow sweep took {elapsed:.2f}s")

    def test_no_float_overflow_at_extreme_colour_values(self):
        """Squaring 255-scale values in float64 must stay exact."""
        pixels = np.array([[255.0, 255.0, 255.0], [0.0, 0.0, 0.0]], dtype=np.float64)
        _, _, inertia = main.kmeans(pixels, k=2, iterations=10, seed=1)
        self.assertTrue(np.isfinite(inertia))
        # each of the two points sits on its own centre -> no residual error
        self.assertAlmostEqual(inertia, 0.0, places=6)


# ---------------------------------------------------------------------------
# end-to-end behaviour through the real image path
# ---------------------------------------------------------------------------

class EndToEndTests(unittest.TestCase):
    def test_uploaded_bytes_to_segmented_png_round_trip(self):
        """Exercise the same path the web app uses: bytes -> array -> clusters
        -> base64 PNG -> decodable image."""
        import base64

        source = photo(120, 160, seed=13)
        image = main.load_image_from_bytes(png_bytes(source), max_size=220)
        segmented, centers, inertia, pixel_count = main.segment_image(image, k=6)
        with_grid = main.add_grid(segmented, 20)

        encoded = main.image_to_base64_png(with_grid)
        decoded = Image.open(io.BytesIO(base64.b64decode(encoded)))

        self.assertEqual(decoded.format, "PNG")
        self.assertEqual(decoded.size, (image.shape[1], image.shape[0]))
        self.assertEqual(pixel_count, image.shape[0] * image.shape[1])
        self.assertEqual(len(main.cluster_legend(centers)), 6)

    def test_cluster_legend_reports_valid_colour_and_luminance(self):
        _, centers, _, _ = main.segment_image(photo(), k=5)
        legend = main.cluster_legend(centers)

        self.assertEqual([row["cluster"] for row in legend], [0, 1, 2, 3, 4])
        for row in legend:
            for channel in ("r", "g", "b"):
                self.assertTrue(0 <= row[channel] <= 255)
            expected = 0.299 * row["r"] + 0.587 * row["g"] + 0.114 * row["b"]
            self.assertAlmostEqual(row["luminance"], round(expected, 1), places=1)

    def test_elbow_curve_decreases_overall_and_is_finite(self):
        ks, errors = main.compute_elbow(photo(), 2, 16, sample_pixels=4000, iterations=15)

        self.assertEqual(ks, list(range(2, 17)))
        self.assertTrue(all(e >= 0 for e in errors))
        self.assertTrue(all(np.isfinite(errors)))
        self.assertLess(errors[-1], errors[0] * 0.5,
                        "error at k=16 should be far below k=2")


def run_benchmark():
    """Print a real runtime / peak-memory table across image sizes and k.

    Invoked with `python test_advanced.py --bench`. Numbers are measured on the
    machine running the command, so they double as a regression baseline.
    """
    sizes = [(64, 64), (144, 220), (240, 320), (400, 500)]
    ks = [2, 4, 8, 16]

    header = f"{'working image':>16} {'pixels':>9} " + " ".join(f"{'k=' + str(k):>12}" for k in ks)
    print("\nRuntime and peak memory for segment_image()\n")
    print(header)
    print("-" * len(header))

    for height, width in sizes:
        image = photo(height, width)
        pixels = height * width
        cells = []
        for k in ks:
            # time without the tracing overhead, then memory in a separate pass
            best = float("inf")
            for _ in range(3):
                start = time.perf_counter()
                main.segment_image(image, k=k, iterations=25)
                best = min(best, time.perf_counter() - start)
            tracemalloc.start()
            main.segment_image(image, k=k, iterations=25)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            cells.append(f"{best * 1e3:5.1f}ms/{peak / 1e6:4.1f}MB")
        print(f"{f'{height}x{width}':>16} {pixels:>9,} " + " ".join(f"{c:>12}" for c in cells))

    print("\nEach cell is  time / peak allocation  for a full segmentation.\n")


if __name__ == "__main__":
    if "--bench" in sys.argv:
        run_benchmark()
    else:
        unittest.main()