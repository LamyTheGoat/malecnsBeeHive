"""Tests for the BeeHive fly brain.

Run from the ``beehive`` directory:  python3 -m unittest discover -s tests
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flybrain import Fly, MBParams, MushroomBody, SteeringParams  # noqa: E402
from flybrain.connectome import default_source, load_export, resolve  # noqa: E402
from flybrain.eye import EyeParams, FlyEye, build_lattice  # noqa: E402
from flybrain.synthetic import make_profiles  # noqa: E402


class TestLattice(unittest.TestCase):
    def test_size_and_bounds(self):
        lat = build_lattice(28, 27)
        self.assertEqual(len(lat.axial), 756)
        self.assertTrue(np.all(lat.xy >= 0.0) and np.all(lat.xy <= 1.0))

    def test_interior_cells_have_six_neighbours(self):
        lat = build_lattice(12, 12)
        counts = (lat.neighbours >= 0).sum(axis=1)
        # Every cell has at most six; the middle of the lattice has exactly six.
        self.assertTrue(np.all(counts <= 6))
        self.assertEqual(counts.max(), 6)

    def test_neighbourhood_is_reciprocal(self):
        lat = build_lattice(9, 9)
        for i, row in enumerate(lat.neighbours):
            for j in row[row >= 0]:
                self.assertIn(i, lat.neighbours[j], f"{i} -> {j} tek yonlu")


class TestEye(unittest.TestCase):
    def setUp(self):
        self.eye = FlyEye()
        self.rng = np.random.default_rng(0)

    def test_vpn_shape(self):
        img = self.rng.random((self.eye.params.grid,) * 2 + (3,)).astype(np.float32)
        self.assertEqual(self.eye.vpn(img).shape, (self.eye.n_vpn,))

    def test_wrong_size_rejected(self):
        with self.assertRaises(ValueError):
            self.eye.vpn(np.zeros((8, 8, 3), dtype=np.float32))

    def test_contrast_channels_are_exposure_invariant(self):
        """Weber contrast must barely care how brightly the photo was exposed.

        Not perfectly: the fixed photoreceptor dark current makes a dim scene
        slightly lower in apparent contrast, which is also true of the real
        animal.  The tolerance allows for that but not for anything larger.
        """
        img = self.rng.random((self.eye.params.grid,) * 2 + (3,)).astype(np.float32)
        dim = self.eye.look(img * 0.35)
        bright = self.eye.look(img)
        for channel in ("on", "off", "axis_h", "opponent"):
            np.testing.assert_allclose(
                dim[channel], bright[channel], atol=0.05,
                err_msg=f"{channel} pozlamaya duyarli",
            )

    def test_luminance_channel_tracks_brightness(self):
        img = np.full((self.eye.params.grid,) * 2 + (3,), 0.2, dtype=np.float32)
        self.assertLess(
            self.eye.look(img)["lum"].mean(), self.eye.look(img * 3)["lum"].mean()
        )

    def test_ascii_view_has_one_line_per_row(self):
        img = self.rng.random((self.eye.params.grid,) * 2 + (3,)).astype(np.float32)
        self.assertEqual(
            len(self.eye.ascii_view(img).splitlines()), self.eye.params.rows
        )


class TestMushroomBody(unittest.TestCase):
    def setUp(self):
        self.mb = MushroomBody(MBParams(n_input=32, n_kc=500, sparsity=0.06))
        self.rng = np.random.default_rng(1)
        self.stim = self.rng.random(32).astype(np.float32)

    def test_apl_enforces_sparse_code(self):
        kc = self.mb.kenyon_cells(self.stim)
        active = (kc > 0).sum()
        self.assertEqual(active, int(round(0.06 * 500)))
        self.assertAlmostEqual(float(kc.sum()), 1.0, places=5)

    def test_naive_fly_is_indifferent(self):
        self.assertAlmostEqual(self.mb.valence(self.mb.kenyon_cells(self.stim)), 0.0, places=5)

    def test_reward_raises_valence_punishment_lowers_it(self):
        kc = self.mb.kenyon_cells(self.stim)
        self.mb.teach(kc, (0.0, 1.0))  # reward: depress the avoidance MBON
        self.assertGreater(self.mb.valence(self.mb.kenyon_cells(self.stim)), 0.0)

        other = MushroomBody(MBParams(n_input=32, n_kc=500, sparsity=0.06))
        kc = other.kenyon_cells(self.stim)
        other.teach(kc, (1.0, 0.0))  # punishment: depress the approach MBON
        self.assertLess(other.valence(other.kenyon_cells(self.stim)), 0.0)

    def test_learning_is_depression_only(self):
        """Weights may fall towards w_min but never exceed the naive value."""
        for _ in range(50):
            stim = self.rng.random(32).astype(np.float32)
            self.mb.teach(self.mb.kenyon_cells(stim), (1.0, 0.0))
        self.assertLessEqual(self.mb.w_out.max(), self.mb.params.w0 + 1e-6)
        self.assertGreaterEqual(self.mb.w_out.min(), self.mb.params.w_min - 1e-6)

    def test_memory_is_stimulus_specific(self):
        a = self.rng.random(32).astype(np.float32)
        b = self.rng.random(32).astype(np.float32)
        for _ in range(5):
            self.mb.teach(self.mb.kenyon_cells(a), (1.0, 0.0))
        va = self.mb.valence(self.mb.kenyon_cells(a))
        vb = self.mb.valence(self.mb.kenyon_cells(b))
        self.assertLess(va, vb, "ceza tum uyaranlara yayilmis")

    def test_rejects_wrong_input_size(self):
        with self.assertRaises(ValueError):
            self.mb.kenyon_cells(np.zeros(5, dtype=np.float32))


class TestSteering(unittest.TestCase):
    def test_strong_valence_drives_the_matching_turn(self):
        fly = Fly()
        right = sum(fly.steering.decide(+1.0).swipe == "right" for _ in range(200))
        left = sum(fly.steering.decide(-1.0).swipe == "left" for _ in range(200))
        self.assertGreater(right, 170)
        self.assertGreater(left, 170)

    def test_indifference_is_a_coin_flip(self):
        fly = Fly(steering_params=SteeringParams(seed=3))
        right = sum(fly.steering.decide(0.0).swipe == "right" for _ in range(400))
        self.assertTrue(150 < right < 250, f"tarafsiz sinek yanli: {right}/400")

    def test_descending_neurons_are_balanced_when_naive(self):
        left, right = Fly().steering.descending(0.0)
        self.assertAlmostEqual(left, right, places=6)


class TestFly(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        fly = Fly()
        profiles = make_profiles(20, fly.eye.params.grid, seed=4)
        for p in profiles:
            fly.train(p.image, p.swipe)
        before = [fly.perceive(p.image).valence for p in profiles]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fly.npz"
            fly.save(path)
            reloaded = Fly.load(path)
        after = [reloaded.perceive(p.image).valence for p in profiles]
        np.testing.assert_allclose(before, after, atol=1e-6)

    def test_decision_precedes_reinforcement(self):
        """train() must not peek: a naive fly's first call is a coin flip."""
        fly = Fly()
        img = make_profiles(1, fly.eye.params.grid, seed=5)[0].image
        self.assertAlmostEqual(fly.perceive(img).valence, 0.0, places=6)
        fly.train(img, "right")
        self.assertGreater(fly.perceive(img).valence, 0.0)

    def test_rejects_bad_label(self):
        fly = Fly()
        img = make_profiles(1, fly.eye.params.grid, seed=6)[0].image
        with self.assertRaises(ValueError):
            fly.train(img, "up")

    def test_training_beats_a_naive_fly(self):
        """Averaged over seeds, because one run of 150 test items is noisy."""
        naive, trained = [], []
        for seed in (7, 8):
            fly = Fly()
            grid = fly.eye.params.grid
            train = make_profiles(350, grid, seed=seed)
            test = make_profiles(150, grid, seed=900 + seed)
            train_vpn = [fly.eye.vpn(p.image) for p in train]
            test_vpn = [fly.eye.vpn(p.image) for p in test]

            naive.append(np.mean([
                fly.swipe_vpn(v).swipe == p.swipe for v, p in zip(test_vpn, test)
            ]))
            for vpn, profile in zip(train_vpn, train):
                fly.train_vpn(vpn, profile.swipe)
            trained.append(np.mean([
                fly.swipe_vpn(v).swipe == p.swipe for v, p in zip(test_vpn, test)
            ]))

        self.assertLess(np.mean(naive), 0.58, f"naif sinek zaten biliyor: {naive}")
        self.assertGreater(np.mean(trained), 0.65, f"ogrenme zayif: {trained}")


class TestConnectome(unittest.TestCase):
    def test_defaults_are_usable(self):
        source = default_source()
        self.assertEqual(source.n_kc, 2000)
        self.assertIn("Kenyon", source.describe())

    def test_load_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "mb_neurons.csv").write_text(
                "bodyid,type,mbclass,claws\n"
                "1,KCg,KC,5\n2,KCab,KC,7\n3,MBON-g1pedc>a/b,MBON,\n",
                encoding="utf-8",
            )
            (tmp / "mb_kc_mbon.csv").write_text(
                "bodyid_pre,bodyid_post,type_post,weight\n"
                "1,3,MBON-g1pedc>a/b,10\n2,3,MBON-g1pedc>a/b,6\n",
                encoding="utf-8",
            )
            source = load_export(tmp)
        self.assertEqual(source.n_kc, 2)
        self.assertEqual(source.claws, 6)
        self.assertAlmostEqual(source.mbon_scale["MBON-g1pedc>a/b"], 1.0)

    def test_resolve_falls_back_when_export_missing(self):
        source = resolve("/nonexistent/path/for/tests")
        self.assertEqual(source.n_kc, 2000)
        self.assertIn("okunamadi", source.provenance)

    def test_source_overrides_model_size(self):
        source = default_source()
        source.n_kc = 321
        fly = Fly(connectome=source)
        self.assertEqual(fly.mb.params.n_kc, 321)
        self.assertEqual(fly.mb.params.n_input, fly.eye.n_vpn)


class TestSynthetic(unittest.TestCase):
    def test_seed_is_reproducible(self):
        a = make_profiles(5, 72, seed=11)
        b = make_profiles(5, 72, seed=11)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x.image, y.image)
            self.assertEqual(x.swipe, y.swipe)

    def test_images_are_in_range(self):
        for p in make_profiles(10, 72, seed=12):
            self.assertGreaterEqual(p.image.min(), 0.0)
            self.assertLessEqual(p.image.max(), 1.0)

    def test_labels_are_roughly_balanced(self):
        profiles = make_profiles(400, 72, seed=13)
        right = sum(p.swipe == "right" for p in profiles)
        self.assertTrue(150 < right < 250, f"etiketler dengesiz: {right}/400")


class TestEyeParams(unittest.TestCase):
    def test_pool_count_drives_vpn_width(self):
        eye = FlyEye(EyeParams(pools_per_side=3))
        self.assertEqual(eye.n_vpn, 9 * 9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
