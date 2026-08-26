"""Fast regression tests for the runnable MA-PFN tutorial."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from ma_pfn_demo import (
    cache_events_explicitly,
    metric_aware_forward,
    score_cached_pairs_explicitly,
    stock_pfn_forward,
)
from models import MAPFN, PFN
from utils import (
    TUTORIAL_SUBSET_SHA256,
    predict_event_pairs,
    prepare_demo_data,
    reconstruct_split_events,
)
from zenodo.make_tutorial_subset import source_pair_rows


ROOT = Path(__file__).resolve().parent


class TutorialDataTests(unittest.TestCase):
    def test_bundled_archive_checksum_and_provenance(self) -> None:
        archive_path = ROOT / "ma_pfn_tutorial_subset.npz"
        self.assertEqual(
            hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            TUTORIAL_SUBSET_SHA256,
        )
        with np.load(archive_path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"]))
        self.assertEqual(metadata["format"], "ma-pfn-real-tutorial-subset")
        self.assertEqual(
            {
                split: details["selected_event_count"]
                for split, details in metadata["splits"].items()
            },
            {"train": 448, "val": 64, "test": 64},
        )

    def test_source_pair_row_mapping(self) -> None:
        source_events = 8
        selected = np.array([1, 4, 6, 7], dtype=np.int64)
        source_pairs = np.stack(np.triu_indices(source_events, k=1), axis=1)
        selected_pairs = np.stack(np.triu_indices(len(selected), k=1), axis=1)
        expected = selected[selected_pairs]
        np.testing.assert_array_equal(
            source_pairs[source_pair_rows(source_events, selected)], expected
        )

    def test_missing_release_data_extracts_and_reuses_real_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = prepare_demo_data(root / "release", root / "subset")
            second = prepare_demo_data(root / "release", root / "subset")

            self.assertEqual(first["kind"], "real_tutorial_subset")
            self.assertTrue(first["extracted"])
            self.assertFalse(second["extracted"])
            self.assertEqual(first["archive_source"], "bundled")
            self.assertEqual(second["archive_source"], "extracted_cache")
            self.assertEqual(
                reconstruct_split_events(root / "subset").shape, (64, 76, 3)
            )

    def test_partial_release_data_has_an_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release = root / "release"
            release.mkdir()
            np.save(release / "train_targets.npy", np.ones(1, dtype=np.float32))
            with self.assertRaisesRegex(FileNotFoundError, "incomplete; missing"):
                prepare_demo_data(release, root / "subset")


class VisibleImplementationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.data_dir = root / "subset"
        prepare_demo_data(
            root / "release",
            cls.data_dir,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_written_out_forward_passes_match_reusable_models(self) -> None:
        pairs = torch.from_numpy(np.load(self.data_dir / "train_features.npy")[:4])
        torch.manual_seed(5)
        cases = (
            (PFN(4, 5, 7, 9).eval(), stock_pfn_forward),
            (MAPFN(4, 5, 7, 9).eval(), metric_aware_forward),
        )
        with torch.inference_mode():
            for model, written_out in cases:
                torch.testing.assert_close(model(pairs), written_out(model, pairs))

    def test_explicit_cache_matches_ordinary_pair_encoding(self) -> None:
        events = reconstruct_split_events(self.data_dir)
        event_tensor = torch.from_numpy(events)
        first_numpy = np.array([0, 0, 2, 4], dtype=np.int64)
        second_numpy = np.array([1, 3, 5, 6], dtype=np.int64)
        first = torch.from_numpy(first_numpy)
        second = torch.from_numpy(second_numpy)

        for model, expected_passes in ((MAPFN(4, 5, 7, 9), 1), (PFN(4, 5, 7, 9), 2)):
            model.eval()
            cache = cache_events_explicitly(model, event_tensor)
            cached = score_cached_pairs_explicitly(
                model, cache, first, second
            ).detach().numpy()
            ordinary = predict_event_pairs(
                model,
                events,
                first_numpy,
                second_numpy,
                batch_size=4,
                device=torch.device("cpu"),
            )
            self.assertEqual(cache[2], expected_passes)
            np.testing.assert_allclose(cached, ordinary, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
