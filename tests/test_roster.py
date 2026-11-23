import tempfile
import unittest
from contextlib import closing
from pathlib import Path

from speechroster import store
from speechroster.__main__ import export_srt
from speechroster.voiceprint import identify, voice_windows


class RosterTests(unittest.TestCase):
    def test_open_set_match_and_ambiguous_match(self):
        profiles = {"Alice": [[1., 0.]], "Bea": [[0., 1.]]}
        self.assertEqual(identify([[1., 0.]], profiles, .75, .08), ("Alice", 1.))
        self.assertIsNone(identify([[-1., 0.]], profiles, .75, .08)[0])
        self.assertIsNone(identify([[.71, .70]], profiles, .65, .08)[0])

    def test_windows_exclude_short_and_overlapping_segments(self):
        segments = [(0., 4., "S01", "first"), (3., 6., "S02", "overlap"),
                    (7., 8., "S02", "short"), (9., 13., "S02", "eligible")]
        self.assertEqual(voice_windows(segments), {"S02": [(9., 13.)]})

    def test_assignment_and_truncation_are_scoped_to_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            with closing(store.connect(Path(directory) / "test.sqlite3")) as db:
                segments = [(0., 4., "S01", "Hello"), (5., 8., "S01", "Again")]
                store.save_recording(db, "one", "first.wav", True, segments, {})
                store.save_recording(db, "two", "second.wav", False, segments, {})
                store.assign(db, "one", "S01", "Alice")
                first = store.read_recording(db, "one")
                second = store.read_recording(db, "two")
                self.assertTrue(first["possibly_truncated"])
                self.assertEqual(export_srt(first).count("Alice:"), 2)
                self.assertEqual(export_srt(second).count("S01:"), 2)
                self.assertEqual(store.profiles(db), {})


if __name__ == "__main__":
    unittest.main()
