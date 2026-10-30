import contextlib
import io
import json
import os
import tempfile
import unittest

from hamc.cli import main


class TestCli(unittest.TestCase):
    def test_generate_writes_layers(self):
        with tempfile.TemporaryDirectory() as out, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["generate", "--seed", "1", "-o", out]), 0)
            self.assertTrue(os.path.exists(os.path.join(out, "final_tilemap.png")))
            with open(os.path.join(out, "map.json"), encoding="utf-8") as f:
                self.assertEqual(json.load(f)["seed"], 1)

    def test_generate_json_matches_seed(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            main(["generate", "--seed", "9", "--width", "3", "--height", "2", "--json"])
        data = json.loads(buf.getvalue())
        self.assertEqual((data["seed"], data["width"], data["height"]), (9, 3, 2))
        self.assertTrue(data["valid"])


if __name__ == "__main__":
    unittest.main()
