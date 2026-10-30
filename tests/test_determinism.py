"""A seed fully determines the map, regardless of process or concurrency."""

import json
import os
import subprocess
import sys
import threading
import unittest

from hamc.pipeline import generate_map

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARGS = (4, 3, 2, 4)


def _layers(hmap):
    return hmap.regions, hmap.blocks, hmap.tilemap


class TestSeed(unittest.TestCase):
    def test_same_seed_same_map(self):
        self.assertEqual(_layers(generate_map(*ARGS, seed=42)), _layers(generate_map(*ARGS, seed=42)))
        self.assertNotEqual(generate_map(*ARGS, seed=1).tilemap, generate_map(*ARGS, seed=2).tilemap)

    def test_unseeded_run_records_a_reproducible_seed(self):
        first = generate_map(*ARGS)
        self.assertIsInstance(first.seed, int)
        self.assertEqual(_layers(first), _layers(generate_map(*ARGS, seed=first.seed)))

    def test_same_seed_across_processes(self):
        # String hashing is randomised per process; output must not depend on
        # set/dict iteration order.
        code = ("import json; from hamc.pipeline import generate_map; "
                f"print(json.dumps(generate_map(*{ARGS!r}, seed=7).tilemap))")
        outputs = []
        for hash_seed in ("1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=hash_seed)
            out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                                 capture_output=True, text=True, check=True).stdout
            outputs.append(json.loads(out))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0], generate_map(*ARGS, seed=7).tilemap)

    def test_concurrent_runs_do_not_share_state(self):
        seeds = [3, 4, 5, 6]
        expected = {s: generate_map(*ARGS, seed=s).tilemap for s in seeds}
        results = {}

        def run(s):
            results[s] = generate_map(*ARGS, seed=s, step_delay=0.0001).tilemap

        threads = [threading.Thread(target=run, args=(s,)) for s in seeds]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results, expected)


if __name__ == "__main__":
    unittest.main()
