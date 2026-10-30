"""A seed fully determines the map: same seed, same tiles, in any process."""

import hashlib
import json

from hamc import generate_map


def fingerprint(hmap) -> str:
    return hashlib.sha256(json.dumps(hmap.tilemap).encode()).hexdigest()[:16]


a = generate_map(4, 3, 2, 4, seed=1234)
b = generate_map(4, 3, 2, 4, seed=1234)
c = generate_map(4, 3, 2, 4, seed=1235)
print("seed 1234:", fingerprint(a))
print("seed 1234:", fingerprint(b), "(identical)" if a.tilemap == b.tilemap else "(DIFFERENT!)")
print("seed 1235:", fingerprint(c))

# Without a seed one is drawn and recorded, so any map can be regenerated.
random_map = generate_map(4, 3, 2, 4)
again = generate_map(4, 3, 2, 4, seed=random_map.seed)
print(f"unseeded run used seed {random_map.seed}; regenerated identically: {random_map.tilemap == again.tilemap}")
