"""Generate a batch of maps and report timing, validity and composition."""

import statistics
import time
from collections import Counter

from hamc import generate_map

N = 20
times, attempts, regions, blocks = [], [], Counter(), Counter()
valid = 0
for seed in range(N):
    t0 = time.perf_counter()
    hmap = generate_map(6, 4, 3, 6, seed=seed)
    times.append(time.perf_counter() - t0)
    valid += hmap.valid
    attempts.append(hmap.block_attempts / (len(hmap.blocks) * len(hmap.blocks[0])))
    regions.update(r for row in hmap.regions for r in row)
    blocks.update(b for row in hmap.blocks for b in row)

print(f"{N} maps of 6x4 regions / 18x12 blocks / 108x72 tiles")
print(f"valid: {valid}/{N}")
print(f"time per map: median {statistics.median(times):.2f}s, max {max(times):.2f}s")
print(f"local attempts per block: mean {statistics.mean(attempts):.3f}")
total_r = sum(regions.values())
print("regions:", ", ".join(f"{k} {v / total_r:.0%}" for k, v in regions.most_common()))
total_b = sum(blocks.values())
print("blocks: ", ", ".join(f"{k} {v / total_b:.0%}" for k, v in blocks.most_common(8)))
