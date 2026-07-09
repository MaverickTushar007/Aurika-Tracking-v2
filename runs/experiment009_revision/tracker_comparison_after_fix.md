# Tracker Comparison After Zone Correction

## Tracker Benchmarking Summary Table

| Tracker Profile | Unique Track IDs | Fragmentation Gaps | Recoveries | Runtime (s) | Speed (FPS) |
|---|---|---|---|---|---|
| **ByteTrack (Baseline)** | 623 | 1853 | 1240 | 231.95s | 25.6 FPS |
| **BoT-SORT + ReID** | 618 | 1967 | 1358 | 2810.77s | 2.1 FPS |

## Production Choice & Technical Rationale
**Choice:** ByteTrack remains the preferred production tracker.

**Technical Rationale:**
While BoT-SORT + OSNet integrates appearance embeddings, the added extraction latency degrades execution speed by 91.8% (dropping from 25.6 to 2.1 FPS) without yielding statistically significant gains in customer waiting time estimation (difference: 0.44s) or queue stability.
