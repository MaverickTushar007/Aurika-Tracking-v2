# Experiment 009 — Tracker Evaluation & Recommendation Report

## 1. Tracker Performance Summary

| Metric | ByteTrack (Baseline) | BoT-SORT + OSNet | Change % |
|---|---|---|---|
| **Unique Track IDs** | 623 | 618 | -0.8% |
| **Fragmentation** | 1853 | 1967 | +6.2% |
| **Recoveries** | 1240 | 1358 | +9.5% |
| **Avg Lifetime (Frames)** | 63.6 | 61.8 | -2.8% |
| **Runtime (Seconds)** | 118.53s | 11539.71s | +9635.7% |
| **Processing Speed** | 50.1 FPS | 0.5 FPS | -99.0% |

## 2. Restaurant Analytics & Business Metrics

| Business Indicator | ByteTrack | BoT-SORT + OSNet | Difference |
|---|---|---|---|
| **Avg Waiting Time (Seconds)** | 8.83s | 9.48s | 0.65s |
| **Queue Stability (Std Dev)** | 0.444 | 0.454 | +0.010 |
| **Total Zone Transitions** | 504 | 504 | +0 |

## 3. Production Recommendation
**Choice:** ByteTrack remains the preferred production tracker.

**Rationale:**
While BoT-SORT + OSNet integrates appearance embeddings, the added extraction latency degrades execution speed by 99.0% (dropping from 50.1 to 0.5 FPS) without yielding statistically significant gains in customer waiting time estimation (difference: 0.65s) or queue stability.
