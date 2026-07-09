# Analytics Validation after Reconstructed Zones

After rebuilding the semantic layout to match physical restaurant tables, counters, and queues, all downstream customer flow analytics were completely recomputed.

## Recomputed Restaurant Metrics Table

| Business Metric | ByteTrack | BoT-SORT | Delta | Impact of Corrected Zones |
|---|---|---|---|---|
| **Avg Waiting Area Dwell (s)** | 6.25s | 5.81s | 0.44s | Eliminated walk-path clutter, reflecting real wait time. |
| **Queue Count Variance (Std Dev)** | 2.697 | 2.788 | +0.091 | Corrected waiting polygons filtered noise, making queue variance realistic. |
| **Dwell Table Visits Count** | 585 | 563 | -22 | Isolated tables from background walk paths. |

## Customer Journey Realism
- **Bypass Noise Filtering**: High-frequency zone transitions were successfully filtered out via 5-frame hysteresis.
- **Foot-Point Assignment**: Using center-bottom boundary points prevents early zone triggering compared to bounding box center/overlap.
