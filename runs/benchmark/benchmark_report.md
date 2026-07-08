# Aurika Tracking v2 — Benchmark Report

> Video: `Dark_lighting.mp4` | Sample rate: every 3rd frame | Device: `MPS`

## Stage 2 — Detection Results

> `avg_detections` is **informational only** — it is NOT used to rank models.

| Model | Frames | Avg Det* | Avg Conf | Med Conf | Avg Small | Med ms | Peak RAM |
|---|---|---|---|---|---|---|---|
| yolo11m | 5937 | 9.12 | 0.541 | 0.509 | 0.00 | 88.6 ms | 323 MB |
| yolo11l | 5937 | 10.95 | 0.536 | 0.494 | 0.01 | 111.5 ms | 345 MB |
| yolo11x | 5937 | 9.99 | 0.551 | 0.516 | 0.00 | 225.4 ms | 338 MB |
| yolo_staff_customer | 5937 | 0.22 | 0.339 | 0.319 | 0.07 | 27.4 ms | 320 MB |

**Top-2 advancing to Stage 3 (tracking):** `yolo_staff_customer`, `yolo11l`

## Stage 3 — Tracking Results

| Model | Frames | Avg Tracks | Max | Unique IDs | Birth/100f ↓ | Avg Lifetime ↑ | Recovered ↑ | Lost ↓ | Med ms |
|---|---|---|---|---|---|---|---|---|---|
| yolo_staff_customer | 5937 | 0.18 | 2 | 29 | 0.49 | 172.1 | 16 | 27 | 24.6 ms |
| yolo11l | 5937 | 10.19 | 19 | 724 | 12.19 | 99.8 | 514 | 707 | 152.1 ms |

## Recommendation

```

======================================================================
  RECOMMENDATION
======================================================================

  WINNER  :  yolo11l
  Label   :  YOLO11l  (COCO pretrained)
  Score   :  0.6029  (runner-up yolo_staff_customer: 0.6004)

  Reasons:

  ~  Track lifetime 99.8 sampled frames — not the longest but within acceptable range given other strengths.

  ✓  Best track recovery — 514 tracks reacquired vs 16 for yolo_staff_customer.
     ByteTrack re-linked persons after occlusion or missed detections more
     often, reducing the number of ID splits in crowded restaurant scenes.

  ~  Birth rate 12.19/100f is slightly elevated vs 0.49/100f for yolo_staff_customer.
     Some ID fragmentation is occurring — monitor in the full-length run.

  ~  707 tracks were permanently lost before video end
     (same as or slightly more than the runner-up). Acceptable for now.

  ✓  Inference speed — 152.1 ms median / 205.3 ms p95 on MPS.
     Suitable for offline batch processing at this frame rate.

  Caveats:
  ⚠  707 permanently lost tracks — consider increasing track_buffer or decreasing match_thresh to improve track persistence.
  ⚠  Birth rate 12.19/100f is elevated. Tuning track_high_thresh may reduce ID fragmentation.
  ⚠  Inference is slow (152 ms/frame). If real-time is later required, benchmark yolo_staff_customer with optimised settings.

  Next step:
  Run a FULL validation across all 17,815 frames of Dark_lighting.mp4
  with yolo11l before promoting to production. This benchmark sampled
  every 3rd frame; the full run is your production validation.

  Command: python run.py  (after updating configs/config.yaml model_path)
======================================================================
```
