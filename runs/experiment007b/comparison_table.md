# Experiment 007B — Ablation Metrics Comparison Table

Comprehensive tracking metrics recorded for all configurations:

| Config | Name | Avg Lifetime | Total IDs | Fragmentation | Recovered | Runtime | Processing FPS | Peak RAM |
|---|---|---|---|---|---|---|---|---|
| **0** | Baseline | 98.3 frames | 620 | 3331 | 2732 | 43.83s | 135.5 FPS | 378.3 MB |
| **A** | EMA Only | 98.3 frames | 620 | 3331 | 2732 | 60.64s | 97.9 FPS | 192.1 MB |
| **B** | Motion Filter Only | 97.9 frames | 620 | 3331 | 2732 | 40.75s | 145.7 FPS | 165.4 MB |
| **C** | Adaptive Buffer Only | 87.7 frames | 692 | 3255 | 2585 | 36.97s | 160.6 FPS | 173.7 MB |
| **D** | Adaptive Confidence Only | 98.3 frames | 620 | 3331 | 2732 | 54.17s | 109.6 FPS | 154.1 MB |
| **E** | Quality Score Only | 107.7 frames | 558 | 3147 | 2610 | 38.3s | 155.0 FPS | 160.6 MB |
| **F** | All Features Enabled | 95.5 frames | 623 | 3046 | 2445 | 49.39s | 120.2 FPS | 152.6 MB |