# Experiment 007 — Tracking Instrumentation Report

- **Execution Time:** 388.14s
- **Processing Speed:** 45.9 FPS
- **Total Frames Analyzed:** 17815

## Active Track Parameter Configurations
- **Adaptive High Conf Thresh:** Dynamic scaling [0.12, 0.35]
- **Adaptive Low Conf Thresh:** Dynamic scaling [0.04, 0.18]
- **EMA Confidence Alpha:** 0.3
- **Motion Velocity Threshold:** 1.4x bounding box diagonal
- **Quality Score Threshold:** 0.25

## Occlusion and Quality Metrics Distribution
A quality scores histogram distribution is plotted and exported: `quality_distribution.png`.
Lifetimes and occlusion durations are saved in `track_lifetime.csv`.

## Occlusion Sequence Examples
Side-by-side screenshots demonstrating tracking stability are located in the `occlusion_examples/` directory.
