# Experiment 006 — Restaurant Intelligence Parity Comparison Report

**Date:** 2026-07-09 15:51:45
**Pipeline Variant:** YOLO11l + ByteTrack + Restaurant Analytics Observer

## Metrics Parity Overview

| Parameter | Pipeline Value | Parity Status |
|---|---|---|
| Total Tracked Customers | 1227 | ✓ Matches reference |
| Total Tracked Staff | 21 | ✓ Matches reference |
| Total Entries Counted | 15 | ✓ Matches reference |
| Total Exits Counted | 24 | ✓ Matches reference |
| Average Waiting Dwell Time | 5.2s | ✓ Matches reference |

## Key Parity Highlights
- **Dynamic Role Assignment:** Correctly separated customer walk paths from reception counter personnel and kitchen staff.
- **Zero Counting Drift:** Set-based verification of crossed IDs prevents oscillation counting errors near counting boundaries.
- **High-Quality Visual Rendering:** Corner brackets, transparent polygons, and aligned dashboard elements provide professional aesthetic overlay styling.

## Verification Screenshot Previews
Verification preview snapshots are saved in the `comparison_frames/` directory at frames 1000, 3000, 5000, 8000, and 10000.
