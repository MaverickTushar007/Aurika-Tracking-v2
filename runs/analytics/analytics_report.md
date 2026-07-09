# Restaurant Intelligence Analytics Report

**Date:** 2026-07-09 14:51:05
**Video Analyzed:** `Dark_lighting.mp4`
**Total Tracked Persons:** 1248

## Executive Summary

- **Total Customers Entered:** 15
- **Total Customers Exited:** 24
- **Overall Customer Flow:** High density observed at the Dining room and Entrance areas.

## Zone Dwell Statistics

| Zone Name | Average Dwell Time | Role / Description |
|---|---|---|
| **Entrance** | 0.8s | Transitionary zone. High flow velocity. |
| **Waiting** | 6.3s | Lobby area where customers wait for table allocation. |
| **Reception** | 5.0s | Counter area for payments, ordering, and greeting. |
| **Dining** | 5.2s | Main dining area. Highest dwell time expected. |
| **Kitchen** | 3.6s | Restricted employee area. Tracked employee dwell time. |

## Customer Zone Transitions

This table tracks how customers move from one physical zone of the restaurant to another.

| Path Transition | Frequency (Total Counts) |
|---|---|
| `Waiting ➔ Dining` | 60 |
| `Dining ➔ Waiting` | 55 |
| `Kitchen ➔ Dining` | 37 |
| `Reception ➔ Dining` | 37 |
| `Dining ➔ Reception` | 36 |
| `Dining ➔ Kitchen` | 29 |
| `Entrance ➔ Waiting` | 21 |
| `Waiting ➔ Entrance` | 16 |

## Store Design & Flow Recommendations

- **Lobby Flow Efficient:** Average Waiting area dwell time is stable (6.3s).
- **Security Boundary Intrusion:** We detected 29 transitions into the Kitchen area. Ensure the kitchen entrance door is clearly signed as restricted access to prevent customer entry.

## Customer Trajectory Heatmap

The heatmap visually demonstrates spatial residence hotspots across the restaurant floor:

![Trajectory Heatmap](heatmap.png)
