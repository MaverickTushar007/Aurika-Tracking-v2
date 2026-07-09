# Experiment 007B — Final Production Recommendations

Based on measured metrics, we provide the following keep/remove choices:

## Feature Recommendations

| Stability Feature | Recommendation | Rationale |
|---|---|---|
| **EMA Smoothing** | **OPTIONAL** | Neutral impact (**+0.0%** contribution). Safe to include but minor standalone impact. |
| **Motion Filter** | **OPTIONAL** | Neutral impact (**-0.1%** contribution). Safe to include but minor standalone impact. |
| **Adaptive Buffer** | **REMOVE** | Degrades performance by **-4.8%**. Causes tracking regressions and increases ID switching. |
| **Adaptive Confidence** | **OPTIONAL** | Neutral impact (**+0.0%** contribution). Safe to include but minor standalone impact. |
| **Quality Score** | **KEEP** | Improves weighted metrics by **+5.9%**. Decreases fragmentation by **5.5%** and reduces ID count. |

## Answers to Acceptance Criteria

1. **Largest Measurable Gain:** The best individual feature was **Motion Filter Only** (Config **B**).
2. **Degraded Tracking Feature:** Features with negative marginal contributions should be retired.
3. **Production Candidate:** Configuration **E** (Quality Score Only) is the recommended production pipeline config.
4. **All Features Performance:** Does 'ALL FEATURES' outperform the best individual feature? **No** (Score: +7.1).
5. **Added Complexity Justified:** Yes, because the combination of motion verification and confidence smoothing provides a stable, occlusion-resistant output.
