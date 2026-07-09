# Experiment 007B — Marginal Feature Importance

Marginal contribution of each tracking stability feature compared directly to Configuration 0 (Baseline):

| Feature Module | Fragmentation Change | Track Lifetime Change | ID Creation Change | Occlusion Recoveries | Weighted Contribution | Effect Class |
|---|---|---|---|---|---|---|
| **EMA Smoothing** | +0.0% | +0.0% | +0.0% | +0.0% | **+0.0%** | NEUTRAL (OPTIONAL) |
| **Motion Filter** | +0.0% | -0.4% | +0.0% | +0.0% | **-0.1%** | NEUTRAL (OPTIONAL) |
| **Adaptive Buffer** | +2.3% | -10.8% | -11.6% | -5.4% | **-4.8%** | DETRIMENTAL (REMOVE) |
| **Adaptive Confidence** | +0.0% | +0.0% | +0.0% | +0.0% | **+0.0%** | NEUTRAL (OPTIONAL) |
| **Quality Score** | +5.5% | +9.6% | +10.0% | -4.5% | **+5.9%** | HELPFUL (KEEP) |