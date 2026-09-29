# Graph Null Model Comparison: Real vs. Shuffled Surrogate

To determine whether our graph metrics capture genuine, coordinated team shapes rather than just the sum of individual player heatmaps, we compared **10,000 randomly sampled real frames** against **10,000 shuffled surrogates**.

A surrogate frame maintains the players on the pitch, but replaces each player's position with their position from a *different, randomly chosen frame*. This destroys any real-time coordination between players while perfectly preserving each individual player's natural roaming range. We ran a Mann-Whitney U two-sample test (two-sided) to check for statistical significance.

## Robustness Check: The Goalkeeper Effect

Because the goalkeeper tends to stay isolated near their own goal (a small positional range compared to outfielders), shuffling their position often leaves them in roughly the same area. Including them mechanically drags down graph density and connectedness in a way that has nothing to do with outfield tactics. To ensure our metrics are truly detecting team shape and not just "the goalkeeper is always isolated," we ran the test both with **All 11 players** and on just the **10 Outfield players**. *(Note: Because Metrica's tracking data does not provide ground-truth player roles, the goalkeeper was identified algorithmically by finding the player with the lowest positional variance within each half of the match).*

---

## 1. Radius Graph (15m threshold)

### All 11 Players
| Metric | Real Mean (± std) | Shuffled Mean (± std) | MWU p-value | Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Density** | 0.2311 (± 0.1371) | 0.1271 (± 0.0511) | 0.000e+00 | **Significant** |
| **Mean Degree** | 2.2999 (± 1.3679) | 1.2662 (± 0.5099) | 0.000e+00 | **Significant** |
| **Largest Component Frac** | 0.6744 (± 0.2064) | 0.4131 (± 0.1520) | 0.000e+00 | **Significant** |
| **Algebraic Connectivity** | 0.1094 (± 0.7603) | 0.0002 (± 0.0066) | 1.032e-106 | **Significant** |

### 10 Outfield Players (Robustness)
| Metric | Real Mean (± std) | Shuffled Mean (± std) | MWU p-value | Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Density** | 0.2671 (± 0.1442) | 0.1387 (± 0.0626) | 0.000e+00 | **Significant** |
| **Mean Degree** | 2.3907 (± 1.2908) | 1.2437 (± 0.5621) | 0.000e+00 | **Significant** |
| **Largest Component Frac** | 0.7240 (± 0.2137) | 0.4362 (± 0.1663) | 0.000e+00 | **Significant** |
| **Algebraic Connectivity** | 0.1832 (± 0.8087) | 0.0012 (± 0.0222) | 0.000e+00 | **Significant** |

**Takeaway:** The radius graph strongly detects coordinated behavior, and this signal easily survives the removal of the goalkeeper. In fact, metrics like density and largest component fraction naturally rise when analyzing only the outfield (72% connected vs 67% with the GK), confirming the keeper was dragging down the averages. When players roam independently (the surrogate), the team fractures into isolated islands much more often.

*(Statistical Note: In the radius graph, the shuffled surrogate almost always fragments into disconnected components, driving its algebraic connectivity to nearly 0.0. A rank-based test like Mann-Whitney U is on shakier ground when comparing a continuously distributed real metric against a near-constant baseline with near-zero variance. While the conclusion holds—the real graph is connected and the surrogate isn't—the p-value itself should be interpreted with that caveat.)*

---

## 2. Delaunay Triangulation

### All 11 Players
| Metric | Real Mean (± std) | Shuffled Mean (± std) | MWU p-value | Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Density** | 0.4392 (± 0.0222) | 0.4405 (± 0.0220) | 7.197e-13 | **Significant** |
| **Mean Degree** | 4.3701 (± 0.1700) | 4.3832 (± 0.1841) | 1.692e-11 | **Significant** |
| **Largest Component Frac** | 1.0000 (± 0.0000) | 1.0000 (± 0.0000) | NaN | Not Significant |
| **Algebraic Connectivity** | 1.5174 (± 0.2548) | 1.4569 (± 0.3027) | 1.373e-56 | **Significant** |

### 10 Outfield Players (Robustness)
| Metric | Real Mean (± std) | Shuffled Mean (± std) | MWU p-value | Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Density** | 0.4596 (± 0.0268) | 0.4764 (± 0.0252) | 0.000e+00 | **Significant** |
| **Mean Degree** | 4.1143 (± 0.1871) | 4.2656 (± 0.1960) | 0.000e+00 | **Significant** |
| **Largest Component Frac** | 1.0000 (± 0.0000) | 1.0000 (± 0.0000) | NaN | Not Significant |
| **Algebraic Connectivity** | 1.4285 (± 0.2740) | 1.5770 (± 0.3171) | 7.488e-280 | **Significant** |

**Takeaway:** Because Delaunay always builds a planar mesh spanning the entire team, the largest component is always 100%. Interestingly, the *real* shapes actually yield slightly **lower** density/mean degree than the randomized blobs! This suggests that real soccer formations feature more players on the convex hull boundary (e.g. distinct defensive/attacking lines) compared to a randomized cloud of players. This effect becomes *even more pronounced* when isolating the 10 outfield players, dropping real mean degree down to 4.11 vs shuffled 4.26.

***

> [!NOTE] Conclusion
> The 10,000-sample test is consistent with the hypothesis that our graph approach detects macro-level team coordination that is invisible if you only look at individual player heatmaps. Removing the perpetually-isolated goalkeeper as a robustness check confirms that the signal is driven by genuine tactical shape in the outfield, not just goalkeeper isolation. It is important to distinguish statistical significance from a large effect size: while all metrics are statistically significant given the massive sample size, the algebraic connectivity result for Delaunay graphs, in particular, only became detectable due to the high $N$ and represents a subtle structural shift rather than a massive effect.
