import numpy as np
from scipy import stats

from sportsgraph.adapters.metrica import load_metrica_match
from sportsgraph.analytics.null_model import compare_real_vs_shuffled


def main():
    print("Loading data...")
    df = load_metrica_match(
        "data/raw/metrica/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "data/raw/metrica/Sample_Game_1_RawTrackingData_Away_Team.csv",
    )

    n_samples = 100
    team = "home"

    for rule in ["delaunay", "radius"]:
        print(f"\n========== Evaluating Rule: {rule} ==========")
        res = compare_real_vs_shuffled(df, team, rule, n_samples=n_samples, seed=42)

        real = res[res["kind"] == "real"]
        shuff = res[res["kind"] == "shuffled"]

        metrics = ["density", "mean_degree", "largest_component_frac", "algebraic_connectivity"]
        for m in metrics:
            r_vals = real[m].dropna().values
            s_vals = shuff[m].dropna().values

            if len(r_vals) < 2 or len(s_vals) < 2:
                continue

            r_mean, r_std = np.mean(r_vals), np.std(r_vals)
            s_mean, s_std = np.mean(s_vals), np.std(s_vals)

            stat, p = stats.mannwhitneyu(r_vals, s_vals, alternative="two-sided")

            sig = "Significant" if p < 0.05 else "Not Significant"
            print(f"Metric: {m}")
            print(f"  Real:     mean={r_mean:.4f}, std={r_std:.4f}")
            print(f"  Shuffled: mean={s_mean:.4f}, std={s_std:.4f}")
            print(f"  MWU Test: p={p:.3e} -> {sig}")


if __name__ == "__main__":
    main()
