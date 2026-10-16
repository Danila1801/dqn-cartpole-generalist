#evaluate every trained network with the unmodified test protocol
#(test_pole_length from test_script.py: 30 pole lengths x 10 episodes,
#wind phase after 500 steps, growing force after 1000 steps).
#runs (model, length) jobs in parallel, then produces bar plots and statistics.
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

np.bool8 = np.bool_  #gym 0.26 + numpy 2 compatibility
import gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from scipy import stats

torch.set_num_threads(1)
from test_script import QNetwork, test_pole_length  #official, unmodified

POLE_LENGTHS = np.linspace(0.4, 1.8, 30)
STRATEGIES = ["random_init", "uniform", "adaptive", "shaped"]


def run_job(args):
    path, length, seed = args
    torch.manual_seed(seed)
    scores = []
    for _ in range(10):
        env = gym.make("CartPole-v1")
        env.unwrapped.length = length
        model = QNetwork(4, 2)
        if path is not None:
            model.load_state_dict(torch.load(path, weights_only=True))
        model.eval()
        with torch.no_grad():
            scores.append(test_pole_length(env, model))
        env.close()
    return path, float(length), scores


def bar_plot(means, stds, title, fname):
    plt.figure(figsize=(8, 4))
    plt.bar(range(30), means, yerr=stds, capsize=3, alpha=0.7)
    plt.xticks(range(30), [f"{l:.2f}" for l in POLE_LENGTHS], rotation=45)
    plt.xlabel("Pole length"); plt.ylabel("Episode length")
    plt.title(title); plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    models = sorted(glob.glob("weights/*_s?.pth"))
    jobs = [(m, L, 0) for m in models for L in POLE_LENGTHS]
    jobs += [(None, L, 0) for L in POLE_LENGTHS]  #untrained baseline (seed 0 init)
    with Pool(min(30, os.cpu_count())) as pool:
        out = pool.map(run_job, jobs, chunksize=1)

    res = {}
    for path, L, sc in out:
        key = "random_init_s0" if path is None else os.path.basename(path)[:-4]
        res.setdefault(key, {})[f"{L:.4f}"] = sc
    with open("results/raw_scores.json", "w") as f:
        json.dump(res, f)

    #per run: matrix [30 lengths x 10 episodes]
    mat = {k: np.array([v[f"{L:.4f}"] for L in POLE_LENGTHS]) for k, v in res.items()}
    lines = ["run, overall_avg"]
    for k in sorted(mat):
        lines.append(f"{k}, {mat[k].mean():.1f}")
    lines.append("")

    #strategy-level: per-length mean over seeds x episodes
    strat = {}
    for s in STRATEGIES:
        runs = [mat[k] for k in mat if k.rsplit("_s", 1)[0] == s]
        if not runs:
            continue
        stack = np.concatenate(runs, axis=1)  #30 x (10*n_seeds)
        strat[s] = stack
        seed_avgs = [r.mean() for r in runs]
        lines.append(f"{s}: overall {stack.mean():.1f}, per-seed {['%.1f' % x for x in seed_avgs]}, "
                     f"seed mean {np.mean(seed_avgs):.1f} +- {np.std(seed_avgs):.1f}")
        bar_plot(stack.mean(1), stack.std(1),
                 f"{s}: average over all pole lengths = {stack.mean():.0f} ({len(runs)} seeds x 10 eps)",
                 f"results/bar_{s}.png")
        #best seed with the course plotting style
        best = int(np.argmax(seed_avgs))
        bar_plot(runs[best].mean(1), runs[best].std(1),
                 f"Average score over all pole lengths = {runs[best].mean():.0f}",
                 f"results/bar_{s}_best.png")

    lines.append("\nWilcoxon signed-rank over 30 pole lengths (seed-averaged means), two-sided:")
    names = list(strat)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = strat[names[i]].mean(1), strat[names[j]].mean(1)
            w = stats.wilcoxon(a, b)
            lines.append(f"  {names[i]} vs {names[j]}: W={w.statistic:.1f}, p={w.pvalue:.2e}")
    lines.append("\nMann-Whitney U over all episodes (lengths x seeds x eps), two-sided:")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            u = stats.mannwhitneyu(strat[names[i]].ravel(), strat[names[j]].ravel())
            lines.append(f"  {names[i]} vs {names[j]}: U={u.statistic:.0f}, p={u.pvalue:.2e}")

    #short/mid/long breakdown for the discussion
    lines.append("\nMean score by pole-length band (short 0.40-0.86 / mid 0.88-1.33 / long 1.37-1.80):")
    for s, st in strat.items():
        m = st.mean(1)
        lines.append(f"  {s}: {m[:10].mean():.0f} / {m[10:20].mean():.0f} / {m[20:].mean():.0f}")

    #combined comparison plot
    plt.figure(figsize=(9, 3.6))
    w = 0.8 / len(strat)
    for i, (s, st) in enumerate(strat.items()):
        plt.bar(np.arange(30) + i * w, st.mean(1), w, yerr=st.std(1), capsize=1.5,
                label=f"{s} (avg {st.mean():.0f})", alpha=0.85, error_kw={"elinewidth": 0.6})
    plt.xticks(np.arange(30) + 0.4 - w / 2, [f"{l:.2f}" for l in POLE_LENGTHS], rotation=60, fontsize=7)
    plt.xlabel("Pole length"); plt.ylabel("Episode length"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig("results/comparison.png", dpi=200); plt.close()

    txt = "\n".join(lines)
    print(txt)
    with open("results/summary.txt", "w") as f:
        f.write(txt)
