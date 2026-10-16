# DQN generalist for CartPole over pole lengths (RL Assignment 3)

Trains the fixed 'standard' QNetwork (4-128-128-2) with Double DQN to balance CartPole-v1
for pole lengths 0.4-1.8, comparing three training strategies:

| strategy | pole sampling | reward |
|---|---|---|
| `uniform` (baseline) | U(0.4, 1.8) per episode | +1 per step |
| `adaptive` | regret-based: bins where the agent fails are sampled more (PLR-inspired) | +1 per step |
| `shaped` | regret-based | 1 - 0.5·abs(theta)/theta_max - 0.25·abs(x)/x_max, -1 on failure |

## Files
- `dqn.py` - training (network, replay buffer, samplers, shaped reward, validation-based checkpointing)
- `evaluate.py` - runs the unmodified `test_pole_length` from `test_script.py` on every network, makes plots + stats
- `test_script.py` - course test script (unchanged)
- `weights/<strategy>_s<seed>.pth` - trained networks (3 strategies x 3 seeds); `weights/*_log.json` training logs
- `results/` - bar plots, comparison plot, `summary.txt` (scores, Wilcoxon / Mann-Whitney tests), `raw_scores.json`

## Reproduce
```bash
pip install torch gym==0.26.2 numpy scipy matplotlib pandas openpyxl
for s in uniform adaptive shaped; do for seed in 0 1 2; do
  python dqn.py --strategy $s --seed $seed --steps 80000 --out weights; done; done
python evaluate.py
```
To use the official script directly, set `trained_nn = 'shaped_s<best>.pth'` in `test_script.py`.
With numpy >= 2, add `np.bool8 = np.bool_` before `import gym` (gym 0.26 still references it).
