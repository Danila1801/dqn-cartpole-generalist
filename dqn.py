#dqn generalist for cartpole-v1 over varying pole lengths.
#three training strategies (selected with --strategy):
#uniform   : domain randomisation, pole length ~ u(0.4, 1.8) every episode (baseline)
#adaptive  : regret-based pole sampling. lengths on which the agent recently
#failed fastest are sampled more often (inspired by prioritized
#level replay, jiang et al. 2021).
#shaped    : adaptive sampling + a stability-shaped reward that penalises pole
#angle and cart offset, to encourage controllers with a safety margin
#against perturbations.
#only the pole length is changed during training; all other dynamics are default.
import argparse
import json
import os
import random
from collections import deque

import numpy as np

np.bool8 = np.bool_  #gym 0.26 passive checker still references np.bool8 (removed in numpy 2)
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(1)

POLE_MIN, POLE_MAX = 0.4, 1.8
N_BINS = 15  #discretisation of the pole-length range for adaptive sampling


#network
class QNetwork(nn.Module):
    #the fixed 'standard' network from the test script (do not modify).

    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


#replay buffer
class ReplayBuffer:
    #uniform experience replay (mnih et al. 2015) stored in numpy arrays.

    def __init__(self, capacity, state_dim):
        self.capacity, self.ptr, self.size = capacity, 0, 0
        self.s = np.zeros((capacity, state_dim), np.float32)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.s2 = np.zeros((capacity, state_dim), np.float32)
        self.d = np.zeros(capacity, np.float32)

    def add(self, s, a, r, s2, d):
        i = self.ptr
        self.s[i], self.a[i], self.r[i], self.s2[i], self.d[i] = s, a, r, s2, d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        idx = np.random.randint(0, self.size, n)
        return (torch.from_numpy(self.s[idx]), torch.from_numpy(self.a[idx]),
                torch.from_numpy(self.r[idx]), torch.from_numpy(self.s2[idx]),
                torch.from_numpy(self.d[idx]))


#pole samplers
class UniformSampler:
    def sample(self):
        return np.random.uniform(POLE_MIN, POLE_MAX)

    def update(self, length, ep_len):
        pass


class AdaptiveSampler:
    #regret-style curriculum: keep a running score per length bin
    #(ema of normalised episode length). bins are sampled with probability
    #proportional to (1 - score + floor)^(1/temp), so weak lengths are visited
    #more, but every bin keeps non-zero mass.

    def __init__(self, n_bins=N_BINS, alpha=0.1, floor=0.05, temp=0.5):
        self.edges = np.linspace(POLE_MIN, POLE_MAX, n_bins + 1)
        self.score = np.zeros(n_bins)  #start pessimistic -> uniform at first
        self.alpha, self.floor, self.temp = alpha, floor, temp

    def probs(self):
        w = (1.0 - self.score + self.floor) ** (1.0 / self.temp)
        return w / w.sum()

    def sample(self):
        b = np.random.choice(len(self.score), p=self.probs())
        return np.random.uniform(self.edges[b], self.edges[b + 1])

    def update(self, length, ep_len):
        b = min(np.searchsorted(self.edges, length) - 1, len(self.score) - 1)
        self.score[b] += self.alpha * (ep_len / 500.0 - self.score[b])


#reward
def shaped_reward(state, terminated, env):
    #+1 survival minus a penalty on |theta| and |x|, normalised by the failure limits.
    x, _, theta, _ = state
    if terminated:
        return -1.0
    u = env.unwrapped
    return 1.0 - 0.5 * abs(theta) / u.theta_threshold_radians - 0.25 * abs(x) / u.x_threshold


#validation
def validate(q, lengths=(0.4, 0.75, 1.1, 1.45, 1.8), episodes=2, max_steps=500):
    #greedy rollouts on held-out lengths under default dynamics (never the test script).
    #returns (mean episode length, mean |theta|) so ties at 500 are broken by stability.
    lens, thetas = [], []
    env = gym.make("CartPole-v1")
    for L in lengths:
        env.unwrapped.length = L
        for _ in range(episodes):
            s, _ = env.reset()
            t, th = 0, 0.0
            done = False
            while not done and t < max_steps:
                with torch.no_grad():
                    a = q(torch.tensor(s).unsqueeze(0)).argmax().item()
                s, _, term, trunc, _ = env.step(a)
                done = term or trunc
                t += 1
                th += abs(s[2])
            lens.append(t)
            thetas.append(th / t)
    env.close()
    return float(np.mean(lens)), float(np.mean(thetas))


#training
def train(strategy, seed, total_steps, out_dir):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    env = gym.make("CartPole-v1")
    sd, ad = env.observation_space.shape[0], env.action_space.n
    q, q_tgt = QNetwork(sd, ad), QNetwork(sd, ad)
    q_tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=5e-4)
    buf = ReplayBuffer(100_000, sd)

    sampler = UniformSampler() if strategy == "uniform" else AdaptiveSampler()
    use_shaping = strategy == "shaped"

    gamma, batch, tau = 0.99, 128, 0.005
    eps_start, eps_end, eps_decay = 1.0, 0.02, 0.3 * total_steps
    learn_start, eval_every = 2_000, 5_000

    log = {"episode_len": [], "pole": [], "val": []}
    best = (-1, 0.0)
    length = sampler.sample(); env.unwrapped.length = length
    s, _ = env.reset(seed=seed)
    ep_len = 0

    for step in range(1, total_steps + 1):
        eps = max(eps_end, eps_start - (eps_start - eps_end) * step / eps_decay)
        if random.random() < eps:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = q(torch.tensor(s).unsqueeze(0)).argmax().item()

        s2, r, term, trunc, _ = env.step(a)
        if use_shaping:
            r = shaped_reward(s2, term, env)
        #truncation (500-step limit) is not a true terminal -> keep bootstrapping
        buf.add(s, a, r, s2, float(term))
        s = s2
        ep_len += 1

        if term or trunc:
            sampler.update(length, ep_len)
            log["episode_len"].append(ep_len); log["pole"].append(length)
            length = sampler.sample(); env.unwrapped.length = length
            s, _ = env.reset()
            ep_len = 0

        if step >= learn_start:
            bs, ba, br, bs2, bd = buf.sample(batch)
            with torch.no_grad():
                #double dqn target (van hasselt et al. 2016)
                a2 = q(bs2).argmax(1, keepdim=True)
                y = br + gamma * (1 - bd) * q_tgt(bs2).gather(1, a2).squeeze(1)
            qsa = q(bs).gather(1, ba.unsqueeze(1)).squeeze(1)
            loss = F.smooth_l1_loss(qsa, y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(q.parameters(), 10.0)
            opt.step()
            #polyak (soft) target update
            with torch.no_grad():
                for p, pt in zip(q.parameters(), q_tgt.parameters()):
                    pt.mul_(1 - tau).add_(tau * p)

        if step % eval_every == 0:
            v_len, v_theta = validate(q)
            log["val"].append([step, v_len, v_theta])
            #model selection: longest survival, then smallest mean |theta|
            if (v_len, -v_theta) > (best[0], -best[1]) or best[0] < 0:
                best = (v_len, v_theta)
                torch.save(q.state_dict(), os.path.join(out_dir, f"{strategy}_s{seed}.pth"))
            print(f"[{strategy} s{seed}] step {step} val_len {v_len:.0f} val_theta {v_theta:.4f} eps {eps:.2f}", flush=True)

    if isinstance(sampler, AdaptiveSampler):
        log["final_probs"] = sampler.probs().tolist()
    with open(os.path.join(out_dir, f"{strategy}_s{seed}_log.json"), "w") as f:
        json.dump(log, f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", choices=["uniform", "adaptive", "shaped"], required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--out", default="weights")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    train(a.strategy, a.seed, a.steps, a.out)
