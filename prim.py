
# ====================================================================
# TOM2C-XRL (single-file demo): Classical vs Quantum vs Hybrid planners
# ====================================================================
# This single Python script implements a ToM2C-inspired setup on
# PettingZoos MPE simple_spread_v3 (cooperative navigation).
# It trains three variants that differ only in the high-level planner:
#   - classical: MLP planner (PyTorch)
#   - quantum:   PQC planner (PennyLane TorchLayer)
#   - hybrid:    fuse classical + quantum outputs
#
# Install (CPU-only is fine to start):
#     pip install "pettingzoo>=1.24" "mpe>=1.0.3" torch pennylane numpy
#
# Run examples:
#     python tom2c_xrl.py --backend classical --episodes 50
#     python tom2c_xrl.py --backend quantum   --episodes 50
#     python tom2c_xrl.py --backend hybrid    --episodes 50
#
# Notes:
# - Uses analytic gradients for the PQC (default.qubit).
# - Defaults are small for quick runs; increase episodes/steps for better results.
# ====================================================================

import argparse
import math
import random
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# PettingZoo MPE environment (simple_spread_v3)
try:
    from pettingzoo.mpe import simple_spread_v3
except Exception as e:
    print("ERROR: Could not import PettingZoo MPE simple_spread_v3. "
          "Please install pettingzoo and mpe: `pip install pettingzoo mpe`")
    raise e

# PennyLane is optional unless you use --backend quantum or hybrid
try:
    import pennylane as qml
    from pennylane.qnn import TorchLayer
    _HAS_PENNYLANE = True
except Exception:
    _HAS_PENNYLANE = False


# -------------------------------
# Utilities
# -------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(x, device):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().to(device)
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return torch.tensor(x, dtype=torch.float32, device=device)


# -------------------------------
# Environment wrapper & helpers
# -------------------------------

@dataclass
class CNConfig:
    n_agents: int = 3
    max_cycles: int = 100
    local_ratio: float = 0.5
    K: int = 10  # planner replan interval
    gamma: float = 0.95


class CNEnv:
    \"\"\"
    Cooperative Navigation using PettingZoo MPE simple_spread_v3 (parallel API).
    Number of landmarks == number of agents (N).
    \"\"\"
    def __init__(self, cfg: CNConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.env = simple_spread_v3.parallel_env(
            N=cfg.n_agents, max_cycles=cfg.max_cycles, local_ratio=cfg.local_ratio
        )
        self.env.reset(seed=seed)
        self.agents = self.env.agents
        self.n_agents = cfg.n_agents
        self.n_landmarks = cfg.n_agents  # simple_spread has N landmarks by default

    def reset(self):
        obs = self.env.reset(seed=self.seed)
        return obs

    def step(self, actions: Dict[str, int]):
        next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
        dones = {a: terminations[a] or truncations[a] for a in self.agents}
        return next_obs, rewards, dones, infos

    @staticmethod
    def parse_obs_vector(obs_vec: np.ndarray, n_agents: int, n_landmarks: int):
        \"\"\"
        Observation format (typical for simple_spread):
        [self_vel(2), self_pos(2), landmark_rel_pos(2*n_landmarks), other_agent_rel_pos(2*(n_agents-1))]
        \"\"\"
        idx = 0
        self_vel = obs_vec[idx:idx+2]; idx += 2
        self_pos = obs_vec[idx:idx+2]; idx += 2
        lm_rel = obs_vec[idx:idx+2*n_landmarks]; idx += 2*n_landmarks
        other_rel = obs_vec[idx:idx+2*(n_agents-1)]
        lm_rel = lm_rel.reshape(n_landmarks, 2)
        other_rel = other_rel.reshape(n_agents-1, 2) if n_agents > 1 else np.zeros((0,2), dtype=np.float32)
        return self_vel, self_pos, lm_rel, other_rel

    @staticmethod
    def closest_landmark_index_from_obs(obs_vec: np.ndarray, n_agents: int, n_landmarks: int) -> int:
        _, _, lm_rel, _ = CNEnv.parse_obs_vector(obs_vec, n_agents, n_landmarks)
        dists = np.linalg.norm(lm_rel, axis=1)
        return int(np.argmin(dists))

    @staticmethod
    def move_towards_landmark_action(obs_vec: np.ndarray, n_agents: int, n_landmarks: int, target_idx: int) -> int:
        _, _, lm_rel, _ = CNEnv.parse_obs_vector(obs_vec, n_agents, n_landmarks)
        rel = lm_rel[target_idx]
        x, y = rel[0], rel[1]
        if abs(x) < 1e-6 and abs(y) < 1e-6:
            return 0  # noop
        if abs(x) > abs(y):
            return 1 if x < 0 else 2  # left/right
        else:
            return 3 if y < 0 else 4  # down/up


# -------------------------------
# Models: ToM, Router, Planners, Critic
# -------------------------------

class ToMNet(nn.Module):
    \"\"\"
    Lightweight ToM module:
    Input (i about j) features from i's obs:
      - rel pos of j (2)
      - rel pos of landmarks (2*nL)
    Outputs:
      - vis_logits: predicted closest landmark of j
      - goal_logits: predicted goal of j
    \"\"\"
    def __init__(self, n_landmarks: int, hidden: int = 128):
        super().__init__()
        self.n_landmarks = n_landmarks
        in_dim = 2 + 2 * n_landmarks
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.vis_head = nn.Linear(hidden, n_landmarks)
        self.goal_head = nn.Linear(hidden, n_landmarks)

    def forward(self, pair_feat: torch.Tensor):
        h = self.net(pair_feat)
        return self.vis_head(h), self.goal_head(h)


class Router(nn.Module):
    \"\"\"Peer-to-peer message gate with Gumbel-Softmax.\"\"\"
    def __init__(self, in_dim: int, hidden: int = 64, temp: float = 1.0):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2)  # [keep, drop]
        )
        self.temp = temp

    def sample_keep(self, pair_feat: torch.Tensor, training: bool = True):
        logits = self.f(pair_feat)
        if training:
            g = -torch.log(-torch.log(torch.rand_like(logits)))
            y = F.softmax((logits + g) / self.temp, dim=-1)
        else:
            y = F.one_hot(logits.argmax(-1), 2).float()
        keep = y[..., 0:1]  # prob/one-hot of "keep"
        return keep, logits


class PlannerMLP(nn.Module):
    def __init__(self, in_dim: int, n_targets: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, n_targets)
        )
    def forward(self, x):
        return self.net(x)


def make_pqc_torchlayer(n_qubits: int, n_layers: int, n_outputs: int):
    dev = qml.device("default.qubit", wires=n_qubits)
    @qml.qnode(dev, interface="torch")
    def circuit(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_outputs)]
    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return circuit, weight_shapes


class PlannerPQC(nn.Module):
    def __init__(self, in_dim: int, n_targets: int, n_qubits: int = 6, n_layers: int = 2):
        super().__init__()
        if not _HAS_PENNYLANE:
            raise RuntimeError("PennyLane not installed. Please `pip install pennylane`.")
        self.pre = nn.Sequential(nn.Linear(in_dim, n_qubits), nn.Tanh())
        circuit, shapes = make_pqc_torchlayer(n_qubits, n_layers, n_targets)
        self.q = TorchLayer(circuit, shapes)
        self.post = nn.Identity()
    def forward(self, x):
        z = self.pre(x)
        y = self.q(z)
        return y  # treat as logits


class PlannerHybrid(nn.Module):
    def __init__(self, in_dim: int, n_targets: int, n_qubits: int = 6, n_layers: int = 2):
        super().__init__()
        self.classic = PlannerMLP(in_dim, n_targets)
        if not _HAS_PENNYLANE:
            raise RuntimeError("PennyLane not installed. Please `pip install pennylane`.")
        self.quantum = PlannerPQC(in_dim, n_targets, n_qubits=n_qubits, n_layers=n_layers)
        self.fuse = nn.Linear(2*n_targets, n_targets)
    def forward(self, x):
        c = self.classic(x)
        q = self.quantum(x)
        return self.fuse(torch.cat([c, q], dim=-1))


class CentralCritic(nn.Module):
    def __init__(self, per_agent_latent: int, n_agents: int):
        super().__init__()
        self.v = nn.Sequential(
            nn.Linear(per_agent_latent * n_agents, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, latents: List[torch.Tensor]):
        x = torch.cat(latents, dim=-1)
        return self.v(x).squeeze(-1)


# -------------------------------
# Agent
# -------------------------------

class AgentShell(nn.Module):
    def __init__(self, agent_id: int, n_agents: int, n_landmarks: int,
                 planner_kind: str, device: torch.device):
        super().__init__()
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.device = device

        self.tom = ToMNet(n_landmarks=n_landmarks).to(device)
        pair_in_dim = 2 + 2 * n_landmarks
        self.router = Router(in_dim=pair_in_dim).to(device)

        # Planner input = enc(obs:64) + agg_tom(2*nL) + agg_msg(nL) = 64 + 3*nL
        planner_in_dim = 64 + 3 * n_landmarks
        if planner_kind == "classical":
            self.planner = PlannerMLP(planner_in_dim, n_landmarks).to(device)
        elif planner_kind == "quantum":
            self.planner = PlannerPQC(planner_in_dim, n_landmarks).to(device)
        elif planner_kind == "hybrid":
            self.planner = PlannerHybrid(planner_in_dim, n_landmarks).to(device)
        else:
            raise ValueError(f"Unknown planner_kind: {planner_kind}")

        obs_dim_guess = 4 + 2 * n_landmarks + 2 * (n_agents - 1)
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_dim_guess, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU()
        ).to(device)

    def build_pair_features(self, obs_i: np.ndarray, obs_j: np.ndarray) -> np.ndarray:
        nA, nL = self.n_agents, self.n_landmarks
        _, _, lm_rel_i, other_rel_i = CNEnv.parse_obs_vector(obs_i, nA, nL)
        # Approximate: choose closest other agent relative position
        rels = other_rel_i
        if rels.shape[0] == 0:
            rel_j = np.zeros(2, dtype=np.float32)
        else:
            dists = np.linalg.norm(rels, axis=1)
            rel_j = rels[int(np.argmin(dists))]
        pair_feat = np.concatenate([rel_j.astype(np.float32), lm_rel_i.reshape(-1).astype(np.float32)], axis=0)
        return pair_feat

    def forward(self, obs_i_vec: torch.Tensor, agg_tom: torch.Tensor, agg_msg: torch.Tensor):
        enc = self.obs_enc(obs_i_vec)
        x = torch.cat([enc, agg_tom, agg_msg], dim=-1)
        logits = self.planner(x)
        return logits


# -------------------------------
# Trainer
# -------------------------------

class MARLTrainer:
    def __init__(self, cfg: CNConfig, backend: str, device: torch.device, seed: int = 0,
                 lr: float = 3e-4, tau_cr: float = 0.02, K: int = None):
        self.cfg = cfg
        if K is not None: self.cfg.K = K
        self.backend = backend
        self.device = device
        self.seed = seed
        self.env = CNEnv(cfg, seed=seed)
        self.agents = self.env.agents
        self.n_agents = cfg.n_agents
        self.n_landmarks = cfg.n_agents
        self.gamma = cfg.gamma
        self.K = cfg.K
        self.tau_cr = tau_cr

        self.shells = nn.ModuleList([
            AgentShell(i, self.n_agents, self.n_landmarks, backend, device)
            for i in range(self.n_agents)
        ]).to(device)

        per_agent_latent = 64 + 3*self.n_landmarks
        self.critic = CentralCritic(per_agent_latent, self.n_agents).to(device)

        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def parameters(self):
        return list(self.shells.parameters()) + list(self.critic.parameters())

    def collect_episode(self, train: bool = True) -> Dict:
        obs = self.env.reset()
        agent_names = self.agents
        nA, nL = self.n_agents, self.n_landmarks
        max_cycles = self.cfg.max_cycles

        ep_reward = 0.0
        traj = {
            "planner_inputs": [],
            "planner_logits": [],
            "planner_actions": [],
            "planner_logps": [],
            "values": [],
            "rewards": [],
            "tom_vis_logits": [],
            "tom_vis_labels": [],
            "tom_goal_logits": [],
            "tom_goal_labels": [],
            "edges_keep_logits": [],
            "edges_keep_targets": []
        }

        current_targets = [0 for _ in range(nA)]

        for t in range(max_cycles):
            replan = (t % self.K == 0)

            if replan:
                tom_vis_logits_list = []
                tom_vis_labels_list = []
                tom_goal_logits_list = []
                tom_goal_labels_list = []
                edges_keep_logits_list = []

                closest_lm = [
                    CNEnv.closest_landmark_index_from_obs(obs[agent_names[j]], nA, nL)
                    for j in range(nA)
                ]
                goal_labels = list(current_targets)

                agg_msgs = []
                planner_logits = []
                planner_actions = []
                planner_logps = []

                obs_vecs = [to_device(obs[agent_names[i]], self.device) for i in range(nA)]
                agg_toms = []

                with_msg_maps = [[None]*nA for _ in range(nA)]
                msg_contents = [[None]*nA for _ in range(nA)]
                tom_goal_preds = [[None]*nA for _ in range(nA)]

                prior_goal_onehots = torch.stack([F.one_hot(torch.tensor(goal_labels[i], device=self.device),
                                                           num_classes=nL).float() for i in range(nA)], dim=0)

                for i in range(nA):
                    for j in range(nA):
                        if i == j: continue
                        pair_feat_np = self.shells[i].build_pair_features(obs[agent_names[i]], obs[agent_names[j]])
                        pair_feat = to_device(pair_feat_np, self.device).unsqueeze(0)
                        vis_logits, goal_logits = self.shells[i].tom(pair_feat)

                        tom_vis_logits_list.append(vis_logits.squeeze(0))
                        tom_vis_labels_list.append(torch.tensor(closest_lm[j], device=self.device))
                        tom_goal_logits_list.append(goal_logits.squeeze(0))
                        tom_goal_labels_list.append(torch.tensor(goal_labels[j], device=self.device))

                        keep, logits = self.shells[i].router.sample_keep(pair_feat, training=train)
                        edges_keep_logits_list.append(logits.squeeze(0))
                        with_msg_maps[i][j] = keep.squeeze(0)
                        msg_contents[i][j] = prior_goal_onehots[i]
                        tom_goal_preds[i][j] = F.log_softmax(goal_logits.squeeze(0), dim=-1).exp()

                for i in range(nA):
                    preds_about_others = [tom_goal_preds[i][j] for j in range(nA) if j != i]
                    preds_from_others  = [tom_goal_preds[s][i] for s in range(nA) if s != i]
                    max_about = torch.stack(preds_about_others, dim=0).max(dim=0).values if preds_about_others else torch.zeros(nL, device=self.device)
                    sum_from = torch.stack(preds_from_others, dim=0).sum(dim=0) if preds_from_others else torch.zeros(nL, device=self.device)
                    agg_toms.append(torch.cat([max_about, sum_from], dim=-1))  # [2*nL]

                for i in range(nA):
                    incoming = []
                    for j in range(nA):
                        if i == j: continue
                        keep_prob = with_msg_maps[j][i]
                        msg = msg_contents[j][i]
                        incoming.append(keep_prob * msg)
                    agg_msgs.append(torch.stack(incoming, dim=0).sum(dim=0) if incoming else torch.zeros(nL, device=self.device))

                per_agent_latents = []
                for i in range(nA):
                    obs_i = obs_vecs[i].unsqueeze(0)
                    agg_tom_i = agg_toms[i].unsqueeze(0)
                    agg_msg_i = agg_msgs[i].unsqueeze(0)
                    logits_i = self.shells[i](obs_i, agg_tom_i, agg_msg_i)
                    probs_i = F.softmax(logits_i, dim=-1)
                    dist = Categorical(probs_i)
                    action_i = dist.sample().squeeze(0)
                    logp_i = dist.log_prob(action_i)
                    per_agent_latents.append(torch.cat([self.shells[i].obs_enc(obs_i),
                                                        agg_tom_i, agg_msg_i], dim=-1).squeeze(0))
                    planner_logits.append(logits_i.squeeze(0))
                    planner_actions.append(action_i)
                    planner_logps.append(logp_i)

                V = self.critic(per_agent_latents)
                traj["planner_inputs"].append(torch.stack(per_agent_latents, dim=0))
                traj["planner_logits"].append(torch.stack(planner_logits, dim=0))
                traj["planner_actions"].append(torch.stack(planner_actions, dim=0))
                traj["planner_logps"].append(torch.stack(planner_logps, dim=0))
                traj["values"].append(V)

                if len(tom_vis_logits_list) > 0:
                    traj["tom_vis_logits"].append(torch.stack(tom_vis_logits_list, dim=0))
                    traj["tom_vis_labels"].append(torch.stack(tom_vis_labels_list, dim=0))
                    traj["tom_goal_logits"].append(torch.stack(tom_goal_logits_list, dim=0))
                    traj["tom_goal_labels"].append(torch.stack(tom_goal_labels_list, dim=0))
                    traj["edges_keep_logits"].append(torch.stack(edges_keep_logits_list, dim=0))

                current_targets = [int(a.item()) for a in planner_actions]

            # low-level executor step
            act_dict = {}
            for i in range(nA):
                name = agent_names[i]
                a = CNEnv.move_towards_landmark_action(obs[name], nA, nL, current_targets[i])
                act_dict[name] = a

            next_obs, rewards, dones, _ = self.env.step(act_dict)
            team_reward = float(sum(rewards.values()))
            ep_reward += team_reward
            traj["rewards"].append(team_reward)

            obs = next_obs
            if all(dones.values()):
                break

        # Pseudo-labels for communication reduction (approximate)
        edges_targets_batches = []
        if len(traj["planner_inputs"]) > 0:
            with torch.no_grad():
                for step_idx in range(len(traj["planner_inputs"])):
                    per_latents = traj["planner_inputs"][step_idx]  # [nA, latent]
                    nL = self.n_landmarks
                    edges_targets = []
                    for i in range(nA):
                        enc = per_latents[i, :64]
                        agg_tom = per_latents[i, 64:64+2*nL]
                        agg_msg = per_latents[i, 64+2*nL:64+3*nL]
                        logits_with = traj["planner_logits"][step_idx][i]
                        x_no = torch.cat([enc, agg_tom, torch.zeros_like(agg_msg)], dim=-1).unsqueeze(0)
                        logits_no = self.shells[i].planner(x_no).squeeze(0)
                        p_no = F.log_softmax(logits_no, dim=-1)
                        p_w  = F.log_softmax(logits_with, dim=-1)
                        kl = F.kl_div(p_no, p_w.exp(), reduction="batchmean")
                        edges_targets.append(1 if kl.item() >= self.tau_cr else 0)
                    edges_targets_batches.append(torch.tensor(edges_targets, device=self.device, dtype=torch.long))

        if len(edges_targets_batches) > 0 and len(traj["edges_keep_logits"]) == len(edges_targets_batches):
            for k in range(len(traj["edges_keep_logits"])):
                per_agent_targets = edges_targets_batches[k]
                n_pairs = traj["edges_keep_logits"][k].shape[0]
                tiled = per_agent_targets.repeat(math.ceil(n_pairs / self.n_agents))[:n_pairs]
                traj["edges_keep_targets"].append(tiled)
        else:
            traj["edges_keep_targets"] = []

        return {"episode_reward": ep_reward, "traj": traj}

    def update(self, traj: Dict, ent_coef: float = 0.01, vf_coef: float = 0.5,
               tom_coef: float = 1.0, cr_coef: float = 0.1):
        rewards = torch.tensor(traj["rewards"], device=self.device, dtype=torch.float32)
        replan_steps = len(traj["values"])
        max_cycles = len(rewards)
        K = self.K
        block_ids = [min(s // K, replan_steps - 1) for s in range(max_cycles)] if replan_steps > 0 else []
        R_blocks = torch.zeros(replan_steps, device=self.device)
        for s in range(max_cycles):
            b = block_ids[s]
            t_in_block = s - b * K
            R_blocks[b] += (self.gamma ** t_in_block) * rewards[s]

        returns = torch.zeros(replan_steps, device=self.device)
        running = 0.0
        for b in reversed(range(replan_steps)):
            running = R_blocks[b] + self.gamma**K * running
            returns[b] = running

        values = torch.stack(traj["values"], dim=0) if replan_steps > 0 else torch.tensor([], device=self.device)
        advantages = returns - values if replan_steps > 0 else torch.tensor([], device=self.device)

        if replan_steps > 0:
            logps = torch.stack(traj["planner_logps"], dim=0)  # [replan, nA]
            ent = 0.0
            for step_logits in traj["planner_logits"]:
                probs = F.softmax(step_logits, dim=-1)
                ent += Categorical(probs=probs).entropy().mean()
            ent = ent / replan_steps
            policy_loss = -(advantages.unsqueeze(-1) * logps).mean()
            value_loss = F.mse_loss(values, returns)
        else:
            policy_loss = torch.tensor(0.0, device=self.device)
            value_loss = torch.tensor(0.0, device=self.device)
            ent = torch.tensor(0.0, device=self.device)

        tom_loss = torch.tensor(0.0, device=self.device)
        if len(traj["tom_vis_logits"]) > 0:
            vis_logits = torch.cat(traj["tom_vis_logits"], dim=0)
            vis_labels = torch.cat(traj["tom_vis_labels"], dim=0)
            tom_loss += F.cross_entropy(vis_logits, vis_labels)
        if len(traj["tom_goal_logits"]) > 0:
            goal_logits = torch.cat(traj["tom_goal_logits"], dim=0)
            goal_labels = torch.cat(traj["tom_goal_labels"], dim=0)
            tom_loss += F.cross_entropy(goal_logits, goal_labels)

        cr_loss = torch.tensor(0.0, device=self.device)
        if len(traj["edges_keep_logits"]) > 0 and len(traj["edges_keep_targets"]) == len(traj["edges_keep_logits"]):
            all_logits = torch.cat(traj["edges_keep_logits"], dim=0)
            all_targets = torch.cat(traj["edges_keep_targets"], dim=0)
            cr_loss = F.cross_entropy(all_logits, all_targets)

        loss = policy_loss + vf_coef * value_loss - ent_coef * ent + tom_coef * tom_loss + cr_coef * cr_loss

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.opt.step()

        stats = {
            "loss_total": float(loss.item()),
            "loss_policy": float(policy_loss.item()),
            "loss_value": float(value_loss.item()),
            "entropy": float(ent.item() if isinstance(ent, torch.Tensor) else ent),
            "loss_tom": float(tom_loss.item()),
            "loss_cr": float(cr_loss.item()),
        }
        return stats

    def train(self, episodes: int = 50, log_interval: int = 10, eval_episodes: int = 5):
        rewards_hist = []
        for ep in range(1, episodes + 1):
            out = self.collect_episode(train=True)
            ep_rew = out["episode_reward"]
            rewards_hist.append(ep_rew)
            stats = self.update(out["traj"])

            if ep % log_interval == 0 or ep == 1:
                avg_last = np.mean(rewards_hist[-log_interval:]) if len(rewards_hist) >= log_interval else np.mean(rewards_hist)
                print(f"[{self.backend}] Ep {ep:4d} | R_ep={ep_rew:7.2f} | R_{log_interval}avg={avg_last:7.2f} "
                      f"| pol {stats['loss_policy']:.3f} val {stats['loss_value']:.3f} ent {stats['entropy']:.3f} "
                      f"| ToM {stats['loss_tom']:.3f} CR {stats['loss_cr']:.3f}")

        eval_scores = []
        for _ in range(eval_episodes):
            out = self.collect_episode(train=False)
            eval_scores.append(out["episode_reward"])
        avg_eval = float(np.mean(eval_scores))
        print(f"[{self.backend}] Eval over {eval_episodes} episodes: AvgR = {avg_eval:.2f}")
        return avg_eval


def main():
    parser = argparse.ArgumentParser(description="ToM2C-inspired X-RL (single-file demo)")
    parser.add_argument("--backend", type=str, default="classical", choices=["classical", "quantum", "hybrid"])
    parser.add_argument("--episodes", type=int, default=50, help="Training episodes")
    parser.add_argument("--eval_episodes", type=int, default=5, help="Evaluation episodes after training")
    parser.add_argument("--n_agents", type=int, default=3, help="Number of agents/landmarks")
    parser.add_argument("--max_cycles", type=int, default=100, help="Episode length (env steps)")
    parser.add_argument("--K", type=int, default=10, help="Planner replan interval")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval (episodes)")
    args = parser.parse_args()

    if args.backend in ("quantum", "hybrid") and not _HAS_PENNYLANE:
        print("ERROR: --backend quantum/hybrid requires PennyLane. Please `pip install pennylane`.")
        sys.exit(1)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    set_seed(args.seed)

    cfg = CNConfig(n_agents=args.n_agents, max_cycles=args.max_cycles, K=args.K, gamma=args.gamma)
    trainer = MARLTrainer(cfg, backend=args.backend, device=device, seed=args.seed, lr=args.lr, tau_cr=0.02, K=args.K)
    avg_eval = trainer.train(episodes=args.episodes, log_interval=args.log_interval, eval_episodes=args.eval_episodes)
    print(f"Done. Backend={args.backend}, EvalAvgReturn={avg_eval:.2f}")


if __name__ == "__main__":
    main()
"""
with open("/mnt/data/tom2c_xrl.py", "w", encoding="utf-8") as f:
    f.write(script)

print("Saved single-file script to /mnt/data/tom2c_xrl.py")
