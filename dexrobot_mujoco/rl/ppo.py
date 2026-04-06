from dataclasses import dataclass
import json
import math
import os

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class PPOConfig:
    seed: int = 0
    total_steps: int = 200000
    rollout_steps: int = 1024
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    train_epochs: int = 10
    minibatch_size: int = 256
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    # Transformer architecture
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 6
    num_tokens: int = 16
    ffn_dim_factor: int = 4
    dropout: float = 0.0
    checkpoint_interval: int = 10000
    device: str = "auto"


class TransformerActorCritic(nn.Module):
    """PPO Actor-Critic with Transformer encoder.

    The observation vector is projected into ``num_tokens`` tokens of size
    ``d_model``.  A learnable CLS token is prepended.  After ``num_layers``
    of Transformer encoder the CLS representation drives both the actor mean
    and the critic value.
    """

    def __init__(self, obs_dim, action_dim, d_model=512, num_heads=8,
                 num_layers=6, num_tokens=16, ffn_dim_factor=4, dropout=0.0):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        ffn_dim = d_model * ffn_dim_factor

        # Project flat obs → (num_tokens, d_model)
        self.obs_proj = nn.Linear(obs_dim, num_tokens * d_model)

        # Learnable CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_emb = nn.Parameter(torch.zeros(1, num_tokens + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.actor_mean = nn.Linear(d_model, action_dim)
        self.critic = nn.Linear(d_model, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def _encode(self, obs):
        B = obs.shape[0]
        # Project and reshape: (B, num_tokens, d_model)
        tokens = self.obs_proj(obs).view(B, self.num_tokens, self.d_model)
        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)           # (B, num_tokens+1, d_model)
        tokens = tokens + self.pos_emb                      # add positional embeddings
        out = self.transformer(tokens)                      # (B, num_tokens+1, d_model)
        return self.norm(out[:, 0])                         # CLS output: (B, d_model)

    def forward(self, obs):
        latent = self._encode(obs)
        return self.actor_mean(latent), self.critic(latent).squeeze(-1)

    def act(self, obs):
        mean, value = self.forward(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate_actions(self, obs, action):
        mean, value = self.forward(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


class PPOTrainer:
    def __init__(self, env, config=None):
        self.env = env
        self.config = config or PPOConfig()
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        # Support both single env and SubprocVecEnv
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vec_env = self.num_envs > 1

        if self.is_vec_env:
            self.obs_dim = env.obs_dim
            self.action_dim = env.action_dim
            obs, _ = env.reset()  # (num_envs, obs_dim)
        else:
            obs, _ = env.reset()
            self.obs_dim = int(obs.shape[0])
            self.action_dim = int(env.action_dim)

        self.device = self._resolve_device(self.config.device)
        self.policy = TransformerActorCritic(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            d_model=self.config.d_model,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            num_tokens=self.config.num_tokens,
            ffn_dim_factor=self.config.ffn_dim_factor,
            dropout=self.config.dropout,
        ).to(self.device)

        n_params = sum(p.numel() for p in self.policy.parameters())
        print(f"Policy parameters: {n_params:,}")

        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=self.config.actor_lr,
            weight_decay=1e-4,
        )
        self.obs = obs
        self.global_step = 0

    @staticmethod
    def _resolve_device(device):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available in the current PyTorch environment.")
        return torch.device(device)

    def _compute_gae(self, rewards, values, dones, last_values):
        T = rewards.shape[0]
        advantages = np.zeros_like(rewards)
        gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(T)):
            next_values = last_values if t == T - 1 else values[t + 1]
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.config.gamma * next_values * nonterminal - values[t]
            gae = delta + self.config.gamma * self.config.gae_lambda * nonterminal * gae
            advantages[t] = gae
        return advantages, advantages + values

    def collect_rollout(self):
        T = self.config.rollout_steps
        N = self.num_envs

        obs_buf = np.zeros((T, N, self.obs_dim), dtype=np.float32)
        act_buf = np.zeros((T, N, self.action_dim), dtype=np.float32)
        logp_buf = np.zeros((T, N), dtype=np.float32)
        rew_buf = np.zeros((T, N), dtype=np.float32)
        done_buf = np.zeros((T, N), dtype=np.float32)
        val_buf = np.zeros((T, N), dtype=np.float32)
        info_list = []

        for t in range(T):
            if self.is_vec_env:
                obs_tensor = torch.tensor(self.obs, dtype=torch.float32, device=self.device)
            else:
                obs_tensor = torch.tensor(self.obs, dtype=torch.float32, device=self.device).unsqueeze(0)

            with torch.no_grad():
                action_tensor, log_prob_tensor, value_tensor = self.policy.act(obs_tensor)

            actions = action_tensor.cpu().numpy()
            clipped = np.clip(actions, -1.0, 1.0)

            if self.is_vec_env:
                next_obs, rewards, dones, infos = self.env.step(clipped)
            else:
                next_obs, reward, terminated, truncated, info = self.env.step(clipped[0])
                rewards = np.array([reward], dtype=np.float32)
                dones = np.array([float(terminated or truncated)], dtype=np.float32)
                infos = [info]
                if dones[0]:
                    next_obs_reset, _ = self.env.reset()
                    next_obs = np.expand_dims(next_obs_reset, 0)
                else:
                    next_obs = np.expand_dims(next_obs, 0)

            obs_buf[t] = self.obs if self.is_vec_env else self.obs[None]
            act_buf[t] = clipped
            logp_buf[t] = log_prob_tensor.cpu().numpy()
            rew_buf[t] = rewards
            done_buf[t] = dones
            val_buf[t] = value_tensor.cpu().numpy()
            info_list.extend(infos)

            self.obs = next_obs
            self.global_step += N

        last_obs_tensor = torch.tensor(self.obs, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            _, last_values = self.policy.forward(last_obs_tensor)
        last_values = last_values.cpu().numpy()

        advantages, returns = self._compute_gae(rew_buf, val_buf, done_buf, last_values)

        return {
            "obs": obs_buf.reshape(-1, self.obs_dim),
            "actions": act_buf.reshape(-1, self.action_dim),
            "log_probs": logp_buf.reshape(-1),
            "rewards": rew_buf.reshape(-1),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
            "infos": info_list,
        }

    def update(self, rollout):
        obs = torch.tensor(rollout["obs"], device=self.device)
        actions = torch.tensor(rollout["actions"], device=self.device)
        old_log_probs = torch.tensor(rollout["log_probs"], device=self.device)
        returns = torch.tensor(rollout["returns"], device=self.device)
        advantages = torch.tensor(rollout["advantages"], device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        n = obs.shape[0]
        indices = np.arange(n)
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        updates = 0
        for _ in range(self.config.train_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.config.minibatch_size):
                batch_idx = indices[start:start + self.config.minibatch_size]
                log_probs, entropy, values = self.policy.evaluate_actions(
                    obs[batch_idx], actions[batch_idx]
                )
                ratio = torch.exp(log_probs - old_log_probs[batch_idx])
                clipped_ratio = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio)
                policy_loss = -torch.min(ratio * advantages[batch_idx],
                                         clipped_ratio * advantages[batch_idx]).mean()
                value_loss = ((values - returns[batch_idx]) ** 2).mean()
                entropy_bonus = entropy.mean()

                loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy_bonus
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy_bonus.item())
                updates += 1
        for key in metrics:
            metrics[key] /= max(updates, 1)
        return metrics

    def save(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(output_dir, "policy.pt"))
        with open(os.path.join(output_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
            json.dump({
                "global_step": self.global_step,
                "obs_dim": self.obs_dim,
                "action_dim": self.action_dim,
                "num_envs": self.num_envs,
                "device": str(self.device),
            }, f, indent=2)

    def train(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        history = []
        while self.global_step < self.config.total_steps:
            rollout = self.collect_rollout()
            metrics = self.update(rollout)
            success_rate = float(np.mean([info.get("success", False) for info in rollout["infos"]]))
            avg_reward = float(np.mean(rollout["rewards"]))
            row = {
                "step": self.global_step,
                "avg_reward": avg_reward,
                "success_rate": success_rate,
                "num_envs": self.num_envs,
                "device": str(self.device),
                **metrics,
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))
            if self.global_step % self.config.checkpoint_interval < self.config.rollout_steps * self.num_envs:
                self.save(os.path.join(output_dir, f"checkpoint_{self.global_step:07d}"))
        self.save(output_dir)
        with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        return history
