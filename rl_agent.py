# =============================================================================
# SITARAM HFT — RL Agent (PPO)
# Step 2 of 5
#
# Pure numpy implementation of Proximal Policy Optimization (PPO).
# No PyTorch or TensorFlow dependency. Runs on any machine.
#
# Architecture:
#   Actor  network: obs(5) → hidden(64) → hidden(64) → action_mean(2)
#   Critic network: obs(5) → hidden(64) → hidden(64) → value(1)
#   Both use tanh activations. Actions are Gaussian with learned log_std.
#
# PPO specifics:
#   - Clipped surrogate objective (clip_eps = 0.2)
#   - GAE advantage estimation (lambda = 0.95, gamma = 0.99)
#   - Entropy bonus for exploration
#   - Gradient descent via Adam (numpy implementation)
# =============================================================================

import os
import json
import numpy as np
from typing import List, Tuple, Dict, Optional

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
LR          = 3e-4     # Adam learning rate
GAMMA       = 0.99     # Discount factor
GAE_LAMBDA  = 0.95     # GAE smoothing
CLIP_EPS    = 0.20     # PPO clip ratio
ENTROPY_C   = 0.01     # Entropy bonus coefficient
VALUE_C     = 0.50     # Value loss coefficient
HIDDEN      = 64       # Hidden layer size
N_EPOCHS    = 4        # PPO update epochs per batch
BATCH_SIZE  = 512      # Rollout buffer before update
MINI_BATCH  = 64       # Mini-batch size for SGD


# =============================================================================
# NUMPY NEURAL NETWORK PRIMITIVES
# =============================================================================

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2

def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -30, 30)))


class AdamOptimizer:
    """Per-parameter Adam optimizer."""

    def __init__(self, lr: float = LR, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.t     = 0
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}

    def step(self, param: np.ndarray, grad: np.ndarray) -> np.ndarray:
        pid = id(param)
        self.t += 1
        if pid not in self.m:
            self.m[pid] = np.zeros_like(grad)
            self.v[pid] = np.zeros_like(grad)
        self.m[pid] = self.beta1 * self.m[pid] + (1 - self.beta1) * grad
        self.v[pid] = self.beta2 * self.v[pid] + (1 - self.beta2) * grad ** 2
        m_hat = self.m[pid] / (1 - self.beta1 ** self.t)
        v_hat = self.v[pid] / (1 - self.beta2 ** self.t)
        return param - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class MLP:
    """
    2-hidden-layer MLP.
    Layers: [in_dim → HIDDEN → HIDDEN → out_dim]
    Activation: tanh on hidden layers, linear on output.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden: int = HIDDEN,
                 out_scale: float = 0.01):
        # Xavier initialisation for stable gradients
        def xavier(fan_in, fan_out):
            lim = np.sqrt(6.0 / (fan_in + fan_out))
            return np.random.uniform(-lim, lim, (fan_in, fan_out))

        self.W1 = xavier(in_dim, hidden)
        self.b1 = np.zeros(hidden)
        self.W2 = xavier(hidden, hidden)
        self.b2 = np.zeros(hidden)
        self.W3 = xavier(hidden, out_dim) * out_scale
        self.b3 = np.zeros(out_dim)

        self._params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]
        self._cache: dict = {}

    def forward(self, x: np.ndarray, store: bool = False) -> np.ndarray:
        h1 = tanh(x  @ self.W1 + self.b1)
        h2 = tanh(h1 @ self.W2 + self.b2)
        out = h2 @ self.W3 + self.b3
        if store:
            self._cache = {'x': x, 'h1': h1, 'h2': h2, 'out': out}
        return out

    def backward(self, d_out: np.ndarray) -> List[np.ndarray]:
        c   = self._cache
        x   = c['x']; h1 = c['h1']; h2 = c['h2']

        # Layer 3
        dW3 = h2.T @ d_out
        db3 = d_out.sum(axis=0)
        dh2 = d_out @ self.W3.T

        # Layer 2
        dh2_pre = dh2 * tanh_grad(h1 @ self.W2 + self.b2)
        dW2 = h1.T @ dh2_pre
        db2 = dh2_pre.sum(axis=0)
        dh1 = dh2_pre @ self.W2.T

        # Layer 1
        dh1_pre = dh1 * tanh_grad(x @ self.W1 + self.b1)
        dW1 = x.T @ dh1_pre
        db1 = dh1_pre.sum(axis=0)

        return [dW1, db1, dW2, db2, dW3, db3]

    def params(self) -> List[np.ndarray]:
        return self._params

    def apply_grads(self, grads: List[np.ndarray], opt: AdamOptimizer):
        for i, (p, g) in enumerate(zip(self._params, grads)):
            updated = opt.step(p, g)
            self._params[i][:] = updated

    def save(self) -> dict:
        return {
            'W1': self.W1.tolist(), 'b1': self.b1.tolist(),
            'W2': self.W2.tolist(), 'b2': self.b2.tolist(),
            'W3': self.W3.tolist(), 'b3': self.b3.tolist(),
        }

    def load(self, d: dict):
        self.W1[:] = np.array(d['W1'])
        self.b1[:] = np.array(d['b1'])
        self.W2[:] = np.array(d['W2'])
        self.b2[:] = np.array(d['b2'])
        self.W3[:] = np.array(d['W3'])
        self.b3[:] = np.array(d['b3'])
        self._params = [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3]


# =============================================================================
# PPO POLICY
# =============================================================================

class PPOPolicy:
    """
    Actor-Critic policy for continuous actions.

    Actor  → outputs [gamma_raw, kappa_raw]  (unbounded)
    log_std → learnable parameter per action dim
    Critic → outputs scalar value estimate

    Actions are sampled from N(actor_out, exp(log_std))
    and clipped to [GAMMA_MIN, GAMMA_MAX] × [KAPPA_MIN, KAPPA_MAX].
    """

    from src.rl_environment import GAMMA_MIN, GAMMA_MAX, KAPPA_MIN, KAPPA_MAX

    def __init__(self, obs_dim: int = 5, act_dim: int = 2):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.actor   = MLP(obs_dim, act_dim, HIDDEN, out_scale=0.01)
        self.critic  = MLP(obs_dim, 1,       HIDDEN, out_scale=1.00)
        self.log_std = np.full(act_dim, -1.5)  # initial std ≈ 0.22

        self.actor_opt  = AdamOptimizer(LR)
        self.critic_opt = AdamOptimizer(LR)
        self.std_opt    = AdamOptimizer(LR)

    # ------------------------------------------------------------------
    def predict(self, obs: np.ndarray, deterministic: bool = False
                ) -> Tuple[np.ndarray, float, float]:
        """
        Given obs, return (action, log_prob, value).
        action is clipped to valid [gamma, kappa] ranges.
        """
        obs2d  = obs.reshape(1, -1)
        mean   = self.actor.forward(obs2d)[0]
        value  = self.critic.forward(obs2d)[0, 0]
        std    = np.exp(self.log_std)

        if deterministic:
            raw_action = mean
        else:
            raw_action = mean + std * np.random.randn(self.act_dim)

        log_prob = self._log_prob(raw_action, mean, std)
        action   = self._clip_action(raw_action)
        return action, float(log_prob), float(value)

    def _clip_action(self, raw: np.ndarray) -> np.ndarray:
        from src.rl_environment import GAMMA_MIN, GAMMA_MAX, KAPPA_MIN, KAPPA_MAX
        return np.array([
            np.clip(raw[0], GAMMA_MIN, GAMMA_MAX),
            np.clip(raw[1], KAPPA_MIN, KAPPA_MAX),
        ], dtype=np.float32)

    @staticmethod
    def _log_prob(action: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
        """Log probability of action under N(mean, std²)."""
        var      = std ** 2
        log_prob = -0.5 * np.sum(((action - mean) ** 2) / var
                                  + 2 * np.log(std) + np.log(2 * np.pi))
        return float(log_prob)

    # ------------------------------------------------------------------
    def update(self, rollout: 'RolloutBuffer') -> dict:
        """
        PPO update from collected rollout.
        Returns dict of loss stats for logging.
        """
        n       = getattr(rollout, '_valid_n', rollout._ptr)
        obs     = rollout.obs[:n]
        acts    = rollout.actions[:n]
        old_lp  = rollout.log_probs[:n]
        advs    = rollout.advantages[:n]
        returns = rollout.returns[:n]

        # Normalise advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        stats = {'policy_loss': [], 'value_loss': [], 'entropy': []}

        for _ in range(N_EPOCHS):
            idx = np.random.permutation(len(obs))
            for start in range(0, len(obs), MINI_BATCH):
                mb = idx[start:start + MINI_BATCH]
                mb_obs  = obs[mb]
                mb_acts = acts[mb]
                mb_olp  = old_lp[mb]
                mb_adv  = advs[mb]
                mb_ret  = returns[mb]

                # Forward pass
                means  = self.actor.forward(mb_obs,  store=True)
                values = self.critic.forward(mb_obs, store=True).squeeze()
                std    = np.exp(self.log_std)

                # Log probs under current policy
                diff   = mb_acts - means
                var    = std ** 2
                new_lp = -0.5 * np.sum(diff**2 / var + 2*np.log(std)
                                        + np.log(2*np.pi), axis=1)

                # PPO ratio
                ratio  = np.exp(new_lp - mb_olp)
                clip   = np.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
                p_loss = -np.minimum(ratio * mb_adv, clip * mb_adv).mean()

                # Value loss
                v_loss = 0.5 * ((values - mb_ret) ** 2).mean()

                # Entropy bonus
                entropy = 0.5 * np.sum(np.log(2 * np.pi * np.e * var))

                total_loss = p_loss + VALUE_C * v_loss - ENTROPY_C * entropy

                # ── Actor backward ──────────────────────────────────────
                # d_loss / d_means
                d_ratio  = np.exp(new_lp - mb_olp)
                d_clip_m = ((ratio > 1 + CLIP_EPS) | (ratio < 1 - CLIP_EPS))
                d_adv    = -mb_adv * np.where(d_clip_m, 0, d_ratio)
                d_lp     = d_adv / (len(mb) + 1e-8)

                # d_log_prob / d_mean
                d_means  = d_lp[:, None] * (-(mb_acts - means) / var)
                a_grads  = self.actor.backward(d_means)
                self.actor.apply_grads(a_grads, self.actor_opt)

                # log_std gradient
                d_std    = d_lp[:, None] * (((mb_acts - means)**2 / var) - 1)
                d_logstd = (d_std * std).mean(axis=0) - ENTROPY_C * 1.0
                self.log_std = np.clip(self.std_opt.step(self.log_std, d_logstd), -3.0, 0.5)

                # ── Critic backward ─────────────────────────────────────
                d_values = VALUE_C * (values - mb_ret) / (len(mb) + 1e-8)
                c_grads  = self.critic.backward(d_values.reshape(-1, 1))
                self.critic.apply_grads(c_grads, self.critic_opt)

                stats['policy_loss'].append(float(p_loss))
                stats['value_loss'].append(float(v_loss))
                stats['entropy'].append(float(entropy))

        return {k: float(np.mean(v)) for k, v in stats.items()}

    # ------------------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        data = {
            'actor':   self.actor.save(),
            'critic':  self.critic.save(),
            'log_std': self.log_std.tolist(),
            'obs_dim': self.obs_dim,
            'act_dim': self.act_dim,
        }
        with open(path, 'w') as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> 'PPOPolicy':
        with open(path, 'r') as f:
            data = json.load(f)
        policy = cls(data['obs_dim'], data['act_dim'])
        policy.actor.load(data['actor'])
        policy.critic.load(data['critic'])
        policy.log_std = np.array(data['log_std'])
        return policy


# =============================================================================
# ROLLOUT BUFFER
# =============================================================================

class RolloutBuffer:
    """Stores one batch of experience for PPO update."""

    def __init__(self, capacity: int = BATCH_SIZE, obs_dim: int = 5, act_dim: int = 2):
        self.capacity = capacity
        self.obs_dim  = obs_dim
        self.act_dim  = act_dim
        self.obs        = np.zeros((capacity, obs_dim),  dtype=np.float32)
        self.actions    = np.zeros((capacity, act_dim),  dtype=np.float32)
        self.rewards    = np.zeros(capacity,              dtype=np.float32)
        self.values     = np.zeros(capacity,              dtype=np.float32)
        self.log_probs  = np.zeros(capacity,              dtype=np.float32)
        self.dones      = np.zeros(capacity,              dtype=np.float32)
        self.advantages = np.zeros(capacity,              dtype=np.float32)
        self.returns    = np.zeros(capacity,              dtype=np.float32)
        self._ptr = 0
        self._full = False
        self._valid_n = 0

    def add(self, obs, action, reward, value, log_prob, done):
        if self._ptr >= self.capacity:
            return   # buffer full — caller should have flushed before this
        i = self._ptr
        self.obs[i]       = obs
        self.actions[i]   = action
        self.rewards[i]   = reward
        self.values[i]    = value
        self.log_probs[i] = log_prob
        self.dones[i]     = float(done)
        self._ptr += 1
        if self._ptr >= self.capacity:
            self._full = True

    def is_ready(self) -> bool:
        return self._full or self._ptr >= self.capacity

    def compute_gae(self, last_value: float = 0.0):
        """Compute GAE advantages and discounted returns."""
        n       = min(self._ptr, self.capacity)
        rewards = self.rewards[:n]
        values  = self.values[:n]
        dones   = self.dones[:n]

        adv = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_val = last_value if t == n - 1 else values[t + 1]
            delta    = rewards[t] + GAMMA * next_val * (1 - dones[t]) - values[t]
            last_gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * last_gae
            adv[t]   = last_gae

        self.advantages[:n] = adv
        self.returns[:n]    = adv + values[:n]
        # Store actual size so update() knows how many valid entries exist
        self._valid_n = n

    def reset(self):
        # Reallocate to full capacity — compute_gae() trims the arrays
        # so we must rebuild them to avoid IndexError on next episode.
        self.obs        = np.zeros((self.capacity, self.obs_dim),  dtype=np.float32)
        self.actions    = np.zeros((self.capacity, self.act_dim),  dtype=np.float32)
        self.rewards    = np.zeros(self.capacity,                   dtype=np.float32)
        self.values     = np.zeros(self.capacity,                   dtype=np.float32)
        self.log_probs  = np.zeros(self.capacity,                   dtype=np.float32)
        self.dones      = np.zeros(self.capacity,                   dtype=np.float32)
        self.advantages = np.zeros(self.capacity,                   dtype=np.float32)
        self.returns    = np.zeros(self.capacity,                   dtype=np.float32)
        self._ptr  = 0
        self._full = False
