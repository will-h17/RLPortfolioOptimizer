"""
agent.py — SB3 PPO with a Dirichlet policy head for simplex actions, version-safe.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict, Any
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3 import PPO


# ---------------- Dirichlet distribution wrapper ---------------- #

class _DirichletWrapper:
    """Light wrapper around torch.distributions.Dirichlet with helpers SB3 expects."""
    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.alpha: Optional[th.Tensor] = None  # (batch, K)

    def set_params_from_logits(self, logits: th.Tensor) -> "_DirichletWrapper":
        # Softplus ensures strictly positive concentration; add tiny epsilon
        self.alpha = F.softplus(logits) + 1e-3
        return self

    def sample(self, deterministic: bool = False) -> th.Tensor:
        assert self.alpha is not None, "Dirichlet alphas not set"
        if deterministic:
            x = self.mean()
        else:
            x = th.distributions.Dirichlet(self.alpha).rsample()  # reparam where possible
        # Normalize / clamp for numeric safety
        x = th.clamp(x, 1e-8, 1.0)
        return x / x.sum(dim=-1, keepdim=True)

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        assert self.alpha is not None, "Dirichlet alphas not set"
        a = th.clamp(actions, 1e-8, 1.0)
        a = a / a.sum(dim=-1, keepdim=True)
        return th.distributions.Dirichlet(self.alpha).log_prob(a)

    def entropy(self) -> th.Tensor:
        assert self.alpha is not None, "Dirichlet alphas not set"
        return th.distributions.Dirichlet(self.alpha).entropy()

    def mean(self) -> th.Tensor:
        assert self.alpha is not None, "Dirichlet alphas not set"
        alpha = th.clamp(self.alpha, 1e-6, None)
        return alpha / alpha.sum(dim=-1, keepdim=True)


# ---------------- Custom policy: add Dirichlet head, override forward paths ---------------- #

class DirichletActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-Critic policy whose actor head outputs Dirichlet alphas.
    We DO NOT modify SB3's internal distribution registration.
    """

    def _build_mlp_extractor(self) -> None:
        # Build default extractor (sets self.mlp_extractor, value_net later in base _build)
        super()._build_mlp_extractor()

        # Determine latent dims in a version-safe way
        if hasattr(self.mlp_extractor, "latent_dim_pi"):
            latent_dim_pi = self.mlp_extractor.latent_dim_pi
            latent_dim_vf = self.mlp_extractor.latent_dim_vf
        else:
            # Fall back: infer from last layers (older SB3s)
            pol_net = getattr(self.mlp_extractor, "policy_net", None)
            val_net = getattr(self.mlp_extractor, "value_net", None)
            if isinstance(pol_net, nn.Sequential) and len(pol_net) > 0 and hasattr(pol_net[-1], "out_features"):
                latent_dim_pi = pol_net[-1].out_features
            else:
                raise RuntimeError("Could not resolve latent_dim_pi from mlp_extractor.")
            if isinstance(val_net, nn.Sequential) and len(val_net) > 0 and hasattr(val_net[-1], "out_features"):
                latent_dim_vf = val_net[-1].out_features
            else:
                raise RuntimeError("Could not resolve latent_dim_vf from mlp_extractor.")

        # Our Dirichlet head: latent_pi -> logits (then softplus -> alphas)
        self._dirichlet = _DirichletWrapper(self.action_space.shape[0])
        self.action_net = nn.Linear(latent_dim_pi, self.action_space.shape[0])

        # Rebuild value head explicitly
        self.value_net = nn.Linear(latent_dim_vf, 1)

    # ---- Core overrides SB3 PPO uses ---- #

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        logits = self.action_net(latent_pi)
        dist = self._dirichlet.set_params_from_logits(logits)
        actions = dist.sample(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        values = self.value_net(latent_vf).flatten()

        # normalize precisely to simplex
        actions = th.clamp(actions, 1e-8, 1.0)
        actions = actions / actions.sum(dim=-1, keepdim=True)
        return actions, values, log_prob

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def get_distribution(self, obs: th.Tensor) -> _DirichletWrapper:
        features = self.extract_features(obs)
        latent_pi, _ = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        return self._dirichlet.set_params_from_logits(logits)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        PPO training path: returns (values, log_prob, entropy)
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        logits = self.action_net(latent_pi)
        dist = self._dirichlet.set_params_from_logits(logits)

        # Normalize actions to simplex before scoring (in case of tiny drift)
        a = th.clamp(actions, 1e-8, 1.0)
        a = a / a.sum(dim=-1, keepdim=True)

        log_prob = dist.log_prob(a)
        values = self.value_net(latent_vf).flatten()
        entropy = dist.entropy()
        return values, log_prob, entropy


# ---------------- Helper to create PPO with our policy ---------------- #

def make_sb3_ppo(
    env,
    learning_rate: float = 3e-4,
    policy_hidden_sizes: Optional[List[int]] = None,
    policy_kwargs: Optional[Dict[str, Any]] = None,
    **ppo_kwargs,   # <--- accept arbitrary PPO kwargs
):
    """
    Factory for SB3 PPO using the DirichletActorCriticPolicy.
    - learning_rate: float
    - policy_hidden_sizes: e.g., [128, 128] for pi/vf MLPs
    - policy_kwargs: forwarded to policy
    - **ppo_kwargs: forwarded directly to PPO(...) (gamma, gae_lambda, n_steps, batch_size, clip_range, ent_coef, etc.)
    """
    hidden = policy_hidden_sizes or [128, 128]

    default_policy_kwargs = dict(
        net_arch=dict(pi=hidden, vf=hidden)
    )

    if policy_kwargs is None:
        final_policy_kwargs = default_policy_kwargs
    else:
        # shallow-merge defaults with user overrides
        final_policy_kwargs = {**default_policy_kwargs, **policy_kwargs}

    model = PPO(
        DirichletActorCriticPolicy,
        env,
        learning_rate=learning_rate,
        policy_kwargs=final_policy_kwargs,
        verbose=1,
        **ppo_kwargs,   # <--- forward everything else to SB3
    )
    return model
