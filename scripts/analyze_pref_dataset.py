import os
import sys

import numpy as np
import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


class RewardModel(nn.Module):
    def __init__(self, obs_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def main():
    dataset_path = "pref_logs/pref_dataset_debug.npz"
    reward_model_path = "pref_logs/reward_model.pt"

    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return
    if not os.path.exists(reward_model_path):
        print(f"Reward model not found at {reward_model_path}")
        return

    data = np.load(dataset_path, allow_pickle=True)
    traj_obs = data["traj_observations"]  # object array: list of (T_i, obs_dim)
    pair_indices = data["pair_indices"]  # (N, 2)
    pref_probs = data["pref_probs"].astype(np.float32)  # (N,)

    print(f"Loaded {len(traj_obs)} trajectories from {dataset_path}")
    lengths = np.array([x.shape[0] for x in traj_obs])
    print(f"Trajectory length min/mean/max: {lengths.min()} / {lengths.mean():.1f} / {lengths.max()}")
    print(
        f"Pref probs stats: min={pref_probs.min():.3f}, "
        f"mean={pref_probs.mean():.3f}, max={pref_probs.max():.3f}, "
        f"std={pref_probs.std():.3f}"
    )

    obs_dim = traj_obs[0].shape[1]
    rm = RewardModel(obs_dim, hidden_dim=256)
    state = torch.load(reward_model_path, map_location="cpu")
    rm.load_state_dict(state)
    rm.eval()

    def traj_return(x_np: np.ndarray) -> float:
        x = torch.from_numpy(x_np).float()
        with torch.no_grad():
            return rm(x).sum().item()

    returns = np.array([traj_return(x) for x in traj_obs], dtype=np.float32)
    logits = returns[pair_indices[:, 0]] - returns[pair_indices[:, 1]]
    p_hat = 1.0 / (1.0 + np.exp(-logits))

    print(
        f"p_hat stats: min={p_hat.min():.3f}, mean={p_hat.mean():.3f}, "
        f"max={p_hat.max():.3f}, std={p_hat.std():.3f}"
    )

    if len(pref_probs) > 1:
        corr = np.corrcoef(pref_probs, p_hat)[0, 1]
    else:
        corr = float("nan")
    print(f"corr(pref_probs, p_hat) = {corr:.3f}")

    eps = 1e-6
    bce = -(
        pref_probs * np.log(p_hat + eps)
        + (1.0 - pref_probs) * np.log(1.0 - p_hat + eps)
    ).mean()
    print(f"Soft-label BCE between VLM prefs and reward model: {bce:.4f}")


if __name__ == "__main__":
    main()

