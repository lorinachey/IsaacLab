"""Offline reward computation for Spot warehouse navigation demos.

Computes per-step reward components from HDF5 episode data collected by
collect_demos.py.  All signals are derived from logged arrays — no
re-simulation required.

Reward terms
------------
1. time_penalty        -c per step (prevent loitering)
2. termination_penalty -C on last step of a failed episode
3. goal_reached        +R on last step of a successful episode
4. goal_approach       d(t-1) - d(t)  (potential-based shaping)
5. command_tracking    -||vel_cmd - vel_achieved||  (L2 error)
6. command_smoothness  -||vel_cmd(t) - vel_cmd(t-1)||
7. body_contact        -sum(||F_i|| for non-foot bodies)
8. uprightness         projected_gravity_z  (≈ -1 when upright → flipped to +1)

Usage
-----
    from rewards import RewardConfig, compute_episode_rewards
    cfg = RewardConfig()
    result = compute_episode_rewards(episode_dict, cfg)
    total  = result["total"]       # (T,) ndarray
    terms  = result["components"]  # dict[str, (T,) ndarray]
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

# Body indices for Spot's four feet in the 17-body contact sensor.
# Empirically identified: bodies with highest mean ground-reaction force.
FOOT_BODY_INDICES = {3, 7, 11, 15}
ALL_BODY_INDICES = set(range(17))
NON_FOOT_BODY_INDICES = sorted(ALL_BODY_INDICES - FOOT_BODY_INDICES)

# policy_obs layout (48-dim)
OBS_LIN_VEL = slice(0, 3)     # base linear velocity, robot frame
OBS_ANG_VEL = slice(3, 6)     # base angular velocity, robot frame
OBS_GRAVITY = slice(6, 9)     # projected gravity vector, robot frame
OBS_VEL_CMD = slice(9, 12)    # velocity command [v_x, v_y, omega_z]


@dataclasses.dataclass
class RewardConfig:
    """Weights and parameters for each reward term.

    All weights multiply the raw signal so the final reward is:
        r_total(t) = sum(w_i * r_i(t))
    Set a weight to 0.0 to disable a term.
    """

    # 1. Time penalty
    w_time_penalty: float = -0.1

    # 2. Termination penalty (applied once on last step of failed episode)
    w_termination: float = -10.0

    # 3. Goal reached bonus (applied once on last step of successful episode)
    w_goal_reached: float = 10.0

    # 4. Goal approach (potential-based shaping)
    w_goal_approach: float = 1.0

    # 5. Command tracking: -||vel_cmd - vel_achieved||_2
    w_command_tracking: float = 1.0

    # 6. Command smoothness: -||delta_vel_cmd||_2
    w_command_smoothness: float = 0.5

    # 7. Non-foot body contact: -sum(||F_i||) for non-foot bodies
    w_body_contact: float = 0.01
    body_contact_threshold_n: float = 1.0  # ignore forces below this (sensor noise)

    # 8. Uprightness: projected_gravity_z flipped to [0, 1] range
    w_uprightness: float = 0.5


def compute_episode_rewards(
    episode: dict[str, Any],
    cfg: RewardConfig | None = None,
) -> dict[str, Any]:
    """Compute per-step rewards for a single episode.

    Parameters
    ----------
    episode : dict
        Episode data as loaded from HDF5.  Expected keys:
            - "policy_obs"       (T, 48)  float32
            - "velocity_cmd"     (T, 3)   float32
            - "goal_relative"    (T, 3)   float32
            - "contact_forces"   (T, 17, 3) float32
            - "success"          bool
            - "episode_length"   int
    cfg : RewardConfig, optional
        Reward weights.  Defaults to RewardConfig().

    Returns
    -------
    dict with keys:
        "total"      : (T,) ndarray — weighted sum of all terms
        "components" : dict[str, (T,) ndarray] — individual reward signals
        "config"     : RewardConfig — the config used
    """
    if cfg is None:
        cfg = RewardConfig()

    policy_obs = episode["policy_obs"]        # (T, 48)
    vel_cmd    = episode["velocity_cmd"]       # (T, 3)
    goal_rel   = episode["goal_relative"]      # (T, 3)
    cf         = episode["contact_forces"]     # (T, 17, 3)
    success    = episode["success"]
    T          = episode["episode_length"]

    components: dict[str, np.ndarray] = {}

    # ── 1. Time penalty ───────────────────────────────────────────────────
    components["time_penalty"] = np.full(T, cfg.w_time_penalty, dtype=np.float32)

    # ── 2. Termination penalty ────────────────────────────────────────────
    term = np.zeros(T, dtype=np.float32)
    if not success:
        term[-1] = cfg.w_termination
    components["termination_penalty"] = term

    # ── 3. Goal reached bonus ─────────────────────────────────────────────
    goal = np.zeros(T, dtype=np.float32)
    if success:
        goal[-1] = cfg.w_goal_reached
    components["goal_reached"] = goal

    # ── 4. Goal approach (potential-based shaping) ────────────────────────
    dist_xy = np.linalg.norm(goal_rel[:, :2], axis=-1)   # (T,)
    approach = np.zeros(T, dtype=np.float32)
    approach[1:] = dist_xy[:-1] - dist_xy[1:]
    components["goal_approach"] = approach * cfg.w_goal_approach

    # ── 5. Command tracking: -||vel_cmd - vel_achieved||_2 ───────────────
    vel_achieved = np.column_stack([
        policy_obs[:, 0],   # v_x  (robot frame)
        policy_obs[:, 1],   # v_y  (robot frame)
        policy_obs[:, 5],   # omega_z (yaw rate, robot frame)
    ])                              # (T, 3)
    tracking_error = np.linalg.norm(vel_cmd - vel_achieved, axis=-1)  # (T,)
    components["command_tracking"] = -tracking_error * cfg.w_command_tracking

    # ── 6. Command smoothness: -||delta_vel_cmd||_2 ──────────────────────
    smoothness = np.zeros(T, dtype=np.float32)
    if T > 1:
        delta = np.linalg.norm(np.diff(vel_cmd, axis=0), axis=-1)  # (T-1,)
        smoothness[1:] = -delta
    components["command_smoothness"] = smoothness * cfg.w_command_smoothness

    # ── 7. Non-foot body contact ─────────────────────────────────────────
    non_foot_forces = cf[:, NON_FOOT_BODY_INDICES, :]      # (T, 13, 3)
    non_foot_mag = np.linalg.norm(non_foot_forces, axis=-1) # (T, 13)
    non_foot_mag[non_foot_mag < cfg.body_contact_threshold_n] = 0.0
    body_contact = -non_foot_mag.sum(axis=-1)               # (T,)
    components["body_contact"] = body_contact.astype(np.float32) * cfg.w_body_contact

    # ── 8. Uprightness ───────────────────────────────────────────────────
    # projected_gravity_z is ≈ -1.0 when perfectly upright (gravity points
    # down in body frame).  Flip sign so reward is ≈ +1 when upright.
    grav_z = policy_obs[:, 8]         # (T,)  z-component of projected gravity
    uprightness = -grav_z             # +1 when upright, ~0 when sideways
    uprightness = np.clip(uprightness, 0.0, 1.0)
    components["uprightness"] = uprightness.astype(np.float32) * cfg.w_uprightness

    # ── Total ─────────────────────────────────────────────────────────────
    total = np.zeros(T, dtype=np.float32)
    for v in components.values():
        total += v

    return {
        "total": total,
        "components": components,
        "config": cfg,
    }


def compute_dataset_rewards(
    episodes: dict[str, dict[str, Any]],
    cfg: RewardConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute rewards for every episode in a dataset.

    Parameters
    ----------
    episodes : dict[str, dict]
        Mapping of episode keys to episode data dicts (as loaded from HDF5).
    cfg : RewardConfig, optional

    Returns
    -------
    dict[str, result_dict] — keyed by episode key, each value is the
    output of ``compute_episode_rewards``.
    """
    if cfg is None:
        cfg = RewardConfig()
    return {k: compute_episode_rewards(ep, cfg) for k, ep in episodes.items()}
