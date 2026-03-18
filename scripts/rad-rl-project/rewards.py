"""Reward computation for Spot warehouse navigation.

Provides both offline (batch over HDF5 arrays) and online (step-by-step)
reward computation.  Both APIs share the same ``RewardConfig`` and the same
per-term math so the reward definition is a single source of truth.

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

Usage — offline (batch)
-----------------------
    from rewards import RewardConfig, compute_episode_rewards
    cfg = RewardConfig()
    result = compute_episode_rewards(episode_dict, cfg)
    total  = result["total"]       # (T,) ndarray
    terms  = result["components"]  # dict[str, (T,) ndarray]

Usage — online (step-by-step, for collect_demos.py and future DPPO)
-------------------------------------------------------------------
    from rewards import RewardConfig, StepRewardComputer, REWARD_COMPONENT_NAMES
    cfg = RewardConfig()
    rc  = StepRewardComputer(cfg)
    # each 10 Hz step:
    components, total = rc.step(policy_obs, vel_cmd, goal_relative, contact_forces)
    # at episode end:
    sparse = rc.finalize_episode(success=True)  # returns sparse dict
    rc.reset()
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

REWARD_COMPONENT_NAMES = [
    "time_penalty",
    "termination_penalty",
    "goal_reached",
    "goal_approach",
    "command_tracking",
    "command_smoothness",
    "body_contact",
    "uprightness",
]


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
    w_termination: float = -100.0

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

    def to_dict(self) -> dict[str, float]:
        """Return all fields as a flat dict (for HDF5 attribute storage)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> RewardConfig:
        return cls(**{k: v for k, v in d.items() if k in {f.name for f in dataclasses.fields(cls)}})


# ── Online step-by-step API ──────────────────────────────────────────────────

class StepRewardComputer:
    """Stateful, step-by-step reward computation for online use.

    Maintains the minimal inter-step state needed by ``goal_approach``
    (previous goal distance) and ``command_smoothness`` (previous vel_cmd).

    Typical lifecycle::

        rc = StepRewardComputer(cfg)
        for each episode:
            for each 10 Hz step:
                comps, total = rc.step(policy_obs, vel_cmd, goal_rel, contact_forces)
            sparse = rc.finalize_episode(success)
            rc.reset()
    """

    def __init__(self, cfg: RewardConfig | None = None):
        self.cfg = cfg or RewardConfig()
        self._prev_goal_dist: float | None = None
        self._prev_vel_cmd: np.ndarray | None = None

    def reset(self):
        """Clear inter-step state for a new episode."""
        self._prev_goal_dist = None
        self._prev_vel_cmd = None

    def step(
        self,
        policy_obs: np.ndarray,       # (48,)
        vel_cmd: np.ndarray,           # (3,)
        goal_relative: np.ndarray,     # (3,)
        contact_forces: np.ndarray,    # (N_bodies, 3)
    ) -> tuple[dict[str, float], float]:
        """Compute reward components for a single timestep.

        Parameters
        ----------
        policy_obs : (48,) — raw locomotion policy input
        vel_cmd : (3,) — [v_x, v_y, omega_z] post-negation/clamping
        goal_relative : (3,) — goal_world - base_pos
        contact_forces : (N_bodies, 3) — net forces, world frame

        Returns
        -------
        (components, total) where components is a dict of raw (unweighted)
        signal values and total is the weighted sum.
        """
        cfg = self.cfg
        comps: dict[str, float] = {}

        # 1. Time penalty (raw signal is always 1.0; weight carries the sign)
        comps["time_penalty"] = 1.0

        # 2 & 3. Sparse — zero during normal steps, set by finalize_episode
        comps["termination_penalty"] = 0.0
        comps["goal_reached"] = 0.0

        # 4. Goal approach
        dist_xy = float(np.linalg.norm(goal_relative[:2]))
        if self._prev_goal_dist is not None:
            comps["goal_approach"] = self._prev_goal_dist - dist_xy
        else:
            comps["goal_approach"] = 0.0
        self._prev_goal_dist = dist_xy

        # 5. Command tracking: -||vel_cmd - vel_achieved||
        vel_achieved = np.array([policy_obs[0], policy_obs[1], policy_obs[5]])
        comps["command_tracking"] = -float(np.linalg.norm(vel_cmd - vel_achieved))

        # 6. Command smoothness
        if self._prev_vel_cmd is not None:
            comps["command_smoothness"] = -float(np.linalg.norm(vel_cmd - self._prev_vel_cmd))
        else:
            comps["command_smoothness"] = 0.0
        self._prev_vel_cmd = vel_cmd.copy()

        # 7. Non-foot body contact
        non_foot = contact_forces[NON_FOOT_BODY_INDICES]   # (13, 3)
        mag = np.linalg.norm(non_foot, axis=-1)            # (13,)
        mag[mag < cfg.body_contact_threshold_n] = 0.0
        comps["body_contact"] = -float(mag.sum())

        # 8. Uprightness
        grav_z = float(policy_obs[8])
        comps["uprightness"] = float(np.clip(-grav_z, 0.0, 1.0))

        # Weighted total
        total = (
            cfg.w_time_penalty      * comps["time_penalty"]
            + cfg.w_termination     * comps["termination_penalty"]
            + cfg.w_goal_reached    * comps["goal_reached"]
            + cfg.w_goal_approach   * comps["goal_approach"]
            + cfg.w_command_tracking * comps["command_tracking"]
            + cfg.w_command_smoothness * comps["command_smoothness"]
            + cfg.w_body_contact    * comps["body_contact"]
            + cfg.w_uprightness     * comps["uprightness"]
        )

        return comps, float(total)

    def finalize_episode(self, success: bool) -> dict[str, float]:
        """Return the sparse end-of-episode reward components.

        Call this *after* the last ``step()`` of the episode. The caller
        is responsible for adding these values to the last buffered step.

        Returns
        -------
        dict with "termination_penalty" and "goal_reached" raw signals.
        Only one will be nonzero.
        """
        cfg = self.cfg
        sparse: dict[str, float] = {}
        if success:
            sparse["goal_reached"] = 1.0
            sparse["termination_penalty"] = 0.0
        else:
            sparse["goal_reached"] = 0.0
            sparse["termination_penalty"] = 1.0
        return sparse


# ── Offline batch API (unchanged) ────────────────────────────────────────────

def compute_episode_rewards(
    episode: dict[str, Any],
    cfg: RewardConfig | None = None,
) -> dict[str, Any]:
    """Compute per-step rewards for a single episode (offline / batch).

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
        "components" : dict[str, (T,) ndarray] — individual raw (unweighted) signals
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

    # 1. Time penalty (raw signal = 1.0 everywhere; weight carries sign)
    components["time_penalty"] = np.ones(T, dtype=np.float32)

    # 2. Termination penalty (raw = 1.0 on last step of failed episode)
    term = np.zeros(T, dtype=np.float32)
    if not success:
        term[-1] = 1.0
    components["termination_penalty"] = term

    # 3. Goal reached (raw = 1.0 on last step of successful episode)
    goal = np.zeros(T, dtype=np.float32)
    if success:
        goal[-1] = 1.0
    components["goal_reached"] = goal

    # 4. Goal approach
    dist_xy = np.linalg.norm(goal_rel[:, :2], axis=-1)
    approach = np.zeros(T, dtype=np.float32)
    approach[1:] = dist_xy[:-1] - dist_xy[1:]
    components["goal_approach"] = approach

    # 5. Command tracking
    vel_achieved = np.column_stack([
        policy_obs[:, 0],
        policy_obs[:, 1],
        policy_obs[:, 5],
    ])
    tracking_error = np.linalg.norm(vel_cmd - vel_achieved, axis=-1)
    components["command_tracking"] = -tracking_error

    # 6. Command smoothness
    smoothness = np.zeros(T, dtype=np.float32)
    if T > 1:
        delta = np.linalg.norm(np.diff(vel_cmd, axis=0), axis=-1)
        smoothness[1:] = -delta
    components["command_smoothness"] = smoothness

    # 7. Non-foot body contact
    non_foot_forces = cf[:, NON_FOOT_BODY_INDICES, :]
    non_foot_mag = np.linalg.norm(non_foot_forces, axis=-1)
    non_foot_mag[non_foot_mag < cfg.body_contact_threshold_n] = 0.0
    components["body_contact"] = -non_foot_mag.sum(axis=-1).astype(np.float32)

    # 8. Uprightness
    grav_z = policy_obs[:, 8]
    components["uprightness"] = np.clip(-grav_z, 0.0, 1.0).astype(np.float32)

    # Weighted total
    weights = {
        "time_penalty": cfg.w_time_penalty,
        "termination_penalty": cfg.w_termination,
        "goal_reached": cfg.w_goal_reached,
        "goal_approach": cfg.w_goal_approach,
        "command_tracking": cfg.w_command_tracking,
        "command_smoothness": cfg.w_command_smoothness,
        "body_contact": cfg.w_body_contact,
        "uprightness": cfg.w_uprightness,
    }
    total = np.zeros(T, dtype=np.float32)
    for name, arr in components.items():
        total += weights[name] * arr

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
