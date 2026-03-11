"""Validate an HDF5 dataset produced by collect_demos.py.

Checks performed
----------------
1.  File-level attributes (session_timestamp present).
2.  Episode group structure: every expected dataset path exists.
3.  Episode-level attributes (success, episode_length, goal_position_world, timestamp).
4.  Shape consistency: each dataset matches the expected shape template.
5.  Dtype consistency: each dataset has the expected dtype.
6.  Velocity command ranges: vel_cmd values are within training limits.
7.  Policy-obs / velocity-cmd consistency:
        policy_obs[:, 9:12]  ==  velocity_cmd  (the vel-cmd obs slice that the
        script injects before policy inference, so they must match exactly).
8.  Goal-relative consistency:
        goal_relative  ==  goal_position_world - base_pose[:, :3]
        (within a small floating-point tolerance).
9.  Base quaternion normalisation: ||q|| ≈ 1.0 for every step.
10. Camera sanity: RGB frames are uint8 and not uniformly black.
11. Contact-forces shape: (T, N_bodies, 3) — confirms 3-D force vectors.
12. Episode-length attribute matches actual stored length.
13. Temporal continuity: base-position does not jump > 1 m between consecutive
    10 Hz samples (catches teleportation or corrupt data).

Usage
-----
    python scripts/rad-rl-project/tests/validate_dataset.py \\
        --dataset scripts/rad-rl-project/datasets/warehouse_demos_20260311_122230.hdf5

    # or glob the latest file automatically:
    python scripts/rad-rl-project/tests/validate_dataset.py
"""

import argparse
import glob
import os
import sys

import h5py
import numpy as np

# ── Constants mirrored from collect_demos.py ──────────────────────────────────
VEL_CMD_OBS_SLICE = slice(9, 12)   # indices in the 48-dim policy obs

EXPECTED_DATASETS = {
    # key: (ndim, dtype_kind)   — 'f' = float, 'u' = unsigned int
    "obs/camera_rgb":     (4, "u"),   # (T, H, W, 3)
    "obs/base_pose":      (2, "f"),   # (T, 7)
    "obs/base_lin_vel":   (2, "f"),   # (T, 3)
    "obs/base_ang_vel":   (2, "f"),   # (T, 3)
    "obs/joint_pos":      (2, "f"),   # (T, 12)
    "obs/joint_vel":      (2, "f"),   # (T, 12)
    "obs/contact_forces": (3, "f"),   # (T, N_bodies, 3)
    "obs/goal_relative":  (2, "f"),   # (T, 3)
    "obs/policy_obs":     (2, "f"),   # (T, 48)
    "action/velocity_cmd":  (2, "f"),   # (T, 3)
    "action/joint_targets": (2, "f"),   # (T, 12)
}

EXPECTED_DIMS = {
    "obs/base_pose":        7,
    "obs/base_lin_vel":     3,
    "obs/base_ang_vel":     3,
    "obs/joint_pos":        12,
    "obs/joint_vel":        12,
    "obs/goal_relative":    3,
    "obs/policy_obs":       48,
    "obs/camera_rgb":       3,   # trailing channel dim
    "obs/contact_forces":   3,   # trailing force-component dim
    "action/velocity_cmd":  3,
    "action/joint_targets": 12,
}

# Training velocity command ranges (from SpotFlatEnvCfg)
VEL_CMD_RANGES = [
    (-2.0, 3.0),   # v_x  m/s
    (-1.5, 1.5),   # v_y  m/s
    (-2.0, 2.0),   # omega_z  rad/s
]

# Maximum plausible 10 Hz base-position jump (metres)
MAX_POSITION_JUMP_M = 1.0

# Goal-relative tolerance (metres) — float32 arithmetic headroom
GOAL_REL_TOL = 1e-3


def _pass(msg: str) -> str:
    return f"  \033[32m[PASS]\033[0m {msg}"


def _fail(msg: str) -> str:
    return f"  \033[31m[FAIL]\033[0m {msg}"


def _warn(msg: str) -> str:
    return f"  \033[33m[WARN]\033[0m {msg}"


def validate(hdf5_path: str) -> bool:
    print(f"\n{'='*70}")
    print(f"  Validating: {hdf5_path}")
    print(f"{'='*70}\n")

    all_passed = True

    with h5py.File(hdf5_path, "r") as f:

        # ── 1. File-level attributes ───────────────────────────────────────
        print("[ File attributes ]")
        if "session_timestamp" in f.attrs:
            print(_pass(f"session_timestamp = '{f.attrs['session_timestamp']}'"))
        else:
            print(_fail("Missing file attribute: session_timestamp"))
            all_passed = False

        # ── 2. Episode inventory ───────────────────────────────────────────
        episode_keys = sorted(f.keys())
        n_episodes = len(episode_keys)
        print(f"\n[ Episodes: {n_episodes} total ]")
        if n_episodes == 0:
            print(_fail("No episodes found in file — nothing to validate."))
            return False

        n_success = sum(1 for k in episode_keys if f[k].attrs.get("success", False))
        print(_pass(f"Found {n_episodes} episodes  ({n_success} success, {n_episodes - n_success} failure/terminated)"))

        episode_lengths = []
        per_episode_errors = 0

        for ep_key in episode_keys:
            ep = f[ep_key]
            ep_errors = []

            # ── 3. Episode attributes ──────────────────────────────────────
            for attr in ("success", "episode_length", "goal_position_world", "timestamp"):
                if attr not in ep.attrs:
                    ep_errors.append(f"missing attribute '{attr}'")

            # ── 4–5. Dataset presence, ndim, dtype ────────────────────────
            for ds_path, (expected_ndim, dtype_kind) in EXPECTED_DATASETS.items():
                if ds_path not in ep:
                    ep_errors.append(f"missing dataset '{ds_path}'")
                    continue

                ds = ep[ds_path]

                if ds.ndim != expected_ndim:
                    ep_errors.append(
                        f"'{ds_path}': expected ndim={expected_ndim}, got {ds.ndim}"
                    )

                if ds.dtype.kind != dtype_kind:
                    ep_errors.append(
                        f"'{ds_path}': expected dtype kind '{dtype_kind}', got '{ds.dtype.kind}' ({ds.dtype})"
                    )

                # Trailing-dimension check
                if ds_path in EXPECTED_DIMS and ds.ndim >= 2:
                    if ds.shape[-1] != EXPECTED_DIMS[ds_path]:
                        ep_errors.append(
                            f"'{ds_path}': expected last dim={EXPECTED_DIMS[ds_path]}, got {ds.shape[-1]}"
                        )

            if ep_errors:
                for err in ep_errors:
                    print(_fail(f"[{ep_key}] {err}"))
                all_passed = False
                per_episode_errors += 1
                continue  # skip value checks for malformed episodes

            # ── Load arrays for value-level checks ────────────────────────
            T = ep["obs/base_pose"].shape[0]
            episode_lengths.append(T)

            vel_cmd    = ep["action/velocity_cmd"][()]       # (T, 3)
            policy_obs = ep["obs/policy_obs"][()]            # (T, 48)
            base_pose  = ep["obs/base_pose"][()]             # (T, 7)
            goal_rel   = ep["obs/goal_relative"][()]         # (T, 3)
            camera_rgb = ep["obs/camera_rgb"][()]            # (T, H, W, 3)
            goal_world = ep.attrs["goal_position_world"]     # (3,)
            ep_len_attr = int(ep.attrs["episode_length"])

            # ── 6. Velocity command ranges ────────────────────────────────
            cmd_ok = True
            for i, (lo, hi) in enumerate(VEL_CMD_RANGES):
                axis_vals = vel_cmd[:, i]
                if axis_vals.min() < lo - 1e-4 or axis_vals.max() > hi + 1e-4:
                    ep_errors.append(
                        f"vel_cmd axis {i}: range [{axis_vals.min():.4f}, {axis_vals.max():.4f}]"
                        f" outside training limits [{lo}, {hi}]"
                    )
                    cmd_ok = False
            if cmd_ok:
                pass  # reported per-episode summary below

            # ── 7. policy_obs[9:12] == velocity_cmd ──────────────────────
            obs_vel_slice = policy_obs[:, VEL_CMD_OBS_SLICE]   # (T, 3)
            max_cmd_diff = np.abs(obs_vel_slice - vel_cmd).max()
            if max_cmd_diff > 1e-5:
                ep_errors.append(
                    f"policy_obs[9:12] ≠ velocity_cmd  (max |diff|={max_cmd_diff:.6f})"
                )

            # ── 8. Goal-relative consistency ──────────────────────────────
            base_pos = base_pose[:, :3]                          # (T, 3)
            expected_goal_rel = goal_world[np.newaxis, :] - base_pos
            goal_rel_err = np.abs(goal_rel - expected_goal_rel).max()
            if goal_rel_err > GOAL_REL_TOL:
                ep_errors.append(
                    f"goal_relative ≠ goal_world - base_pos  (max |diff|={goal_rel_err:.6f} m)"
                )

            # ── 9. Quaternion normalisation ───────────────────────────────
            quats = base_pose[:, 3:]   # (T, 4)  wxyz
            norms = np.linalg.norm(quats, axis=-1)
            if not np.allclose(norms, 1.0, atol=1e-3):
                bad = int(np.sum(np.abs(norms - 1.0) > 1e-3))
                ep_errors.append(f"quaternion not normalised at {bad}/{T} steps (min norm={norms.min():.5f})")

            # ── 10. Camera sanity ─────────────────────────────────────────
            mean_brightness = camera_rgb.mean()
            if mean_brightness < 1.0:
                ep_errors.append(
                    f"camera_rgb appears all-black (mean pixel={mean_brightness:.2f})"
                )

            # ── 11. Contact-forces trailing dim ───────────────────────────
            if ep["obs/contact_forces"].shape[-1] != 3:
                ep_errors.append(
                    f"contact_forces last dim={ep['obs/contact_forces'].shape[-1]}, expected 3"
                )

            # ── 12. Episode-length attribute matches actual length ────────
            if ep_len_attr != T:
                ep_errors.append(
                    f"episode_length attribute={ep_len_attr} but dataset has T={T} steps"
                )

            # ── 13. Temporal continuity ───────────────────────────────────
            if T > 1:
                deltas = np.linalg.norm(np.diff(base_pos, axis=0), axis=-1)  # (T-1,)
                max_delta = deltas.max()
                if max_delta > MAX_POSITION_JUMP_M:
                    bad_step = int(np.argmax(deltas))
                    ep_errors.append(
                        f"position jump of {max_delta:.3f} m at step {bad_step} "
                        f"(threshold={MAX_POSITION_JUMP_M} m)"
                    )

            # ── Per-episode summary ───────────────────────────────────────
            if ep_errors:
                for err in ep_errors:
                    print(_fail(f"[{ep_key}] {err}"))
                all_passed = False
                per_episode_errors += 1
            else:
                vel_summary = (
                    f"vx=[{vel_cmd[:,0].min():+.2f},{vel_cmd[:,0].max():+.2f}]  "
                    f"vy=[{vel_cmd[:,1].min():+.2f},{vel_cmd[:,1].max():+.2f}]  "
                    f"wz=[{vel_cmd[:,2].min():+.2f},{vel_cmd[:,2].max():+.2f}]"
                )
                print(_pass(
                    f"[{ep_key}]  T={T:4d} steps  success={bool(ep.attrs['success'])}  {vel_summary}"
                ))

        # ── Summary ───────────────────────────────────────────────────────
        print(f"\n{'─'*70}")
        if episode_lengths:
            lengths = np.array(episode_lengths)
            print(
                f"  Episode lengths:  mean={lengths.mean():.1f}  "
                f"min={lengths.min()}  max={lengths.max()}  total={lengths.sum()} steps"
            )

        if per_episode_errors == 0 and all_passed:
            print(f"\n  \033[32m✓  All checks passed ({n_episodes} episodes).\033[0m\n")
        else:
            print(f"\n  \033[31m✗  {per_episode_errors}/{n_episodes} episodes had errors.\033[0m\n")

    return all_passed


def find_latest_dataset(datasets_dir: str) -> str | None:
    pattern = os.path.join(datasets_dir, "*.hdf5")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_datasets_dir = os.path.join(script_dir, "..", "datasets")

    parser = argparse.ArgumentParser(description="Validate a collect_demos HDF5 dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=(
            "Path to the HDF5 file to validate. "
            "If omitted, the most recently modified .hdf5 in the datasets/ folder is used."
        ),
    )
    args = parser.parse_args()

    if args.dataset:
        hdf5_path = args.dataset
    else:
        hdf5_path = find_latest_dataset(default_datasets_dir)
        if hdf5_path is None:
            print(f"[ERROR] No .hdf5 files found in {default_datasets_dir}")
            sys.exit(1)
        print(f"[INFO] No --dataset specified; using latest: {os.path.basename(hdf5_path)}")

    if not os.path.isfile(hdf5_path):
        print(f"[ERROR] File not found: {hdf5_path}")
        sys.exit(1)

    passed = validate(hdf5_path)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
