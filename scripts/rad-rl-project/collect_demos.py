# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Collect human-teleoperated demonstrations with Spot in the warehouse environment.

The script spawns Spot in the warehouse USD scene, accepts velocity commands from
an Xbox controller, runs the trained low-level locomotion policy to execute those
commands, and logs raw robot state, camera frames, and actions to HDF5 at 10 Hz.

No reward function is computed during collection — all raw state components needed
to define rewards offline are stored so the reward design can iterate independently.

Xbox controller bindings:
    Left stick X/Y      Strafe left-right / forward-backward (lin_x, lin_y)
    Right stick X       Turn left-right (omega_z)
    A button            Mark episode as SUCCESS and reset
    B button            Mark episode as FAILURE / force reset
    START               Quit collection

Usage:
    ./isaaclab.sh -p scripts/rad-rl-project/collect_demos.py \\
        --checkpoint logs/rsl_rl/spot_flat/2026-03-09_15-47-17/model_19999.pt \\
        --dataset_file datasets/warehouse_demos.hdf5 \\
        --enable_cameras
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect teleoperated Spot demos in the warehouse scene.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained locomotion policy checkpoint.")
parser.add_argument(
    "--dataset_file",
    type=str,
    default="./datasets/warehouse_demos.hdf5",
    help="Output HDF5 file path.",
)
parser.add_argument("--num_episodes", type=int, default=0, help="Episodes to collect before quitting (0 = infinite).")
parser.add_argument("--camera_height", type=int, default=240, help="Camera image height in pixels.")
parser.add_argument("--camera_width", type=int, default=320, help="Camera image width in pixels.")
parser.add_argument(
    "--goal_position",
    type=float,
    nargs=3,
    default=[8.0, 0.0, 0.0],
    metavar=("X", "Y", "Z"),
    help="Goal marker position in world frame (metres). Default: 8.0 0.0 0.0",
)
parser.add_argument(
    "--success_radius",
    type=float,
    default=0.5,
    help="Distance to goal (m) for automatic success detection.",
)
parser.add_argument(
    "--min_episode_steps",
    type=int,
    default=10,
    help="Minimum 10 Hz steps required to save an episode (discards accidental resets).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True  # always required for camera collection
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import os
from collections import defaultdict
from datetime import datetime

import carb.input
import gymnasium as gym
import h5py
import numpy as np
import torch
from packaging import version
from rsl_rl.runners import OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.devices.gamepad import Se2Gamepad, Se2GamepadCfg
from isaaclab.sensors import CameraCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.agents.rsl_rl_ppo_cfg import SpotFlatPPORunnerCfg
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# ── Constants ──────────────────────────────────────────────────────────────────
# Slice of the 48-dim policy obs vector corresponding to velocity commands:
#   [0:3]  base_lin_vel   [3:6]  base_ang_vel  [6:9]  projected_gravity
#   [9:12] velocity_cmds  [12:24] joint_pos     [24:36] joint_vel  [36:48] actions
VEL_CMD_OBS_SLICE = slice(9, 12)

POLICY_HZ = 50        # SpotFlatEnvCfg: sim.dt=0.002, decimation=10 → 50 Hz
LOG_HZ = 10           # camera / dataset logging rate
LOG_EVERY_N = POLICY_HZ // LOG_HZ  # log every 5 policy steps

WAREHOUSE_USD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "spot-in-simple-warehouse.usd")


# ── Data collector ─────────────────────────────────────────────────────────────
class HDF5DataCollector:
    """Buffers per-step data in memory and writes completed episodes to HDF5.

    Each episode is stored as a group ``episode_<N>`` with the following layout::

        episode_N/
            obs/
                camera_rgb          (T, H, W, 3)  uint8
                base_pose           (T, 7)         float32  pos(3) + quat_wxyz(4)
                base_lin_vel        (T, 3)         float32  world frame
                base_ang_vel        (T, 3)         float32  world frame
                joint_pos           (T, 12)        float32
                joint_vel           (T, 12)        float32
                contact_forces      (T, N_feet, 3) float32  net forces world frame
                goal_relative       (T, 3)         float32  goal minus robot pos
                policy_obs          (T, 48)        float32  raw locomotion policy input
            action/
                velocity_cmd        (T, 3)         float32  gamepad [v_x, v_y, omega_z]
                joint_targets       (T, 12)        float32  raw policy output
            attrs:
                success             bool
                episode_length      int
                goal_position_world (3,)  float32
                timestamp           str   ISO-8601 wall-clock time at flush
    """

    def __init__(self, filepath: str, session_timestamp: str):
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self._file = h5py.File(filepath, "a")
        self._episode_idx = len(self._file.keys())
        self._buf: dict[str, list] = defaultdict(list)
        self._file.attrs["session_timestamp"] = session_timestamp
        print(f"[DataCollector] Writing to {filepath} (starting at episode {self._episode_idx})")

    def buffer_step(
        self,
        camera_rgb: np.ndarray,      # (H, W, 3) uint8
        base_pose: np.ndarray,        # (7,)
        base_lin_vel: np.ndarray,     # (3,)
        base_ang_vel: np.ndarray,     # (3,)
        joint_pos: np.ndarray,        # (12,)
        joint_vel: np.ndarray,        # (12,)
        contact_forces: np.ndarray,   # (N_feet, 3)
        goal_relative: np.ndarray,    # (3,)
        policy_obs: np.ndarray,       # (48,)
        velocity_cmd: np.ndarray,     # (3,)
        joint_targets: np.ndarray,    # (12,)
    ):
        self._buf["obs/camera_rgb"].append(camera_rgb)
        self._buf["obs/base_pose"].append(base_pose)
        self._buf["obs/base_lin_vel"].append(base_lin_vel)
        self._buf["obs/base_ang_vel"].append(base_ang_vel)
        self._buf["obs/joint_pos"].append(joint_pos)
        self._buf["obs/joint_vel"].append(joint_vel)
        self._buf["obs/contact_forces"].append(contact_forces)
        self._buf["obs/goal_relative"].append(goal_relative)
        self._buf["obs/policy_obs"].append(policy_obs)
        self._buf["action/velocity_cmd"].append(velocity_cmd)
        self._buf["action/joint_targets"].append(joint_targets)

    def flush_episode(self, success: bool, goal_position_world: np.ndarray) -> int:
        """Write buffered steps as a new episode group and clear the buffer."""
        if not self._buf:
            return self._episode_idx

        n_steps = len(next(iter(self._buf.values())))
        grp = self._file.create_group(f"episode_{self._episode_idx}")

        for key, frames in self._buf.items():
            arr = np.stack(frames, axis=0)
            opts = {"compression": "gzip", "compression_opts": 4 if "camera" in key else 1}
            grp.create_dataset(key, data=arr, **opts)

        grp.attrs["success"] = bool(success)
        grp.attrs["episode_length"] = n_steps
        grp.attrs["goal_position_world"] = goal_position_world
        grp.attrs["timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        self._file.flush()
        print(f"[DataCollector] Episode {self._episode_idx} saved — {n_steps} steps @ {LOG_HZ} Hz, success={success}")

        self._episode_idx += 1
        self._buf.clear()
        return self._episode_idx - 1

    def discard_episode(self):
        """Discard buffered steps without writing (e.g. accidental resets)."""
        n = len(next(iter(self._buf.values()))) if self._buf else 0
        self._buf.clear()
        if n:
            print(f"[DataCollector] Discarded episode ({n} steps — below minimum).")

    def close(self):
        self._file.close()


# ── Environment configuration ──────────────────────────────────────────────────
def build_env_cfg(args):
    """Return a SpotFlatEnvCfg modified for the warehouse scene."""
    env_cfg = parse_env_cfg("Isaac-Velocity-Flat-Spot-Play-v0", device=args.device, num_envs=1)

    # Replace procedural terrain generator with the warehouse USD
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=WAREHOUSE_USD,
        env_spacing=10.0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    # Attach a forward-facing camera to Spot's body link
    env_cfg.scene.camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/body/front_cam",
        update_period=1.0 / LOG_HZ,
        height=args.camera_height,
        width=args.camera_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        # ~51 cm forward of body centre, facing forward (ROS convention)
        offset=CameraCfg.OffsetCfg(pos=(0.510, 0.0, 0.015), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
    )

    # Visual goal marker — bright green sphere at the goal position
    goal_pos = tuple(args.goal_position)
    env_cfg.scene.goal_marker = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GoalMarker",
        spawn=sim_utils.SphereCfg(
            radius=0.3,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), opacity=0.85),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=goal_pos),
    )

    # Disable timeout — episodes end via human gamepad input or termination
    env_cfg.terminations.time_out = None

    return env_cfg


# ── Policy loading ─────────────────────────────────────────────────────────────
def load_policy(env_wrapped, checkpoint_path: str, device: str):
    """Load the trained low-level locomotion policy from a checkpoint."""
    agent_cfg = SpotFlatPPORunnerCfg()
    agent_cfg.device = device
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    runner = OnPolicyRunner(env_wrapped, agent_cfg.to_dict(), log_dir=None, device=device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=device)
    print(f"[Policy] Loaded from: {checkpoint_path}")
    return policy, agent_cfg


# ── Main collection loop ───────────────────────────────────────────────────────
def run_collection(env_wrapped, policy, agent_cfg, gamepad: Se2Gamepad, collector: HDF5DataCollector, args):
    env = env_wrapped.unwrapped
    device = env.device

    goal_pos_world = np.array(args.goal_position, dtype=np.float32)
    goal_pos_tensor = torch.tensor(goal_pos_world[:2], device=device)  # XY only for distance check
    success_radius_sq = args.success_radius**2

    rsl_rl_version = version.parse(metadata.version("rsl-rl-lib"))

    # Gamepad button callbacks via shared flags dict
    flags = {"success": False, "failure": False, "quit": False}

    gamepad.add_callback(carb.input.GamepadInput.A, lambda: flags.update({"success": True}))
    gamepad.add_callback(carb.input.GamepadInput.B, lambda: flags.update({"failure": True}))
    gamepad.add_callback(carb.input.GamepadInput.START, lambda: flags.update({"quit": True}))

    obs, _ = env_wrapped.reset()
    gamepad.reset()

    episode_count = 0
    physics_step = 0   # policy steps within current episode
    data_step = 0      # 10 Hz samples within current episode

    print("\n[INFO] Collection running. Drive Spot with the Xbox controller.")
    print("       A = success  |  B = failure/reset  |  START = quit\n")

    while simulation_app.is_running() and not flags["quit"]:
        # ── 1. Read velocity command from gamepad ──────────────────────────
        vel_cmd = gamepad.advance()  # tensor (3,) on device: [v_x, v_y, omega_z]
        # Clamp to the ranges seen during training so that asymmetric axes
        # (e.g. lin_vel_x: [-2.0, +3.0]) are never commanded out-of-distribution.
        vel_cmd = torch.stack([
            vel_cmd[0].clamp(-2.0, 3.0),   # lin_vel_x: [-2.0, +3.0] m/s
            vel_cmd[1].clamp(-1.5, 1.5),   # lin_vel_y: [-1.5, +1.5] m/s
            vel_cmd[2].clamp(-2.0, 2.0),   # ang_vel_z: [-2.0, +2.0] rad/s
        ])

        # ── 2. Inject gamepad command into obs for the locomotion policy ───
        # The env's internal command manager may have resampled; we replace
        # the velocity command slice so the policy always follows the human.
        policy_obs = obs["policy"].clone()
        policy_obs[0, VEL_CMD_OBS_SLICE] = vel_cmd

        # ── 3. Policy inference → joint targets ────────────────────────────
        with torch.inference_mode():
            actions = policy({"policy": policy_obs, "critic": policy_obs})

        # ── 4. Step environment ────────────────────────────────────────────
        obs, _, dones, _ = env_wrapped.step(actions)
        physics_step += 1
        env_done = bool(dones[0])

        # ── 5. Log at 10 Hz (skip on env_done — state reflects reset) ─────
        if physics_step % LOG_EVERY_N == 0 and not env_done:
            robot = env.scene["robot"]
            camera = env.scene["camera"]
            contacts = env.scene["contact_forces"]

            # Base state
            base_pos = robot.data.root_pos_w[0].cpu().numpy()         # (3,)
            base_quat = robot.data.root_quat_w[0].cpu().numpy()        # (4,) wxyz
            base_lin = robot.data.root_lin_vel_w[0].cpu().numpy()      # (3,)
            base_ang = robot.data.root_ang_vel_w[0].cpu().numpy()      # (3,)
            j_pos = robot.data.joint_pos[0].cpu().numpy()              # (12,)
            j_vel = robot.data.joint_vel[0].cpu().numpy()              # (12,)

            # Contact forces — (num_envs, num_bodies, 3) → (num_bodies, 3)
            cf = contacts.data.net_forces_w[0].cpu().numpy()

            # Camera — Isaac Sim returns RGBA (H, W, 4); take RGB
            rgb = camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)

            # Goal-relative position (world frame, XYZ)
            goal_rel = goal_pos_world - base_pos

            collector.buffer_step(
                camera_rgb=rgb,
                base_pose=np.concatenate([base_pos, base_quat]),
                base_lin_vel=base_lin,
                base_ang_vel=base_ang,
                joint_pos=j_pos,
                joint_vel=j_vel,
                contact_forces=cf,
                goal_relative=goal_rel,
                policy_obs=policy_obs[0].cpu().numpy(),
                velocity_cmd=vel_cmd.cpu().numpy(),
                joint_targets=actions[0].cpu().numpy(),
            )
            data_step += 1

            # Auto-success: reached goal radius in XY plane
            robot_xy = robot.data.root_pos_w[0, :2]
            if float(torch.sum((robot_xy - goal_pos_tensor) ** 2)) < success_radius_sq:
                print(f"[INFO] Goal reached automatically.")
                flags["success"] = True

        # ── 6. Episode end: env termination or human button ────────────────
        human_success = flags["success"]
        human_failure = flags["failure"]

        if env_done or human_success or human_failure:
            # Env termination (fall, out-of-bounds) overrides human success
            success = human_success and not env_done

            if env_done:
                print("[INFO] Termination: robot fell or left bounds.")

            if data_step >= args.min_episode_steps:
                collector.flush_episode(success=success, goal_position_world=goal_pos_world)
                episode_count += 1
            else:
                collector.discard_episode()

            # Reset recurrent policy state if applicable
            if rsl_rl_version >= version.parse("4.0.0"):
                policy.reset(dones)

            # Clear episode state (env already auto-reset inside the wrapper)
            physics_step = 0
            data_step = 0
            flags["success"] = False
            flags["failure"] = False
            gamepad.reset()

            if args.num_episodes > 0 and episode_count >= args.num_episodes:
                print(f"[INFO] Target of {args.num_episodes} episodes reached. Exiting.")
                break


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    device = args_cli.device if args_cli.device else "cuda:0"
    checkpoint_path = retrieve_file_path(args_cli.checkpoint)

    # Stamp the output filename so each session produces a distinct file.
    # e.g. datasets/warehouse_demos.hdf5 → datasets/warehouse_demos_20260309_144216.hdf5
    session_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base, ext = os.path.splitext(args_cli.dataset_file)
    stamped_filepath = f"{base}_{session_ts}{ext}"

    # Build and create environment
    env_cfg = build_env_cfg(args_cli)
    env = gym.make("Isaac-Velocity-Flat-Spot-Play-v0", cfg=env_cfg)

    # Wrap for RSL-RL policy compatibility
    agent_cfg_tmp = SpotFlatPPORunnerCfg()
    env_wrapped = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg_tmp, "clip_actions", False))

    # Load locomotion policy
    policy, agent_cfg = load_policy(env_wrapped, checkpoint_path, device)

    # Xbox controller (Se2Gamepad outputs [v_x, v_y, omega_z])
    # v_x_sensitivity=3.0 maps full-forward stick to the training max of +3 m/s.
    # Commands are clamped to training ranges inside run_collection so that
    # full-backward stick (-3.0) is clipped to the trained minimum of -2.0 m/s.
    gamepad = Se2Gamepad(
        Se2GamepadCfg(
            v_x_sensitivity=3.0,    # lin_vel_x trained range: [-2.0, +3.0] m/s
            v_y_sensitivity=1.5,    # lin_vel_y trained range: [-1.5, +1.5] m/s
            omega_z_sensitivity=2.0,  # ang_vel_z trained range: [-2.0, +2.0] rad/s
            dead_zone=0.05,
            sim_device=device,
        )
    )

    # HDF5 data collector
    collector = HDF5DataCollector(stamped_filepath, session_timestamp=session_ts)

    try:
        run_collection(env_wrapped, policy, agent_cfg, gamepad, collector, args_cli)
    finally:
        collector.close()
        env.close()
        print(f"\n[INFO] Dataset saved to: {stamped_filepath}")


if __name__ == "__main__":
    main()
    simulation_app.close()
