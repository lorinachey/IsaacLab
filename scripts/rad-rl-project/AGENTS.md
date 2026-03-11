# AGENTS.md — rad-rl-project

This file is the canonical briefing for any AI agent assisting on this project.
Read it in full before making any changes. Key decisions and hard-won debugging
knowledge are marked **[IMPORTANT]** — do not reverse them without understanding why.

---

## 1. What this project is

**Goal:** Collect a human-teleoperated demonstration dataset of Boston Dynamics Spot
navigating a warehouse, then use those demonstrations to train a high-level navigation
policy with Reinforcement Learning from Demonstrations (RfD) or Imitation Learning.

**Current stage:** Demo collection is complete and working. The next stage is training
a high-level policy that takes camera + state observations and outputs velocity commands
(which the low-level locomotion policy then executes).

**System overview:**

```
Xbox controller
    │  [v_x, v_y, ω_z] velocity command
    ▼
collect_demos.py
    │  injects command into 48-dim obs vector (indices 9–11)
    ▼
Low-level locomotion policy (MLP, 50 Hz)
    │  outputs 12 joint position targets
    ▼
Isaac Sim / PhysX  ←  spot-in-simple-warehouse.usd
    │  robot state, camera frames
    ▼
HDF5 dataset  (datasets/warehouse_demos_<timestamp>.hdf5)
```

---

## 2. File map

```
scripts/rad-rl-project/
├── collect_demos.py              # Main collection script — primary file
├── spot-in-simple-warehouse.usd  # Warehouse scene with collision geometry
├── environment.yml               # Conda env spec (conda env create -f ...)
├── README.md                     # Human-facing setup & usage docs
├── AGENTS.md                     # This file — AI agent briefing
├── utils.ipynb                   # HuggingFace checkpoint upload
├── datasets/
│   └── warehouse_demos_<ts>.hdf5 # Collected episode data
└── tests/
    ├── validate_dataset.py        # CLI validation: 13 structural/semantic checks
    └── explore_dataset.ipynb      # Jupyter visualisation notebook (9 sections)
```

**Policy checkpoint** (trained, do not re-train unless explicitly asked):
```
logs/rsl_rl/spot_flat/2026-03-09_15-47-17/model_19999.pt
```
Also mirrored on HuggingFace: `lorinachey/rad-rl-checkpoints` →
`checkpoints/spot_flat/model_19999.pt`

---

## 3. Environment

- **Conda environment name:** `isaaclab`
- **Isaac Lab repo root:** `/home/lorin-cairo/Documents/IsaacLab/`
- **Launch wrapper:** `./isaaclab.sh -p <script.py> [args]`  — always use this,
  never `python` directly. It sets up the Isaac Sim / Carbonite paths.
- **CUDA:** Required. Policy runs on `cuda:0` by default.
- **Controller:** Xbox 360-compatible, wired, via Linux `xpad` kernel driver.
  User must be in the `input` group (`sudo usermod -aG input $USER`).

---

## 4. How to run collection

```bash
./isaaclab.sh -p scripts/rad-rl-project/collect_demos.py \
    --checkpoint logs/rsl_rl/spot_flat/2026-03-09_15-47-17/model_19999.pt
```

Output is automatically stamped: `datasets/warehouse_demos_<YYYYMMDD_HHMMSS>.hdf5`.
See `README.md` for the full argument table.

---

## 5. [IMPORTANT] Hard-won technical decisions

These were all debugged over many iterations. Do not change them without understanding
the reasoning.

### 5.1 Joystick axis negation

```python
# collect_demos.py  ~line 368
vel_cmd = torch.stack([
    vel_cmd[0].clamp(-2.0, 3.0),    # v_x: no inversion needed
    (-vel_cmd[1]).clamp(-1.5, 1.5), # v_y: NEGATED — raw axis is inverted vs robot frame
    (-vel_cmd[2]).clamp(-2.0, 2.0),  # ω_z: NEGATED — raw axis is inverted vs robot frame
])
```

`Se2Gamepad` returns raw axis values where left-stick-right = positive `v_y` but the
simulator's Y-axis convention means left-stick-right should produce *negative* `v_y`
(strafe right). Same issue applies to right-stick turning. **These negations are
correct and intentional.** Removing them will invert controls AND corrupt training
labels (the logged `velocity_cmd` must match the human's intent).

### 5.2 Velocity command injection into policy obs

```python
# collect_demos.py  ~line 384
VEL_CMD_OBS_SLICE = slice(9, 12)   # indices in the 48-dim policy obs
policy_obs[0, VEL_CMD_OBS_SLICE] = vel_cmd
```

The environment's internal command manager generates its own random velocity commands.
We *overwrite* indices 9–11 with the human's gamepad command before every policy
inference call. The **same `vel_cmd` tensor** is then logged as
`action/velocity_cmd`. This is why `validate_dataset.py` check #7 can assert
`policy_obs[:, 9:12] == velocity_cmd` to within floating-point precision.

### 5.3 Curriculum and termination disabled for USD terrain

```python
env_cfg.curriculum = None
env_cfg.terminations.terrain_out_of_bounds = None
env_cfg.terminations.time_out = None
```

The training env uses a procedural terrain generator; the warehouse uses a static USD.
These three managers crash at reset when `terrain_generator is None`. They must remain
`None` for the warehouse scene.

### 5.4 Reset events must be explicitly configured

Isaac Lab's `Articulation.reset()` restores `default_root_state`, which is captured
from the USD prim position at startup — **not** from `init_state.pos`. Without an
explicit `reset_base` event, Spot teleports back to the warehouse origin
(inside a wall) every episode and immediately falls.

```python
# reset_base: teleports Spot to init_state.pos ± 0.5 m, fixed 180° yaw, zero velocity
env_cfg.events.reset_base = EventTermCfg(
    func=reset_root_state_uniform,
    params={
        "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (3.14159, 3.14159)},
        "velocity_range": {},   # zero velocity — non-empty dict caused Spot to flip at spawn
    },
)
# reset_robot_joints: writes joint positions to PhysX at reset; without this,
# joints remain at end-of-episode positions (Spot starts mid-fall next episode)
env_cfg.events.reset_robot_joints = EventTermCfg(
    func=reset_joints_around_default,
    params={"position_range": (-0.1, 0.1), "velocity_range": (0.0, 0.0)},
)
env_cfg.events.push_robot = None   # must be explicitly disabled
```

### 5.5 Spawn position

```python
env_cfg.scene.robot.init_state.pos = (6.0, -8.5, 0.5)
```

This places Spot on the warehouse floor with 0.5 m clearance. The warehouse USD has
its floor near Z = 0 world-space. Do not use Z = 0 or lower — Spot spawns inside
the floor and the body-contact termination triggers immediately.

### 5.6 Gamepad dead zone

`dead_zone=0.12` in `Se2GamepadCfg`. The physical controller has ~0.09 stick drift
at rest; 0.12 clears it. Do not lower this below 0.10.

---

## 6. Policy and observation space

**Architecture:** MLP [512 → 256 → 128], ELU activations  
**Algorithm:** PPO via RSL-RL  
**Policy frequency:** 50 Hz (`POLICY_HZ = 50`)  
**Logging frequency:** 10 Hz (`LOG_HZ = 10`, `LOG_EVERY_N = 5`)

### 48-dim observation vector layout

| Indices | Content | Notes |
|---------|---------|-------|
| 0–2 | `base_lin_vel` (3) | Robot frame |
| 3–5 | `base_ang_vel` (3) | Robot frame |
| 6–8 | `projected_gravity` (3) | Tilt indicator |
| **9–11** | **`velocity_cmd` (3)** | **[v_x, v_y, ω_z] — injected from gamepad** |
| 12–23 | `joint_pos` (12) | All 12 joints |
| 24–35 | `joint_vel` (12) | All 12 joints |
| 36–47 | `actions` (12) | Previous policy output |

### Velocity command training ranges

| Axis | Min | Max |
|------|-----|-----|
| v_x (forward) | −2.0 m/s | +3.0 m/s |
| v_y (lateral) | −1.5 m/s | +1.5 m/s |
| ω_z (yaw) | −2.0 rad/s | +2.0 rad/s |

Note: v_x is *asymmetric* — the policy was trained to go faster forward than backward.

### 12-joint order (approximate Isaac Lab Spot ordering)

```
FL_hx, FL_hy, FL_kn,   # Front-left:  hip-abduction, hip-flexion, knee
FR_hx, FR_hy, FR_kn,   # Front-right
RL_hx, RL_hy, RL_kn,   # Rear-left
RR_hx, RR_hy, RR_kn    # Rear-right
```

---

## 7. HDF5 dataset structure

```
<file>.hdf5
├── attrs: session_timestamp (str)
└── episode_N/
    ├── attrs: success (bool), episode_length (int),
    │         goal_position_world (float32[3]), timestamp (str ISO-8601)
    ├── obs/
    │   ├── camera_rgb       (T, 240, 320, 3)  uint8
    │   ├── base_pose        (T, 7)   float32  — pos(3) + quat_wxyz(4)
    │   ├── base_lin_vel     (T, 3)   float32  — world frame
    │   ├── base_ang_vel     (T, 3)   float32  — world frame
    │   ├── joint_pos        (T, 12)  float32
    │   ├── joint_vel        (T, 12)  float32
    │   ├── contact_forces   (T, 17, 3) float32  — 17 bodies, net force world frame
    │   ├── goal_relative    (T, 3)   float32  — goal_world − base_pos
    │   └── policy_obs       (T, 48)  float32
    └── action/
        ├── velocity_cmd     (T, 3)   float32  — gamepad [v_x, v_y, ω_z], post-negation
        └── joint_targets    (T, 12)  float32  — raw policy output
```

`T` = number of 10 Hz steps. `contact_forces` has 17 bodies (full Spot articulation),
not just the 4 feet — identify foot bodies by finding the 4 with highest mean force.

---

## 8. Testing and validation

```bash
# Validate structural integrity and semantic consistency of any HDF5 file
python scripts/rad-rl-project/tests/validate_dataset.py
# (auto-picks latest .hdf5 in datasets/)

# Or specify explicitly:
python scripts/rad-rl-project/tests/validate_dataset.py \
    --dataset scripts/rad-rl-project/datasets/warehouse_demos_20260311_122230.hdf5
```

Returns exit code 0 (all pass) or 1 (failures). The 13 checks include:
- All dataset keys, shapes, dtypes present
- `velocity_cmd` within training ranges
- `policy_obs[9:12] == velocity_cmd` exactly (data-integrity invariant)
- `goal_relative == goal_world - base_pos` (tolerance 1e-3 m)
- Quaternion norms ≈ 1.0
- Camera not all-black
- No position jumps > 1 m at 10 Hz

**Visualisation:**
Open `tests/explore_dataset.ipynb` in Jupyter (kernel: `isaaclab`). Sections:
1. Dataset overview (episode lengths, velocity distributions)
2. Interactive camera frame scrubber
3. XY bird's-eye trajectory map
4. Velocity commanded vs actual (with training-range bands)
5. Policy-obs / velocity-cmd consistency scatter
6. Joint position heatmaps (measured + policy targets)
7. Contact force heatmaps
8. Frame strip (8 evenly-spaced frames)
9. Goal-distance over time

---

## 9. Known issues and status

| Issue | Status |
|-------|--------|
| All collected episodes end with failure (robot falls) | **Known** — Spot falls on warehouse obstacles; collect more/longer episodes |
| No successful episodes in current dataset | **Known** — goal not yet reached; data is still valid for IL pre-training |
| `time_out` termination disabled | **Intentional** — episode ends only on fall or human button press |
| `terrain_out_of_bounds` disabled | **Intentional** — USD terrain has no procedural bounds |

---

## 10. What to work on next

The demo collection infrastructure is complete. Likely next steps:

1. **Collect more episodes** — run more sessions until the dataset has enough
   successful episodes (goal reached) for imitation learning.
2. **High-level policy training** — train a navigation policy that:
   - Takes `camera_rgb` + `goal_relative` (or `base_pose`) as input
   - Outputs `velocity_cmd` [v_x, v_y, ω_z]
   - Can be trained with BC, GAIL, or IQL on the HDF5 dataset
3. **Identify foot body indices** — the 17 contact-force bodies need to be mapped
   to named links so foot-contact rewards can be computed offline.
4. **Reward shaping offline** — all raw state components needed to define training
   rewards are stored in the dataset (no re-simulation needed).

---

## 11. Key source files outside this directory

| File | Relevance |
|------|-----------|
| `source/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/` | Spot env cfg (`SpotFlatEnvCfg`, `SpotFlatEnvCfg_PLAY`) |
| `source/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/` | `reset_joints_around_default` function |
| `source/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/agents/rsl_rl_ppo_cfg.py` | `SpotFlatPPORunnerCfg` |
| `source/isaaclab/devices/gamepad/se2_gamepad.py` | `Se2Gamepad`, `Se2GamepadCfg` |
| `source/isaaclab/envs/mdp/events.py` | `reset_root_state_uniform` |
