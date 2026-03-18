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
├── rewards.py                    # Reward computation (offline batch + online step-by-step)
├── spot-in-simple-warehouse.usd  # Warehouse scene with collision geometry
├── environment.yml               # Conda env spec (conda env create -f ...)
├── README.md                     # Human-facing setup & usage docs
├── AGENTS.md                     # This file — AI agent briefing
├── utils.ipynb                   # HuggingFace checkpoint upload
├── datasets/
│   └── warehouse_demos_<ts>.hdf5 # Collected episode data (with reward/ group)
└── tests/
    ├── validate_dataset.py        # CLI validation: 14 structural/semantic checks
    └── explore_dataset.ipynb      # Jupyter visualisation notebook (11 sections)
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
# collect_demos.py  ~line 536
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
# collect_demos.py  ~line 546
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
env_cfg.scene.robot.init_state.pos = (7.0, -8.5, 0.5)
```

This places Spot on the warehouse floor with 0.5 m clearance. The warehouse USD has
its floor near Z = 0 world-space. Do not use Z = 0 or lower — Spot spawns inside
the floor and the body-contact termination triggers immediately.

### 5.6 Controller input backend: EvdevGamepad (not Se2Gamepad / carb.input)

**[IMPORTANT]** The script uses a custom `EvdevGamepad` class (defined in
`collect_demos.py`) that reads `/dev/input/js0` directly via the Linux kernel
joystick interface. It does **not** use `Se2Gamepad` or `carb.input`.

**Why:** Isaac Sim's GLFW plugin uses an embedded SDL gamecontroller database
to identify gamepads. Controllers absent from that database are silently
rejected with:
```
Joystick with unknown remapping detected (will be ignored): <name> [<guid>]
```
causing `omni.appwindow.get_default_app_window().get_gamepad(0)` to return a
null handle. The current controller — **PowerA Xbox Series X EnWired**
(VID=20d6, PID=2002) — is not in GLFW's database.

`EvdevGamepad` bypasses GLFW entirely. The `xpad` kernel driver handles this
controller perfectly and exposes it at `/dev/input/js0`. The user must be in
the `input` group (`sudo usermod -aG input $USER`) for read access.

**`dead_zone=0.05`** is applied inside `EvdevGamepad.advance()`. Axis values
below 0.05 are zeroed. If you see stick drift, verify with:
```bash
cat /dev/input/js0 | xxd   # raw bytes, or use jstest if installed
```

**Button constants** (xpad, 0-indexed in order of BTN_* kernel codes):
```python
JSPAD_A = 0   # success
JSPAD_B = 1   # failure / reset
JSPAD_START = 7   # quit
```

**Axis layout** (xpad → `/dev/input/js0`):
```
axis 0  ABS_X   Left stick X   (left=−1, right=+1)
axis 1  ABS_Y   Left stick Y   (up=−1,   down=+1)   ← negated for v_x
axis 3  ABS_RX  Right stick X  (left=−1, right=+1)
```

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
│         reward_config/*    (float — one attr per RewardConfig field)
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
    ├── action/
    │   ├── velocity_cmd     (T, 3)   float32  — gamepad [v_x, v_y, ω_z], post-negation
    │   └── joint_targets    (T, 12)  float32  — raw policy output
    └── reward/
        ├── total            (T,)     float32  — weighted sum
        ├── time_penalty     (T,)     float32  — raw (unweighted) signal
        ├── termination_penalty (T,)  float32  — raw signal (1.0 on last step if failed)
        ├── goal_reached     (T,)     float32  — raw signal (1.0 on last step if success)
        ├── goal_approach    (T,)     float32  — d(t-1) - d(t)
        ├── command_tracking (T,)     float32  — -||vel_cmd - vel_achieved||
        ├── command_smoothness (T,)   float32  — -||delta_vel_cmd||
        ├── body_contact     (T,)     float32  — -sum(||F_nonfoot||)
        └── uprightness      (T,)     float32  — clamp(-grav_z, 0, 1)
```

`T` = number of 10 Hz steps. `contact_forces` has 17 bodies (full Spot articulation),
not just the 4 feet — foot body indices are {3, 7, 11, 15}.

**Reward storage design:** Raw (unweighted) component signals are stored alongside a
pre-computed weighted total. The `RewardConfig` weights used are stored as file-level
attributes (`reward_config/*`). This allows offline re-weighting without re-collecting,
while providing a ready-to-use total for downstream RL/IL training. The same
`rewards.py` module (`StepRewardComputer`) will be used for online DPPO training,
ensuring reward consistency between offline and online phases.

**Backward compatibility:** Datasets collected before the reward integration do not
have a `reward/` group. `validate_dataset.py` and `explore_dataset.ipynb` handle both
formats — missing rewards are computed offline from the obs/action data.

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

Returns exit code 0 (all pass) or 1 (failures). The 14 checks include:
- All dataset keys, shapes, dtypes present
- `velocity_cmd` within training ranges
- `policy_obs[9:12] == velocity_cmd` exactly (data-integrity invariant)
- `goal_relative == goal_world - base_pos` (tolerance 1e-3 m)
- Quaternion norms ≈ 1.0
- Camera not all-black
- No position jumps > 1 m at 10 Hz
- Reward datasets present, consistent length, `total ≈ weighted sum` (when rewards exist)

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
10. Reward computation + component breakdown + cumulative return
11. Body orientation / tilt (roll, pitch, projected gravity)

---

## 9. Known issues and status

| Issue | Status |
|-------|--------|
| All collected episodes end with failure (robot falls) | **Known** — Spot falls on warehouse obstacles; collect more/longer episodes |
| No successful episodes in current dataset | **Known** — goal not yet reached; data is still valid for IL pre-training |
| `time_out` termination disabled | **Intentional** — episode ends only on fall or human button press |
| `terrain_out_of_bounds` disabled | **Intentional** — USD terrain has no procedural bounds |

---

## 10. Reward system

**[IMPORTANT] Termination penalty weight history:**
- Original: `w_termination = -10.0` — too weak. A long failure episode (e.g. T=200 steps) could accumulate enough per-step reward to match or exceed a safe episode, making failure and success distributions indistinguishable.
- Current: `w_termination = -100.0` — 10× increase so a single fall dominates the cumulative return regardless of episode length. In principle, a robot falling or breaking should carry a very high cost.



`rewards.py` defines all 8 reward terms in one place. Two APIs share the same math:

- **`compute_episode_rewards()`** — offline batch over numpy arrays (for HDF5 analysis)
- **`StepRewardComputer`** — online step-by-step (used in `collect_demos.py`, will be
  used by DPPO training loop)

Both use `RewardConfig` (a dataclass) for weights. The config is serializable via
`to_dict()` / `from_dict()` and is stored as HDF5 file-level attributes.

**Raw vs weighted storage:** Component signals are stored *unweighted*. The weighted
total is also stored for convenience. To re-weight offline, just re-run
`compute_episode_rewards()` with a different `RewardConfig`.

---

## 11. What to work on next

The demo collection infrastructure is complete. Likely next steps:

1. **Collect more episodes** — run more sessions until the dataset has enough
   successful episodes (goal reached) for imitation learning.
2. **Distributional PPO (DPPO)** — implement online RL training using the same
   `StepRewardComputer` from `rewards.py` as the reward function. The high-level
   policy should take `camera_rgb` + `goal_relative` as input and output
   `velocity_cmd` [v_x, v_y, ω_z].
3. **Reward weight tuning** — iterate on `RewardConfig` weights using the offline
   analysis in `explore_dataset.ipynb` before committing to DPPO training.

---

## 12. Key source files outside this directory

| File | Relevance |
|------|-----------|
| `source/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/` | Spot env cfg (`SpotFlatEnvCfg`, `SpotFlatEnvCfg_PLAY`) |
| `source/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/mdp/` | `reset_joints_around_default` function |
| `source/isaaclab_tasks/manager_based/locomotion/velocity/config/spot/agents/rsl_rl_ppo_cfg.py` | `SpotFlatPPORunnerCfg` |
| `source/isaaclab/devices/gamepad/se2_gamepad.py` | `Se2Gamepad`, `Se2GamepadCfg` |
| `source/isaaclab/envs/mdp/events.py` | `reset_root_state_uniform` |
